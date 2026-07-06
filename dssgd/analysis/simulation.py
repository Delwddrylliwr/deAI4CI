"""NMH-aware gossip-SGD simulation runner.

Wraps the dssgd Agent / Simulator infrastructure with hierarchy-level
bookkeeping: per-level consensus distances, parameter snapshots, spectral
gaps, and gradient-noise estimation for effective-temperature calculations.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import GossipAveraging
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology

from .results import AnalysisRun, LevelTrace, TopologyStats
from . import theory


@dataclass
class NMHSimConfig:
    """All degrees of freedom for one NMH simulation experiment."""
    name: str
    branching: int
    depth: int
    leaf_size: int
    p: float
    seed: int = 0
    n_rounds: int = 200
    lr: float = 0.0
    local_steps: int = 1
    snapshot_every: int = 10
    layer_name: str = "social"
    model_factory: Optional[Callable[[], nn.Module]] = None
    loss_fn: Optional[Callable] = None
    data_loaders_factory: Optional[Callable[[int], List[DataLoader]]] = None
    start_from_consensus: bool = False
    skip_spectral_gaps: bool = False
    n_warmup: int = 0
    escape_data_loaders_factory: Optional[Callable[[int], List[DataLoader]]] = None


# ---------------------------------------------------------------------------
# Internal measurement helpers
# ---------------------------------------------------------------------------

def _flat_params(agent: Agent, model_key: str = "model") -> np.ndarray:
    parts = [p.detach().float().flatten() for p in agent.registry[model_key].model.parameters()]
    return torch.cat(parts).numpy()


def _consensus_distance(agents: List[Agent], ids: List[int]) -> float:
    subset = [a for a in agents if a.id in set(ids)]
    if len(subset) < 2:
        return 0.0
    states = [
        torch.cat([p.detach().float().flatten() for p in a.registry["model"].model.parameters()])
        for a in subset
    ]
    stacked = torch.stack(states)
    mean = stacked.mean(0)
    return float(torch.stack([(s - mean).norm(2) for s in stacked]).mean())


def _module_centroid(
    agents: List[Agent], ids: List[int]
) -> Optional[torch.Tensor]:
    subset = [a for a in agents if a.id in set(ids)]
    if not subset:
        return None
    states = [
        torch.cat([p.detach().float().flatten() for p in a.registry["model"].model.parameters()])
        for a in subset
    ]
    return torch.stack(states).mean(0)


def _cross_centroid_distance(
    agents: List[Agent], modules: List[List[int]]
) -> float:
    centroids = [_module_centroid(agents, mod) for mod in modules]
    centroids = [c for c in centroids if c is not None]
    if len(centroids) < 2:
        return 0.0
    stacked = torch.stack(centroids)
    grand = stacked.mean(0)
    return float(torch.stack([(c - grand).norm(2) for c in centroids]).mean())


def _spectral_gap(W: np.ndarray) -> float:
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    return float(1.0 - eigvals[1]) if len(eigvals) > 1 else 1.0


def _measure_topology(
    topo: NestedModularTopology, config: NMHSimConfig
) -> TopologyStats:
    G = topo._G
    W = topo._W
    degrees = np.array([d for _, d in G.degree()])
    modules_by_level = theory.nmh_modules_by_level(
        config.branching, config.depth, config.leaf_size
    )

    level_expected: List[float] = []
    level_observed: List[float] = []

    for level in range(1, config.depth + 1):
        level_expected.append(
            theory.expected_level_neighbours(
                config.p, config.branching, config.leaf_size, level
            )
        )
        prev_mods = modules_by_level[level - 1]
        curr_mods = modules_by_level[level]
        node_to_prev = {nid: m for m, mod in enumerate(prev_mods) for nid in mod}
        node_to_curr = {nid: m for m, mod in enumerate(curr_mods) for nid in mod}

        cross_counts: List[float] = []
        for node in G.nodes():
            own_prev = node_to_prev[node]
            own_curr = node_to_curr[node]
            count = sum(
                1
                for nb in G.neighbors(node)
                if node_to_prev[nb] != own_prev and node_to_curr[nb] == own_curr
            )
            cross_counts.append(float(count))
        level_observed.append(float(np.mean(cross_counts)))

    # BFS cumulative neighbourhood growth: n(r) = mean |{nodes within distance r}|
    n_nodes = G.number_of_nodes()
    dist_hist: dict = {}
    for node in G.nodes():
        for target, dist in nx.single_source_shortest_path_length(G, node).items():
            if dist > 0:
                dist_hist[dist] = dist_hist.get(dist, 0) + 1
    nbhd_by_r: List[float] = []
    if dist_hist:
        cumulative = 0
        for r in range(1, max(dist_hist.keys()) + 1):
            cumulative += dist_hist.get(r, 0)
            nbhd_by_r.append(cumulative / n_nodes)

    return TopologyStats(
        n_nodes=n_nodes,
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        seed=config.seed,
        degrees=degrees,
        level_expected_neighbours=level_expected,
        level_observed_neighbours=level_observed,
        spectral_gap=_spectral_gap(W),
        mean_nbhd_by_radius=nbhd_by_r,
    )


def _default_dummy_loader(model_dim: int = 4) -> DataLoader:
    x = torch.randn(16, model_dim)
    y = torch.zeros(16, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8)


# ---------------------------------------------------------------------------
# Synthetic regression helpers (for training-enabled experiments)
# ---------------------------------------------------------------------------


def make_regression_loaders(
    n_agents: int,
    branching: int,
    depth: int,
    leaf_size: int,
    n_dim: int = 8,
    noise_std: float = 0.5,
    n_samples: int = 64,
    batch_size: int = 16,
) -> List[DataLoader]:
    """Per-agent regression data with module-specific true weight vectors.

    Agents in the same leaf module share a true weight, creating within-module
    homogeneity and cross-module heterogeneity — the type-heterogeneous landscape
    the SDE theory describes.
    """
    modules = theory.nmh_modules_by_level(branching, depth, leaf_size)[0]
    torch.manual_seed(0)
    module_weights = torch.stack([torch.randn(n_dim) for _ in range(len(modules))])
    node_to_mod = {nid: m for m, mod in enumerate(modules) for nid in mod}

    loaders: List[DataLoader] = []
    for agent_id in range(n_agents):
        true_w = module_weights[node_to_mod[agent_id]]
        torch.manual_seed(1000 + agent_id)
        x = torch.randn(n_samples, n_dim)
        y = x @ true_w + noise_std * torch.randn(n_samples)
        loaders.append(
            DataLoader(TensorDataset(x, y.unsqueeze(1)), batch_size=batch_size, shuffle=True)
        )
    return loaders


def make_homogeneous_loaders(
    n_agents: int,
    n_dim: int = 8,
    noise_std: float = 0.5,
    n_samples: int = 64,
    batch_size: int = 16,
) -> List[DataLoader]:
    """Per-agent regression data with a shared zero true weight vector.

    Removes the module-specific gradient bias (heterogeneous pinning), leaving
    only gradient noise.  Used in Phase 2 of the two-phase Kramers test to
    dissolve the cluster structure established during heterogeneous warmup.
    """
    true_w = torch.zeros(n_dim)
    loaders: List[DataLoader] = []
    for agent_id in range(n_agents):
        torch.manual_seed(3000 + agent_id)
        x = torch.randn(n_samples, n_dim)
        y = x @ true_w + noise_std * torch.randn(n_samples)
        loaders.append(
            DataLoader(TensorDataset(x, y.unsqueeze(1)), batch_size=batch_size, shuffle=True)
        )
    return loaders


def regression_loss(model: nn.Module, batch) -> torch.Tensor:
    x, y = batch
    return F.mse_loss(model(x), y)


def _estimate_gradient_noise(agents: List[Agent]) -> float:
    """Estimate per-coordinate gradient variance from 4 forward passes per agent."""
    variances: List[float] = []
    for agent in agents:
        entry = agent.registry["model"]
        if not entry.train_locally:
            continue
        grads: List[torch.Tensor] = []
        for _ in range(4):
            batch = agent._next_batch()
            entry.optimizer.zero_grad()
            loss = entry.loss_fn(entry.model, batch)
            loss.backward()
            g_parts = [
                p.grad.float().flatten()
                for p in entry.model.parameters()
                if p.grad is not None
            ]
            if g_parts:
                grads.append(torch.cat(g_parts).detach())
        if len(grads) >= 2:
            stacked = torch.stack(grads)
            variances.append(float(stacked.var(0).mean()))
    return float(np.mean(variances)) if variances else 0.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_nmh_simulation(config: NMHSimConfig) -> AnalysisRun:
    """Run a complete NMH gossip-SGD experiment; return a picklable AnalysisRun."""
    n_agents = config.branching ** config.depth * config.leaf_size
    train_locally = config.loss_fn is not None and config.lr > 0.0

    topo = NestedModularTopology(
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        seed=config.seed,
    )
    topology_stats = _measure_topology(topo, config)
    modules_by_level = theory.nmh_modules_by_level(
        config.branching, config.depth, config.leaf_size
    )

    if config.data_loaders_factory is not None:
        loaders = config.data_loaders_factory(n_agents)
    else:
        loaders = [_default_dummy_loader() for _ in range(n_agents)]

    def _make_agent(agent_id: int) -> Agent:
        # All agents share the same init seed when start_from_consensus=True
        init_seed = config.seed * 10000 if config.start_from_consensus else config.seed * 10000 + agent_id
        torch.manual_seed(init_seed)
        model = (
            config.model_factory()
            if config.model_factory
            else nn.Linear(4, 1, bias=False)
        )
        loss_fn = (
            config.loss_fn
            if train_locally
            else (lambda m, b: torch.tensor(0.0))
        )
        entry = ModelEntry(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
            loss_fn=loss_fn,
            layers=frozenset({config.layer_name}),
            train_locally=train_locally,
            local_steps=config.local_steps,
        )
        return Agent(agent_id, ModelRegistry({"model": entry}), loaders[agent_id])

    agents = [_make_agent(i) for i in range(n_agents)]
    ml_topo = MultiLayerTopology({config.layer_name: topo})
    compositor = CoupledCompositor()
    protocol = GossipAveraging()

    level_trace = LevelTrace(
        within_distances={lv: [] for lv in range(config.depth + 1)},
        cross_distances={lv: [] for lv in range(1, config.depth + 1)},
        global_distances=[],
    )
    snapshots: Dict[int, Dict[int, np.ndarray]] = {}
    spectral_gaps: List[float] = []

    use_two_phase = (
        config.n_warmup > 0
        and config.escape_data_loaders_factory is not None
    )

    for round_idx in range(config.n_rounds):
        # Phase transition: swap to homogeneous loaders at warmup boundary
        if use_two_phase and round_idx == config.n_warmup:
            escape_loaders = config.escape_data_loaders_factory(n_agents)
            for agent in agents:
                agent.data_loader = escape_loaders[agent.id]
                agent._data_iter = iter(agent.data_loader)

        if train_locally:
            for agent in agents:
                agent.local_step()

        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)

        _, W = list(layer_graphs.values())[0]
        if not config.skip_spectral_gaps:
            spectral_gaps.append(_spectral_gap(W))

        for level, mods in modules_by_level.items():
            dists = [_consensus_distance(agents, mod) for mod in mods if len(mod) >= 2]
            level_trace.within_distances[level].append(
                float(np.mean(dists)) if dists else 0.0
            )

        for level in range(1, config.depth + 1):
            level_trace.cross_distances[level].append(
                _cross_centroid_distance(agents, modules_by_level[level])
            )

        level_trace.global_distances.append(
            _consensus_distance(agents, list(range(n_agents)))
        )

        if round_idx % config.snapshot_every == 0:
            snapshots[round_idx] = {a.id: _flat_params(a) for a in agents}

    sigma2 = _estimate_gradient_noise(agents) if train_locally else 0.0

    return AnalysisRun(
        name=config.name,
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        seed=config.seed,
        n_rounds=config.n_rounds,
        lr=config.lr,
        sigma2=sigma2,
        warmup_rounds=config.n_warmup,
        topology=topology_stats,
        level_trace=level_trace,
        snapshots=snapshots,
        spectral_gaps=spectral_gaps,
    )
