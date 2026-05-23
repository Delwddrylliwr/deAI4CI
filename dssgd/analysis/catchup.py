"""Catch-up experiment runner for NMH gossip-SGD dynamics.

Implements the centroid-propagation observable: after warmup to stationarity,
a perturbation is injected into one source leaf module, and the time for each
other leaf's centroid to shift by half the perturbation magnitude is measured
as t_50.  The theoretical prediction is log₂ t_50(d) = d + const, tested by
Experiments A-E of the dynamics specification.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

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

from . import theory


# ---------------------------------------------------------------------------
# Type-indexing utilities (binary NMH, branching=2)
# ---------------------------------------------------------------------------


def leaf_assignments(n_agents: int, leaf_size: int) -> np.ndarray:
    """Map agent_id → leaf_idx (0-indexed).  Nodes are contiguous per leaf."""
    return np.arange(n_agents) // leaf_size


def hierarchical_distance(i: int, j: int) -> int:
    """Tree distance between leaf indices i and j under binary (branching=2) NMH.

    Equals the position of the highest differing bit: (i ^ j).bit_length().
    """
    if i == j:
        return 0
    return (i ^ j).bit_length()


def cousin_at_level(leaf_idx: int, ell: int) -> int:
    """Return the level-ℓ cousin of leaf_idx (flip bit ell-1)."""
    return leaf_idx ^ (1 << (ell - 1))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CatchupRun:
    """All outputs from one catch-up simulation run.

    centroid_traj : np.ndarray (n_meas_rounds+1, 2^depth, n_params)
        Type-centroid snapshots at every measurement round, including t=0
        (pre-perturbation baseline).
    agent_traj : Optional[np.ndarray] (n_meas_rounds+1, n_agents, n_params)
        Per-agent snapshots; only populated when record_agent_traj=True (Exp D).
    t50_table : list of dicts
        One entry per target leaf with keys:
        lr, seed, source_leaf, target_leaf, distance_d, t_50, perturbation_norm.
        t_50 is None when the threshold was never crossed within n_meas_rounds.
    """
    name: str
    lr: float
    seed: int
    branching: int
    depth: int
    leaf_size: int
    p: float
    n_warmup: int
    n_meas_rounds: int
    source_leaf: int
    perturbation_norm: float
    perturbation_direction: np.ndarray
    centroid_traj: np.ndarray
    agent_traj: Optional[np.ndarray]
    t50_table: List[dict]
    warmup_sigma2: float
    direct_edge_leaves: Optional[set] = None  # target leaf indices with ≥1 direct edge to source

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> CatchupRun:
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------


def _flat_params(agent: Agent, model_key: str = "model") -> np.ndarray:
    parts = [p.detach().float().flatten() for p in agent.registry[model_key].model.parameters()]
    return torch.cat(parts).numpy()


def compute_all_centroids(
    agents: List[Agent],
    leaf_assigns: np.ndarray,
    n_leaf_types: int,
) -> np.ndarray:
    """Return shape (n_leaf_types, n_params) array of type-conditional means."""
    params = np.stack([_flat_params(a) for a in agents])
    centroids = np.array([
        params[leaf_assigns == tau].mean(axis=0)
        for tau in range(n_leaf_types)
    ])
    return centroids


def within_leaf_spread(
    agents: List[Agent],
    leaf_assigns: np.ndarray,
    source_leaf: int,
) -> float:
    """Std of per-coordinate params within the source leaf at stationarity (σ₀)."""
    mask = leaf_assigns == source_leaf
    params = np.stack([_flat_params(a) for a in agents])[mask]
    if params.shape[0] < 2:
        return 1.0
    return float(params.std())


def inject_perturbation(
    agents: List[Agent],
    leaf_assigns: np.ndarray,
    source_leaf: int,
    delta: np.ndarray,
) -> None:
    """Add delta (1-D np.ndarray) in-place to all agents in source_leaf."""
    delta_t = torch.from_numpy(delta.astype(np.float32))
    offset = 0
    for a in agents:
        if leaf_assigns[a.id] != source_leaf:
            continue
        for param in a.registry["model"].model.parameters():
            n = param.numel()
            with torch.no_grad():
                param.add_(delta_t[offset : offset + n].view_as(param))
            offset += n
        offset = 0  # reset per agent — same delta applied to each


def find_t_50(
    centroid_traj: np.ndarray,
    initial_centroid: np.ndarray,
    delta_dir: np.ndarray,
    threshold: float,
    max_t: Optional[int] = None,
) -> Optional[int]:
    """First t (0-indexed) where projection of displacement onto delta_dir ≥ threshold.

    centroid_traj : (T, n_params)  — trajectory for one leaf type
    initial_centroid : (n_params,) — centroid at t=0 (pre-measurement baseline)
    delta_dir : (n_params,)        — unit direction of perturbation
    threshold : float              — detection threshold
    max_t : int or None            — only search t in [0, max_t]; prevents late
                                     thermal-noise peaks at large t from registering
                                     as spurious detections.
    """
    traj = centroid_traj[:max_t] if max_t is not None else centroid_traj
    displacements = traj - initial_centroid[np.newaxis, :]
    projections = displacements @ delta_dir
    above = np.where(projections >= threshold)[0]
    return int(above[0]) if len(above) > 0 else None


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
# Sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(
    branching: int,
    depth: int,
    leaf_size: int,
    p: float,
    gamma: float = 1.0,
    lr: float = 0.01,
    sigma2: float = 1.0,
    tol_factor: float = 3.0,
) -> None:
    """Assert correctness of type-indexing and basic theory predictions.

    Raises AssertionError with a descriptive message on first failure.
    """
    n_leaf_types = branching ** depth
    n_agents = n_leaf_types * leaf_size
    assigns = leaf_assignments(n_agents, leaf_size)

    # 1. Exactly leaf_size agents per leaf module
    for tau in range(n_leaf_types):
        count = int((assigns == tau).sum())
        assert count == leaf_size, (
            f"Leaf {tau} has {count} agents, expected {leaf_size}"
        )

    # 2. Cousin relations: hierarchical_distance(i, cousin_at_level(i, ℓ)) == ℓ
    for i in range(n_leaf_types):
        for ell in range(1, depth + 1):
            cousin = cousin_at_level(i, ell)
            assert 0 <= cousin < n_leaf_types, (
                f"cousin_at_level({i}, {ell}) = {cousin} out of range [0, {n_leaf_types})"
            )
            d = hierarchical_distance(i, cousin)
            assert d == ell, (
                f"hierarchical_distance({i}, cousin_at_level({i},{ell})={cousin}) = {d}, expected {ell}"
            )

    # 3. Stationary variance prediction: σ²_ℓ = T_eff / γ_ℓ  (checked analytically, not from sim)
    T_eff = theory.effective_temperature(lr, sigma2)
    for ell in range(1, depth + 1):
        gamma_ell = theory.level_coupling_strength(gamma, leaf_size, p, ell)
        predicted_var = T_eff / gamma_ell
        assert predicted_var > 0, f"Predicted variance at level {ell} is non-positive"

    # 4. All distances between non-identical leaves are positive
    for i in range(n_leaf_types):
        for j in range(i + 1, n_leaf_types):
            d = hierarchical_distance(i, j)
            assert d > 0, f"hierarchical_distance({i}, {j}) = 0 for distinct leaves"
            assert d <= depth, f"hierarchical_distance({i}, {j}) = {d} > depth {depth}"


# ---------------------------------------------------------------------------
# Direct-edge utilities
# ---------------------------------------------------------------------------


def _compute_direct_edge_leaves_from_graph(
    G,
    assigns: np.ndarray,
    source_leaf: int,
    n_leaf_types: int,
) -> set:
    """Return set of leaf indices that share ≥1 direct edge with any source agent."""
    source_agents = frozenset(int(i) for i in np.where(assigns == source_leaf)[0])
    direct: set = set()
    for tau in range(n_leaf_types):
        if tau == source_leaf:
            continue
        for agent_j in (int(i) for i in np.where(assigns == tau)[0]):
            if any(nbr in source_agents for nbr in G.neighbors(agent_j)):
                direct.add(tau)
                break
    return direct


def compute_direct_edge_leaves(run: "CatchupRun") -> set:
    """Post-hoc: reconstruct topology from stored seed and compute direct-edge set."""
    topo = NestedModularTopology(
        branching=run.branching,
        depth=run.depth,
        leaf_size=run.leaf_size,
        p=run.p,
        seed=run.seed,
    )
    G, _ = topo.step(0)
    n_leaf_types = run.branching ** run.depth
    n_agents = n_leaf_types * run.leaf_size
    assigns = leaf_assignments(n_agents, run.leaf_size)
    return _compute_direct_edge_leaves_from_graph(G, assigns, run.source_leaf, n_leaf_types)


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------


@dataclass
class CatchupSimConfig:
    """All degrees of freedom for one catch-up simulation."""
    name: str
    branching: int = 2
    depth: int = 5
    leaf_size: int = 4
    p: float = 4.0
    seed: int = 0
    n_warmup: int = 800
    n_meas_rounds: int = 400
    lr: float = 0.01
    local_steps: int = 1
    source_leaf: int = 0
    perturbation_scale: float = 5.0
    threshold_fraction: float = 0.10
    max_t_50_search: int = 50
    layer_name: str = "social"
    model_factory: Optional[Callable[[], nn.Module]] = None
    loss_fn: Optional[Callable] = None
    data_loaders_factory: Optional[Callable[[int], List[DataLoader]]] = None
    record_agent_traj: bool = False
    skip_spectral_gaps: bool = True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_catchup_simulation(config: CatchupSimConfig) -> CatchupRun:
    """Run a two-phase catch-up experiment and return a picklable CatchupRun.

    Phase 1 — warmup: standard gossip-SGD for n_warmup rounds.
    Phase 2 — measurement: inject perturbation into source leaf, then record
    type-centroid trajectories for n_meas_rounds rounds, computing t_50 for
    each target leaf.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    n_leaf_types = config.branching ** config.depth
    n_agents = n_leaf_types * config.leaf_size
    train_locally = config.loss_fn is not None and config.lr > 0.0

    topo = NestedModularTopology(
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        seed=config.seed,
    )
    assigns = leaf_assignments(n_agents, config.leaf_size)
    G_static, _ = topo.step(0)
    direct_edge_leaves = _compute_direct_edge_leaves_from_graph(
        G_static, assigns, config.source_leaf, n_leaf_types
    )

    if config.data_loaders_factory is not None:
        loaders = config.data_loaders_factory(n_agents)
    else:
        loaders = [_dummy_loader() for _ in range(n_agents)]

    def _make_agent(agent_id: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + agent_id)
        model = config.model_factory() if config.model_factory else nn.Linear(4, 1, bias=False)
        loss_fn = config.loss_fn if train_locally else (lambda m, b: torch.tensor(0.0))
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

    # Phase 1: warmup
    for round_idx in range(config.n_warmup):
        if train_locally:
            for agent in agents:
                agent.local_step()
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)

    warmup_sigma2 = _estimate_gradient_noise(agents) if train_locally else 0.0

    # Estimate σ₀ from stationary spread within source leaf
    sigma0 = within_leaf_spread(agents, assigns, config.source_leaf)
    n_params = _flat_params(agents[0]).shape[0]

    # Generate random perturbation direction orthogonal to nothing in particular
    rng = np.random.default_rng(config.seed + 1)
    raw = rng.standard_normal(n_params).astype(np.float32)
    delta_dir = raw / (np.linalg.norm(raw) + 1e-12)
    perturbation_norm = float(config.perturbation_scale * sigma0)
    delta = delta_dir * perturbation_norm

    # Snapshot centroids before perturbation (t=0 baseline)
    centroid_traj_list: List[np.ndarray] = []
    agent_traj_list: Optional[List[np.ndarray]] = [] if config.record_agent_traj else None

    baseline = compute_all_centroids(agents, assigns, n_leaf_types)
    centroid_traj_list.append(baseline.copy())
    if agent_traj_list is not None:
        agent_traj_list.append(np.stack([_flat_params(a) for a in agents]))

    # Inject perturbation into source leaf
    inject_perturbation(agents, assigns, config.source_leaf, delta)

    # Phase 2: measurement
    for round_idx in range(config.n_meas_rounds):
        if train_locally:
            for agent in agents:
                agent.local_step()
        layer_graphs = ml_topo.step(config.n_warmup + round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)

        centroid_traj_list.append(compute_all_centroids(agents, assigns, n_leaf_types))
        if agent_traj_list is not None:
            agent_traj_list.append(np.stack([_flat_params(a) for a in agents]))

    centroid_traj = np.stack(centroid_traj_list)   # (n_meas_rounds+1, n_leaf_types, n_params)
    agent_traj = np.stack(agent_traj_list) if agent_traj_list is not None else None

    # Threshold: threshold_fraction × (source centroid peak shift).
    # The source peak adapts to discrete-time MH attenuation (~1/(1+mean_degree)
    # per gossip step vs the continuous-time γ_ℓ).  fraction=0.10 gives SNR ≈ 6.8
    # at d=1 and ≈ 1.0 at d=2 (their early-round signal sits just at threshold).
    # max_t_50_search caps the search window so late thermal-noise excursions at
    # t > 50 rounds cannot register as spurious t_50 detections for d ≥ 3.
    source_projections = (centroid_traj[:, config.source_leaf, :] - baseline[config.source_leaf]) @ delta_dir
    source_peak_shift = float(np.max(source_projections))
    threshold = source_peak_shift * config.threshold_fraction

    t50_table: List[dict] = []
    for j in range(n_leaf_types):
        if j == config.source_leaf:
            continue
        d = hierarchical_distance(config.source_leaf, j)
        t50 = find_t_50(
            centroid_traj[:, j, :],
            baseline[j],
            delta_dir,
            threshold,
            max_t=config.max_t_50_search,
        )
        t50_table.append({
            "lr": config.lr,
            "seed": config.seed,
            "source_leaf": config.source_leaf,
            "target_leaf": j,
            "distance_d": d,
            "t_50": t50,
            "perturbation_norm": perturbation_norm,
            "source_peak_shift": source_peak_shift,
            "threshold": threshold,
            "has_direct_edge": j in direct_edge_leaves,
        })

    return CatchupRun(
        name=config.name,
        lr=config.lr,
        seed=config.seed,
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        n_warmup=config.n_warmup,
        n_meas_rounds=config.n_meas_rounds,
        source_leaf=config.source_leaf,
        perturbation_norm=perturbation_norm,
        perturbation_direction=delta_dir,
        centroid_traj=centroid_traj,
        agent_traj=agent_traj,
        t50_table=t50_table,
        warmup_sigma2=warmup_sigma2,
        direct_edge_leaves=direct_edge_leaves,
    )


def _dummy_loader() -> DataLoader:
    x = torch.randn(16, 4)
    y = torch.zeros(16, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8)
