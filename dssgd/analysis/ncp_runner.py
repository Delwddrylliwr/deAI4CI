"""NCP (Network Core Periphery) Regime C simulation runner.

Uses a ForestFireTopology whose k-core shell decomposition defines the
NCP structure.  One agent per node (no within-shell clique grouping).
State is tracked at both node level (centroid_traj) and shell level
(shell_traj computed post-hoc).

The clamped_shell mechanism holds a chosen shell fixed at basin B
throughout the measurement phase, enabling asymmetric nucleation
experiments (NCP-2).
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import AsynchronousGossip, SynchronousPairwiseGossip
from dssgd.topology.forest_fire import ForestFireTopology
from dssgd.topology.multilayer import MultiLayerTopology

from .active_escape import (
    BistableParameterModule,
    _dummy_loader_bistable,
    basin_label,
    find_t_flip,
    make_bistable_loss_fn,
    verify_bistable_loss,
)


# ---------------------------------------------------------------------------
# Config and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NCPSimConfig:
    """Degrees of freedom for one NCP Regime C simulation."""

    name: str
    n_nodes: int = 200
    p_f: float = 0.35
    r: float = 0.5
    seed: int = 0
    n_warmup: int = 400
    n_meas_rounds: int = 800
    lr: float = 0.1
    local_steps: int = 50
    d_param: int = 1
    theta_A: Optional[np.ndarray] = None
    theta_B: Optional[np.ndarray] = None
    a: float = 0.5
    b: float = 0.042
    epsilon: float = 0.2
    persistence: int = 3
    layer_name: str = "social"
    # NCP-2: clamp this shell to B throughout measurement (None = free run)
    clamped_shell: Optional[int] = None
    gossip_protocol: str = "async_poisson"   # "async_poisson" or "synchronous"
    gossip_rate: Optional[float] = None     # None → auto-set to n_nodes (see runner)
    gossip_alpha: float = 0.5               # initiator mixing weight toward the neighbour


@dataclass
class NCPRun:
    """All outputs from one NCP simulation run.

    centroid_traj : (n_meas_rounds+1, n_nodes, d_param) — per-node parameters.
    shell_traj    : {shell_idx: (n_meas_rounds+1, d_param)} — per-shell means.
    graph_stats   : {n_nodes, n_shells, mean_degree, max_shell}.
    """

    name: str
    seed: int
    n_nodes: int
    p_f: float
    r: float
    a: float
    b: float
    d_param: int
    local_steps: int
    n_warmup: int
    n_meas_rounds: int
    theta_A: np.ndarray
    theta_B: np.ndarray
    centroid_traj: np.ndarray          # (T+1, n_nodes, d_param)
    shell_assigns: Dict[int, int]      # node → k-core shell
    shell_traj: Dict[int, np.ndarray]  # shell → (T+1, d_param)
    warmup_ok: bool
    nucleation_node: Optional[int]
    t_nucleation: Optional[int]
    flip_table: List[dict]             # per-node flip info
    graph_stats: dict
    # NCP-2 (Prop. 5.2): which shell was held at B throughout measurement,
    # None for a free (unclamped) run. Read by compute_ncp2_directionality
    # to classify a run as an outward (clamped == max_shell) or inward
    # (clamped == min shell) directionality trial.
    clamped_shell: Optional[int] = None

    def save(self, path: Union[str, Path]) -> None:
        p_obj = Path(path)
        p_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(p_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NCPRun":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Shell trajectory helper
# ---------------------------------------------------------------------------


def compute_shell_trajs(
    centroid_traj: np.ndarray,
    shell_assigns: Dict[int, int],
) -> Dict[int, np.ndarray]:
    """Compute per-shell mean parameter trajectories.

    Parameters
    ----------
    centroid_traj : (T, n_nodes, d_param)
    shell_assigns : {node_id: shell_idx}

    Returns
    -------
    {shell_idx: (T, d_param)} — mean over nodes in each shell.
    """
    shells = sorted(set(shell_assigns.values()))
    result: Dict[int, np.ndarray] = {}
    for s in shells:
        nodes = [node for node, sh in shell_assigns.items() if sh == s]
        result[s] = centroid_traj[:, nodes, :].mean(axis=1)
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_ncp_simulation(config: NCPSimConfig) -> NCPRun:
    """Two-phase NCP Regime C simulation.

    Phase 1 — warmup: gossip-SGD for n_warmup rounds from near theta_A.
    Phase 2 — measurement: n_meas_rounds with optional shell clamping.

    Timescale separation: local_steps gradient steps precede each gossip
    event (same Regime C mechanism as natural_cascade.py).
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    theta_A_np = (
        config.theta_A if config.theta_A is not None
        else np.zeros(config.d_param, dtype=np.float32)
    )
    theta_B_np = (
        config.theta_B if config.theta_B is not None
        else np.array([1.0] + [0.0] * (config.d_param - 1), dtype=np.float32)
    )
    theta_A_t = torch.from_numpy(theta_A_np)
    theta_B_t = torch.from_numpy(theta_B_np)

    require_barrier = (config.b == 0.0)
    verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b,
                         require_barrier=require_barrier)

    # Build Forest Fire topology
    ff_topo = ForestFireTopology(
        n=config.n_nodes,
        p_f=config.p_f,
        r=config.r,
        seed=config.seed,
        ensure_connected=True,
    )
    G, _ = ff_topo.step(0)
    shell_assigns = ff_topo.shell_assignment()

    # Graph stats
    degrees = [d for _, d in G.degree()]
    graph_stats = {
        "n_nodes": config.n_nodes,
        "n_shells": len(set(shell_assigns.values())),
        "mean_degree": float(np.mean(degrees)),
        "max_shell": max(shell_assigns.values()),
    }

    # Build agents — one per node
    loss_fn = make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b)
    loaders = [_dummy_loader_bistable() for _ in range(config.n_nodes)]

    def _make_agent(node_id: int) -> Agent:
        torch.manual_seed(config.seed * 100000 + node_id)
        init_noise = torch.randn(config.d_param) * 0.05
        init_val = theta_A_t + init_noise
        model = BistableParameterModule(d_param=config.d_param, init_value=init_val)
        entry = ModelEntry(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
            loss_fn=loss_fn,
            layers=frozenset({config.layer_name}),
            train_locally=True,
            local_steps=1,   # runner controls gradient step granularity
        )
        return Agent(node_id, ModelRegistry({"model": entry}), loaders[node_id])

    agents = [_make_agent(i) for i in range(config.n_nodes)]
    ml_topo = MultiLayerTopology({config.layer_name: ff_topo})
    compositor = CoupledCompositor()
    if config.gossip_protocol == "async_poisson":
        per_step_rate = (
            config.gossip_rate if config.gossip_rate is not None else float(config.n_nodes)
        ) / float(config.local_steps)
        protocol = AsynchronousGossip(
            rate=per_step_rate,
            mode="poisson",
            alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )
    else:
        protocol = SynchronousPairwiseGossip(
            alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )

    # Phase 1: warmup
    for round_idx in range(config.n_warmup):
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)

    # Post-warmup check
    rng = np.random.default_rng(config.seed + 1)
    post_warmup_params = _snapshot_params(agents, config.d_param)
    warmup_ok = all(
        basin_label(post_warmup_params[i], theta_A_np, theta_B_np, config.epsilon) == 'A'
        for i in range(config.n_nodes)
    )

    # Clamped nodes: those in the specified shell
    clamped_nodes = (
        [node for node, sh in shell_assigns.items() if sh == config.clamped_shell]
        if config.clamped_shell is not None else []
    )

    # Phase 2: measurement
    traj_list: List[np.ndarray] = [post_warmup_params.copy()]

    for round_idx in range(config.n_meas_rounds):
        layer_graphs = ml_topo.step(config.n_warmup + round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)

        # Re-clamp shell if specified
        if clamped_nodes:
            _clamp_nodes(agents, clamped_nodes, theta_B_t, rng)

        traj_list.append(_snapshot_params(agents, config.d_param))

    centroid_traj = np.stack(traj_list)  # (T+1, n_nodes, d_param)
    shell_traj = compute_shell_trajs(centroid_traj, shell_assigns)

    # Post-hoc nucleation detection
    t_flip_per_node: List[Optional[int]] = [
        find_t_flip(centroid_traj[:, i, :], theta_A_np, theta_B_np,
                    config.epsilon, config.persistence)
        if warmup_ok else None
        for i in range(config.n_nodes)
    ]

    nucleation_node: Optional[int] = None
    t_nucleation: Optional[int] = None
    valid = [(i, t) for i, t in enumerate(t_flip_per_node) if t is not None]
    if valid:
        nucleation_node, t_nucleation = min(valid, key=lambda x: x[1])

    flip_table: List[dict] = [
        {
            "node": i,
            "shell": shell_assigns[i],
            "t_flip_absolute": t_flip_per_node[i],
            "t_flip_relative": (
                t_flip_per_node[i] - t_nucleation
                if (t_flip_per_node[i] is not None and t_nucleation is not None)
                else None
            ),
            "warmup_ok": warmup_ok,
        }
        for i in range(config.n_nodes)
        if i != nucleation_node
    ]

    return NCPRun(
        name=config.name,
        seed=config.seed,
        n_nodes=config.n_nodes,
        p_f=config.p_f,
        r=config.r,
        a=config.a,
        b=config.b,
        d_param=config.d_param,
        local_steps=config.local_steps,
        n_warmup=config.n_warmup,
        n_meas_rounds=config.n_meas_rounds,
        theta_A=theta_A_np,
        theta_B=theta_B_np,
        centroid_traj=centroid_traj,
        shell_assigns=shell_assigns,
        shell_traj=shell_traj,
        warmup_ok=warmup_ok,
        nucleation_node=nucleation_node,
        t_nucleation=t_nucleation,
        flip_table=flip_table,
        graph_stats=graph_stats,
        clamped_shell=config.clamped_shell,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _snapshot_params(agents: List[Agent], d_param: int) -> np.ndarray:
    """Return (n_nodes, d_param) array of current agent parameters."""
    result = np.zeros((len(agents), d_param), dtype=np.float32)
    for agent in agents:
        params = list(agent.registry["model"].model.parameters())
        vec = torch.cat([p.detach().flatten() for p in params]).numpy()
        result[agent.id] = vec[:d_param]
    return result


def _clamp_nodes(
    agents: List[Agent],
    node_ids: List[int],
    theta_B: torch.Tensor,
    rng: np.random.Generator,
    noise_scale: float = 0.01,
) -> None:
    """Set the parameters of specified nodes to theta_B + small noise."""
    for agent in agents:
        if agent.id not in node_ids:
            continue
        with torch.no_grad():
            for p in agent.registry["model"].model.parameters():
                noise = torch.from_numpy(
                    rng.standard_normal(p.numel()).astype(np.float32) * noise_scale
                ).view_as(p)
                p.copy_(theta_B.view_as(p) + noise)
