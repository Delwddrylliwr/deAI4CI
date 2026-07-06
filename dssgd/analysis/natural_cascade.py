"""Natural-cascade runner for Regime C (separation-of-timescales) experiments.

Implements gossip-SGD with local_steps >> 1 and no source clamping.  In this
regime each agent fully relaxes to its local basin attractor between gossip
events, so the state space collapses to {A,B}^N (Regime C / PDMP limit).

Unlike run_active_escape_simulation, there is no force-flip: the first B-entry
is detected post-hoc from the centroid trajectory.  All flip times are indexed
relative to t_nucleation (the round at which the first leaf entered B).

Supports per-leaf loss functions for heterogeneous-landscape experiments (NMH-6).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import AsynchronousGossip, GossipAveraging
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology

from . import theory
from .active_escape import (
    BistableParameterModule,
    _dummy_loader_bistable,
    basin_label,
    find_t_flip,
    force_flip_module,
    make_bistable_loss_fn,
    verify_bistable_loss,
)
from .catchup import (
    _flat_params,
    compute_all_centroids,
    hierarchical_distance,
    leaf_assignments,
)


# ---------------------------------------------------------------------------
# Config and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NaturalCascadeConfig:
    """All degrees of freedom for one natural-cascade Regime C simulation."""

    name: str
    branching: int = 2
    depth: int = 5
    leaf_size: int = 4
    p: float = 2.0
    seed: int = 0
    n_warmup: int = 400
    n_meas_rounds: int = 1000
    lr: float = 0.1
    local_steps: int = 50          # Regime C knob: gradient steps per gossip round
    d_param: int = 1
    theta_A: Optional[np.ndarray] = None   # default: zeros(d_param)
    theta_B: Optional[np.ndarray] = None   # default: e_1
    a: float = 0.5
    b: float = 0.042               # near bistability limit; override per experiment
    epsilon: float = 0.2
    persistence: int = 3
    flip_noise_scale: float = 0.0   # Langevin noise std per gossip round; 0 = pure gossip cascade
    force_flip_source: bool = False  # if True, force-flip one random leaf post-warmup (zero noise)
    layer_name: str = "social"
    gossip_protocol: str = "async_poisson"   # "async_poisson" or "synchronous"
    gossip_rate: Optional[float] = None     # None → auto-set to n_agents (see runner)
    gossip_alpha: float = 0.5               # initiator mixing weight toward the neighbour
    # NMH-6 heterogeneous loss: one (a_i, b_i) per leaf; None = uniform
    per_leaf_loss_params: Optional[List[Tuple[float, float]]] = None
    t_horizon: int = 800           # round index used by cascade_size observable


@dataclass
class NaturalCascadeRun:
    """All outputs from one natural-cascade simulation.

    centroid_traj : np.ndarray (n_meas_rounds+1, n_leaf_types, d_param)
        Type-centroid snapshots at every measurement round including t=0.
    flip_table : list of dicts
        One entry per leaf (excluding nucleation_leaf when nucleation_leaf
        is not None).  Keys: target_leaf, distance_from_source,
        t_flip_absolute (int or None), t_flip_relative (int or None),
        warmup_ok (bool).
    nucleation_leaf : int or None
        Leaf index that first entered basin B.  None if no leaf flipped.
    t_nucleation : int or None
        Absolute measurement round of first B-entry.
    """

    name: str
    lr: float
    seed: int
    branching: int
    depth: int
    leaf_size: int
    p: float
    a: float
    b: float
    d_param: int
    local_steps: int
    n_warmup: int
    n_meas_rounds: int
    theta_A: np.ndarray
    theta_B: np.ndarray
    centroid_traj: np.ndarray      # (n_meas_rounds+1, n_leaf_types, d_param)
    warmup_ok: bool
    nucleation_leaf: Optional[int]
    t_nucleation: Optional[int]
    flip_table: List[dict]
    regime: str
    ell_c: int

    def save(self, path: Union[str, Path]) -> None:
        p_obj = Path(path)
        p_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(p_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NaturalCascadeRun":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_natural_cascade_simulation(
    config: NaturalCascadeConfig,
) -> NaturalCascadeRun:
    """Two-phase Regime C simulation with post-hoc nucleation detection.

    Phase 1 — warmup: gossip-SGD for n_warmup rounds from near theta_A.
    Phase 2 — measurement: n_meas_rounds with no force-flip and no clamping.
        The full centroid trajectory is recorded.  The first leaf to enter
        basin B is identified post-hoc and used as the cascade source.

    Timescale separation: local_steps gradient steps precede each gossip
    event, ensuring full basin relaxation before the next kick (Regime C).
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Resolve theta_A / theta_B defaults
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

    # For b=0 (NMH-7 symmetric), L(A)=L(B)=0 exactly — verify_bistable_loss
    # rejects this since it checks L(B) < L(A).  The symmetric quartic is valid
    # by construction when a>0, so skip verification for b=0.
    # For b>0 (all other experiments), skip barrier check since b near the
    # bistability limit makes A metastable rather than a local min.
    if config.b != 0.0:
        verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b,
                             require_barrier=False)

    regime = theory.classify_regime(
        config.a, 1.0, config.leaf_size, config.p, config.depth
    )
    ell_c = theory.critical_level(
        config.a, 1.0, config.leaf_size, config.p, config.depth
    )

    n_leaf_types = config.branching ** config.depth
    n_agents = n_leaf_types * config.leaf_size
    assigns = leaf_assignments(n_agents, config.leaf_size)

    topo = NestedModularTopology(
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        seed=config.seed,
    )

    # Build per-leaf loss functions
    if config.per_leaf_loss_params is None:
        uniform_loss = make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b)
        loss_fn_for_leaf: Dict[int, Callable] = {
            tau: uniform_loss for tau in range(n_leaf_types)
        }
    else:
        if len(config.per_leaf_loss_params) != n_leaf_types:
            raise ValueError(
                f"per_leaf_loss_params has {len(config.per_leaf_loss_params)} entries "
                f"but there are {n_leaf_types} leaf types."
            )
        loss_fn_for_leaf = {
            tau: make_bistable_loss_fn(theta_A_t, theta_B_t, a_i, b_i)
            for tau, (a_i, b_i) in enumerate(config.per_leaf_loss_params)
        }

    loaders = [_dummy_loader_bistable() for _ in range(n_agents)]

    def _make_agent(agent_id: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + agent_id)
        init_noise = torch.randn(config.d_param) * 0.05
        init_val = theta_A_t + init_noise
        model = BistableParameterModule(d_param=config.d_param, init_value=init_val)
        leaf_idx = int(assigns[agent_id])
        entry = ModelEntry(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
            loss_fn=loss_fn_for_leaf[leaf_idx],
            layers=frozenset({config.layer_name}),
            train_locally=True,
            local_steps=1,   # runner controls gradient step granularity
        )
        return Agent(agent_id, ModelRegistry({"model": entry}), loaders[agent_id])

    agents = [_make_agent(i) for i in range(n_agents)]
    ml_topo = MultiLayerTopology({config.layer_name: topo})
    compositor = CoupledCompositor()
    if config.gossip_protocol == "async_poisson":
        # Rate per individual gradient step: N/local_steps events per step gives
        # N expected events per block of local_steps steps (matching synchronous throughput).
        per_step_rate = (
            config.gossip_rate if config.gossip_rate is not None else float(n_agents)
        ) / float(config.local_steps)
        protocol = AsynchronousGossip(
            rate=per_step_rate,
            mode="poisson",
            alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )
    else:
        protocol = GossipAveraging()

    # Phase 1: warmup — drives all agents toward basin A.
    # Plan is computed once per round (topology is static); gossip fires after
    # each individual gradient step so waiting times are in gradient-step units.
    for round_idx in range(config.n_warmup):
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)

    # Post-warmup check: all modules should be in basin A
    post_warmup_centroids = compute_all_centroids(agents, assigns, n_leaf_types)
    warmup_ok = all(
        basin_label(post_warmup_centroids[tau], theta_A_np, theta_B_np, config.epsilon) == 'A'
        for tau in range(n_leaf_types)
    )
    # For b=0 (NMH-7), agents start near A but both basins are symmetric;
    # warmup_ok may be False for some seeds — acceptable, flagged not raised.

    # Phase 2: measurement
    centroid_traj_list: List[np.ndarray] = []

    # Optional force-flip: set one random leaf to basin B at t=0 so that the
    # cascade propagation is driven purely by gossip (zero Langevin noise).
    # The source leaf appears as nucleation_leaf with t_nucleation=0; all
    # other flip times are relative to this forced starting point.
    if config.force_flip_source:
        source_leaf = int(np.random.randint(n_leaf_types))
        force_flip_module(agents, assigns, source_leaf, theta_B_t, noise_scale=0.0)

    baseline = compute_all_centroids(agents, assigns, n_leaf_types)
    centroid_traj_list.append(baseline.copy())

    for round_idx in range(config.n_meas_rounds):
        layer_graphs = ml_topo.step(config.n_warmup + round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)
        # Langevin noise: injected once per measurement round (not per gradient step)
        # to model low-frequency thermal fluctuations.
        if config.flip_noise_scale > 0:
            for agent in agents:
                model = agent.registry["model"].model
                with torch.no_grad():
                    for param in model.parameters():
                        noise = torch.from_numpy(
                            np.random.randn(*param.shape).astype(np.float32)
                        ) * config.flip_noise_scale
                        param.add_(noise)
        centroid_traj_list.append(
            compute_all_centroids(agents, assigns, n_leaf_types)
        )

    centroid_traj = np.stack(centroid_traj_list)  # (n_meas+1, n_leaf_types, d_param)

    # Post-hoc nucleation detection
    t_flip_per_leaf: List[Optional[int]] = []
    for tau in range(n_leaf_types):
        t_flip_per_leaf.append(
            find_t_flip(
                centroid_traj[:, tau, :],
                theta_A_np,
                theta_B_np,
                config.epsilon,
                config.persistence,
            ) if warmup_ok else None
        )

    # Identify nucleation leaf: first leaf to enter B
    nucleation_leaf: Optional[int] = None
    t_nucleation: Optional[int] = None
    valid = [(tau, t) for tau, t in enumerate(t_flip_per_leaf) if t is not None]
    if valid:
        nucleation_leaf, t_nucleation = min(valid, key=lambda x: x[1])

    # Build flip table relative to nucleation
    flip_table: List[dict] = []
    for tau in range(n_leaf_types):
        if tau == nucleation_leaf:
            continue
        t_abs = t_flip_per_leaf[tau]
        d = (
            hierarchical_distance(nucleation_leaf, tau)
            if nucleation_leaf is not None
            else None
        )
        t_rel = (
            (t_abs - t_nucleation) if (t_abs is not None and t_nucleation is not None)
            else None
        )
        flip_table.append({
            "target_leaf": tau,
            "distance_from_source": d,
            "t_flip_absolute": t_abs,
            "t_flip_relative": t_rel,
            "warmup_ok": warmup_ok,
            "a": config.a,
            "b": config.b,
            "lr": config.lr,
            "seed": config.seed,
            "local_steps": config.local_steps,
        })

    return NaturalCascadeRun(
        name=config.name,
        lr=config.lr,
        seed=config.seed,
        branching=config.branching,
        depth=config.depth,
        leaf_size=config.leaf_size,
        p=config.p,
        a=config.a,
        b=config.b,
        d_param=config.d_param,
        local_steps=config.local_steps,
        n_warmup=config.n_warmup,
        n_meas_rounds=config.n_meas_rounds,
        theta_A=theta_A_np,
        theta_B=theta_B_np,
        centroid_traj=centroid_traj,
        warmup_ok=warmup_ok,
        nucleation_leaf=nucleation_leaf,
        t_nucleation=t_nucleation,
        flip_table=flip_table,
        regime=regime,
        ell_c=ell_c,
    )
