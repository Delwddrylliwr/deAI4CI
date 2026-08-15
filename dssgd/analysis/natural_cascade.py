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
from dssgd.protocols.gossip import (
    AsynchronousGossip,
    BoundedStalenessGossip,
    GossipAveraging,
    HybridGossip,
    SynchronousPairwiseGossip,
)
from dssgd.topology.base import Topology
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology, OverlappingModularTopology

from . import theory
from .active_escape import (
    BistableParameterModule,
    _dummy_loader_bistable,
    basin_label,
    find_t_flip,
    force_flip_module,
    make_asymmetric_bistable_loss_fn,
    make_bistable_loss_fn,
    verify_bistable_loss,
)
from .catchup import (
    _flat_params,
    compute_all_centroids,
    hierarchical_distance,
    leaf_assignments,
)
from .provenance import GossipEvent, ProvenanceAsyncGossip, SeveredTopology


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
    graph_seed: Optional[int] = None  # None: reuse `seed` (default/legacy behaviour).
    # Decouples the topology draw from the dynamics RNG seed so a quenched
    # graph realisation can be replayed against many dynamics seeds, and vice
    # versa (Experiment E4's quenched-vs-annealed cascade-size comparison).
    init_basin: str = "A"  # "A" or "B": which basin warmup drives agents toward
    # (Experiment E3's hysteresis test: does the reachable phase depend on
    # whether agents start near A or near B?).
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
    force_flip_source: bool = False  # if True, force-flip a leaf post-warmup (zero noise)
    # Which leaf to force-flip when force_flip_source=True. None (default):
    # pick uniformly at random -- correct for homogeneous-loss experiments
    # (NMH-1/NMH-3/E11) where every leaf is equivalent. Experiments with a
    # per-leaf heterogeneity model keyed to a specific origin (E6/E12a's
    # per_leaf_loss_params_for_generality(source_leaf=...), Def. 4.3's
    # generality level) MUST set this to that same leaf, or the force-flipped
    # leaf and the leaf the bias pattern / scoring (cascade_depth's
    # source_leaf) were built around silently diverge.
    source_leaf: Optional[int] = None
    layer_name: str = "social"
    # "async_poisson", "sync_pairwise" (SynchronousPairwiseGossip, Remark 4.3's
    # H-sched class), "sync_neighbourhood" (GossipAveraging, simultaneous m-way
    # mean -- Lemma 6.1's basin-destroying mechanism), "bounded_staleness", or
    # "hybrid" (HybridGossip, the two-jump Type-P+Type-N protocol of
    # paper1_computing_hybrid_gossip.md Sec. 2.1 -- see round_ratio below).
    # See dssgd.protocols.gossip.protocol_suffix / gossip_mechanisms.md: the
    # bare "synchronous" string is retired -- it used to mean the m-way mean,
    # then briefly meant the pairwise class, an ambiguity that made "S"-suffixed
    # files impossible to interpret without knowing when they were generated.
    gossip_protocol: str = "async_poisson"
    gossip_rate: Optional[float] = None     # None → auto-set to n_agents (see runner)
    gossip_alpha: float = 0.5               # initiator mixing weight toward the neighbour

    # Hybrid protocol only (gossip_protocol="hybrid"): the round ratio K
    # (eq. 2.2c). If set, overrides gossip_rate -- the per-round aggregate
    # Type-P rate is derived from K via HybridGossip.pairwise_rate_for_K,
    # exact for a single isolated clique (the calibration-ladder scope: E0,
    # E7, E14 / phases H1-H2). None (default): fall back to gossip_rate (or
    # n_agents), matching every other protocol's default when K isn't the
    # thing being controlled (e.g. topology-scale experiments calibrating
    # gossip_rate empirically instead -- see HybridGossip's docstring).
    round_ratio: Optional[float] = None
    # Type-N fires every this many rounds (1/epsilon_n in round units).
    # Only consulted when gossip_protocol == "hybrid".
    epsilon_n_rounds: int = 1
    # NMH-6 heterogeneous loss: one (a_i, b_i) per leaf; None = uniform
    per_leaf_loss_params: Optional[List[Tuple[float, float]]] = None
    t_horizon: int = 800           # round index used by cascade_size observable

    # Experiment E1: kick-vs-escape provenance tracking + gossip-severed control.
    track_provenance: bool = False  # log cross-boundary events (async_poisson only)
    sever_min_distance: Optional[int] = None  # prune edges >= this hierarchical
    # distance apart before simulating (Prop. 3.4's severed-system null); None = no severing.

    # Experiments E7 / E12(b): curvature ratio r of the asymmetric well (Lemma 10.1).
    # r=1.0 (default) is the equal-curvature family of Lemma 2.1 (unchanged behaviour).
    curvature_ratio: float = 1.0

    # Experiment E11: bounded-staleness / per-boundary seed-locking scheduling.
    # Only consulted when gossip_protocol == "bounded_staleness".
    staleness_bound: int = 0

    # Experiment E13: DAG-nested / overlapping module hierarchy (Section 11).
    # overlap_level=None (default) builds the plain tree (NestedModularTopology).
    overlap_level: Optional[int] = None
    delta_in: int = 1        # boundary in-degree Delta_in at overlap_level
    n_overlap: int = 0       # number of level-overlap_level modules granted extra parents


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
    events: Optional[List[GossipEvent]] = None  # populated iff track_provenance=True

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
# Topology construction with connectivity retry
# ---------------------------------------------------------------------------


def build_nmh_topology_with_retry(
    branching: int, depth: int, leaf_size: int, p: float, seed: int,
    overlap_level: Optional[int] = None, delta_in: int = 1, n_overlap: int = 0,
    max_retries: int = 200,
) -> Topology:
    """NestedModularTopology / OverlappingModularTopology construction,
    retrying with a perturbed graph seed on a disconnected-graph ValueError
    (topology/static.py's own sanity check) instead of propagating it as a
    task failure.

    At low p, the deepest level's cross-edge probability p/4^depth can sit
    below this shape's percolation threshold, making disconnection common
    (confirmed empirically: 16/20 seeds failed at p=0.5, branching=2,
    depth=5, leaf_size=4 -- Annex B.2's E16 ceiling-location experiment).
    Silently losing most of a seed sweep to this, rather than getting an
    unbiased sample of the intended size, biases whatever survives toward
    atypically well-connected draws -- exactly the wrong direction for an
    experiment measuring how connectivity affects transmission. Retrying is
    strictly better than failing outright: a disconnected draw is not a
    meaningful data point to lose seeds over, it is simply the wrong graph
    for the (branching, depth, leaf_size, p) shape requested.

    Each retry uses `seed + attempt * 104729` (a large prime offset),
    deterministic given the original seed, so results stay reproducible.
    Shared by both `run_natural_cascade_simulation` here and its
    checkpointable duplicate in `hpc/worker.py`'s `_run_nmh` -- imported,
    not reimplemented, so the retry policy itself can't silently diverge
    between the two even though the surrounding runner logic does (by
    design, for checkpointing).
    """
    last_exc: Optional[ValueError] = None
    for attempt in range(max_retries):
        trial_seed = seed if attempt == 0 else seed + attempt * 104729
        try:
            if overlap_level is not None:
                return OverlappingModularTopology(
                    branching=branching, depth=depth, leaf_size=leaf_size, p=p,
                    overlap_level=overlap_level, delta_in=delta_in, n_overlap=n_overlap,
                    seed=trial_seed,
                )
            return NestedModularTopology(
                branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=trial_seed,
            )
        except ValueError as exc:
            if "disconnected graph" not in str(exc):
                raise
            last_exc = exc
    raise ValueError(
        f"Could not build a connected topology after {max_retries} seed retries "
        f"(branching={branching}, depth={depth}, leaf_size={leaf_size}, p={p}, "
        f"base_seed={seed}): {last_exc}"
    )


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

    if config.init_basin not in ("A", "B"):
        raise ValueError(f"init_basin must be 'A' or 'B', got {config.init_basin!r}")

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

    graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
    topo: Topology = build_nmh_topology_with_retry(
        branching=config.branching, depth=config.depth, leaf_size=config.leaf_size,
        p=config.p, seed=graph_seed, overlap_level=config.overlap_level,
        delta_in=config.delta_in, n_overlap=config.n_overlap,
    )
    if config.sever_min_distance is not None:
        topo = SeveredTopology(topo, assigns, config.sever_min_distance)

    # Build per-leaf loss functions
    _make_loss = (
        (lambda a_i, b_i: make_bistable_loss_fn(theta_A_t, theta_B_t, a_i, b_i))
        if config.curvature_ratio == 1.0
        else (lambda a_i, b_i: make_asymmetric_bistable_loss_fn(
            theta_A_t, theta_B_t, a_i, b_i, r=config.curvature_ratio))
    )
    if config.per_leaf_loss_params is None:
        uniform_loss = _make_loss(config.a, config.b)
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
            tau: _make_loss(a_i, b_i)
            for tau, (a_i, b_i) in enumerate(config.per_leaf_loss_params)
        }

    loaders = [_dummy_loader_bistable() for _ in range(n_agents)]

    init_target_t = theta_A_t if config.init_basin == "A" else theta_B_t

    def _make_agent(agent_id: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + agent_id)
        init_noise = torch.randn(config.d_param) * 0.05
        init_val = init_target_t + init_noise
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

    if config.track_provenance and config.gossip_protocol not in ("async_poisson", "hybrid"):
        raise ValueError(
            "track_provenance requires gossip_protocol='async_poisson' or "
            f"'hybrid' (event-level attribution is undefined for {config.gossip_protocol!r})"
        )

    agents = [_make_agent(i) for i in range(n_agents)]
    ml_topo = MultiLayerTopology({config.layer_name: topo})
    compositor = CoupledCompositor()
    if config.gossip_protocol == "async_poisson":
        # Rate per individual gradient step: N/local_steps events per step gives
        # N expected events per block of local_steps steps (matching synchronous throughput).
        per_step_rate = (
            config.gossip_rate if config.gossip_rate is not None else float(n_agents)
        ) / float(config.local_steps)
        if config.track_provenance:
            protocol = ProvenanceAsyncGossip(
                rate=per_step_rate,
                mode="poisson",
                alpha=config.gossip_alpha,
                rng=np.random.default_rng(config.seed + 42),
                leaf_assigns=assigns,
            )
        else:
            protocol = AsynchronousGossip(
                rate=per_step_rate,
                mode="poisson",
                alpha=config.gossip_alpha,
                rng=np.random.default_rng(config.seed + 42),
            )
    elif config.gossip_protocol == "bounded_staleness":
        per_step_rate = (
            config.gossip_rate if config.gossip_rate is not None else float(n_agents)
        ) / float(config.local_steps)
        protocol = BoundedStalenessGossip(
            rate=per_step_rate,
            mode="poisson",
            alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
            staleness_bound=config.staleness_bound,
        )
    elif config.gossip_protocol == "sync_pairwise":
        protocol = SynchronousPairwiseGossip(
            alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )
    elif config.gossip_protocol == "sync_neighbourhood":
        protocol = GossipAveraging()
    elif config.gossip_protocol == "hybrid":
        pairwise_rate = (
            HybridGossip.pairwise_rate_for_K(config.round_ratio, config.epsilon_n_rounds)
            if config.round_ratio is not None
            else (config.gossip_rate if config.gossip_rate is not None else float(n_agents))
        )
        per_step_pairwise_rate = pairwise_rate / float(config.local_steps)
        if config.track_provenance:
            # Type-P sub-protocol gets the same event-level logging as the
            # async_poisson path (see HybridGossip's pairwise_protocol
            # docstring) -- Type-N never logs events (Lemma 2.4's clique
            # round map has no per-kick provenance content), so this is the
            # full two-axis instrumentation B.0.2 asks for at hierarchy
            # scale: "did a GossipEvent happen" already distinguishes the
            # jump type (always Type-P) from a flip with no recent event
            # (Type-N-consolidated or Kramers escape).
            inner_pairwise = ProvenanceAsyncGossip(
                rate=per_step_pairwise_rate,
                mode="poisson",
                alpha=config.gossip_alpha,
                rng=np.random.default_rng(config.seed + 42),
                leaf_assigns=assigns,
            )
            protocol = HybridGossip(
                pairwise_rate=per_step_pairwise_rate,
                epsilon_n_rounds=config.epsilon_n_rounds,
                pairwise_protocol=inner_pairwise,
            )
        else:
            protocol = HybridGossip(
                pairwise_rate=per_step_pairwise_rate,
                epsilon_n_rounds=config.epsilon_n_rounds,
                alpha=config.gossip_alpha,
                rng=np.random.default_rng(config.seed + 42),
            )
    else:
        raise ValueError(
            f"Unknown gossip_protocol {config.gossip_protocol!r}; expected "
            f"'async_poisson', 'sync_pairwise', 'sync_neighbourhood', "
            f"'bounded_staleness', or 'hybrid'."
        )
    _has_round_idx = hasattr(protocol, "round_idx")

    # Phase 1: warmup — drives all agents toward basin A.
    # Async: gossip fires after each gradient step (Poisson-calibrated rate gives
    # ~n_agents pairwise events total per round). Sync: one all-neighbour averaging
    # event per round, after all local gradient steps (preserves timescale separation).
    # Hybrid dispatches per-local-step too (its Type-P sub-step needs that
    # granularity; its Type-N sub-step self-limits to once per round via the
    # externally-set .round_idx, same convention bounded_staleness/provenance use).
    _async = config.gossip_protocol in ("async_poisson", "bounded_staleness", "hybrid")
    for round_idx in range(config.n_warmup):
        if _has_round_idx:
            protocol.round_idx = round_idx
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            if _async:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
        if not _async:
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)

    # Post-warmup check: all modules should be in the init_basin target
    post_warmup_centroids = compute_all_centroids(agents, assigns, n_leaf_types)
    warmup_ok = all(
        basin_label(post_warmup_centroids[tau], theta_A_np, theta_B_np, config.epsilon)
        == config.init_basin
        for tau in range(n_leaf_types)
    )
    # For b=0 (NMH-7), agents start near A but both basins are symmetric;
    # warmup_ok may be False for some seeds — acceptable, flagged not raised.

    # Phase 2: measurement
    centroid_traj_list: List[np.ndarray] = []

    # Optional force-flip: set one leaf to basin B at t=0 so that the cascade
    # propagation is driven purely by gossip (zero Langevin noise). The
    # source leaf appears as nucleation_leaf with t_nucleation=0; all other
    # flip times are relative to this forced starting point. config.source_leaf
    # pins which leaf, for experiments whose per-leaf heterogeneity (and
    # scoring) is keyed to a specific origin (E6/E12a); otherwise a leaf is
    # chosen uniformly at random (NMH-1/NMH-3/E11: every leaf is equivalent).
    if config.force_flip_source:
        source_leaf = (
            config.source_leaf if config.source_leaf is not None
            else int(np.random.randint(n_leaf_types))
        )
        flip_target_t = theta_B_t if config.init_basin == "A" else theta_A_t
        force_flip_module(agents, assigns, source_leaf, flip_target_t, noise_scale=0.0)

    baseline = compute_all_centroids(agents, assigns, n_leaf_types)
    centroid_traj_list.append(baseline.copy())

    for round_idx in range(config.n_meas_rounds):
        if _has_round_idx:
            protocol.round_idx = config.n_warmup + round_idx
        layer_graphs = ml_topo.step(config.n_warmup + round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            if _async:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
        if not _async:
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
        events=protocol.events if config.track_provenance else None,
    )
