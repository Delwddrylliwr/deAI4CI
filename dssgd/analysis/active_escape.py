"""Active-escape experiment runner for NMH gossip-SGD dynamics (v2).

Implements the basin-flip observable: each module has an explicit bistable
loss landscape with basins A and B.  The source module is force-flipped from
A to B, and we measure the first sustained entry of every other module's
centroid into basin B (t_flip).  The theoretical prediction is

    T_flip(d) ~ 2^(d+1) / (γ·m·p)

i.e. log₂(t_flip) is linear in hierarchical distance d with slope 1.

This supersedes the passive-perturbation catch-up experiment (catchup.py),
which measured t_50 without a genuine barrier and was confounded by graph
distance.
"""

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import GossipAveraging
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology

from . import theory

# Re-export type-indexing utilities from catchup so callers need only one import
from .catchup import (
    leaf_assignments,
    hierarchical_distance,
    cousin_at_level,
    compute_all_centroids,
    _flat_params,
)


# ---------------------------------------------------------------------------
# Bistable landscape: model, loss factory, sanity check
# ---------------------------------------------------------------------------


class BistableParameterModule(nn.Module):
    """A module whose sole purpose is to hold a d-dimensional parameter vector θ.

    The bistable loss is defined over θ, not over data — batch input to
    forward() is unused.  GossipAveraging operates on self.theta via the
    standard state_dict / load_state_dict interface without any modifications.
    """

    def __init__(self, d_param: int = 1, init_value: Optional[torch.Tensor] = None):
        super().__init__()
        self.theta = nn.Parameter(
            init_value.clone().float()
            if init_value is not None
            else torch.zeros(d_param)
        )

    def forward(self, x):
        return self.theta


def make_bistable_loss_fn(
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
    a: float = 0.5,
    b: float = 0.05,
) -> Callable:
    """Return a loss_fn(model, batch) -> torch.Tensor for the quartic double-well.

    L(θ) = (a/2)·‖θ−A‖²·‖θ−B‖² / ‖B−A‖²  −  b·(θ−M)·dir

    where M = (A+B)/2 and dir = (B−A)/‖B−A‖.  The bias term tilts the
    landscape so B is the global minimum; the quartic term creates a barrier.

    The batch argument is ignored — loss depends only on model parameters.
    PyTorch autograd computes gradients via standard backward().
    """
    A = theta_A.float()
    B = theta_B.float()
    delta_AB = B - A
    delta_norm_sq = (delta_AB ** 2).sum()
    delta_norm = delta_norm_sq.sqrt()
    M = (A + B) / 2.0
    direction = delta_AB / (delta_norm + 1e-12)

    def _loss(model: nn.Module, batch) -> torch.Tensor:
        theta = torch.cat([p.flatten() for p in model.parameters()])
        diff_A = theta - A
        diff_B = theta - B
        well = (a / 2.0) * (diff_A ** 2).sum() * (diff_B ** 2).sum() / (delta_norm_sq + 1e-12)
        bias = b * ((theta - M) * direction).sum()
        return well - bias

    return _loss


def make_asymmetric_bistable_loss_fn(
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
    a: float = 0.5,
    b: float = 0.05,
    r: float = 1.0,
) -> Callable:
    """Curvature-asymmetric generalisation of make_bistable_loss_fn (Lemma 10.1).

    L_r(theta) = (a/2)*Q(theta)*(1 + eps*T(theta)) - b*Bias(theta)

    where Q, Bias are exactly as in make_bistable_loss_fn, T(theta) =
    proj(theta) - 1/2 is the signed projection onto the A->B axis centred at
    the midpoint, and eps = theory.curvature_epsilon(r) = 2(r-1)/(r+1).

    r=1 (eps=0) recovers make_bistable_loss_fn exactly. r>1 makes B the
    sharper (higher-curvature) minimum and A the flatter one, at mu=0 (b=0)
    exactly; for b != 0 this holds only approximately (paper's own scope:
    "exact in the small-tilt, moderate-r regime"). See theory.chord_geometry
    for the corresponding threshold/lambda computation.
    """
    from . import theory

    A = theta_A.float()
    B = theta_B.float()
    delta_AB = B - A
    delta_norm_sq = (delta_AB ** 2).sum()
    delta_norm = delta_norm_sq.sqrt()
    M = (A + B) / 2.0
    direction = delta_AB / (delta_norm + 1e-12)
    eps = theory.curvature_epsilon(r)

    def _loss(model: nn.Module, batch) -> torch.Tensor:
        theta = torch.cat([p.flatten() for p in model.parameters()])
        diff_A = theta - A
        diff_B = theta - B
        Q = (diff_A ** 2).sum() * (diff_B ** 2).sum() / (delta_norm_sq + 1e-12)
        proj = ((theta - A) * direction).sum() / (delta_norm + 1e-12)
        T = proj - 0.5
        well = (a / 2.0) * Q * (1.0 + eps * T)
        bias = b * ((theta - M) * direction).sum()
        return well - bias

    return _loss


def verify_bistable_loss(
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
    a: float,
    b: float,
    n_scan: int = 100,
    require_barrier: bool = True,
) -> None:
    """Assert structural correctness of the bistable loss landscape.

    Checks:
    1. theta_A != theta_B (non-degenerate landscape).
    2. L(B) < L(A) — B is the global minimum (bias b > 0 tilts toward B).
    3. (require_barrier=True only) A barrier exists: some interior point along
       the A→B axis has loss strictly greater than L(A).

    For clamped-source experiments (require_barrier=True), check 3 confirms that
    A is a metastable attractor — agents won't spontaneously escape without
    being driven.

    For natural-cascade experiments (require_barrier=False, clamp_source=False),
    check 3 is skipped.  In this regime b is deliberately close to the bistability
    limit so the energy barrier falls below L(A) (b > 2a/27), but a gradient-based
    saddle still exists at some θ_s ∈ (A, B).  A is not a stable attractor in
    this regime — gradient descent from A flows directly toward B — so the energy
    barrier condition is not meaningful.

    Note: A and B are NOT exact critical points of the full loss L (the bias
    term adds a constant -b gradient everywhere).  The gradient at A is -b,
    not zero.  Only the quartic term has critical points at A and B.

    Raises ValueError with a descriptive message on the first failure.
    """
    A = theta_A.float()
    B = theta_B.float()
    d = A.shape[0]

    if torch.allclose(A, B):
        raise ValueError("theta_A and theta_B must differ (degenerate landscape)")

    loss_fn = make_bistable_loss_fn(A, B, a, b)

    def _eval_loss(point: torch.Tensor) -> float:
        model = BistableParameterModule(d, init_value=point.clone())
        with torch.no_grad():
            return float(loss_fn(model, None).item())

    L_A = _eval_loss(A)
    L_B = _eval_loss(B)

    # Check B is global minimum
    if L_B >= L_A:
        raise ValueError(
            f"L(B)={L_B:.4f} >= L(A)={L_A:.4f}; B must be the global minimum.  "
            f"Increase b or check that a > 0."
        )

    if not require_barrier:
        return

    # Scan interior: check a barrier (local max > L_A) exists between A and B
    t_vals = torch.linspace(0.05, 0.95, n_scan)
    interior_losses = [_eval_loss(A + t * (B - A)) for t in t_vals]
    max_interior = max(interior_losses)
    if max_interior <= L_A:
        raise ValueError(
            f"No barrier detected between A and B (max interior loss {max_interior:.4f} "
            f"<= L(A)={L_A:.4f}).  Reduce b (current b={b}) or increase a (current a={a}).  "
            f"Bistability requires b < a * 0.096 * ||B-A||."
        )


# ---------------------------------------------------------------------------
# Basin labeling and flip-time detection
# ---------------------------------------------------------------------------


def basin_label(
    centroid: np.ndarray,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
) -> str:
    """Return 'A', 'B', or 'X' based on normalized position along the A→B axis.

    Computes the scalar projection s of (centroid − A) onto the unit direction
    (B − A) / ‖B − A‖, then normalises by ‖B − A‖ to get s ∈ [0, 1] for
    points between A and B.

    s < epsilon        → 'A'
    s > 1 − epsilon    → 'B'
    otherwise          → 'X' (near saddle or in transition)
    """
    delta = theta_B - theta_A
    norm = np.linalg.norm(delta) + 1e-12
    direction = delta / norm
    s = float(np.dot(centroid - theta_A, direction) / norm)
    if s < epsilon:
        return 'A'
    if s > 1.0 - epsilon:
        return 'B'
    return 'X'


def find_t_flip(
    centroid_traj: np.ndarray,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
    persistence: int = 5,
) -> Optional[int]:
    """First t at which basin_label == 'B' for `persistence` consecutive rounds.

    centroid_traj : (T, d_param) — trajectory for one leaf type.
    Returns the index of the FIRST round of the first sustained B-run, or None.

    The persistence requirement filters out thermal fluctuations that briefly
    visit B before returning to A, without committing to the new basin.
    """
    T = len(centroid_traj)
    labels = [
        basin_label(centroid_traj[t], theta_A, theta_B, epsilon) for t in range(T)
    ]
    run_len = 0
    run_start: Optional[int] = None
    for t, label in enumerate(labels):
        if label == 'B':
            if run_len == 0:
                run_start = t
            run_len += 1
            if run_len >= persistence:
                return run_start
        else:
            run_len = 0
            run_start = None
    return None


# ---------------------------------------------------------------------------
# Force-flip (replaces source leaf params; distinct from v1's inject_perturbation)
# ---------------------------------------------------------------------------


def force_flip_module(
    agents: List[Agent],
    leaf_assigns: np.ndarray,
    source_leaf: int,
    theta_B: torch.Tensor,
    noise_scale: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Set all parameters of source-leaf agents to theta_B + small noise.

    Unlike v1's inject_perturbation() which ADDS a delta, this REPLACES the
    parameter values using p.copy_().  Noise keeps agents within basin B
    without being exactly at the basin centre.
    """
    if rng is None:
        rng = np.random.default_rng()
    for agent in agents:
        if leaf_assigns[agent.id] != source_leaf:
            continue
        with torch.no_grad():
            for p in agent.registry["model"].model.parameters():
                assert p.numel() == theta_B.numel(), (
                    f"Parameter has {p.numel()} elements but theta_B has {theta_B.numel()}"
                )
                noise = torch.from_numpy(
                    rng.standard_normal(p.numel()).astype(np.float32) * noise_scale
                ).view_as(p)
                p.copy_(theta_B.view_as(p) + noise)


# ---------------------------------------------------------------------------
# Dummy data loader (batch content is irrelevant for bistable experiments)
# ---------------------------------------------------------------------------


def _dummy_loader_bistable() -> DataLoader:
    x = torch.zeros(4, 1)
    y = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=4)


# ---------------------------------------------------------------------------
# Config and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActiveEscapeSimConfig:
    """All degrees of freedom for one active-escape simulation."""

    name: str
    branching: int = 2
    depth: int = 5
    leaf_size: int = 4
    p: float = 2.0
    seed: int = 0
    n_warmup: int = 400
    n_meas_rounds: int = 800
    lr: float = 0.01
    local_steps: int = 1
    source_leaf: int = 0
    # Bistable landscape
    d_param: int = 1
    theta_A: Optional[np.ndarray] = None   # default: zeros(d_param)
    theta_B: Optional[np.ndarray] = None   # default: e_1 = [1, 0, ..., 0]
    a: float = 0.5
    b: float = 0.01
    # Flip detection
    epsilon: float = 0.2
    persistence: int = 5
    flip_noise_scale: float = 0.01
    layer_name: str = "social"
    # Source clamping: re-apply force-flip after each gossip round so source
    # stays in B throughout the measurement phase.  The shallow bistable well
    # (a<1, Regime I) cannot self-sustain against gossip from A-neighbors —
    # gossip coupling (~0.5/round) >> loss gradient step (~0.003/round) — so
    # one-shot flip reverts to A within 10 rounds, killing the cascade.
    # Clamping implements the theory's implicit "fixed source BC" assumption.
    clamp_source: bool = True


@dataclass
class ActiveEscapeRun:
    """All outputs from one active-escape simulation run.

    centroid_traj : np.ndarray (n_meas_rounds+1, n_leaf_types, d_param)
        Type-centroid snapshots at every measurement round, including t=0
        (immediately post-flip baseline).
    flip_table : list of dicts
        One entry per target leaf with keys:
        lr, seed, a, b, regime, ell_c, source_leaf, target_leaf, distance_d,
        t_flip (int or None), warmup_ok (bool).
    warmup_ok : bool
        False if any module was found in basin B after warmup.  Runs with
        warmup_ok=False are excluded from analysis.
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
    n_warmup: int
    n_meas_rounds: int
    source_leaf: int
    theta_A: np.ndarray
    theta_B: np.ndarray
    centroid_traj: np.ndarray      # (n_meas_rounds+1, n_leaf_types, d_param)
    flip_table: List[dict]
    warmup_ok: bool
    regime: str
    ell_c: int

    def save(self, path: Union[str, Path]) -> None:
        p_obj = Path(path)
        p_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(p_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ActiveEscapeRun":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_active_escape_simulation(config: ActiveEscapeSimConfig) -> ActiveEscapeRun:
    """Two-phase active-escape experiment; returns a picklable ActiveEscapeRun.

    Phase 1 — warmup: gossip-SGD on bistable loss for n_warmup rounds.
        All agents initialise near theta_A.  After warmup, all modules must
        be in basin A (verified via basin_label); warmup_ok=False if not.

    Phase 2 — measurement: force-flip source leaf to basin B, then record
        type-centroid trajectories for n_meas_rounds rounds.  t_flip is
        extracted per target leaf using find_t_flip with persistence check.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Resolve theta_A / theta_B defaults
    theta_A_np = (
        config.theta_A
        if config.theta_A is not None
        else np.zeros(config.d_param, dtype=np.float32)
    )
    theta_B_np = (
        config.theta_B
        if config.theta_B is not None
        else np.array([1.0] + [0.0] * (config.d_param - 1), dtype=np.float32)
    )
    theta_A_t = torch.from_numpy(theta_A_np)
    theta_B_t = torch.from_numpy(theta_B_np)

    # Validate landscape; skip energy-barrier check for natural-cascade configs
    verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b,
                         require_barrier=config.clamp_source)

    # Classify regime
    regime = theory.classify_regime(config.a, 1.0, config.leaf_size, config.p, config.depth)
    ell_c = theory.critical_level(config.a, 1.0, config.leaf_size, config.p, config.depth)

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

    loss_fn = make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b)
    loaders = [_dummy_loader_bistable() for _ in range(n_agents)]

    def _make_agent(agent_id: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + agent_id)
        # Initialise near theta_A with small noise
        init_noise = torch.randn(config.d_param) * 0.05
        init_val = theta_A_t + init_noise
        model = BistableParameterModule(d_param=config.d_param, init_value=init_val)
        entry = ModelEntry(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
            loss_fn=loss_fn,
            layers=frozenset({config.layer_name}),
            train_locally=True,
            local_steps=config.local_steps,
        )
        return Agent(agent_id, ModelRegistry({"model": entry}), loaders[agent_id])

    agents = [_make_agent(i) for i in range(n_agents)]
    ml_topo = MultiLayerTopology({config.layer_name: topo})
    compositor = CoupledCompositor()
    protocol = GossipAveraging()

    # Phase 1: warmup
    for round_idx in range(config.n_warmup):
        for agent in agents:
            agent.local_step()
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)

    # Post-warmup sanity check: all modules must be in basin A
    post_warmup_centroids = compute_all_centroids(agents, assigns, n_leaf_types)
    warmup_ok = all(
        basin_label(post_warmup_centroids[tau], theta_A_np, theta_B_np, config.epsilon) == 'A'
        for tau in range(n_leaf_types)
    )
    # Record flag but do not raise — lets batch runs complete and flag failures post-hoc

    # Phase 2: force-flip source leaf to basin B
    rng = np.random.default_rng(config.seed + 1)
    force_flip_module(agents, assigns, config.source_leaf, theta_B_t, config.flip_noise_scale, rng)

    # Snapshot t=0: centroids immediately after force-flip
    centroid_traj_list: List[np.ndarray] = []
    baseline = compute_all_centroids(agents, assigns, n_leaf_types)
    centroid_traj_list.append(baseline.copy())

    # Measurement phase
    for round_idx in range(config.n_meas_rounds):
        for agent in agents:
            agent.local_step()
        layer_graphs = ml_topo.step(config.n_warmup + round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)
        # Re-clamp source leaf so it acts as a fixed B-boundary for the next
        # round's gossip.  Without this, the shallow Regime-I basin (a<1) is
        # overwhelmed by gossip from A-neighbors and the source reverts within
        # ~10 rounds, before any cascade can propagate.
        if config.clamp_source:
            force_flip_module(
                agents, assigns, config.source_leaf, theta_B_t,
                config.flip_noise_scale, rng,
            )
        centroid_traj_list.append(compute_all_centroids(agents, assigns, n_leaf_types))

    centroid_traj = np.stack(centroid_traj_list)  # (n_meas_rounds+1, n_leaf_types, d_param)

    # Extract flip times
    flip_table: List[dict] = []
    for j in range(n_leaf_types):
        if j == config.source_leaf:
            continue
        d = hierarchical_distance(config.source_leaf, j)
        t_flip = (
            find_t_flip(
                centroid_traj[:, j, :],
                theta_A_np,
                theta_B_np,
                config.epsilon,
                config.persistence,
            )
            if warmup_ok
            else None
        )
        flip_table.append({
            "lr": config.lr,
            "seed": config.seed,
            "a": config.a,
            "b": config.b,
            "regime": regime,
            "ell_c": ell_c,
            "source_leaf": config.source_leaf,
            "target_leaf": j,
            "distance_d": d,
            "t_flip": t_flip,
            "warmup_ok": warmup_ok,
        })

    return ActiveEscapeRun(
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
        n_warmup=config.n_warmup,
        n_meas_rounds=config.n_meas_rounds,
        source_leaf=config.source_leaf,
        theta_A=theta_A_np,
        theta_B=theta_B_np,
        centroid_traj=centroid_traj,
        flip_table=flip_table,
        warmup_ok=warmup_ok,
        regime=regime,
        ell_c=ell_c,
    )
