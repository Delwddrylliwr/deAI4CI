"""Post-hoc observable functions for NMH Regime C experiments.

All functions are pure: they operate on recorded centroid_traj arrays
(shape (T, n_leaf_types, d_param)) and return scalar or dict observables.
Simulations can be re-analysed without re-running.

Imports reuse stable utilities from active_escape and catchup modules.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .active_escape import basin_label, find_t_flip
from .catchup import hierarchical_distance
from . import theory


# ---------------------------------------------------------------------------
# Cascade propagation observables (NMH-1, NMH-3, NMH-4, NMH-5)
# ---------------------------------------------------------------------------


def cascade_depth(
    centroid_traj: np.ndarray,
    source_leaf: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
    persistence: int = 3,
) -> int:
    """Highest hierarchical distance reached by any leaf that flips to B.

    Parameters
    ----------
    centroid_traj : (T, n_leaf_types, d_param)
    source_leaf : leaf index of the cascade origin.
    Returns 0 if no other leaf enters basin B.
    """
    n_leaf_types = centroid_traj.shape[1]
    max_d = 0
    for j in range(n_leaf_types):
        if j == source_leaf:
            continue
        t = find_t_flip(centroid_traj[:, j, :], theta_A, theta_B, epsilon, persistence)
        if t is not None:
            d = hierarchical_distance(source_leaf, j)
            if d > max_d:
                max_d = d
    return max_d


def fraction_reaching_level(
    centroid_traj: np.ndarray,
    source_leaf: int,
    target_level: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
    persistence: int = 3,
) -> float:
    """Fraction of leaves at hierarchical distance `target_level` that flip.

    Returns NaN if there are no leaves at that distance.
    """
    n_leaf_types = centroid_traj.shape[1]
    total = 0
    flipped = 0
    for j in range(n_leaf_types):
        if j == source_leaf:
            continue
        if hierarchical_distance(source_leaf, j) != target_level:
            continue
        total += 1
        t = find_t_flip(centroid_traj[:, j, :], theta_A, theta_B, epsilon, persistence)
        if t is not None:
            flipped += 1
    return flipped / total if total > 0 else float("nan")


def cascade_size(
    centroid_traj: np.ndarray,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    t_horizon: int,
    epsilon: float = 0.2,
) -> int:
    """Number of leaves in basin B at round t_horizon.

    Parameters
    ----------
    t_horizon : index into the T dimension of centroid_traj.
        Clamped to T-1 if it exceeds the trajectory length.
    """
    T, n_leaf_types, _ = centroid_traj.shape
    t = min(t_horizon, T - 1)
    return sum(
        1 for j in range(n_leaf_types)
        if basin_label(centroid_traj[t, j, :], theta_A, theta_B, epsilon) == 'B'
    )


# ---------------------------------------------------------------------------
# Nucleation probability (NMH-2)
# ---------------------------------------------------------------------------


def nucleation_prob_per_level(
    centroid_traj: np.ndarray,
    assigns: np.ndarray,
    source_leaf: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
) -> Dict[int, float]:
    """Empirical per-level A→B nucleation rate inferred from centroid jumps.

    For each consecutive pair of rounds (t-1, t) and each leaf tau, detects
    if the leaf was in basin A at t-1 and B at t (a 'nucleation event').
    The rate is the count of such events divided by the number of (t, tau)
    pairs where the leaf was in A at t-1.

    Returns {hierarchical_distance: estimated_rate}.  Entries are NaN for
    distances where no A-at-(t-1) observations were recorded.
    """
    T, n_leaf_types, _ = centroid_traj.shape
    max_d = max(
        hierarchical_distance(source_leaf, j)
        for j in range(n_leaf_types) if j != source_leaf
    ) if n_leaf_types > 1 else 0

    nucleations: Dict[int, int] = {d: 0 for d in range(1, max_d + 1)}
    exposures: Dict[int, int] = {d: 0 for d in range(1, max_d + 1)}

    for j in range(n_leaf_types):
        if j == source_leaf:
            continue
        d = hierarchical_distance(source_leaf, j)
        if d not in exposures:
            continue
        for t in range(1, T):
            prev_label = basin_label(
                centroid_traj[t - 1, j, :], theta_A, theta_B, epsilon
            )
            curr_label = basin_label(
                centroid_traj[t, j, :], theta_A, theta_B, epsilon
            )
            if prev_label == 'A':
                exposures[d] += 1
                if curr_label == 'B':
                    nucleations[d] += 1

    return {
        d: (nucleations[d] / exposures[d] if exposures[d] > 0 else float("nan"))
        for d in sorted(nucleations)
    }


# ---------------------------------------------------------------------------
# Variance decomposition (NMH-6)
# ---------------------------------------------------------------------------


def hierarchical_variance_decomposition(
    centroid_traj: np.ndarray,
    branching: int,
    depth: int,
    t_range: Optional[Tuple[int, int]] = None,
) -> Dict[int, float]:
    """Between-level variance V_ell averaged over a time window.

    V_ell is the variance of level-ell super-module centroids averaged over
    t_range rounds.  The decomposition follows:
        V_ell = Var(level-ell module means) - Var(level-(ell-1) module means)
    where level-0 module means are the leaf centroids themselves.

    Returns {level: V_level} for level in 1..depth.

    Parameters
    ----------
    centroid_traj : (T, n_leaf_types, d_param)
        Leaf-centroid trajectories.
    branching : binary branching factor (typically 2).
    depth : hierarchy depth (number of levels above leaves).
    t_range : (t_start, t_end) slice for time averaging.  Defaults to full
        trajectory.
    """
    T, n_leaf_types, d_param = centroid_traj.shape
    t_start, t_end = t_range if t_range is not None else (0, T)
    traj = centroid_traj[t_start:t_end]   # (T', n_leaf_types, d_param)
    T_window = traj.shape[0]

    # Get leaf groupings for each level
    modules_by_level = theory.nmh_modules_by_level(branching, depth, leaf_size=1)
    # modules_by_level[ell] = list of lists of LEAF indices in each level-ell module.
    # We need to map these to indices into centroid_traj (which is indexed by leaf type).

    # level 0: each leaf is its own module
    # level ell: super-modules group branching^ell leaves contiguously

    result: Dict[int, float] = {}
    # Compute module centroid variance at level 0 (leaf level)
    prev_var = float(np.var(traj.mean(axis=0), axis=0).mean())  # mean over d_param dims

    for ell in range(1, depth + 1):
        # Each level-ell module contains branching^ell leaf types
        leaves_per_module = branching ** ell
        n_modules = n_leaf_types // leaves_per_module
        # Module centroid: mean over leaves within module, then over time
        module_means = np.zeros((T_window, n_modules, d_param))
        for m in range(n_modules):
            start = m * leaves_per_module
            end = (m + 1) * leaves_per_module
            module_means[:, m, :] = traj[:, start:end, :].mean(axis=1)
        # Variance across modules, averaged over time and d_param dims
        curr_var = float(np.var(module_means.mean(axis=0), axis=0).mean())
        result[ell] = curr_var - prev_var
        prev_var = curr_var

    return result


# ---------------------------------------------------------------------------
# Detailed balance (NMH-7)
# ---------------------------------------------------------------------------


def detailed_balance_ratio(
    centroid_traj: np.ndarray,
    source_leaf: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
) -> Dict[int, float]:
    """Empirical A→B rate / B→A rate per hierarchical distance from source.

    Counts forward (A→B) and backward (B→A) transition events across all
    leaves at each distance d and all time steps.

    Returns {distance: ratio}.  NaN where the B→A count is zero (expected
    for b > 0 absorbing dynamics; use b=0 for NMH-7 to get both counts).
    """
    T, n_leaf_types, _ = centroid_traj.shape
    max_d = max(
        hierarchical_distance(source_leaf, j)
        for j in range(n_leaf_types) if j != source_leaf
    ) if n_leaf_types > 1 else 0

    fwd: Dict[int, int] = {d: 0 for d in range(1, max_d + 1)}
    bwd: Dict[int, int] = {d: 0 for d in range(1, max_d + 1)}

    for j in range(n_leaf_types):
        if j == source_leaf:
            continue
        d = hierarchical_distance(source_leaf, j)
        if d not in fwd:
            continue
        for t in range(1, T):
            prev = basin_label(centroid_traj[t - 1, j, :], theta_A, theta_B, epsilon)
            curr = basin_label(centroid_traj[t, j, :], theta_A, theta_B, epsilon)
            if prev == 'A' and curr == 'B':
                fwd[d] += 1
            elif prev == 'B' and curr == 'A':
                bwd[d] += 1

    return {
        d: (fwd[d] / bwd[d] if bwd[d] > 0 else float("nan"))
        for d in sorted(fwd)
    }
