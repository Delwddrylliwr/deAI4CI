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
    """Highest hierarchical distance reached by any leaf that flips to B:
    the d_max of Theorem 4.5's level-matching filter (paper1_PDMP_wDAG_wData
    .md Sec. 4.4), compared against a candidate innovation's generality level
    G(b) (Def. 4.3) by E6/E12a's scoring.

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


def source_module_consensus(
    centroid_traj: np.ndarray,
    source_leaf: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
    persistence: int = 3,
) -> Optional[int]:
    """Whether/when the SOURCE leaf's own centroid reaches and holds basin B
    -- `cascade_depth`'s counterpart restricted to source_leaf itself (which
    `cascade_depth` explicitly excludes from its scan). Returns None if the
    source leaf's own module never commits to B.

    Diagnostic for Experiment E16's p-inversion (Prop. 4.10): `mean_d_max`
    declined as p increased from 2 to 32 even though `theory.
    cross_module_ceiling` predicts the OPPOSITE (larger p -> more permissive
    ceiling -> more propagation). One candidate explanation is that this is
    a SOURCE-side artifact, not a transport property: Lemma 2.4's clean
    "clique converts to consensus every round" guarantee only holds for
    workers whose entire neighbourhood lies inside the module, and at large
    p more of the source module's OWN members may become boundary workers
    (cross-linked outward), diluting the source module's ability to
    consolidate on B internally before propagation to other modules is ever
    tested. If `source_module_consensus` returns None at large p (source
    never even commits) while `cascade_depth` is 0, the inversion is
    source-side; if the source DOES commit but propagation still fails,
    the ceiling itself is implicated instead.
    """
    return find_t_flip(centroid_traj[:, source_leaf, :], theta_A, theta_B, epsilon, persistence)


def fraction_reaching_level(
    centroid_traj: np.ndarray,
    source_leaf: int,
    target_level: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
    persistence: int = 3,
) -> float:
    """Fraction of leaves at hierarchical distance `target_level` that flip:
    the per-level attainment rate underlying Proposition 3.3's renewal
    composition and NMH-5's filter-composition test (Theorem 4.5).

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
# Realised cross-edge count (Experiment E4)
# ---------------------------------------------------------------------------


def realized_cross_edge_count(
    branching: int,
    depth: int,
    leaf_size: int,
    p: float,
    graph_seed: int,
    source_leaf: int,
    target_leaf: int,
) -> int:
    """Realised cross-edge count K at the boundary between source_leaf and
    target_leaf's lowest common (level-`hierarchical_distance`) ancestor
    (Theorem 4.1's quenched disorder, K_l ~ Poisson(p*m^2/4)).

    Reconstructs the exact NestedModularTopology graph from its construction
    parameters (post-hoc; NaturalCascadeRun does not store the graph itself)
    and counts edges crossing the boundary between source_leaf's level-
    (level-1) module and the rest of the shared level-`level` supermodule.
    Requires the caller to know `graph_seed` -- the config used to run the
    simulation, not the run's own `seed` field, since Experiment E4
    deliberately decouples the two (NaturalCascadeConfig.graph_seed).
    """
    from dssgd.topology.static import NestedModularTopology

    level = hierarchical_distance(source_leaf, target_leaf)
    if level == 0:
        return 0
    topo = NestedModularTopology(
        branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=graph_seed,
    )
    G, _ = topo.step(0)

    leaves_per_module = branching ** (level - 1)
    super_leaves_per_module = branching ** level
    src_module = source_leaf // leaves_per_module
    super_module = source_leaf // super_leaves_per_module

    src_nodes = set(range(
        src_module * leaves_per_module * leaf_size,
        (src_module + 1) * leaves_per_module * leaf_size,
    ))
    super_nodes = set(range(
        super_module * super_leaves_per_module * leaf_size,
        (super_module + 1) * super_leaves_per_module * leaf_size,
    ))
    other_nodes = super_nodes - src_nodes

    return sum(
        1 for u, v in G.edges()
        if (u in src_nodes and v in other_nodes) or (v in src_nodes and u in other_nodes)
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
    """Empirical per-level A→B nucleation rate inferred from centroid jumps:
    the q_l of Lemma 3.1's fixation formula / Proposition 4.2's per-level
    nucleation probability, measured directly (NMH-2) rather than fitted.

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
    """Nested-ANOVA variance components across hierarchy levels (Sec. 4.8's
    V_l^between, ANOVA identity V_L = sum_l V_l^between).

    This is a proper nested (hierarchical) random-effects ANOVA decomposition,
    NOT a difference of raw variances-of-means at successive grains. That
    naive difference (curr_var - prev_var, this function's form before this
    fix) is structurally, asymptotically negative regardless of the true
    variance components: writing X_leaf = mu + sum_j alpha^(j) + eps, with
    alpha^(j) ~ N(0, sigma_j^2) shared within a level-j module and eps the
    private leaf residual, the true variance of a level-k module mean is
        Var(mean_k) = sum_{j>k} sigma_j^2 + sum_{j<=k} sigma_j^2 / b^(k-j)
    (b = branching): levels above k contribute their FULL variance (shared,
    not averaged away within the module) while levels at/below k are averaged
    over b^(k-j) i.i.d. draws. Differencing Var(mean_k) - Var(mean_{k-1}) makes
    the sigma_k^2 term cancel exactly and leaves only a negative combination of
    LOWER-level variances (-(b-1) * sum_{j<k} sigma_j^2/b^(k-j)) -- i.e. the old
    formula could never recover sigma_k^2 at any sample size.

    The correct estimator instead computes, at each level ell, the nested-ANOVA
    mean square MS_ell = SS_ell / df_ell, where SS_ell is the sum of squared
    deviations of level-(ell-1) module means from their level-ell parent's
    mean (summed over all level-ell parents) and df_ell = n_{ell-1} - n_ell.
    This has the exact expectation E[MS_ell] = sigma_{ell-1}^2 + E[MS_{ell-1}]/b
    (standard balanced nested-ANOVA result), so the unbiased variance component
    is recovered by the simple recursive subtraction
        sigma_{ell-1}^2_hat = MS_ell - MS_{ell-1} / b,   MS_0 := 0.
    This is exact (no bias) for any module/sample size, including the small
    branching factors (b=2..4) used throughout this codebase, unlike the old
    formula whose bias did not vanish even at large sample sizes.

    Returns {level: sigma_{level-1}^2_hat} for level in 1..depth -- i.e. key 1
    is the leaf-residual/private-noise component (sigma_0^2), key 2 is the
    level-1-module component (sigma_1^2), ..., key `depth` is the
    second-to-last, coarsest *estimable* component (sigma_{depth-1}^2). The
    outermost/root-level component sigma_depth^2 is NOT included: a single
    run has exactly one root, so there is nothing to compare it against within
    one run (this is the rigorous version of the spec's informal "V_L is
    trivially zero" note -- it is not zero, it is inestimable from one run,
    and pooling multiple independent runs/seeds would be required to recover
    it as an across-seed variance).

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
    b = branching

    # level_means[ell]: (n_modules_at_ell, d_param) time-averaged module means,
    # for ell=0 (individual leaves) through ell=depth (root).
    level_means: Dict[int, np.ndarray] = {0: traj.mean(axis=0)}  # (n_leaf_types, d_param)
    for ell in range(1, depth + 1):
        prev = level_means[ell - 1]
        n_ell = prev.shape[0] // b
        level_means[ell] = prev.reshape(n_ell, b, d_param).mean(axis=1)

    ms: Dict[int, float] = {0: 0.0}
    for ell in range(1, depth + 1):
        prev = level_means[ell - 1]              # (n_{ell-1}, d_param)
        n_prev = prev.shape[0]
        n_ell = n_prev // b
        parent_broadcast = np.repeat(level_means[ell], b, axis=0)  # (n_{ell-1}, d_param)
        ss = float(np.sum((prev - parent_broadcast) ** 2, axis=0).mean())  # sum over samples, mean over d_param
        df = n_prev - n_ell
        ms[ell] = ss / df if df > 0 else float("nan")

    result: Dict[int, float] = {}
    for ell in range(1, depth + 1):
        result[ell] = ms[ell] - ms[ell - 1] / b if not math.isnan(ms[ell]) else float("nan")

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
    """Empirical A→B rate / B→A rate per hierarchical distance from source:
    the R_l of Theorem 2/Proposition 4.6's detailed-balance test
    (R_l = e^{2*beta_eff*J_l} under reversibility at the symmetric b=0 point).

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
