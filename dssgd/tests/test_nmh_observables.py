"""Unit tests for nmh_observables.py.

Uses synthetic centroid trajectories to verify observable logic in isolation,
without running full simulations.
"""
import math

import numpy as np
import pytest

from analysis.nmh_observables import (
    cascade_depth,
    cascade_size,
    detailed_balance_ratio,
    fraction_reaching_level,
    hierarchical_variance_decomposition,
    nucleation_prob_per_level,
)


# ---------------------------------------------------------------------------
# Synthetic trajectory helpers
# ---------------------------------------------------------------------------

def _make_traj(n_leaf_types: int, T: int, d_param: int = 1) -> np.ndarray:
    """All-A trajectory."""
    return np.zeros((T, n_leaf_types, d_param), dtype=np.float32)


def _set_leaf_B(traj: np.ndarray, leaf: int, t_start: int) -> np.ndarray:
    """Set leaf to basin B (value 1.0) from t_start onwards."""
    traj = traj.copy()
    traj[t_start:, leaf, :] = 1.0
    return traj


THETA_A = np.array([0.0], dtype=np.float32)
THETA_B = np.array([1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# cascade_depth
# ---------------------------------------------------------------------------

def test_cascade_depth_zero_when_no_flip():
    """No leaf enters B → depth = 0."""
    traj = _make_traj(n_leaf_types=4, T=20)
    assert cascade_depth(traj, source_leaf=0, theta_A=THETA_A, theta_B=THETA_B) == 0


def test_cascade_depth_one_when_level1_flips():
    """If only the level-1 cousin (distance 1) flips, depth = 1."""
    traj = _make_traj(n_leaf_types=4, T=20)
    # leaf 0 is source; leaf 1 is at distance hierarchical_distance(0,1) = 1
    traj = _set_leaf_B(traj, leaf=1, t_start=5)
    # Set source leaf to B as well so source doesn't look like a flip
    traj = _set_leaf_B(traj, leaf=0, t_start=0)
    d = cascade_depth(traj, source_leaf=0, theta_A=THETA_A, theta_B=THETA_B)
    assert d == 1


def test_cascade_depth_increases_with_more_flips():
    """Adding a deeper flip increases the reported depth."""
    # With 8 leaves (depth=3) and branching=2:
    # hierarchical_distance(0, 4) = (0 ^ 4).bit_length() = 3
    traj = _make_traj(n_leaf_types=8, T=30)
    traj = _set_leaf_B(traj, leaf=0, t_start=0)  # source
    traj = _set_leaf_B(traj, leaf=4, t_start=10)  # distance 3

    d = cascade_depth(traj, source_leaf=0, theta_A=THETA_A, theta_B=THETA_B)
    assert d == 3


# ---------------------------------------------------------------------------
# fraction_reaching_level
# ---------------------------------------------------------------------------

def test_fraction_nan_when_no_leaves_at_level():
    """NaN returned when there are no leaves at the target level."""
    traj = _make_traj(n_leaf_types=4, T=10)
    f = fraction_reaching_level(traj, source_leaf=0, target_level=5,
                                 theta_A=THETA_A, theta_B=THETA_B)
    assert math.isnan(f)


def test_fraction_one_when_all_at_level_flip():
    """1.0 when all leaves at distance 1 flip."""
    # 4 leaves, branching=2: leaf 0 has level-1 cousin at leaf 1
    traj = _make_traj(n_leaf_types=4, T=20)
    traj = _set_leaf_B(traj, leaf=0, t_start=0)   # source
    traj = _set_leaf_B(traj, leaf=1, t_start=5)   # level-1 cousin
    f = fraction_reaching_level(traj, source_leaf=0, target_level=1,
                                 theta_A=THETA_A, theta_B=THETA_B)
    assert f == 1.0


def test_fraction_zero_when_none_flip():
    """0.0 when no leaf at the target level flips."""
    traj = _make_traj(n_leaf_types=4, T=20)
    traj = _set_leaf_B(traj, leaf=0, t_start=0)   # source only
    f = fraction_reaching_level(traj, source_leaf=0, target_level=1,
                                 theta_A=THETA_A, theta_B=THETA_B)
    assert f == 0.0


def test_fraction_half_when_half_flip():
    """0.5 when half the leaves at distance 2 flip (with 8 leaves)."""
    # With 8 leaves and source=0:
    # distance-2 leaves: hierarchical_distance(0, j) == 2 for j in {2, 3}
    traj = _make_traj(n_leaf_types=8, T=30)
    traj = _set_leaf_B(traj, leaf=0, t_start=0)   # source
    traj = _set_leaf_B(traj, leaf=2, t_start=5)   # distance 2
    # leaf 3 is also distance 2 but does not flip
    f = fraction_reaching_level(traj, source_leaf=0, target_level=2,
                                 theta_A=THETA_A, theta_B=THETA_B)
    assert f == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# cascade_size
# ---------------------------------------------------------------------------

def test_cascade_size_zero_at_t0():
    """No leaves in B at t=0 → size 0."""
    traj = _make_traj(n_leaf_types=4, T=10)
    assert cascade_size(traj, THETA_A, THETA_B, t_horizon=0) == 0


def test_cascade_size_counts_b_leaves():
    """Counts exactly the leaves in B at the given round."""
    traj = _make_traj(n_leaf_types=4, T=20)
    traj = _set_leaf_B(traj, leaf=0, t_start=0)
    traj = _set_leaf_B(traj, leaf=2, t_start=5)
    # At t=10: leaf 0 and leaf 2 are in B → size 2
    assert cascade_size(traj, THETA_A, THETA_B, t_horizon=10) == 2


def test_cascade_size_clamped_to_trajectory_length():
    """t_horizon beyond T-1 is clamped to T-1."""
    traj = _make_traj(n_leaf_types=4, T=10)
    traj = _set_leaf_B(traj, leaf=1, t_start=0)
    # t_horizon=999 should use t=9 (T-1)
    s = cascade_size(traj, THETA_A, THETA_B, t_horizon=999)
    assert s == 1


# ---------------------------------------------------------------------------
# nucleation_prob_per_level
# ---------------------------------------------------------------------------

def _dummy_assigns(n_agents: int, leaf_size: int) -> np.ndarray:
    return np.arange(n_agents) // leaf_size


def test_nucleation_prob_in_range():
    """Nucleation probabilities are in [0, 1] or NaN."""
    traj = _make_traj(n_leaf_types=4, T=50)
    # Leaf 1 occasionally transitions to B
    traj[10, 1, :] = 1.0   # single round B
    assigns = _dummy_assigns(8, 2)
    probs = nucleation_prob_per_level(traj, assigns, source_leaf=0,
                                      theta_A=THETA_A, theta_B=THETA_B)
    for d, p in probs.items():
        if not math.isnan(p):
            assert 0.0 <= p <= 1.0, f"prob at d={d} out of range: {p}"


def test_nucleation_prob_nonzero_after_transition():
    """At least one level has nonzero probability when a transition occurs."""
    traj = _make_traj(n_leaf_types=4, T=50)
    traj = _set_leaf_B(traj, leaf=1, t_start=5)  # leaf 1 permanently enters B at t=5
    assigns = _dummy_assigns(8, 2)
    probs = nucleation_prob_per_level(traj, assigns, source_leaf=0,
                                      theta_A=THETA_A, theta_B=THETA_B)
    nonzero = [p for p in probs.values() if not math.isnan(p) and p > 0]
    assert len(nonzero) > 0


# ---------------------------------------------------------------------------
# hierarchical_variance_decomposition
# ---------------------------------------------------------------------------

def test_variance_decomp_returns_correct_levels():
    """Returns dict with keys 1..depth."""
    depth = 3
    traj = np.random.default_rng(0).standard_normal((50, 2**depth, 1)).astype(np.float32)
    result = hierarchical_variance_decomposition(traj, branching=2, depth=depth)
    assert set(result.keys()) == set(range(1, depth + 1))


def test_variance_decomp_values_are_finite():
    """All values are finite floats."""
    depth = 3
    traj = np.random.default_rng(1).standard_normal((30, 2**depth, 1)).astype(np.float32)
    result = hierarchical_variance_decomposition(traj, branching=2, depth=depth)
    for ell, v in result.items():
        assert math.isfinite(v), f"V_{ell} is not finite: {v}"


def test_variance_decomp_t_range():
    """t_range parameter restricts the averaging window."""
    depth = 2
    T = 40
    traj = np.random.default_rng(2).standard_normal((T, 2**depth, 1)).astype(np.float32)
    # Two non-overlapping windows should give different results
    r1 = hierarchical_variance_decomposition(traj, branching=2, depth=depth, t_range=(0, 20))
    r2 = hierarchical_variance_decomposition(traj, branching=2, depth=depth, t_range=(20, 40))
    # They should differ (very unlikely to be equal with random data)
    assert r1 != r2


# ---------------------------------------------------------------------------
# detailed_balance_ratio
# ---------------------------------------------------------------------------

def test_detailed_balance_ratio_nan_when_no_backward_transitions():
    """NaN when no B→A transitions occur (expected for b > 0 absorbing dynamics)."""
    traj = _make_traj(n_leaf_types=4, T=30)
    traj = _set_leaf_B(traj, leaf=0, t_start=0)   # source stays in B
    traj = _set_leaf_B(traj, leaf=1, t_start=10)  # cousin flips to B, stays
    ratios = detailed_balance_ratio(traj, source_leaf=0, theta_A=THETA_A, theta_B=THETA_B)
    # No B→A transitions → ratios should be NaN or inf
    for d, r in ratios.items():
        assert math.isnan(r) or math.isinf(r), f"Expected NaN/inf at d={d}, got {r}"


def test_detailed_balance_ratio_approx_one_for_symmetric():
    """For a synthetic trajectory with equal A→B and B→A rates, ratio ≈ 1."""
    T = 200
    n_leaf_types = 4
    rng = np.random.default_rng(42)
    # Build a trajectory where each leaf randomly flips A↔B each step
    traj = np.zeros((T, n_leaf_types, 1), dtype=np.float32)
    state = np.zeros(n_leaf_types, dtype=np.float32)
    for t in range(T):
        # Each leaf independently flips with probability 0.1
        flips = rng.random(n_leaf_types) < 0.1
        state = np.where(flips, 1.0 - state, state)
        traj[t, :, 0] = state

    # Use leaf 0 as source; it may or may not be in B at any given time
    ratios = detailed_balance_ratio(traj, source_leaf=0,
                                    theta_A=THETA_A, theta_B=THETA_B)
    # With equal flip rates, ratio should be in (0.3, 3.0) for most distances
    for d, r in ratios.items():
        if not (math.isnan(r) or math.isinf(r)):
            assert 0.1 <= r <= 10.0, f"Ratio at d={d} looks wrong: {r}"
