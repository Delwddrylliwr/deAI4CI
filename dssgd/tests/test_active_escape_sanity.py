"""Sanity tests for the v2 active-escape experiment infrastructure.

Tests in order (as specified in §6 of the implementation plan):
1.  Bistable loss gradient ≈ 0 at A and B
2.  Loss landscape structure: L(B) < L(A), L(midpoint) > L(A)
3.  verify_bistable_loss raises ValueError on degenerate input
4.  force_flip_module changes ONLY source leaf
5.  basin_label correctness (parametrized)
6.  find_t_flip: returns run_start, requires persistence, returns None when not reached
7.  classify_regime and critical_level against default parameters
8.  Smoke test: run_active_escape_simulation with tiny config
"""

import math
import pickle

import numpy as np
import pytest
import torch

from analysis.active_escape import (
    ActiveEscapeSimConfig,
    BistableParameterModule,
    basin_label,
    find_t_flip,
    force_flip_module,
    leaf_assignments,
    make_bistable_loss_fn,
    run_active_escape_simulation,
    verify_bistable_loss,
)
from analysis import theory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THETA_A = torch.tensor([0.0])
THETA_B = torch.tensor([1.0])
A_NP = np.array([0.0])
B_NP = np.array([1.0])
DEFAULT_A = 0.5   # barrier curvature
DEFAULT_B = 0.01  # bias (must be < a * 0.096 ≈ 0.048 for bistability)


# ---------------------------------------------------------------------------
# 1 & 2. Bistable loss gradient and landscape structure
# ---------------------------------------------------------------------------


def test_bistable_loss_gradient_at_A_is_small():
    """At A, quartic gradient = 0; only the bias term contributes (-b).
    Total |grad| = b, which should be small but nonzero.
    """
    loss_fn = make_bistable_loss_fn(THETA_A, THETA_B, a=DEFAULT_A, b=DEFAULT_B)
    model = BistableParameterModule(1, init_value=THETA_A.clone())
    model.zero_grad()
    loss = loss_fn(model, None)
    loss.backward()
    grad_norm = float(model.theta.grad.norm().item())
    # Gradient at A equals the bias magnitude b=0.01 (not zero)
    assert abs(grad_norm - DEFAULT_B) < 1e-4, (
        f"Gradient at A has norm {grad_norm:.4f}, expected b={DEFAULT_B}"
    )


def test_bistable_loss_gradient_at_B_is_small():
    """At B, quartic gradient = 0; only the bias term contributes (-b)."""
    loss_fn = make_bistable_loss_fn(THETA_A, THETA_B, a=DEFAULT_A, b=DEFAULT_B)
    model = BistableParameterModule(1, init_value=THETA_B.clone())
    model.zero_grad()
    loss = loss_fn(model, None)
    loss.backward()
    grad_norm = float(model.theta.grad.norm().item())
    assert abs(grad_norm - DEFAULT_B) < 1e-4, (
        f"Gradient at B has norm {grad_norm:.4f}, expected b={DEFAULT_B}"
    )


def test_bistable_loss_B_is_global_minimum():
    """L(B) < L(A) because the linear bias term tilts toward B."""
    loss_fn = make_bistable_loss_fn(THETA_A, THETA_B, a=DEFAULT_A, b=DEFAULT_B)

    model_A = BistableParameterModule(1, init_value=THETA_A.clone())
    model_B = BistableParameterModule(1, init_value=THETA_B.clone())

    L_A = float(loss_fn(model_A, None).item())
    L_B = float(loss_fn(model_B, None).item())

    assert L_B < L_A, f"L(B)={L_B:.4f} should be less than L(A)={L_A:.4f}"


def test_bistable_loss_barrier_exists():
    """Some interior point along A→B has higher loss than L(A), confirming a barrier."""
    loss_fn = make_bistable_loss_fn(THETA_A, THETA_B, a=DEFAULT_A, b=DEFAULT_B)

    model_A = BistableParameterModule(1, init_value=THETA_A.clone())
    L_A = float(loss_fn(model_A, None).item())

    # Scan interior points; expect max > L(A)
    max_interior = max(
        float(loss_fn(BistableParameterModule(1, init_value=THETA_A + t * (THETA_B - THETA_A)), None).item())
        for t in torch.linspace(0.1, 0.9, 20)
    )
    assert max_interior > L_A, (
        f"max interior loss {max_interior:.4f} should exceed L(A)={L_A:.4f}; "
        f"no barrier detected"
    )


# ---------------------------------------------------------------------------
# 3. verify_bistable_loss raises on degenerate input
# ---------------------------------------------------------------------------


def test_verify_bistable_loss_raises_on_equal_A_B():
    theta = torch.tensor([0.5])
    with pytest.raises(ValueError, match="differ"):
        verify_bistable_loss(theta, theta, a=DEFAULT_A, b=DEFAULT_B)


def test_verify_bistable_loss_passes_on_valid_params():
    """Should not raise for standard parameters (b=0.01 << 0.048 bistability limit)."""
    verify_bistable_loss(THETA_A, THETA_B, a=DEFAULT_A, b=DEFAULT_B)


def test_verify_bistable_loss_raises_when_no_barrier():
    """b=0.05 with a=0.5 exceeds the bistability limit (b > a*0.096 ≈ 0.048)."""
    with pytest.raises(ValueError, match="barrier"):
        verify_bistable_loss(THETA_A, THETA_B, a=0.5, b=0.05)


# ---------------------------------------------------------------------------
# 4. force_flip_module changes ONLY source leaf
# ---------------------------------------------------------------------------


def test_force_flip_isolates_to_source_leaf():
    """Agents outside source_leaf must be unchanged after force_flip_module."""
    from analysis.active_escape import _dummy_loader_bistable
    from dssgd.nodes.agent import Agent
    from dssgd.nodes.registry import ModelEntry, ModelRegistry

    n_agents = 8
    leaf_size = 2
    assigns = leaf_assignments(n_agents, leaf_size)
    source_leaf = 0
    theta_B_t = THETA_B

    agents = []
    for i in range(n_agents):
        model = BistableParameterModule(1, init_value=THETA_A.clone())
        entry = ModelEntry(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=lambda m, b: torch.tensor(0.0),
            layers=frozenset({"social"}),
            train_locally=False,
        )
        agents.append(Agent(i, ModelRegistry({"model": entry}), _dummy_loader_bistable()))

    # Record pre-flip values for non-source agents
    pre_flip = {
        a.id: float(a.registry["model"].model.theta.data.item())
        for a in agents
        if assigns[a.id] != source_leaf
    }

    force_flip_module(agents, assigns, source_leaf, theta_B_t,
                      noise_scale=0.001, rng=np.random.default_rng(0))

    # Source leaf agents should now be near theta_B
    for agent in agents:
        val = float(agent.registry["model"].model.theta.data.item())
        if assigns[agent.id] == source_leaf:
            assert abs(val - 1.0) < 0.05, f"Source agent {agent.id} not flipped: {val}"
        else:
            assert abs(val - pre_flip[agent.id]) < 1e-6, (
                f"Non-source agent {agent.id} was modified: {val} vs {pre_flip[agent.id]}"
            )


# ---------------------------------------------------------------------------
# 5. basin_label correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_val,expected", [
    (0.05, 'A'),
    (0.10, 'A'),
    (0.95, 'B'),
    (0.90, 'B'),
    (0.50, 'X'),
    (0.50, 'X'),
    (0.21, 'X'),  # just outside A zone (epsilon=0.2)
    (0.79, 'X'),  # just outside B zone
])
def test_basin_label_values(theta_val, expected):
    centroid = np.array([theta_val])
    result = basin_label(centroid, A_NP, B_NP, epsilon=0.2)
    assert result == expected, f"basin_label({theta_val}) = {result!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# 6. find_t_flip behaviour
# ---------------------------------------------------------------------------


def _make_traj(labels_pattern: list) -> np.ndarray:
    """Build a centroid trajectory matching the given label pattern.

    'A' → theta=0.05, 'B' → theta=0.95, 'X' → theta=0.50.
    """
    val_map = {'A': 0.05, 'B': 0.95, 'X': 0.50}
    return np.array([[val_map[l]] for l in labels_pattern])


def test_find_t_flip_returns_run_start():
    """Should return the index where the B-run STARTS, not where persistence is confirmed."""
    # B-run starts at t=5, runs through t=15 (11 rounds)
    pattern = ['A'] * 5 + ['B'] * 11
    traj = _make_traj(pattern)
    result = find_t_flip(traj, A_NP, B_NP, epsilon=0.2, persistence=5)
    assert result == 5, f"Expected run_start=5, got {result}"


def test_find_t_flip_requires_persistence():
    """Short B-runs below persistence threshold should not trigger detection."""
    # B-run of length 3 at t=3 (< persistence=5), then B-run of length 6 at t=10
    pattern = ['A'] * 3 + ['B'] * 3 + ['A'] * 4 + ['B'] * 6
    traj = _make_traj(pattern)
    result = find_t_flip(traj, A_NP, B_NP, epsilon=0.2, persistence=5)
    assert result == 10, f"Expected 10 (second B-run), got {result}"


def test_find_t_flip_returns_none_when_never_reached():
    pattern = ['A'] * 20
    traj = _make_traj(pattern)
    result = find_t_flip(traj, A_NP, B_NP, epsilon=0.2, persistence=5)
    assert result is None


def test_find_t_flip_returns_none_when_only_short_runs():
    # Only runs of length 4, never reaches persistence=5
    pattern = ['A'] * 2 + ['B'] * 4 + ['A'] * 2 + ['B'] * 4
    traj = _make_traj(pattern)
    result = find_t_flip(traj, A_NP, B_NP, epsilon=0.2, persistence=5)
    assert result is None


# ---------------------------------------------------------------------------
# 7. Theory: classify_regime and critical_level
# ---------------------------------------------------------------------------


def test_classify_regime_default_params():
    """Default parameters: γ=1, m=4, p=2, L=5 → γmp=8, γ_1=2, γ_L=0.125.

    With p=2 (sparse Safari regime):
    Regime I  boundary: κ_loc/2 < γ_5=0.125, i.e. a < 0.5.
    Regime III boundary: κ_loc/2 ≥ γ_1=2,   i.e. a ≥ 8.
    """
    # a=0.3: κ_loc=0.15, κ_loc/2=0.075 < γ_5=0.125 → Regime I
    assert theory.classify_regime(0.3, 1.0, 4, 2.0, 5) == 'I'
    # a=2.0: κ_loc/2=0.5 ≥ γ_5=0.125 and < γ_1=2 → Regime II
    assert theory.classify_regime(2.0, 1.0, 4, 2.0, 5) == 'II'
    # a=10.0: κ_loc/2=2.5 ≥ γ_1=2 → Regime III
    assert theory.classify_regime(10.0, 1.0, 4, 2.0, 5) == 'III'


def test_classify_regime_boundary():
    """Regime II/III boundary: κ_loc/2 = γ_1=2, i.e. a/4 = 2, i.e. a = 8.

    With p=2: boundary shifts from a=16 (p=4) to a=8.
    """
    # a=7: κ_loc/2=1.75 < γ_1=2 → Regime II
    r = theory.classify_regime(7.0, 1.0, 4, 2.0, 5)
    assert r == 'II', f"a=7 should be Regime II with p=2, got {r}"
    # a=8: κ_loc/2=2.0 ≥ γ_1=2 → Regime III (boundary, >= condition)
    r_boundary = theory.classify_regime(8.0, 1.0, 4, 2.0, 5)
    assert r_boundary == 'III', f"a=8 should be Regime III with p=2, got {r_boundary}"
    # a=9: clearly Regime III
    r_iii = theory.classify_regime(9.0, 1.0, 4, 2.0, 5)
    assert r_iii == 'III', f"a=9 should be Regime III with p=2, got {r_iii}"


def test_critical_level_regime_I():
    """In Regime I with p=2 (a=0.3), all 5 levels flip deterministically → ell_c=5."""
    ell_c = theory.critical_level(0.3, 1.0, 4, 2.0, 5)
    assert ell_c == 5, f"Expected ell_c=5 for a=0.3 p=2, got {ell_c}"


def test_critical_level_regime_III():
    """In Regime III with p=2 (a=20), no levels flip → ell_c=0."""
    ell_c = theory.critical_level(20.0, 1.0, 4, 2.0, 5)
    assert ell_c == 0, f"Expected ell_c=0 for a=20 p=2, got {ell_c}"


def test_deterministic_flip_time_scaling():
    """T_flip(d) should double with each unit increase in d (independent of p)."""
    gamma, m, p = 1.0, 4, 2.0
    t1 = theory.deterministic_flip_time(1, gamma, m, p)
    t2 = theory.deterministic_flip_time(2, gamma, m, p)
    t3 = theory.deterministic_flip_time(3, gamma, m, p)
    assert abs(t2 / t1 - 2.0) < 1e-10, f"T_flip(2)/T_flip(1) = {t2/t1}, expected 2"
    assert abs(t3 / t2 - 2.0) < 1e-10, f"T_flip(3)/T_flip(2) = {t3/t2}, expected 2"


def test_kappa_loc_formula():
    assert abs(theory.kappa_loc(0.5, 1.0) - 0.25) < 1e-10
    assert abs(theory.kappa_loc(2.0, 1.0) - 1.00) < 1e-10
    assert abs(theory.kappa_loc(0.5, 2.0) - 1.00) < 1e-10  # δ_norm=2


# ---------------------------------------------------------------------------
# 8. Smoke test — full simulation with tiny config
# ---------------------------------------------------------------------------


def test_run_active_escape_smoke():
    """Full end-to-end smoke test with depth=2 (4 leaf types, 8 agents)."""
    config = ActiveEscapeSimConfig(
        name="smoke/test",
        branching=2,
        depth=2,
        leaf_size=2,
        p=4.0,
        seed=42,
        n_warmup=5,
        n_meas_rounds=10,
        lr=0.01,
        a=0.5,
        b=DEFAULT_B,
        source_leaf=0,
        d_param=1,
    )
    run = run_active_escape_simulation(config)

    n_leaf_types = 2 ** config.depth  # = 4
    assert run.centroid_traj.shape == (config.n_meas_rounds + 1, n_leaf_types, config.d_param), (
        f"Unexpected centroid_traj shape: {run.centroid_traj.shape}"
    )
    # flip_table has one entry per non-source leaf
    assert len(run.flip_table) == n_leaf_types - 1, (
        f"Expected {n_leaf_types-1} rows in flip_table, got {len(run.flip_table)}"
    )
    # All distances are 1 or 2 for depth=2
    assert all(r["distance_d"] in [1, 2] for r in run.flip_table)
    # regime and ell_c are populated
    assert run.regime in ('I', 'II', 'III')
    assert isinstance(run.ell_c, int)
    # Result is picklable
    dumped = pickle.dumps(run)
    restored = pickle.loads(dumped)
    assert restored.name == run.name
    assert restored.centroid_traj.shape == run.centroid_traj.shape


def test_run_active_escape_warmup_flag():
    """warmup_ok should be True when agents start near theta_A."""
    config = ActiveEscapeSimConfig(
        name="smoke/warmup_flag",
        branching=2,
        depth=2,
        leaf_size=2,
        p=4.0,
        seed=0,
        n_warmup=20,
        n_meas_rounds=5,
        lr=0.01,
        a=0.5,
        b=DEFAULT_B,
        source_leaf=0,
    )
    run = run_active_escape_simulation(config)
    assert run.warmup_ok, (
        "warmup_ok should be True for standard Regime I parameters with small init noise"
    )


def test_source_leaf_centroid_near_B_after_flip():
    """After force-flip, source leaf centroid should be in basin B at t=0."""
    config = ActiveEscapeSimConfig(
        name="smoke/source_flip",
        branching=2,
        depth=2,
        leaf_size=2,
        p=4.0,
        seed=7,
        n_warmup=5,
        n_meas_rounds=5,
        lr=0.01,
        a=0.5,
        b=DEFAULT_B,
        source_leaf=0,
    )
    run = run_active_escape_simulation(config)
    source_centroid_t0 = run.centroid_traj[0, config.source_leaf, :]
    label = basin_label(source_centroid_t0, run.theta_A, run.theta_B, epsilon=0.2)
    assert label == 'B', (
        f"Source leaf centroid at t=0 should be in B; got {label}, value={source_centroid_t0}"
    )


def test_natural_cascade_flips_without_clamping():
    """With b≈0.042 and local_steps=50, the source-leaf should self-sustain in B
    and the d=1 cousin should flip within a short measurement window.

    Two-timescale regime: local SGD equilibrates agents before each gossip round.
    b≈0.042 shifts the saddle from θ≈0.46 to θ≈0.30, below the gossip mixing
    point (~0.33), so the cousin's local gradient points toward B after one round.
    """
    cfg = ActiveEscapeSimConfig(
        name="test_natural/smoke",
        depth=3,
        leaf_size=4,
        p=4.0,
        seed=0,
        n_warmup=20,
        n_meas_rounds=200,
        lr=0.1,
        local_steps=50,
        a=0.5,
        b=0.042,
        clamp_source=False,
    )
    run = run_active_escape_simulation(cfg)
    assert run.warmup_ok
    d1_flips = [r["t_flip"] for r in run.flip_table if r["distance_d"] == 1]
    assert any(t is not None for t in d1_flips), (
        "Natural cascade should flip d=1 leaf within 200 rounds "
        "(b=0.042 shifts saddle to ~0.30, local_steps=50 approximates local equilibration)"
    )
