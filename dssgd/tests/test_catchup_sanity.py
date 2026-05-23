"""Sanity checks for the catch-up experiment infrastructure.

Fast tests (< 60 s) that verify:
  1. Type assignment correctness (exactly leaf_size agents per leaf)
  2. Cousin relations (hierarchical_distance(i, cousin_at_level(i, ℓ)) == ℓ)
  3. find_t_50 returns correct index on a synthetic trajectory
  4. inject_perturbation shifts params by the expected amount
  5. run_catchup_simulation smoke test (short warmup + measurement, picklable)
  6. run_sanity_checks passes without raising
"""
from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch

from analysis.catchup import (
    CatchupSimConfig,
    cousin_at_level,
    find_t_50,
    hierarchical_distance,
    inject_perturbation,
    leaf_assignments,
    run_catchup_simulation,
    run_sanity_checks,
    within_leaf_spread,
)
from analysis.catchup_experiments import experiment_A
from analysis import theory


# ---------------------------------------------------------------------------
# 1. Leaf assignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf_size,depth", [(4, 3), (2, 4), (8, 2)])
def test_leaf_assignments_count(leaf_size: int, depth: int):
    n_leaf_types = 2 ** depth
    n_agents = n_leaf_types * leaf_size
    assigns = leaf_assignments(n_agents, leaf_size)
    for tau in range(n_leaf_types):
        count = int((assigns == tau).sum())
        assert count == leaf_size, f"Leaf {tau}: got {count}, expected {leaf_size}"


def test_leaf_assignments_contiguous():
    assigns = leaf_assignments(32, 4)
    for i in range(32):
        assert assigns[i] == i // 4


# ---------------------------------------------------------------------------
# 2. Hierarchical distance and cousin relations
# ---------------------------------------------------------------------------


def test_hierarchical_distance_self():
    for i in range(32):
        assert hierarchical_distance(i, i) == 0


def test_hierarchical_distance_siblings():
    # leaves 0 (00000) and 1 (00001) differ only in bit 0 → distance 1
    assert hierarchical_distance(0, 1) == 1
    assert hierarchical_distance(2, 3) == 1
    assert hierarchical_distance(4, 5) == 1


def test_hierarchical_distance_larger():
    # 0 (00000) vs 2 (00010): highest differing bit = bit 1 → distance 2
    assert hierarchical_distance(0, 2) == 2
    # 0 (00000) vs 4 (00100): bit 2 → distance 3
    assert hierarchical_distance(0, 4) == 3


def test_cousin_at_level_roundtrip():
    depth = 5
    n_leaf_types = 2 ** depth
    for i in range(n_leaf_types):
        for ell in range(1, depth + 1):
            cousin = cousin_at_level(i, ell)
            assert 0 <= cousin < n_leaf_types
            d = hierarchical_distance(i, cousin)
            assert d == ell, (
                f"hierarchical_distance({i}, cousin_at_level({i},{ell})={cousin}) = {d}, expected {ell}"
            )


def test_cousin_involution():
    # Applying cousin_at_level twice returns the original leaf
    depth = 5
    for i in range(2 ** depth):
        for ell in range(1, depth + 1):
            assert cousin_at_level(cousin_at_level(i, ell), ell) == i


# ---------------------------------------------------------------------------
# 3. find_t_50 on synthetic trajectory
# ---------------------------------------------------------------------------


def test_find_t50_detects_threshold():
    n_params = 4
    n_types = 1
    delta_dir = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    threshold = 0.5

    # Centroid increases linearly from 0; crosses threshold at t=5
    traj = np.zeros((20, n_params), dtype=np.float32)
    for t in range(20):
        traj[t, 0] = t * 0.1

    initial = np.zeros(n_params, dtype=np.float32)
    t50 = find_t_50(traj, initial, delta_dir, threshold)
    assert t50 == 5


def test_find_t50_returns_none_when_not_reached():
    n_params = 4
    delta_dir = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    threshold = 10.0

    traj = np.zeros((10, n_params), dtype=np.float32)
    traj[:, 0] = 0.1  # always below threshold

    t50 = find_t_50(traj, np.zeros(n_params), delta_dir, threshold)
    assert t50 is None


def test_find_t50_first_crossing():
    # Multiple crossings: should return the first
    n_params = 2
    delta_dir = np.array([1.0, 0.0], dtype=np.float32)
    traj = np.array([[0.0, 0.0], [0.6, 0.0], [0.4, 0.0], [0.8, 0.0]], dtype=np.float32)
    t50 = find_t_50(traj, np.zeros(2), delta_dir, 0.5)
    assert t50 == 1


# ---------------------------------------------------------------------------
# 4. inject_perturbation modifies parameters in place
# ---------------------------------------------------------------------------


def _make_simple_agents(n: int, n_params: int):
    """Create minimal agents with nn.Linear for perturbation testing."""
    from dssgd.nodes.agent import Agent
    from dssgd.nodes.registry import ModelEntry, ModelRegistry
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    agents = []
    for i in range(n):
        model = nn.Linear(n_params, 1, bias=False)
        nn.init.zeros_(model.weight)
        entry = ModelEntry(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=lambda m, b: torch.tensor(0.0),
            layers=frozenset({"social"}),
            train_locally=False,
            local_steps=1,
        )
        x = torch.zeros(4, n_params)
        y = torch.zeros(4, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        agents.append(Agent(i, ModelRegistry({"model": entry}), loader))
    return agents


def test_inject_perturbation_correct_agents():
    n_agents = 8
    leaf_size = 4
    n_params = 3
    assigns = leaf_assignments(n_agents, leaf_size)
    agents = _make_simple_agents(n_agents, n_params)

    delta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    source_leaf = 0
    inject_perturbation(agents, assigns, source_leaf, delta)

    for a in agents:
        w = a.registry["model"].model.weight.detach().numpy().flatten()
        if assigns[a.id] == source_leaf:
            np.testing.assert_allclose(w, delta, atol=1e-5)
        else:
            np.testing.assert_allclose(w, np.zeros(n_params), atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Smoke test: run_catchup_simulation is fast and picklable
# ---------------------------------------------------------------------------


def test_run_catchup_smoke():
    config = CatchupSimConfig(
        name="smoke/test",
        branching=2,
        depth=3,
        leaf_size=2,
        p=4.0,
        seed=0,
        n_warmup=5,
        n_meas_rounds=10,
        lr=0.05,
        source_leaf=0,
        model_factory=lambda: torch.nn.Linear(8, 1, bias=False),
        loss_fn=lambda m, b: torch.nn.functional.mse_loss(m(b[0]), b[1]),
        data_loaders_factory=lambda n: _make_regression_loaders_simple(n),
    )
    run = run_catchup_simulation(config)

    # All 2^3 - 1 = 7 target leaves should have an entry
    assert len(run.t50_table) == 2 ** config.depth - 1

    # Every entry has a distance_d in [1, depth]
    for row in run.t50_table:
        assert 1 <= row["distance_d"] <= config.depth

    # Centroid trajectory has the right shape
    n_leaf_types = 2 ** config.depth
    assert run.centroid_traj.shape[0] == config.n_meas_rounds + 1
    assert run.centroid_traj.shape[1] == n_leaf_types

    # Must be picklable
    dumped = pickle.dumps(run)
    restored = pickle.loads(dumped)
    assert len(restored.t50_table) == len(run.t50_table)


def _make_regression_loaders_simple(n_agents: int):
    from torch.utils.data import DataLoader, TensorDataset
    loaders = []
    for i in range(n_agents):
        torch.manual_seed(i)
        x = torch.randn(16, 8)
        y = x @ torch.randn(8) + 0.1 * torch.randn(16)
        loaders.append(DataLoader(TensorDataset(x, y.unsqueeze(1)), batch_size=8))
    return loaders


# ---------------------------------------------------------------------------
# 6. run_sanity_checks passes for default parameters
# ---------------------------------------------------------------------------


def test_sanity_checks_pass():
    run_sanity_checks(branching=2, depth=5, leaf_size=4, p=4.0)


def test_sanity_checks_small():
    run_sanity_checks(branching=2, depth=3, leaf_size=2, p=2.0)


# ---------------------------------------------------------------------------
# 7. Theory formulas sanity
# ---------------------------------------------------------------------------


def test_catch_up_time_doubles_with_distance():
    gamma, m, p = 1.0, 4, 4.0
    for d in range(1, 5):
        ratio = theory.catch_up_time(d + 1, gamma, m, p) / theory.catch_up_time(d, gamma, m, p)
        assert abs(ratio - 2.0) < 1e-10, f"T_catch ratio at d={d}: {ratio}"


def test_total_coupling_strength_positive():
    gt = theory.total_coupling_strength(1.0, 4, 4.0, 5)
    assert gt > 0


def test_gossip_timescale_decreases_with_level():
    gamma, m, p = 1.0, 4, 4.0
    taus = [theory.gossip_timescale(gamma, m, p, ell) for ell in range(1, 6)]
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i + 1], "τ_gossip should increase with level"
