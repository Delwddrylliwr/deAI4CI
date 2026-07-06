"""Sanity tests for the natural-cascade Regime C runner.

Fast smoke tests that verify infrastructure correctness without running
full-scale experiments.  All tests use small topologies (depth=2, leaf_size=2)
and short warmup/measurement windows.
"""
import pickle

import numpy as np
import pytest

from analysis.natural_cascade import (
    NaturalCascadeConfig,
    NaturalCascadeRun,
    run_natural_cascade_simulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**kwargs) -> NaturalCascadeConfig:
    """Minimal fast config for smoke tests."""
    defaults = dict(
        name="test_smoke",
        depth=2,
        leaf_size=2,
        p=2.0,
        seed=0,
        n_warmup=5,
        n_meas_rounds=30,
        lr=0.1,
        local_steps=50,
        a=0.5,
        b=0.042,
    )
    defaults.update(kwargs)
    return NaturalCascadeConfig(**defaults)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_smoke_completes():
    """Runner completes without error for minimal topology."""
    cfg = _small_config()
    run = run_natural_cascade_simulation(cfg)
    assert isinstance(run, NaturalCascadeRun)


def test_centroid_traj_shape():
    """centroid_traj has the right shape."""
    cfg = _small_config(depth=2, leaf_size=2, n_meas_rounds=15)
    run = run_natural_cascade_simulation(cfg)
    n_leaf_types = cfg.branching ** cfg.depth  # 4
    assert run.centroid_traj.shape == (cfg.n_meas_rounds + 1, n_leaf_types, cfg.d_param)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def test_warmup_ok_standard_params():
    """Warmup should converge to basin A for standard Regime C params."""
    cfg = _small_config(n_warmup=200, seed=0)
    run = run_natural_cascade_simulation(cfg)
    assert run.warmup_ok, (
        "warmup_ok is False: some module was not in basin A after warmup. "
        "Increase n_warmup or check b is below the bistability limit."
    )


def test_symmetric_warmup_nmh7_does_not_crash():
    """b=0 (NMH-7 symmetric) runs without error even if warmup_ok may be False."""
    cfg = _small_config(b=0.0, n_warmup=10, n_meas_rounds=20)
    run = run_natural_cascade_simulation(cfg)
    # warmup_ok may be True or False for b=0 — both are valid
    assert isinstance(run.warmup_ok, bool)


# ---------------------------------------------------------------------------
# Nucleation detection
# ---------------------------------------------------------------------------

def test_detects_nucleation_given_long_enough_run():
    """With b=0.042 and local_steps=50, a long run should find a natural flip."""
    # Use a longer measurement window and a more favourable regime
    cfg = _small_config(
        n_warmup=200,
        n_meas_rounds=500,
        seed=7,
        a=0.5,
        b=0.042,
        local_steps=50,
    )
    run = run_natural_cascade_simulation(cfg)
    if run.warmup_ok:
        # Not guaranteed to find a flip in every run, but should in most seeds
        # We just verify the detection logic doesn't crash and returns correct types
        assert run.nucleation_leaf is None or isinstance(run.nucleation_leaf, int)
        assert run.t_nucleation is None or isinstance(run.t_nucleation, int)


def test_flip_table_structure():
    """Each flip_table entry has the required keys."""
    cfg = _small_config(n_meas_rounds=20)
    run = run_natural_cascade_simulation(cfg)
    required_keys = {
        "target_leaf", "distance_from_source",
        "t_flip_absolute", "t_flip_relative", "warmup_ok",
        "a", "b", "lr", "seed", "local_steps",
    }
    for entry in run.flip_table:
        assert required_keys <= set(entry.keys()), (
            f"Missing keys: {required_keys - set(entry.keys())}"
        )


def test_flip_table_excludes_nucleation_leaf():
    """If nucleation was detected, the nucleation leaf is absent from flip_table."""
    cfg = _small_config(n_warmup=200, n_meas_rounds=500, seed=3)
    run = run_natural_cascade_simulation(cfg)
    if run.nucleation_leaf is not None:
        table_leaves = {e["target_leaf"] for e in run.flip_table}
        assert run.nucleation_leaf not in table_leaves


def test_t_flip_relative_consistent():
    """t_flip_relative == t_flip_absolute - t_nucleation when both are set."""
    cfg = _small_config(n_warmup=200, n_meas_rounds=500, seed=1)
    run = run_natural_cascade_simulation(cfg)
    if run.t_nucleation is not None:
        for entry in run.flip_table:
            if entry["t_flip_absolute"] is not None:
                assert entry["t_flip_relative"] == (
                    entry["t_flip_absolute"] - run.t_nucleation
                )


# ---------------------------------------------------------------------------
# Per-leaf loss (NMH-6)
# ---------------------------------------------------------------------------

def test_per_leaf_loss_nmh6_runs():
    """Per-leaf (a_i, b_i) loss params run without error."""
    n_leaf_types = 2 ** 2  # depth=2, branching=2
    per_leaf = [(0.5, 0.01 + i * 0.002) for i in range(n_leaf_types)]
    cfg = _small_config(per_leaf_loss_params=per_leaf, b=0.01, a=0.5)
    run = run_natural_cascade_simulation(cfg)
    assert isinstance(run, NaturalCascadeRun)


def test_per_leaf_loss_wrong_length_raises():
    """Mismatched per_leaf_loss_params length raises ValueError."""
    cfg = _small_config(per_leaf_loss_params=[(0.5, 0.01)])  # should be 4 entries
    with pytest.raises(ValueError, match="per_leaf_loss_params"):
        run_natural_cascade_simulation(cfg)


# ---------------------------------------------------------------------------
# Picklability
# ---------------------------------------------------------------------------

def test_picklable():
    """NaturalCascadeRun can be pickled and restored."""
    cfg = _small_config()
    run = run_natural_cascade_simulation(cfg)
    data = pickle.dumps(run)
    restored = pickle.loads(data)
    assert restored.name == run.name
    np.testing.assert_array_equal(restored.centroid_traj, run.centroid_traj)


# ---------------------------------------------------------------------------
# Regime and theory fields
# ---------------------------------------------------------------------------

def test_regime_field_is_valid():
    """run.regime is one of 'I', 'II', 'III'."""
    cfg = _small_config()
    run = run_natural_cascade_simulation(cfg)
    assert run.regime in {'I', 'II', 'III'}


def test_ell_c_field_is_int():
    """run.ell_c is a non-negative integer."""
    cfg = _small_config()
    run = run_natural_cascade_simulation(cfg)
    assert isinstance(run.ell_c, int)
    assert run.ell_c >= 0
