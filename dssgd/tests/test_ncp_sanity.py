"""Sanity tests for the NCP Regime C runner and observables.

Uses small graphs (n ≤ 30) and short runs for speed.
"""
import math
import pickle

import numpy as np
import pytest

from analysis.ncp_runner import NCPSimConfig, NCPRun, run_ncp_simulation
from analysis.ncp_observables import (
    shell_modal_basin,
    decoupling_chi,
    conditional_mutual_information,
)
from dssgd.topology.forest_fire import ForestFireTopology


THETA_A = np.array([0.0], dtype=np.float32)
THETA_B = np.array([1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**kwargs) -> NCPSimConfig:
    defaults = dict(
        name="test_ncp_smoke",
        n_nodes=20,
        p_f=0.35,
        seed=0,
        n_warmup=5,
        n_meas_rounds=20,
        lr=0.1,
        local_steps=10,
        a=0.5,
        b=0.042,
    )
    defaults.update(kwargs)
    return NCPSimConfig(**defaults)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_smoke_completes():
    """NCP runner completes without error for small graph."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    assert isinstance(run, NCPRun)


def test_centroid_traj_shape():
    """centroid_traj has shape (n_meas+1, n_nodes, d_param)."""
    cfg = _small_config(n_meas_rounds=15)
    run = run_ncp_simulation(cfg)
    assert run.centroid_traj.shape == (16, cfg.n_nodes, cfg.d_param)


def test_shell_traj_keys_match_shell_assigns():
    """shell_traj keys match the set of shell indices from shell_assigns."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    assert set(run.shell_traj.keys()) == set(run.shell_assigns.values())


def test_shell_traj_shapes():
    """Each shell_traj entry has shape (n_meas+1, d_param)."""
    cfg = _small_config(n_meas_rounds=10)
    run = run_ncp_simulation(cfg)
    T = cfg.n_meas_rounds + 1
    for shell, traj in run.shell_traj.items():
        assert traj.shape == (T, cfg.d_param), (
            f"shell {shell}: expected ({T}, {cfg.d_param}), got {traj.shape}"
        )


def test_graph_stats_populated():
    """graph_stats dict has the expected keys with sensible values."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    assert "n_nodes" in run.graph_stats
    assert run.graph_stats["n_nodes"] == cfg.n_nodes
    assert run.graph_stats["n_shells"] >= 1
    assert run.graph_stats["mean_degree"] > 0


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def test_warmup_ok_type():
    """warmup_ok is a bool."""
    cfg = _small_config(n_warmup=50)
    run = run_ncp_simulation(cfg)
    assert isinstance(run.warmup_ok, bool)


# ---------------------------------------------------------------------------
# Clamped shell
# ---------------------------------------------------------------------------

def test_clamped_shell_nodes_near_B():
    """Clamped-shell nodes have parameters close to theta_B at final round."""
    # First build the topology to find the innermost (highest k-core) shell
    topo = ForestFireTopology(n=25, p_f=0.35, seed=3)
    shells = topo.shell_assignment()
    max_shell = max(shells.values())

    cfg = _small_config(
        n_nodes=25,
        seed=3,
        n_meas_rounds=10,
        clamped_shell=max_shell,
    )
    run = run_ncp_simulation(cfg)

    clamped_nodes = [n for n, s in run.shell_assigns.items() if s == max_shell]
    final_t = run.centroid_traj.shape[0] - 1
    for node in clamped_nodes:
        param = run.centroid_traj[final_t, node, 0]
        assert param > 0.7, (
            f"Clamped node {node} at t={final_t} has param={param:.3f}; "
            "expected near theta_B=1.0"
        )


# ---------------------------------------------------------------------------
# Flip table
# ---------------------------------------------------------------------------

def test_flip_table_structure():
    """Each flip_table entry has the required keys."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    required = {"node", "shell", "t_flip_absolute", "t_flip_relative", "warmup_ok"}
    for entry in run.flip_table:
        assert required <= set(entry.keys())


def test_flip_table_excludes_nucleation_node():
    """nucleation_node is absent from flip_table (if detected)."""
    cfg = _small_config(n_meas_rounds=30, n_warmup=50)
    run = run_ncp_simulation(cfg)
    if run.nucleation_node is not None:
        table_nodes = {e["node"] for e in run.flip_table}
        assert run.nucleation_node not in table_nodes


# ---------------------------------------------------------------------------
# Picklability
# ---------------------------------------------------------------------------

def test_picklable():
    """NCPRun can be pickled and restored."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    data = pickle.dumps(run)
    restored = pickle.loads(data)
    assert restored.name == run.name
    np.testing.assert_array_equal(restored.centroid_traj, run.centroid_traj)
    assert restored.shell_assigns == run.shell_assigns


# ---------------------------------------------------------------------------
# Observable: shell_modal_basin
# ---------------------------------------------------------------------------

def test_shell_modal_basin_returns_valid_labels():
    """shell_modal_basin returns only 'A', 'B', or 'X' labels."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    modal = shell_modal_basin(run.shell_traj, THETA_A, THETA_B)
    for shell, label in modal.items():
        assert label in {'A', 'B', 'X'}, f"shell {shell} got label {label!r}"


# ---------------------------------------------------------------------------
# Observable: decoupling_chi
# ---------------------------------------------------------------------------

def test_decoupling_chi_in_range():
    """χ is in [0, 1] for valid shells."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    shells = sorted(run.shell_traj.keys())
    if len(shells) < 2:
        pytest.skip("Fewer than 2 shells — skip chi test")
    chi = decoupling_chi(run.shell_traj, shells[0], shells[-1], THETA_A, THETA_B)
    if not math.isnan(chi):
        assert 0.0 <= chi <= 1.0


def test_decoupling_chi_nan_for_missing_shell():
    """NaN returned when a shell index is absent from shell_traj."""
    cfg = _small_config()
    run = run_ncp_simulation(cfg)
    chi = decoupling_chi(run.shell_traj, inner_shell=9999, outer_shell=0,
                         theta_A=THETA_A, theta_B=THETA_B)
    assert math.isnan(chi)


# ---------------------------------------------------------------------------
# Observable: conditional_mutual_information
# ---------------------------------------------------------------------------

def test_cmi_non_negative():
    """CMI is non-negative (zero or small positive due to numerical noise)."""
    cfg = _small_config(n_meas_rounds=50)
    run = run_ncp_simulation(cfg)
    shells = sorted(run.shell_traj.keys())
    if len(shells) < 3:
        pytest.skip("Fewer than 3 shells — skip CMI test")
    cmi = conditional_mutual_information(
        run.shell_traj, shells[0], shells[1], shells[2], THETA_A, THETA_B
    )
    if not math.isnan(cmi):
        assert cmi >= 0.0
