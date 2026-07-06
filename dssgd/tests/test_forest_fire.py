"""Sanity tests for ForestFireTopology and shell_assignment().

Verifies graph properties, interface compliance, k-core decomposition,
and determinism.  All tests use small graphs (n ≤ 100) for speed.
"""
import networkx as nx
import numpy as np
import pytest

from dssgd.topology.forest_fire import ForestFireTopology


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_step_returns_correct_types():
    """step(0) returns (nx.Graph, np.ndarray) with shape (n, n)."""
    n = 30
    topo = ForestFireTopology(n=n, p_f=0.3, seed=0)
    G, W = topo.step(0)
    assert isinstance(G, nx.Graph)
    assert isinstance(W, np.ndarray)
    assert W.shape == (n, n)


def test_graph_has_n_nodes():
    """Generated graph has exactly n nodes."""
    for n in [10, 50, 100]:
        topo = ForestFireTopology(n=n, p_f=0.3, seed=0)
        G, _ = topo.step(0)
        assert G.number_of_nodes() == n, f"Expected {n} nodes, got {G.number_of_nodes()}"


def test_connected_for_typical_params():
    """Graph is connected for p_f in the typical NCP range."""
    for p_f in [0.3, 0.35, 0.37]:
        topo = ForestFireTopology(n=50, p_f=p_f, seed=0, ensure_connected=True)
        G, _ = topo.step(0)
        assert nx.is_connected(G), f"Graph disconnected for p_f={p_f}"


def test_static_topology_same_every_round():
    """step() returns the same graph regardless of round index."""
    topo = ForestFireTopology(n=30, p_f=0.3, seed=0)
    G0, W0 = topo.step(0)
    G5, W5 = topo.step(5)
    assert G0 is G5, "Graph object should be identical across rounds"
    np.testing.assert_array_equal(W0, W5)


# ---------------------------------------------------------------------------
# Mixing matrix correctness
# ---------------------------------------------------------------------------

def test_mh_matrix_row_stochastic():
    """Metropolis-Hastings W has rows summing to 1."""
    topo = ForestFireTopology(n=40, p_f=0.3, seed=0)
    _, W = topo.step(0)
    np.testing.assert_allclose(W.sum(axis=1), np.ones(40), atol=1e-10)


def test_mh_matrix_non_negative():
    """W has no negative entries."""
    topo = ForestFireTopology(n=40, p_f=0.3, seed=0)
    _, W = topo.step(0)
    assert (W >= 0).all()


def test_mh_matrix_symmetric():
    """W is symmetric (undirected graph → doubly stochastic MH matrix)."""
    topo = ForestFireTopology(n=40, p_f=0.3, seed=0)
    _, W = topo.step(0)
    np.testing.assert_allclose(W, W.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Shell decomposition
# ---------------------------------------------------------------------------

def test_shell_assignment_returns_dict_with_all_nodes():
    """shell_assignment() covers all n nodes."""
    n = 50
    topo = ForestFireTopology(n=n, p_f=0.35, seed=0)
    shells = topo.shell_assignment()
    assert isinstance(shells, dict)
    assert set(shells.keys()) == set(range(n))


def test_shell_assignment_non_trivial_for_ncp_params():
    """For NCP-producing p_f values, max shell > 1 (at least two shell levels)."""
    topo = ForestFireTopology(n=100, p_f=0.37, seed=0)
    shells = topo.shell_assignment()
    max_shell = max(shells.values())
    assert max_shell > 1, (
        f"Expected max shell > 1 for NCP structure, got {max_shell}. "
        "Forest Fire may not be producing NCP structure at this p_f."
    )


def test_shell_values_are_positive_integers():
    """All shell indices are positive integers (k-core numbers ≥ 1)."""
    topo = ForestFireTopology(n=50, p_f=0.35, seed=0)
    shells = topo.shell_assignment()
    for node, s in shells.items():
        assert isinstance(s, int) and s >= 1, f"Node {node} has shell {s}"


def test_shell_assignment_agrees_with_networkx():
    """shell_assignment() matches networkx.core_number() directly."""
    topo = ForestFireTopology(n=50, p_f=0.35, seed=0)
    G, _ = topo.step(0)
    expected = nx.core_number(G)
    actual = topo.shell_assignment()
    assert actual == expected


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_same_graph():
    """Two instances with the same seed produce identical graphs."""
    topo1 = ForestFireTopology(n=40, p_f=0.3, seed=42)
    topo2 = ForestFireTopology(n=40, p_f=0.3, seed=42)
    G1, W1 = topo1.step(0)
    G2, W2 = topo2.step(0)
    assert set(G1.edges()) == set(G2.edges())
    np.testing.assert_array_equal(W1, W2)


def test_different_seeds_different_graphs():
    """Different seeds (very likely) produce different graphs."""
    topo1 = ForestFireTopology(n=40, p_f=0.3, seed=0)
    topo2 = ForestFireTopology(n=40, p_f=0.3, seed=99)
    G1, _ = topo1.step(0)
    G2, _ = topo2.step(0)
    assert set(G1.edges()) != set(G2.edges()), (
        "Different seeds produced identical graphs — very unlikely unless RNG is broken."
    )


# ---------------------------------------------------------------------------
# Invalid parameters
# ---------------------------------------------------------------------------

def test_invalid_p_f_zero_raises():
    with pytest.raises(ValueError, match="p_f"):
        ForestFireTopology(n=10, p_f=0.0, seed=0)


def test_invalid_p_f_one_raises():
    with pytest.raises(ValueError, match="p_f"):
        ForestFireTopology(n=10, p_f=1.0, seed=0)


# ---------------------------------------------------------------------------
# Connectivity retry
# ---------------------------------------------------------------------------

def test_ensure_connected_produces_connected_graph():
    """ensure_connected=True guarantees a connected output across seeds."""
    for seed in range(5):
        topo = ForestFireTopology(n=20, p_f=0.25, seed=seed, ensure_connected=True)
        G, _ = topo.step(0)
        assert nx.is_connected(G), f"Graph disconnected at seed={seed}"
