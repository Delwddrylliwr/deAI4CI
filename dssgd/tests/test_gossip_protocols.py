"""Regression tests for the gossip protocol classes in dssgd.protocols.gossip.

Focused on SynchronousPairwiseGossip's matching diffusion: nx.maximal_matching
is a deterministic greedy scan, so calling it on the SAME static graph every
round (the common case -- most topologies don't change round to round) used
to return the identical matching forever, permanently excluding whichever
edges the greedy scan didn't reach first. On a hierarchical/modular topology
this meant cross-module edges could go entirely unused for the whole run --
found via NMH-1 showing zero nucleation events under sync_pairwise. The fix
(_randomized_maximal_matching) reshuffles the greedy order every call.
"""
import networkx as nx
import numpy as np
import pytest

from dssgd.protocols.gossip import SynchronousPairwiseGossip
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology


def _hierarchical_graph(branching=2, depth=3, leaf_size=4, seed=0) -> nx.Graph:
    topo = NestedModularTopology(branching=branching, depth=depth, leaf_size=leaf_size, p=2.0, seed=seed)
    ml = MultiLayerTopology({"social": topo})
    return ml.step(0)["social"][0]


def test_matching_varies_across_rounds():
    """A static graph must not yield the identical matching every round."""
    g = _hierarchical_graph()
    proto = SynchronousPairwiseGossip(rng=np.random.default_rng(0))
    matchings = set()
    for _ in range(20):
        m = proto._randomized_maximal_matching(g)
        matchings.add(tuple(sorted(tuple(sorted(e)) for e in m)))
    assert len(matchings) > 1, "matching is identical every round -- diffusion is impossible"


def test_cross_module_edges_eventually_used():
    """Over many rounds, cross-leaf-module edges must get selected at least
    once -- this is exactly the condition that was violated by the bug
    (nx.maximal_matching(static_graph) picked zero of 232 available
    cross-module edges at NMH-1's default scale, every single round)."""
    g = _hierarchical_graph(branching=2, depth=5, leaf_size=4)
    leaf_size = 4
    leaf_of = {n: n // leaf_size for n in g.nodes()}
    cross_module_edges = {tuple(sorted((u, v))) for u, v in g.edges() if leaf_of[u] != leaf_of[v]}
    assert cross_module_edges, "test topology has no cross-module edges to check"

    proto = SynchronousPairwiseGossip(rng=np.random.default_rng(0))
    used = set()
    for _ in range(50):
        for u, v in proto._randomized_maximal_matching(g):
            edge = tuple(sorted((u, v)))
            if leaf_of[u] != leaf_of[v]:
                used.add(edge)

    assert used, "no cross-module edge was ever selected across 50 rounds"
    # Loose bound: diffusion should reach a meaningful fraction, not just one edge.
    assert len(used) > len(cross_module_edges) * 0.5


def test_matching_respects_maximality_and_no_double_booking():
    """Every round's matching must still be a valid matching: no node
    appears twice, and every edge used must be a real edge of the graph."""
    g = _hierarchical_graph()
    proto = SynchronousPairwiseGossip(rng=np.random.default_rng(1))
    for _ in range(10):
        m = proto._randomized_maximal_matching(g)
        seen = set()
        for u, v in m:
            assert u not in seen and v not in seen, "node matched more than once in a single round"
            seen.add(u)
            seen.add(v)
            assert g.has_edge(u, v)


def test_matching_is_reproducible_given_seeded_rng():
    """Same seed -> same sequence of matchings (determinism for a fixed seed,
    not for a fixed graph across repeated unseeded calls)."""
    g = _hierarchical_graph()
    proto_a = SynchronousPairwiseGossip(rng=np.random.default_rng(7))
    proto_b = SynchronousPairwiseGossip(rng=np.random.default_rng(7))
    for _ in range(5):
        m_a = proto_a._randomized_maximal_matching(g)
        m_b = proto_b._randomized_maximal_matching(g)
        assert sorted(tuple(sorted(e)) for e in m_a) == sorted(tuple(sorted(e)) for e in m_b)
