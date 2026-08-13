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
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.clique_fixation import CliqueFixationConfig, run_clique_fixation_trial
from dssgd.protocols.gossip import HybridGossip, SynchronousPairwiseGossip
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


# ---------------------------------------------------------------------------
# HybridGossip (Sec. 2.1's two-jump protocol, gossip_protocol="hybrid")
# ---------------------------------------------------------------------------


class _CountingStub:
    """Records execute() calls without touching comm_round/agents -- used to
    unit-test HybridGossip's round-scheduling logic in isolation from the
    rest of the simulation stack (Agent/CommunicationRound machinery), the
    same way the SynchronousPairwiseGossip tests above isolate
    _randomized_maximal_matching from full simulation."""

    def __init__(self):
        self.calls = 0

    def execute(self, comm_round, agents):
        self.calls += 1


def test_type_n_fires_exactly_once_per_qualifying_round():
    """The bug class this guards against is the HybridGossip analogue of
    SynchronousPairwiseGossip's frozen-matching regression above: if Type-N
    fired on every execute() call instead of once per round, it would run
    `local_steps` times too often per round, destroying Regime C's
    timescale separation between gossip events and local relaxation
    (Assumption R). Type-P must fire on every call; Type-N must fire only
    on the first call of each round whose round_idx is a multiple of
    epsilon_n_rounds."""
    proto = HybridGossip(pairwise_rate=1.0, epsilon_n_rounds=3, rng=np.random.default_rng(0))
    pairwise_stub = _CountingStub()
    neighbourhood_stub = _CountingStub()
    proto._pairwise = pairwise_stub
    proto._neighbourhood = neighbourhood_stub

    local_steps = 5
    n_rounds = 9  # rounds 0, 3, 6 qualify -> 3 Type-N firings
    for round_idx in range(n_rounds):
        proto.round_idx = round_idx
        for _ in range(local_steps):
            proto.execute(comm_round=None, agents=[])

    assert pairwise_stub.calls == n_rounds * local_steps
    assert neighbourhood_stub.calls == 3


def test_type_n_fires_every_round_at_epsilon_n_rounds_one():
    proto = HybridGossip(pairwise_rate=0.0, epsilon_n_rounds=1, rng=np.random.default_rng(0))
    neighbourhood_stub = _CountingStub()
    proto._pairwise = _CountingStub()
    proto._neighbourhood = neighbourhood_stub

    n_rounds = 6
    for round_idx in range(n_rounds):
        proto.round_idx = round_idx
        proto.execute(comm_round=None, agents=[])  # local_steps=1

    assert neighbourhood_stub.calls == n_rounds


def test_pairwise_rate_for_K_inverts_round_ratio_definition():
    """eq. 2.2c: K = epsilon_p*C(m,2) / epsilon_n = pairwise_rate * epsilon_n_rounds
    for a single isolated clique, so pairwise_rate = K / epsilon_n_rounds."""
    K, epsilon_n_rounds = 10.0, 5
    rate = HybridGossip.pairwise_rate_for_K(K, epsilon_n_rounds)
    assert rate == pytest.approx(2.0)
    assert rate * epsilon_n_rounds == pytest.approx(K)


def test_epsilon_n_rounds_must_be_positive():
    with pytest.raises(ValueError):
        HybridGossip(pairwise_rate=1.0, epsilon_n_rounds=0)
    with pytest.raises(ValueError):
        HybridGossip.pairwise_rate_for_K(K=1.0, epsilon_n_rounds=0)


def _clique_config(gossip_protocol: str, seed: int, **kwargs) -> CliqueFixationConfig:
    return CliqueFixationConfig(
        name=f"test/{gossip_protocol}/seed={seed}",
        m=8, j_seeds=1, a=0.5, b=0.03, r=1.0, seed=seed,
        local_steps=20, n_rounds=15, gossip_protocol=gossip_protocol,
        **kwargs,
    )


def test_hybrid_K_to_infinity_limit_matches_async_poisson():
    """K -> infinity (epsilon_n_rounds far beyond the run's horizon, so
    Type-N never fires) must reduce to plain async_poisson: HybridGossip's
    internal AsynchronousGossip sub-step is byte-for-byte the same class,
    constructed with the same rate/alpha/seed, and Type-N contributes zero
    RNG draws when it never fires -- so the two configs' fixation outcomes
    must agree exactly, not just statistically, across every seed tested."""
    for seed in range(5):
        async_run = run_clique_fixation_trial(
            _clique_config("async_poisson", seed, gossip_rate=8.0)
        )
        hybrid_run = run_clique_fixation_trial(
            _clique_config("hybrid", seed, gossip_rate=8.0, epsilon_n_rounds=10**6)
        )
        assert hybrid_run.fixed_at_B == async_run.fixed_at_B, (
            f"seed={seed}: K->inf hybrid diverged from pure async_poisson"
        )


def test_hybrid_K_to_zero_limit_matches_sync_neighbourhood():
    """K -> 0 (pairwise_rate=0, so Type-P is a guaranteed no-op every call)
    with Type-N firing every round must reduce to plain sync_neighbourhood:
    GossipAveraging has no RNG at all, so the two configs' final states are
    deterministic given the same initial conditions and must agree exactly."""
    for seed in range(5):
        sync_run = run_clique_fixation_trial(_clique_config("sync_neighbourhood", seed))
        hybrid_run = run_clique_fixation_trial(
            _clique_config("hybrid", seed, gossip_rate=0.0, epsilon_n_rounds=1)
        )
        assert hybrid_run.fixed_at_B == sync_run.fixed_at_B, (
            f"seed={seed}: K->0 hybrid diverged from pure sync_neighbourhood"
        )


def test_hybrid_occupancy_log_tracks_round_boundaries():
    """B.0.2's mandatory n(t) logging: with track_occupancy=True, the log
    must have one entry per local step (Type-P granularity) and its round
    indices must span every round of the run, including the Type-N-firing
    ones -- the guarantee B.0.1 relies on ("round boundaries are a
    sufficient statistic")."""
    config = _clique_config(
        "hybrid", seed=0, gossip_rate=8.0, epsilon_n_rounds=4, track_occupancy=True,
    )
    run = run_clique_fixation_trial(config)
    assert run.occupancy_log is not None
    assert len(run.occupancy_log) == config.n_rounds * config.local_steps
    rounds_seen = {round_idx for round_idx, _ in run.occupancy_log}
    assert rounds_seen == set(range(config.n_rounds))
    assert all(0 <= n_b <= config.m for _, n_b in run.occupancy_log)


def test_track_occupancy_requires_hybrid_protocol():
    config = _clique_config("async_poisson", seed=0, track_occupancy=True)
    with pytest.raises(ValueError):
        run_clique_fixation_trial(config)
