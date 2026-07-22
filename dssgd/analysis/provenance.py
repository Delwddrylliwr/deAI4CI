"""Provenance tracking for kick-vs-escape attribution (Experiment E1).

Round-level flip detection (active_escape.find_t_flip) records *when* a leaf
enters basin B but not *why*: whether the transition was driven by a
cross-boundary gossip kick (social transport) or occurred with no recent
cross-boundary contact (independent noise-driven / Kramers escape via
flip_noise_scale).  This module provides the reusable primitives:

  - GossipEvent / ProvenanceAsyncGossip: event-level logging of cross-leaf-
    boundary kicks inside AsynchronousGossip, without changing its dynamics
    (the RNG draw sequence is identical to the base class; logging is a pure
    side effect).
  - classify_flip_provenance / crossover_stage: post-hoc classifiers that
    answer "kick or escape?" per flipped leaf, and the deepest hierarchical
    distance at which flips are still kick-attributed (paper's l_c, eq 3.6).
  - sever_boundaries / SeveredTopology: the gossip-severed control
    (epsilon=0 across a chosen hierarchy level) used as the transport null
    against which E1's provenance statistics are compared (Proposition 3.4).

These primitives are consumed directly by natural_cascade.py's
run_natural_cascade_simulation (via NaturalCascadeConfig.track_provenance /
sever_min_distance), which is the production, checkpointed HPC path for E1
-- this module intentionally has no dependency on natural_cascade.py (kept
one-directional to avoid a circular import) and is not itself a runner.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from dssgd.nodes.agent import Agent
from dssgd.protocols.gossip import AsynchronousGossip
from dssgd.topology.base import Topology, metropolis_hastings

from .catchup import hierarchical_distance


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


@dataclass
class GossipEvent:
    """One cross-leaf-boundary initiator<-sender pull inside AsynchronousGossip.

    Same-leaf events are the overwhelming majority of traffic and carry no
    provenance information (they cannot cause a leaf to nucleate from
    outside), so ProvenanceAsyncGossip only retains crossing events.
    """

    round_idx: int
    step_idx: int
    initiator_id: int
    sender_id: int
    initiator_leaf: int
    sender_leaf: int


class ProvenanceAsyncGossip(AsynchronousGossip):
    """AsynchronousGossip that additionally logs cross-boundary events.

    Mechanically identical to the base class -- the exact same sequence of
    RNG draws produces the exact same kicks -- with an `events` log appended
    to as a side effect.  Callers must set `.round_idx` before each
    measurement round; `.step_idx` auto-increments once per execute() call
    (i.e. once per local gradient step, matching how natural_cascade.py
    interleaves gossip with local_steps).
    """

    def __init__(self, *args, leaf_assigns: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.leaf_assigns = leaf_assigns
        self.events: List[GossipEvent] = []
        self.round_idx: int = 0
        self.step_idx: int = 0

    def execute(self, comm_round, agents: List[Agent]):
        self._round_counter += 1

        if self.mode == "fixed":
            if self._round_counter % self.interval != 0:
                self.step_idx += 1
                return
            n_events = int(self.rate)
        else:
            n_events = int(self._rng.poisson(self.rate))

        if n_events == 0 or not agents:
            self.step_idx += 1
            return

        agent_map = {a.id: a for a in agents}
        G = comm_round.graph

        for _ in range(n_events):
            initiator = agents[int(self._rng.integers(len(agents)))]
            neighbours = list(G.neighbors(initiator.id))
            if not neighbours:
                continue
            sender_id = neighbours[int(self._rng.integers(len(neighbours)))]
            sender = agent_map.get(sender_id)
            if sender is None:
                continue

            sender_state = sender.get_state(comm_round.state_keys, comm_round.param_mask)
            self_state = initiator.get_state(comm_round.state_keys, comm_round.param_mask)

            initiator_leaf = int(self.leaf_assigns[initiator.id])
            sender_leaf = int(self.leaf_assigns[sender_id])
            if initiator_leaf != sender_leaf:
                self.events.append(GossipEvent(
                    round_idx=self.round_idx,
                    step_idx=self.step_idx,
                    initiator_id=initiator.id,
                    sender_id=sender_id,
                    initiator_leaf=initiator_leaf,
                    sender_leaf=sender_leaf,
                ))

            initiator.aggregate(
                comm_round,
                {initiator.id: self_state, sender_id: sender_state},
                {initiator.id: 1.0 - self.alpha, sender_id: self.alpha},
            )

        self.step_idx += 1


# ---------------------------------------------------------------------------
# Post-hoc provenance classification
# ---------------------------------------------------------------------------


def classify_flip_provenance(
    flip_table: List[dict],
    events: List[GossipEvent],
    lookback_rounds: int = 5,
) -> Dict[int, str]:
    """Classify each flipped leaf's first sustained B-entry as 'kick' or 'escape'.

    'kick'   -- at least one cross-boundary event landed on this leaf (as the
                initiator, i.e. the leaf pulled state from a different leaf)
                within `lookback_rounds` rounds before its flip.
    'escape' -- no such event: the flip is attributed to local noise
                (flip_noise_scale) rather than social transport, matching the
                paper's attribution requirement (Corollary 3.5).

    Leaves with t_flip_absolute is None (never flipped) are omitted.
    """
    by_leaf: Dict[int, List[GossipEvent]] = {}
    for ev in events:
        by_leaf.setdefault(ev.initiator_leaf, []).append(ev)

    result: Dict[int, str] = {}
    for row in flip_table:
        leaf = row["target_leaf"]
        t_flip = row.get("t_flip_absolute")
        if t_flip is None:
            continue
        window_lo = t_flip - lookback_rounds
        touched = any(
            window_lo <= ev.round_idx <= t_flip for ev in by_leaf.get(leaf, [])
        )
        result[leaf] = "kick" if touched else "escape"
    return result


def crossover_stage(
    flip_table: List[dict],
    events: List[GossipEvent],
    lookback_rounds: int = 5,
) -> Optional[int]:
    """Highest hierarchical distance at which a flip is still kick-attributed.

    This is the paper's l_c (eq 3.6): beyond this stage, transmission is no
    faster than local rediscovery, and observed transitions cannot be
    attributed to social transport.  Returns None if no leaf ever flipped.
    """
    provenance = classify_flip_provenance(flip_table, events, lookback_rounds)
    kick_distances = [
        row["distance_from_source"]
        for row in flip_table
        if provenance.get(row["target_leaf"]) == "kick"
        and row["distance_from_source"] is not None
    ]
    return max(kick_distances) if kick_distances else None


# ---------------------------------------------------------------------------
# Gossip-severed control (Proposition 3.4's transport null)
# ---------------------------------------------------------------------------


def sever_boundaries(
    G: nx.Graph,
    leaf_assigns: np.ndarray,
    min_distance: int,
    distance_fn: Callable[[int, int], int] = hierarchical_distance,
) -> nx.Graph:
    """Remove every edge whose endpoints' leaves are >= min_distance apart.

    Node set is unchanged; the graph may become disconnected across modules
    at that boundary -- that is the point: no cascade can cross a severed
    boundary via gossip, isolating the noise (Kramers) channel.
    """
    H = G.copy()
    to_remove = [
        (u, v)
        for u, v in H.edges()
        if distance_fn(int(leaf_assigns[u]), int(leaf_assigns[v])) >= min_distance
    ]
    H.remove_edges_from(to_remove)
    return H


class SeveredTopology(Topology):
    """Wraps a base Topology, pruning boundary edges >= min_distance apart.

    W is recomputed via metropolis_hastings on the pruned (static) graph.
    """

    def __init__(
        self,
        base_topo: Topology,
        leaf_assigns: np.ndarray,
        min_distance: int,
        distance_fn: Callable[[int, int], int] = hierarchical_distance,
    ):
        G0, _ = base_topo.step(0)
        self._G = sever_boundaries(G0, leaf_assigns, min_distance, distance_fn)
        self._W = metropolis_hastings(self._G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W
