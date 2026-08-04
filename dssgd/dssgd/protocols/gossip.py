from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from ..nodes.agent import Agent
from ..compositor.plan import CommunicationRound


class Protocol(ABC):
    @abstractmethod
    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        ...


class GossipAveraging(Protocol):
    """Synchronous NEIGHBOURHOOD-averaging gossip (gossip_protocol="sync_neighbourhood",
    suffix "SN"): all agents snapshot state, then each averages with its
    whole current neighbourhood + self -- a simultaneous m-way mean, not a
    pairwise kick. This is the mechanism E9/Lemma 6.1's basin-destruction
    result deliberately uses (uniform averaging is basin-destroying). It is
    NOT the H-sched round-synchronous scheduling of Remark 4.3 -- that is
    SynchronousPairwiseGossip ("sync_pairwise", suffix "SP") below. The two
    were conflated under a single "synchronous"/"S" label until this
    distinction was introduced; see gossip_mechanisms.md for the history and
    rationale, and never assume an "S"-suffixed file predating that note
    means this class rather than the other."""

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        # Snapshot before any agent mutates its model
        all_states = {
            a.id: a.get_state(comm_round.state_keys, comm_round.param_mask)
            for a in agents
        }
        for agent in agents:
            neighbours = list(comm_round.graph.neighbors(agent.id))
            participants = neighbours + [agent.id]
            neighbour_states = {j: all_states[j] for j in participants if j in all_states}
            mixing_weights = {j: float(comm_round.W[agent.id, j]) for j in participants}
            agent.aggregate(comm_round, neighbour_states, mixing_weights)


class AllReduce(Protocol):
    """Synchronous all-reduce baseline: every agent sees every other agent with equal weight."""

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        n = len(agents)
        all_states = {
            a.id: a.get_state(comm_round.state_keys, comm_round.param_mask)
            for a in agents
        }
        uniform = {a.id: 1.0 / n for a in agents}
        for agent in agents:
            agent.aggregate(comm_round, all_states, uniform)


class PushSum(Protocol):
    """Push-sum gossip for directed or asymmetric graphs.

    Each agent pushes half its accumulated weight + model to each out-neighbour
    and to itself. Dividing by the accumulated weight yields an unbiased estimate
    of the network average even on non-doubly-stochastic graphs.
    """

    def __init__(self):
        self._weights: dict = {}

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        for a in agents:
            if a.id not in self._weights:
                self._weights[a.id] = 1.0

        all_states = {
            a.id: a.get_state(comm_round.state_keys, comm_round.param_mask)
            for a in agents
        }

        new_weights: dict = {a.id: 0.0 for a in agents}
        incoming: dict = {a.id: {} for a in agents}

        G = comm_round.graph
        for agent in agents:
            out_nbrs = (
                list(G.successors(agent.id)) if G.is_directed()
                else list(G.neighbors(agent.id))
            )
            out_nbrs.append(agent.id)
            share = self._weights[agent.id] / len(out_nbrs)
            for j in out_nbrs:
                new_weights[j] = new_weights.get(j, 0.0) + share
                incoming[j][agent.id] = all_states[agent.id]

        for agent in agents:
            senders = incoming[agent.id]
            if not senders:
                continue
            # Equal weight across senders; push-sum bias removed by weight accumulation
            uniform = {j: 1.0 / len(senders) for j in senders}
            agent.aggregate(comm_round, senders, uniform)
            self._weights[agent.id] = new_weights[agent.id]


class AsynchronousGossip(Protocol):
    """Asynchronous unilateral gossip: each round, a number of (initiator, neighbour)
    pairs are sampled and only the initiator updates toward the neighbour.

    At each event, initiator i picks a random neighbour j and pulls:
        w_i  ←  (1 - alpha) * w_i  +  alpha * w_j
    Events within a round are sequential: each event sees the state updated by
    the previous one, matching the standard asynchronous-gossip process.

    Parameters
    ----------
    rate : float
        mode='poisson' — Poisson mean; expected number of gossip events per round.
        mode='fixed'   — number of events fired once every ``interval`` rounds.
    mode : str
        'poisson' or 'fixed'.
    interval : int
        Only for mode='fixed'.  Gossip fires every this many rounds (default 1).
    alpha : float
        Initiator mixing weight toward the neighbour.  0.5 gives a symmetric
        pairwise average on the initiator side; 1.0 is a full copy (pure push
        from the neighbour's perspective).  Default 0.5. Ignored if
        `alpha_sampler` is set.
    alpha_sampler : Callable[[], float] | None
        If set, draws a FRESH kick weight from this callable for every
        event, instead of using the fixed `alpha` -- this is this repo's
        degenerate kick-weight law (a point mass at `alpha`, see
        theory.fixation_bias's docstring) generalised to a genuinely
        continuous one (theory.fixation_bias_distributed). Use
        `make_uniform_alpha_sampler` for the natural Uniform(0,1) default
        that pairs with `fixation_bias_distributed`'s own default CDF --
        passing a sampler here whose distribution doesn't match whatever
        CDF a prediction assumes reintroduces a theory/simulation mismatch,
        just a subtler one than the all-fixed-alpha case. None (default)
        preserves the exact original fixed-alpha behaviour.
    rng : np.random.Generator | None
        Seeded RNG for reproducibility; a fresh generator is created if None.
    """

    def __init__(
        self,
        rate: float,
        mode: str = "poisson",
        interval: int = 1,
        alpha: float = 0.5,
        alpha_sampler: Optional[Callable[[], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if mode not in ("poisson", "fixed"):
            raise ValueError(f"mode must be 'poisson' or 'fixed', got {mode!r}")
        self.rate = rate
        self.mode = mode
        self.interval = interval
        self.alpha = alpha
        self.alpha_sampler = alpha_sampler
        self._rng = rng if rng is not None else np.random.default_rng()
        self._round_counter = 0

    def _current_alpha(self) -> float:
        return self.alpha_sampler() if self.alpha_sampler is not None else self.alpha

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        self._round_counter += 1

        if self.mode == "fixed":
            if self._round_counter % self.interval != 0:
                return
            n_events = int(self.rate)
        else:
            n_events = int(self._rng.poisson(self.rate))

        if n_events == 0 or not agents:
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

            # Sequential snapshots: use states as they are at this moment in the round
            sender_state = sender.get_state(comm_round.state_keys, comm_round.param_mask)
            self_state = initiator.get_state(comm_round.state_keys, comm_round.param_mask)

            alpha = self._current_alpha()
            initiator.aggregate(
                comm_round,
                {initiator.id: self_state, sender_id: sender_state},
                {initiator.id: 1.0 - alpha, sender_id: alpha},
            )


class BoundedStalenessGossip(AsynchronousGossip):
    """AsynchronousGossip with a per-edge staleness lock (Experiment E11,
    Remark 4.3's "restorable by per-boundary seed-locking or bounded staleness").

    Identical kick mechanics to AsynchronousGossip, except an edge (i,j) that
    fired within the last `staleness_bound` rounds is skipped on subsequent
    draws (the event is dropped, not retried) until the lock expires. This
    approximates round-synchronous, renewal-enforcing scheduling (H-sched)
    without going fully synchronous: a single seed's fixation contest along
    a given edge gets `staleness_bound` rounds to resolve before that edge
    can re-fire, capping (rather than eliminating) seed accumulation.

    `staleness_bound=0` recovers AsynchronousGossip exactly (no lock).
    Callers must set `.round_idx` each round (same convention as
    provenance.ProvenanceAsyncGossip) so the lock can measure elapsed rounds.
    """

    def __init__(self, *args, staleness_bound: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.staleness_bound = staleness_bound
        self.round_idx: int = 0
        self._last_fired: dict = {}  # frozenset({i,j}) -> round_idx last fired

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        self._round_counter += 1

        if self.mode == "fixed":
            if self._round_counter % self.interval != 0:
                return
            n_events = int(self.rate)
        else:
            n_events = int(self._rng.poisson(self.rate))

        if n_events == 0 or not agents:
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

            edge_key = frozenset((initiator.id, sender_id))
            last = self._last_fired.get(edge_key)
            if last is not None and self.round_idx - last < self.staleness_bound:
                continue  # edge locked: drop this event rather than retry
            self._last_fired[edge_key] = self.round_idx

            sender_state = sender.get_state(comm_round.state_keys, comm_round.param_mask)
            self_state = initiator.get_state(comm_round.state_keys, comm_round.param_mask)

            alpha = self._current_alpha()
            initiator.aggregate(
                comm_round,
                {initiator.id: self_state, sender_id: sender_state},
                {initiator.id: 1.0 - alpha, sender_id: alpha},
            )


class SynchronousPairwiseGossip(Protocol):
    """Round-synchronous scheduling of pairwise kicks (gossip_protocol=
    "sync_pairwise", suffix "SP"; Remark 4.3's H-sched class,
    paper1_PDMP_wDAG_wData.md Section 4.2/Annex D.1: "Round-synchronous scheduling
    (all edges, or a maximal matching, per round, with full relaxation between
    rounds) enforces [single-seed resolution] by construction").

    Unlike GossipAveraging (a simultaneous m-way mean over each agent's whole
    neighbourhood), this computes a maximal matching over the round's graph and
    applies, per matched pair, the *same* one-sided pairwise kick AsynchronousGossip
    uses -- so the scheduling axis (synchronous round-matching vs. free asynchrony)
    is isolated from the update-rule axis (pairwise kick vs. m-way average), which
    Remark 4.3's H-sched hypothesis and Proposition D.3.2's scheduling parameter
    sigma_sched are specifically about. All matched pairs read from a pre-round
    snapshot, so the round is genuinely simultaneous; unmatched agents are
    untouched this round (at most one kick per agent per round, as under a
    maximal matching).

    The matching is recomputed with a FRESH randomized greedy order every round
    (`self._rng`-shuffled edges, not `nx.maximal_matching(graph)` directly).
    `nx.maximal_matching` is a deterministic greedy scan over `graph.edges()`'s
    fixed iteration order: on a static topology (the common case -- the same
    graph object every round) it returns the exact same matching every single
    call, forever. On a hierarchical/modular topology this locks in whichever
    edges the greedy scan reaches first -- typically the dense within-module
    ties, since those get enumerated before the sparser cross-module ones --
    and those cross-module edges then never fire, ever, for the whole run: a
    confirmed real bug (found empirically: at NMH-1's default topology, 232 of
    232 cross-leaf-module edges were excluded from the fixed matching, and
    NMH-1 showed literally zero nucleation events beyond the source's own
    leaf). Reshuffling per round is what makes round-synchronous matching
    diffusive over time, matching every other protocol here (AsynchronousGossip
    picks a fresh random neighbour every event) and the informal gossip/
    consensus literature's convention for matching-based protocols.

    Parameters
    ----------
    alpha : float
        Initiator mixing weight toward the neighbour, matching
        AsynchronousGossip's `alpha` (default 0.5: symmetric pairwise average
        on the initiator side). Ignored if `alpha_sampler` is set.
    alpha_sampler : Callable[[], float] | None
        If set, draws a FRESH kick weight per matched pair instead of using
        the fixed `alpha` -- see AsynchronousGossip's `alpha_sampler` for the
        full rationale (theory.fixation_bias_distributed's simulation-side
        counterpart); `make_uniform_alpha_sampler` gives the matching
        Uniform(0,1) default. None (default) preserves the exact original
        fixed-alpha behaviour.
    rng : np.random.Generator | None
        Seeded RNG for reproducibility (used to pick, per matched pair,
        which endpoint is the initiator, and to draw from `alpha_sampler`
        if you build one from this same generator); a fresh generator is
        created if None.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        alpha_sampler: Optional[Callable[[], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.alpha = alpha
        self.alpha_sampler = alpha_sampler
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def _current_alpha(self) -> float:
        return self.alpha_sampler() if self.alpha_sampler is not None else self.alpha

    def _randomized_maximal_matching(self, graph: nx.Graph) -> List[tuple]:
        """Greedy maximal matching over a FRESH randomized edge order every
        call -- see the class docstring for why `nx.maximal_matching(graph)`
        itself (deterministic given a fixed graph) is unusable here."""
        edges = list(graph.edges())
        self._rng.shuffle(edges)
        matched: set = set()
        matching = []
        for u, v in edges:
            if u not in matched and v not in matched:
                matching.append((u, v))
                matched.add(u)
                matched.add(v)
        return matching

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        all_states = {
            a.id: a.get_state(comm_round.state_keys, comm_round.param_mask)
            for a in agents
        }
        agent_map = {a.id: a for a in agents}
        matching = self._randomized_maximal_matching(comm_round.graph)

        for u, v in matching:
            if self._rng.random() < 0.5:
                initiator_id, sender_id = u, v
            else:
                initiator_id, sender_id = v, u
            initiator = agent_map[initiator_id]
            alpha = self._current_alpha()
            initiator.aggregate(
                comm_round,
                {initiator_id: all_states[initiator_id], sender_id: all_states[sender_id]},
                {initiator_id: 1.0 - alpha, sender_id: alpha},
            )


def make_uniform_alpha_sampler(rng: Optional[np.random.Generator] = None) -> Callable[[], float]:
    """Kick-weight sampler drawing a fresh alpha ~ Uniform(0,1) each call --
    the simulation-side counterpart to theory.uniform_kick_cdf /
    fixation_bias_distributed's default kick-weight law mu_C. Pass as the
    `alpha_sampler` argument to AsynchronousGossip/SynchronousPairwiseGossip
    so the actual kick mechanics draw from the SAME law a
    fixation_bias_distributed prediction assumes, rather than the theory and
    simulation sides independently guessing what "distributed kick weight"
    means (see gossip_mechanisms.md and theory.fixation_bias's docstring for
    why this repo's original fixed-alpha default was a degenerate special
    case of this, not a distribution at all).
    """
    rng = rng if rng is not None else np.random.default_rng()
    return lambda: float(rng.random())


class CompositeProtocol(Protocol):
    """Applies a sequence of protocols in order each communication round.

    Enables mixed strategies, e.g. one round of synchronous GossipAveraging
    followed by a burst of AsynchronousGossip events.  Protocols are executed
    in the order supplied; later protocols see states already updated by earlier
    ones.

    Parameters
    ----------
    protocols : list of Protocol
        Ordered list of protocol instances to execute per round.
    """

    def __init__(self, protocols: List[Protocol]):
        self.protocols = list(protocols)

    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        for protocol in self.protocols:
            protocol.execute(comm_round, agents)


# ---------------------------------------------------------------------------
# gossip_protocol string -> experiment-name/pkl-directory suffix
# ---------------------------------------------------------------------------
#
# Single source of truth for the three mechanisms every NMH/NCP experiment
# factory distinguishes by filename suffix. Introduced to fix a real
# reproducibility hazard: "synchronous"/"S" used to mean GossipAveraging (a
# bug -- see gossip_mechanisms.md), then briefly meant SynchronousPairwiseGossip
# after that bug was fixed, silently reusing the same label for a different
# mechanism. Going forward "S" alone is retired; every sync-family experiment
# must say which of the two it means, so an "SP"/"SN"-suffixed file is
# unambiguous regardless of when it was generated.

PROTOCOL_SUFFIXES = {
    "async_poisson": "",
    "sync_pairwise": "SP",
    "sync_neighbourhood": "SN",
}


def protocol_suffix(gossip_protocol: str) -> str:
    """Experiment-name/pkl-directory suffix for a gossip_protocol string.

    Raises on any value outside the three general-purpose mechanisms (async,
    sync_pairwise, sync_neighbourhood) -- deliberately, rather than silently
    treating an unrecognised or legacy string (e.g. the retired bare
    "synchronous") as one mechanism or another. bounded_staleness is not
    covered here: it is E11-specific and uses its own name-embedded scheme
    (name=f"E11/proto={protocol}/stale={staleness}/...") rather than a
    directory suffix.
    """
    try:
        return PROTOCOL_SUFFIXES[gossip_protocol]
    except KeyError:
        raise ValueError(
            f"Unknown gossip_protocol {gossip_protocol!r}; expected one of "
            f"{sorted(PROTOCOL_SUFFIXES)} for this experiment."
        ) from None


# Reverse lookup for review/reporting scripts (check_phase*.py's --suffix CLI
# argument): given the suffix a caller passes, report which mechanism it
# names. Deliberately has no entry for "" -> bare "S": that string is retired
# and any script still passing it should fail loudly, not silently guess.
SUFFIX_TO_PROTOCOL = {
    "": "async_poisson",
    "SP": "sync_pairwise",
    "SN": "sync_neighbourhood",
}


def protocol_from_suffix(suffix: str) -> str:
    """Inverse of protocol_suffix: gossip_protocol name for a suffix string.

    Raises on the retired bare "S" (ambiguous between sync_pairwise and
    sync_neighbourhood) or any other unrecognised suffix, rather than
    guessing which mechanism produced a given file.
    """
    try:
        return SUFFIX_TO_PROTOCOL[suffix]
    except KeyError:
        raise ValueError(
            f"Unknown suffix {suffix!r}; expected one of "
            f"{sorted(SUFFIX_TO_PROTOCOL)}. If this is the retired bare "
            f"'S' suffix from before sync_pairwise/sync_neighbourhood were "
            f"distinguished, you must know (e.g. from when the data was "
            f"generated) which mechanism it actually is and pass 'SP' or "
            f"'SN' explicitly -- see gossip_mechanisms.md."
        ) from None
