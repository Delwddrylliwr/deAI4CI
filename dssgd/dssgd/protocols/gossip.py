from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..nodes.agent import Agent
from ..compositor.plan import CommunicationRound


class Protocol(ABC):
    @abstractmethod
    def execute(self, comm_round: CommunicationRound, agents: List[Agent]):
        ...


class GossipAveraging(Protocol):
    """Synchronous gossip: all agents snapshot state, then each averages with neighbours + self."""

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
        from the neighbour's perspective).  Default 0.5.
    rng : np.random.Generator | None
        Seeded RNG for reproducibility; a fresh generator is created if None.
    """

    def __init__(
        self,
        rate: float,
        mode: str = "poisson",
        interval: int = 1,
        alpha: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        if mode not in ("poisson", "fixed"):
            raise ValueError(f"mode must be 'poisson' or 'fixed', got {mode!r}")
        self.rate = rate
        self.mode = mode
        self.interval = interval
        self.alpha = alpha
        self._rng = rng if rng is not None else np.random.default_rng()
        self._round_counter = 0

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

            initiator.aggregate(
                comm_round,
                {initiator.id: self_state, sender_id: sender_state},
                {initiator.id: 1.0 - self.alpha, sender_id: self.alpha},
            )


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
