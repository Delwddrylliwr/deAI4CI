from abc import ABC, abstractmethod
from typing import List

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
