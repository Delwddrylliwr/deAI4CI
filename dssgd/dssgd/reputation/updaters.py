from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

import torch

if TYPE_CHECKING:
    from ..nodes.agent import Agent
    from .store import ReputationStore


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class LocalReputationUpdater(ABC):
    """Called from Agent.aggregate() — has access to local information only."""

    @abstractmethod
    def update(
        self,
        observer: "Agent",
        target_id: int,
        received_states: Dict[str, Dict[str, torch.Tensor]],
        store: "ReputationStore",
    ):
        ...


class GlobalReputationUpdater(ABC):
    """Called from Simulator after each round — has access to all agents.

    Should only write to ObjectiveReputationStore; writing to a
    SubjectiveReputationStore defeats the subjectivity invariant.
    """

    @abstractmethod
    def update(self, all_agents: List["Agent"], store: "ReputationStore"):
        ...


# ---------------------------------------------------------------------------
# Local updaters
# ---------------------------------------------------------------------------

class ModelDistanceUpdater(LocalReputationUpdater):
    """Score decays with L2 distance between received weights and observer's weights.

    Agents with similar models are trusted more. Uses an EMA to smooth over rounds.
    """

    def __init__(self, model_key: str = "model", ema_alpha: float = 0.9):
        self.model_key = model_key
        self.alpha = ema_alpha

    def update(self, observer, target_id, received_states, store):
        if self.model_key not in received_states or self.model_key not in observer.registry:
            return
        local_sd = observer.registry[self.model_key].model.state_dict()
        received = received_states[self.model_key]
        dist = sum(
            (local_sd[k] - received[k]).float().norm().item()
            for k in received
            if k in local_sd
        )
        score = 1.0 / (1.0 + dist)
        current = store.get(observer.id, target_id)
        store.update(observer.id, target_id, self.alpha * current + (1 - self.alpha) * score)


class PrivateDriftUpdater(LocalReputationUpdater):
    """Measures divergence between the agent's social and private models.

    High drift → the social model has moved far from what the agent would have
    learned alone → reputation score for all neighbours decreases uniformly.
    Naturally subjective: each agent accumulates its own drift history.
    """

    def __init__(
        self,
        social_key: str = "social",
        private_key: str = "private",
        ema_alpha: float = 0.9,
    ):
        self.social_key = social_key
        self.private_key = private_key
        self.alpha = ema_alpha

    def update(self, observer, target_id, received_states, store):
        if self.social_key not in observer.registry or self.private_key not in observer.registry:
            return
        social_sd = observer.registry[self.social_key].model.state_dict()
        private_sd = observer.registry[self.private_key].model.state_dict()
        drift = sum(
            (social_sd[k] - private_sd[k]).float().norm().item()
            for k in social_sd
            if k in private_sd
        )
        score = 1.0 / (1.0 + drift)
        current = store.get(observer.id, target_id)
        store.update(observer.id, target_id, self.alpha * current + (1 - self.alpha) * score)


# ---------------------------------------------------------------------------
# Global updaters
# ---------------------------------------------------------------------------

class ConsensusDeviationUpdater(GlobalReputationUpdater):
    """Penalises agents whose model deviates far from the global consensus.

    Intended for use with ObjectiveReputationStore only.
    """

    def __init__(self, model_key: str = "model", ema_alpha: float = 0.9):
        self.model_key = model_key
        self.alpha = ema_alpha

    def update(self, all_agents, store):
        agents_with_key = [a for a in all_agents if self.model_key in a.registry]
        if not agents_with_key:
            return

        states = [a.registry[self.model_key].model.state_dict() for a in agents_with_key]
        mean_state = {
            k: torch.stack([s[k].float() for s in states]).mean(0)
            for k in states[0]
        }

        for agent in agents_with_key:
            agent_sd = agent.registry[self.model_key].model.state_dict()
            dist = sum(
                (agent_sd[k].float() - mean_state[k]).norm().item()
                for k in agent_sd
                if k in mean_state
            )
            score = 1.0 / (1.0 + dist)
            # observer=agent.id is ignored by ObjectiveReputationStore
            current = store.get(agent.id, agent.id)
            store.update(agent.id, agent.id, self.alpha * current + (1 - self.alpha) * score)
