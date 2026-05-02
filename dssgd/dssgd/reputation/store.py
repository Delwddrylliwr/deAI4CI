from abc import ABC, abstractmethod
from typing import Dict


class ReputationStore(ABC):
    @abstractmethod
    def get(self, observer: int, target: int) -> float:
        """Return reputation score that `observer` assigns to `target`."""
        ...

    @abstractmethod
    def update(self, observer: int, target: int, value: float):
        ...


class ObjectiveReputationStore(ReputationStore):
    """Single score per agent, identical for all observers.

    The `observer` argument to get/update is ignored; only `target` matters.
    Compatible with GlobalReputationUpdaters.
    """

    def __init__(self, n_agents: int, default: float = 1.0):
        self._scores: Dict[int, float] = {i: default for i in range(n_agents)}

    def get(self, observer: int, target: int) -> float:
        return self._scores.get(target, 1.0)

    def update(self, observer: int, target: int, value: float):
        self._scores[target] = value

    def all_scores(self) -> Dict[int, float]:
        return dict(self._scores)


class SubjectiveReputationStore(ReputationStore):
    """Per-observer score matrix — each agent holds its own view of peers.

    Should only be written to by LocalReputationUpdaters, since subjectivity
    arises from private observations. Writing from a GlobalReputationUpdater
    would make the store effectively objective.
    """

    def __init__(self, n_agents: int, default: float = 1.0):
        self._scores: Dict[int, Dict[int, float]] = {
            i: {j: default for j in range(n_agents) if j != i}
            for i in range(n_agents)
        }

    def get(self, observer: int, target: int) -> float:
        return self._scores.get(observer, {}).get(target, 1.0)

    def update(self, observer: int, target: int, value: float):
        if observer not in self._scores:
            self._scores[observer] = {}
        self._scores[observer][target] = value

    def all_scores(self) -> Dict[int, Dict[int, float]]:
        return {k: dict(v) for k, v in self._scores.items()}
