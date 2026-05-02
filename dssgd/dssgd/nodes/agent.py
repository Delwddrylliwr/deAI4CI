from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional

import torch
from torch.utils.data import DataLoader

from .registry import ModelRegistry

if TYPE_CHECKING:
    from ..compositor.plan import CommunicationRound
    from ..reputation.store import ReputationStore
    from ..reputation.updaters import LocalReputationUpdater


class Agent:
    def __init__(
        self,
        agent_id: int,
        registry: ModelRegistry,
        data_loader: DataLoader,
        reputation_store: Optional["ReputationStore"] = None,
        local_updaters: Optional[List["LocalReputationUpdater"]] = None,
    ):
        self.id = agent_id
        self.registry = registry
        self.data_loader = data_loader
        self.reputation_store = reputation_store
        self.local_updaters = local_updaters or []
        self._data_iter = iter(data_loader)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def local_step(self):
        batch = self._next_batch()
        for entry in self.registry.trainable():
            for _ in range(entry.local_steps):
                entry.optimizer.zero_grad()
                loss = entry.loss_fn(entry.model, batch)
                loss.backward()
                entry.optimizer.step()

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    def get_state(
        self,
        state_keys: List[str],
        param_mask: Optional[FrozenSet[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a snapshot of the requested model keys, filtered by param_mask."""
        result = {}
        for key in state_keys:
            if key not in self.registry:
                continue
            entry = self.registry[key]
            # Round-level mask overrides per-entry surface_params
            mask = param_mask if param_mask is not None else entry.surface_params
            result[key] = entry.get_state(mask)
        return result

    def aggregate(
        self,
        comm_round: "CommunicationRound",
        neighbor_states: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        mixing_weights: Dict[int, float],
    ):
        """Weighted average of received states, optionally modulated by reputation."""
        if not neighbor_states:
            return

        effective = self._reputation_weights(mixing_weights)

        for key in comm_round.state_keys:
            if key not in self.registry:
                continue
            entry = self.registry[key]
            mask = comm_round.param_mask if comm_round.param_mask is not None else entry.surface_params

            received = {j: s[key] for j, s in neighbor_states.items() if key in s}
            if not received:
                continue

            ref = next(iter(received.values()))
            params_to_avg = set(ref.keys()) if mask is None else (mask & set(ref.keys()))

            averaged = {
                param: sum(
                    effective[j] * received[j][param]
                    for j in received
                    if param in received[j]
                )
                for param in params_to_avg
            }
            entry.apply_state(averaged)

        # Fire local reputation updaters after aggregation
        if self.reputation_store is not None:
            for j, states in neighbor_states.items():
                if j == self.id:
                    continue
                for updater in self.local_updaters:
                    updater.update(self, j, states, self.reputation_store)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            return next(self._data_iter)

    def _reputation_weights(self, base_weights: Dict[int, float]) -> Dict[int, float]:
        """Re-weight mixing coefficients by reputation, preserving self-weight."""
        if self.reputation_store is None:
            return base_weights
        weighted = {
            j: w * (self.reputation_store.get(self.id, j) if j != self.id else 1.0)
            for j, w in base_weights.items()
        }
        total = sum(weighted.values())
        return {j: v / total for j, v in weighted.items()} if total > 0 else base_weights
