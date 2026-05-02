from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional

import torch
import torch.nn as nn


@dataclass
class ModelEntry:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    layers: FrozenSet[str] = field(default_factory=frozenset)
    # None = all parameters are surface (shared); named set = only those params are shared
    surface_params: Optional[FrozenSet[str]] = None
    train_locally: bool = True
    local_steps: int = 1

    def get_state(self, param_mask: Optional[FrozenSet[str]] = None) -> Dict[str, torch.Tensor]:
        """Return a cloned snapshot of the requested parameters."""
        sd = self.model.state_dict()
        keys = param_mask if param_mask is not None else set(sd.keys())
        return {k: v.clone() for k, v in sd.items() if k in keys}

    def apply_state(self, partial_state: Dict[str, torch.Tensor]):
        """Write averaged parameters back into the model in-place."""
        current = self.model.state_dict()
        current.update(partial_state)
        self.model.load_state_dict(current)


class ModelRegistry:
    """Per-agent container for named model entries with role-aware accessors."""

    def __init__(self, entries: Dict[str, ModelEntry]):
        self._entries = entries

    def __getitem__(self, key: str) -> ModelEntry:
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def on_layer(self, layer: str) -> Dict[str, ModelEntry]:
        """Return entries whose layers set includes the given layer name."""
        return {k: v for k, v in self._entries.items() if layer in v.layers}

    def trainable(self) -> List[ModelEntry]:
        """Return entries that participate in local SGD steps."""
        return [v for v in self._entries.values() if v.train_locally]

    def items(self):
        return self._entries.items()

    def keys(self):
        return self._entries.keys()
