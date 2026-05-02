from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .plan import CommunicationPlan, CommunicationRound
from ..nodes.registry import ModelRegistry


class LayerCompositor(ABC):
    @abstractmethod
    def compose(
        self,
        layers: Dict[str, Tuple[nx.Graph, np.ndarray]],
        registry: ModelRegistry,
    ) -> CommunicationPlan:
        ...


class CoupledCompositor(LayerCompositor):
    """Blends all layer W matrices into a single W_eff = Σ α_l · W_l.

    Best fit for mean field analysis: the multi-layer structure collapses
    to one effective mixing operator, matching the spectral gap theory.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # None → uniform weights across layers
        self.weights = weights

    def compose(self, layers: Dict, registry: ModelRegistry) -> CommunicationPlan:
        names = list(layers.keys())
        n = len(names)
        alphas = self.weights or {name: 1.0 / n for name in names}

        W_eff = sum(alphas[name] * layers[name][1] for name in names)
        G_eff = nx.compose_all([layers[name][0] for name in names])

        # All models that are social on any constituent layer
        all_keys = list({k for name in names for k in registry.on_layer(name)})
        return CommunicationPlan([CommunicationRound(G_eff, W_eff, all_keys)])


class SequentialCompositor(LayerCompositor):
    """One communication round per layer executed in order.

    Effective mixing matrix is the product W_L · ... · W_1.
    The same model key can appear in multiple rounds, accumulating
    averaging from each layer in sequence.
    """

    def __init__(self, order: Optional[List[str]] = None):
        # None → dict insertion order
        self.order = order

    def compose(self, layers: Dict, registry: ModelRegistry) -> CommunicationPlan:
        names = self.order or list(layers.keys())
        rounds = [
            CommunicationRound(
                layers[name][0],
                layers[name][1],
                list(registry.on_layer(name).keys()),
            )
            for name in names
        ]
        return CommunicationPlan(rounds)


class IndependentCompositor(LayerCompositor):
    """One communication round per layer; each model participates in exactly one layer.

    Best for architectures where different model components live on
    different networks (e.g. social vs backbone models).
    """

    def compose(self, layers: Dict, registry: ModelRegistry) -> CommunicationPlan:
        rounds = [
            CommunicationRound(
                G,
                W,
                list(registry.on_layer(name).keys()),
            )
            for name, (G, W) in layers.items()
        ]
        return CommunicationPlan(rounds)
