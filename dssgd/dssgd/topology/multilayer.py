from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .base import Topology


class MultiLayerTopology:
    """Container for named topology layers. Single-layer setups use one entry."""

    def __init__(self, layers: Dict[str, Topology]):
        self.layers = layers

    def step(self, round: int) -> Dict[str, Tuple[nx.Graph, np.ndarray]]:
        return {name: topo.step(round) for name, topo in self.layers.items()}
