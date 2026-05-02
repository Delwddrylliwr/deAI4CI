from abc import ABC, abstractmethod
from typing import Tuple

import networkx as nx
import numpy as np


class Topology(ABC):
    @abstractmethod
    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        """Return (graph, mixing_matrix_W) for this round."""
        ...


def metropolis_hastings(G: nx.Graph) -> np.ndarray:
    """Doubly-stochastic mixing matrix via Metropolis-Hastings weights."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    W = np.zeros((n, n))
    for u in nodes:
        i = idx[u]
        for v in G.neighbors(u):
            j = idx[v]
            W[i, j] = 1.0 / (1 + max(G.degree(u), G.degree(v)))
        W[i, i] = 1.0 - W[i].sum()
    return W
