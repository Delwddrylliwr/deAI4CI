from typing import Tuple

import networkx as nx
import numpy as np

from .base import Topology, metropolis_hastings


class RandomGeometricTopology(Topology):
    """Agents connect when within a fixed radius; models mobile networks."""

    def __init__(self, n: int, radius: float, seed: int = 0):
        self.n = n
        self.radius = radius
        self.rng = np.random.default_rng(seed)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        G = nx.random_geometric_graph(self.n, self.radius, seed=int(self.rng.integers(int(1e9))))
        _ensure_connected(G)
        return G, metropolis_hastings(G)


class RandomMatchingTopology(Topology):
    """Random perfect matching each round — common in decentralised SGD theory."""

    def __init__(self, n: int, seed: int = 0):
        if n % 2 != 0:
            raise ValueError("n must be even for random matching")
        self.n = n
        self.rng = np.random.default_rng(seed)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        perm = self.rng.permutation(self.n)
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i in range(0, self.n, 2):
            G.add_edge(int(perm[i]), int(perm[i + 1]))
        return G, metropolis_hastings(G)


class FailureTopology(Topology):
    """Wraps any topology and randomly drops edges to simulate link failures."""

    def __init__(self, base: Topology, edge_fail_prob: float = 0.1, seed: int = 0):
        self.base = base
        self.p_fail = edge_fail_prob
        self.rng = np.random.default_rng(seed)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        G_base, _ = self.base.step(round)
        G = G_base.copy()
        for u, v in list(G.edges()):
            if self.rng.random() < self.p_fail:
                G.remove_edge(u, v)
                if not nx.is_connected(G):
                    G.add_edge(u, v)
        return G, metropolis_hastings(G)


def _ensure_connected(G: nx.Graph):
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = min(components[i])
        v = min(components[i + 1])
        G.add_edge(u, v)
