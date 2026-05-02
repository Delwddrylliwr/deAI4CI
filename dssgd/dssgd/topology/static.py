from typing import Tuple

import networkx as nx
import numpy as np

from .base import Topology, metropolis_hastings


class RingTopology(Topology):
    def __init__(self, n: int):
        self._G = nx.cycle_graph(n)
        self._W = metropolis_hastings(self._G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class GridTopology(Topology):
    def __init__(self, rows: int, cols: int):
        G = nx.grid_2d_graph(rows, cols)
        self._G = nx.convert_node_labels_to_integers(G)
        self._W = metropolis_hastings(self._G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class FullyConnectedTopology(Topology):
    def __init__(self, n: int):
        self._G = nx.complete_graph(n)
        self._W = metropolis_hastings(self._G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class ErdosRenyiTopology(Topology):
    def __init__(self, n: int, p: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        while True:
            G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(int(1e9))))
            if nx.is_connected(G):
                break
        self._G = G
        self._W = metropolis_hastings(G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class BarabasiAlbertTopology(Topology):
    def __init__(self, n: int, m: int, seed: int = 0):
        self._G = nx.barabasi_albert_graph(n, m, seed=seed)
        self._W = metropolis_hastings(self._G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class NestedModularTopology(Topology):
    """
    Nested modular hierarchy following Moretti & Munoz.

    Nodes are arranged in a ``branching``-ary tree of ``depth`` levels.
    Bottom-level modules form complete graphs. At each successive level the
    gateway (first) node of each sibling sub-module is wired into a clique,
    propagating connectivity up the hierarchy.

    Total nodes = branching ** depth * leaf_size.

    Parameters
    ----------
    branching : int
        Number of child sub-modules per parent module at each level.
    depth : int
        Number of hierarchy levels (depth=1 gives flat, non-nested clusters).
    leaf_size : int
        Number of nodes per bottom-level module.
    """

    def __init__(self, branching: int, depth: int, leaf_size: int):
        n_leaf_modules = branching ** depth
        n = n_leaf_modules * leaf_size
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Dense intra-module cliques at the leaf level
        for m in range(n_leaf_modules):
            start = m * leaf_size
            nodes = list(range(start, start + leaf_size))
            G.add_edges_from(
                (nodes[i], nodes[j])
                for i in range(len(nodes))
                for j in range(i + 1, len(nodes))
            )

        # Sparse inter-module gateway cliques at each hierarchical level.
        # At level l, group leaf modules into super-modules of size branching^l;
        # the gateway node of each child sub-module connects to its siblings.
        for level in range(1, depth + 1):
            super_size = branching ** level        # leaf modules per super-module
            n_super = n_leaf_modules // super_size
            child_size = branching ** (level - 1)  # leaf modules per child

            for sm in range(n_super):
                gateways = [
                    (sm * super_size + c * child_size) * leaf_size
                    for c in range(branching)
                ]
                G.add_edges_from(
                    (gateways[i], gateways[j])
                    for i in range(len(gateways))
                    for j in range(i + 1, len(gateways))
                )

        self._G = G
        self._W = metropolis_hastings(G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W
