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
    Nested modular hierarchy (Safari–Moretti–Muñoz / Moretti–Muñoz construction).

    Level 0: n/leaf_size disjoint cliques of size leaf_size (m in the paper).
    Level ℓ ≥ 1: super-modules are formed by grouping `branching` child modules.
    Cross-module edges between each pair of distinct child modules are added
    independently with probability p/4^ℓ, so inter-module edge density decays
    geometrically with depth.  Any node within a module can acquire bridging
    edges at any level — there is no fixed gateway node.

    Total nodes = branching ** depth * leaf_size.

    Parameters
    ----------
    branching : int
        Number of child sub-modules per parent module at each level.
    depth : int
        Number of hierarchy levels (depth=1 gives flat, non-nested clusters).
    leaf_size : int
        Number of nodes per bottom-level module.  m in the paper.
    p : float
        Base inter-module edge probability.  At level ℓ, each cross-module
        node-pair is connected independently with probability p / 4**ℓ.
        Default p=2 gives level-1 edge probability 0.5 and expected degree 2
        at level 1 — the stochastic sparse regime studied in Safari et al.
        Using p=leaf_size (e.g. p=4) degenerately makes level-1 a complete
        bipartite graph (probability = 1) and should be avoided.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        branching: int,
        depth: int,
        leaf_size: int,
        p: float = None,
        seed: int = 0,
    ):
        if p is None:
            p = 2.0   # sparse stochastic regime: level-1 edge prob = 0.5
        rng = np.random.default_rng(seed)
        n_leaf_modules = branching ** depth
        n = n_leaf_modules * leaf_size
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Level 0: dense intra-module cliques
        for mod in range(n_leaf_modules):
            start = mod * leaf_size
            nodes = list(range(start, start + leaf_size))
            G.add_edges_from(
                (nodes[i], nodes[j])
                for i in range(len(nodes))
                for j in range(i + 1, len(nodes))
            )

        # Level ℓ ≥ 1: stochastic cross-module edges with probability p / 4^ℓ.
        # For each super-module at this level, edges are sampled independently
        # between every pair of distinct child sub-modules across all their nodes,
        # so bridging connections are distributed throughout the module rather than
        # concentrated at a single gateway.
        for level in range(1, depth + 1):
            super_size = branching ** level        # leaf modules per super-module
            n_super = n_leaf_modules // super_size
            child_size = branching ** (level - 1)  # leaf modules per child
            p_level = p / (4.0 ** level)

            for sm in range(n_super):
                # Node lists for each child sub-module inside this super-module
                children = [
                    list(range(
                        (sm * super_size + c * child_size) * leaf_size,
                        (sm * super_size + c * child_size + child_size) * leaf_size,
                    ))
                    for c in range(branching)
                ]

                # Sample cross-edges independently between every pair of children
                for a in range(branching):
                    for b in range(a + 1, branching):
                        for u in children[a]:
                            for v in children[b]:
                                if rng.random() < p_level:
                                    G.add_edge(u, v)

        if not nx.is_connected(G):
            raise ValueError(
                f"NestedModularTopology(branching={branching}, depth={depth}, "
                f"leaf_size={leaf_size}, p={p}, seed={seed}) produced a disconnected "
                f"graph.  Increase p or the seed."
            )

        self._G = G
        self._W = metropolis_hastings(G)

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W
