from typing import Dict, List, Optional, Tuple

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


class OverlappingModularTopology(Topology):
    """NestedModularTopology with tunable module overlap at one level
    (Section 11's DAG-nested / overlapping-hierarchy generalisation).

    The communication graph and gossip dynamics are byte-for-byte identical
    to NestedModularTopology at the same (branching, depth, leaf_size, p,
    seed) -- overlap does not change the wiring. What changes is the MODULE
    MEMBERSHIP consumed by the generality/loss machinery (tree_loss.py): at
    `overlap_level`, `n_overlap` level-`overlap_level` modules are each
    additionally assigned `delta_in - 1` extra parent modules at level
    `overlap_level + 1`, chosen uniformly from parents other than their
    natural one. This gives those modules Hasse in-degree `delta_in` at the
    overlap boundary -- the tree (delta_in=1) is the n_overlap=0 special case.

    See analysis.tree_loss.in_scope for how this membership metadata is
    consumed to decide whether a worker is in generality-scope of a source.
    """

    def __init__(
        self,
        branching: int,
        depth: int,
        leaf_size: int,
        p: Optional[float],
        overlap_level: int,
        delta_in: int,
        n_overlap: int,
        seed: int = 0,
    ):
        if not (0 <= overlap_level < depth):
            raise ValueError(f"overlap_level must be in [0, depth); got {overlap_level}, depth={depth}")
        if delta_in < 1:
            raise ValueError(f"delta_in must be >= 1; got {delta_in}")

        base = NestedModularTopology(
            branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=seed,
        )
        self._G = base._G
        self._W = base._W
        self.branching = branching
        self.depth = depth
        self.leaf_size = leaf_size
        self.overlap_level = overlap_level
        self.delta_in = delta_in

        n_leaf_modules = branching ** depth
        n_at_ov = n_leaf_modules // (branching ** overlap_level)
        n_at_ov_plus1 = n_leaf_modules // (branching ** (overlap_level + 1))
        rng = np.random.default_rng(seed + 777)
        n_overlap = min(n_overlap, n_at_ov)
        chosen = rng.choice(n_at_ov, size=n_overlap, replace=False) if n_overlap > 0 else []

        # {level-overlap_level module idx: [extra level-(overlap_level+1) module idxs]}
        self.extra_parents: Dict[int, List[int]] = {}
        for k in chosen:
            natural = int(k) // branching
            others = [q for q in range(n_at_ov_plus1) if q != natural]
            n_extra = min(delta_in - 1, len(others))
            if n_extra > 0:
                extra = rng.choice(others, size=n_extra, replace=False).tolist()
                self.extra_parents[int(k)] = [int(q) for q in extra]

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class StarTopology(Topology):
    """Centralised star K_{1,N-1}: one hub (node 0), N-1 leaf workers.

    Used for the centralised-aggregation experiments (E9, Lemma 6.1 / basin
    destruction): the hub has no local loss in the runner that consumes this
    topology, so it plays the role of the paper's server node w_0.
    """

    def __init__(self, n: int):
        self._G = nx.star_graph(n - 1)  # hub is node 0, leaves are 1..n-1
        self._W = metropolis_hastings(self._G)
        self.hub = 0

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        return self._G, self._W


class DumbbellTopology(Topology):
    """Two dense blocks joined by a tunable-density bipartite bridge.

    Used for the matched-spectral-gap comparison against the NMH (E10,
    Remark 7.1): unlike the NMH's geometric ladder of L relaxation scales,
    the dumbbell has exactly one slow mode (the bottleneck), so a bridge
    density tuned to match the NMH's second-largest eigenvalue at the same N
    gives two graphs with identical spectral gap but categorically different
    mode structure.

    Parameters
    ----------
    block_size : int
        Number of nodes in each of the two blocks (total N = 2*block_size).
    bridge_p : float
        Independent bridge-edge probability between every cross-block pair.
        Tune this (e.g. via topology.base.spectral_gap + a bisection search)
        to match a target second eigenvalue.
    """

    def __init__(self, block_size: int, bridge_p: float, seed: int = 0):
        if not (0.0 < bridge_p <= 1.0):
            raise ValueError(f"bridge_p must be in (0, 1]; got {bridge_p}")
        rng = np.random.default_rng(seed)
        n = 2 * block_size
        G = nx.Graph()
        G.add_nodes_from(range(n))
        block_a = list(range(block_size))
        block_b = list(range(block_size, n))
        G.add_edges_from(
            (block_a[i], block_a[j])
            for i in range(block_size) for j in range(i + 1, block_size)
        )
        G.add_edges_from(
            (block_b[i], block_b[j])
            for i in range(block_size) for j in range(i + 1, block_size)
        )
        for u in block_a:
            for v in block_b:
                if rng.random() < bridge_p:
                    G.add_edge(u, v)
        if not nx.is_connected(G):
            # Guarantee connectivity with a single deterministic bridge edge
            # without disturbing the (already-sampled) bridge_p statistics.
            G.add_edge(block_a[0], block_b[0])
        self._G = G
        self._W = metropolis_hastings(G)
        self.block_size = block_size
        self.bridge_p = bridge_p

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
