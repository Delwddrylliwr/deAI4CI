"""Forest Fire topology for NCP (Network Core Periphery) experiments.

Implements the Leskovec et al. (2005) undirected Forest Fire graph model,
which produces NCP structure: a dense core surrounded by nested peripheral
shells, identified via k-core decomposition.

The topology interface matches all other static topologies: step() returns
(graph, Metropolis-Hastings mixing matrix) and is constant across rounds.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .base import Topology, metropolis_hastings


class ForestFireTopology(Topology):
    """Undirected Forest Fire random graph with k-core shell decomposition.

    Parameters
    ----------
    n : int
        Number of nodes.
    p_f : float
        Forward burning probability (Leskovec 2005 parametrisation).
        Typical NCP-producing range: 0.35–0.39.
    r : float
        Backward/forward ratio; backward burning probability = p_f * r.
        Default 0.5 follows Leskovec's report.
    seed : int
        RNG seed for reproducibility.
    ensure_connected : bool
        If True, retries generation (up to max_retries times) until the
        graph is connected.  In practice the ambassador mechanism almost
        always produces a connected graph for n >= 20.
    max_retries : int
        Maximum number of generation attempts when ensure_connected=True.
    """

    def __init__(
        self,
        n: int,
        p_f: float,
        r: float = 0.5,
        seed: int = 0,
        ensure_connected: bool = True,
        max_retries: int = 10,
    ) -> None:
        if not (0.0 < p_f < 1.0):
            raise ValueError(f"p_f must be in (0, 1); got {p_f}")
        if not (0.0 < r):
            raise ValueError(f"r must be positive; got {r}")

        self._n = n
        self._p_f = p_f
        self._r = r

        G: Optional[nx.Graph] = None
        for attempt in range(max_retries):
            G = self._generate(seed + attempt)
            if not ensure_connected or nx.is_connected(G):
                break
        else:
            raise RuntimeError(
                f"Could not generate a connected Forest Fire graph with n={n}, "
                f"p_f={p_f} after {max_retries} attempts."
            )

        self._G: nx.Graph = G
        self._W: np.ndarray = metropolis_hastings(G)

    # ------------------------------------------------------------------
    # Topology interface
    # ------------------------------------------------------------------

    def step(self, round: int) -> Tuple[nx.Graph, np.ndarray]:
        """Return the (static) graph and MH mixing matrix."""
        return self._G, self._W

    # ------------------------------------------------------------------
    # Shell decomposition
    # ------------------------------------------------------------------

    def shell_assignment(self) -> Dict[int, int]:
        """Return {node_id: k-core shell index} for all nodes.

        Uses NetworkX k-core decomposition.  The shell index equals the
        k-core number: node u has shell index k if it belongs to the k-core
        but not the (k+1)-core.  Higher index = denser neighbourhood.
        """
        return nx.core_number(self._G)

    # ------------------------------------------------------------------
    # Graph accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph:
        return self._G

    @property
    def n_nodes(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(self, seed: int) -> nx.Graph:
        """Generate one Forest Fire graph instance."""
        rng = np.random.default_rng(seed)
        p_f = self._p_f
        p_b = self._p_f * self._r

        G = nx.Graph()
        G.add_node(0)

        for v in range(1, self._n):
            G.add_node(v)
            # Pick a uniformly random ambassador from existing nodes
            ambassador = int(rng.integers(0, v))
            G.add_edge(v, ambassador)

            # BFS burning from ambassador
            visited: set = {ambassador}
            frontier: List[int] = [ambassador]

            while frontier:
                next_frontier: List[int] = []
                for u in frontier:
                    # Neighbours of u that exist (excluding v itself and already visited)
                    candidates = [
                        w for w in G.neighbors(u) if w != v and w not in visited
                    ]
                    if not candidates:
                        continue
                    # Sample number of forward and backward links to burn.
                    # Geometric(p) gives number of trials until first failure;
                    # subtract 1 to get the count of successes before stopping.
                    # Cap at the available candidates.
                    k_fwd = int(rng.geometric(1.0 - p_f)) - 1 if p_f > 0 else 0
                    k_bwd = int(rng.geometric(1.0 - p_b)) - 1 if p_b > 0 else 0
                    k = min(k_fwd + k_bwd, len(candidates))
                    if k <= 0:
                        continue
                    burned = rng.choice(
                        candidates, size=k, replace=False
                    ).tolist()
                    for w in burned:
                        if w not in visited:
                            G.add_edge(v, w)
                            visited.add(w)
                            next_frontier.append(w)
                frontier = next_frontier

        return G
