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


def spectral_gap(W: np.ndarray) -> float:
    """1 - |second-largest eigenvalue| of a doubly-stochastic mixing matrix.

    Shared implementation; previously duplicated as a private helper in
    analysis/simulation.py.
    """
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    return float(1.0 - eigvals[1]) if len(eigvals) > 1 else 1.0


def full_spectrum(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition of a symmetric mixing matrix W.

    Returns (eigvals, eigvecs) sorted by descending |eigenvalue|, with
    eigvecs[:, k] the eigenvector for eigvals[k].  Used to test level-
    localisation of slow gossip modes (Remark 7.1 / Experiment E10).
    """
    eigvals, eigvecs = np.linalg.eigh(W)
    order = np.argsort(-np.abs(eigvals))
    return eigvals[order], eigvecs[:, order]


def eigenvector_ipr(v: np.ndarray) -> float:
    """Inverse participation ratio sum(v_i^4) of a normalised eigenvector.

    IPR ~ 1/n for a fully delocalised mode, IPR -> 1 for a mode localised on
    a single node.  v is renormalised internally so callers may pass raw
    eigenvector columns from full_spectrum.
    """
    norm = np.linalg.norm(v)
    v_normed = v / norm if norm > 1e-12 else v
    return float(np.sum(v_normed ** 4))
