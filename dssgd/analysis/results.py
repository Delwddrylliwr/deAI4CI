"""Persistent result containers for NMH simulation experiments.

AnalysisRun holds all raw outputs from one simulation run.  Save with
.save(path) and reload with AnalysisRun.load(path) so that statistics and
plots can be recomputed without re-running.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class TopologyStats:
    """Graph-level measurements for one NMH instance."""
    n_nodes: int
    branching: int
    depth: int
    leaf_size: int
    p: float
    seed: int
    degrees: np.ndarray = field(default_factory=lambda: np.array([]))
    level_expected_neighbours: List[float] = field(default_factory=list)
    level_observed_neighbours: List[float] = field(default_factory=list)
    spectral_gap: float = 0.0
    # BFS cumulative neighbourhood sizes: mean_nbhd_by_radius[r-1] = mean |{nodes within distance r}|
    mean_nbhd_by_radius: List[float] = field(default_factory=list)

    @property
    def mean_degree(self) -> float:
        return float(self.degrees.mean()) if len(self.degrees) > 0 else 0.0


@dataclass
class LevelTrace:
    """Per-round consensus distances split by hierarchy level.

    within_distances[level][round] = mean consensus distance within each
    level-ℓ super-module, averaged over all such modules.
    cross_distances[level][round] = mean distance between level-ℓ module
    centroids (inter-module spread).
    """
    within_distances: Dict[int, List[float]] = field(default_factory=dict)
    cross_distances: Dict[int, List[float]] = field(default_factory=dict)
    global_distances: List[float] = field(default_factory=list)


@dataclass
class AnalysisRun:
    """All outputs from one NMH gossip-SGD simulation.

    Designed for pickle persistence.  Statistics and plots are computed from
    the raw fields, keeping this as the single authoritative record.
    """
    name: str
    branching: int
    depth: int
    leaf_size: int
    p: float
    seed: int
    n_rounds: int
    lr: float
    sigma2: float
    warmup_rounds: int = 0

    topology: TopologyStats = field(
        default_factory=lambda: TopologyStats(0, 0, 0, 0, 0.0, 0)
    )
    level_trace: LevelTrace = field(default_factory=LevelTrace)

    # Raw flat parameter snapshots: snapshots[round_idx][agent_id] = 1-D numpy array
    snapshots: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)

    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    spectral_gaps: List[float] = field(default_factory=list)

    @property
    def n_agents(self) -> int:
        return self.branching ** self.depth * self.leaf_size

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnalysisRun":
        with open(path, "rb") as f:
            return pickle.load(f)
