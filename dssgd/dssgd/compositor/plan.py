from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional

import networkx as nx
import numpy as np


@dataclass
class CommunicationRound:
    graph: nx.Graph
    W: np.ndarray
    state_keys: List[str]
    # Optional round-level override for which parameters to share.
    # None means fall back to each ModelEntry's own surface_params.
    param_mask: Optional[FrozenSet[str]] = None


@dataclass
class CommunicationPlan:
    rounds: List[CommunicationRound] = field(default_factory=list)
