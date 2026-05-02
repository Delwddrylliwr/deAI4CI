from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..nodes.agent import Agent


@dataclass
class RoundMetrics:
    round: int
    mean_loss: Optional[float] = None
    mean_accuracy: Optional[float] = None
    agent_losses: Optional[Dict[int, float]] = None
    consensus_distance: Optional[float] = None
    # spectral gap per layer: 1 - |λ_2(W)|, governs mixing rate
    spectral_gaps: Optional[Dict[str, float]] = None
    # cross_eval_matrix[(i, j)] = cross_eval_fn(agent_i, agent_j.data_loader)
    cross_eval_matrix: Optional[Dict[Tuple[int, int], float]] = None


class MetricsTracker:
    def __init__(
        self,
        eval_fn: Optional[Callable[["Agent"], Tuple[float, float]]] = None,
        eval_every: int = 1,
        model_key: str = "model",
        cross_eval_fn: Optional[Callable[["Agent", DataLoader], float]] = None,
        cross_eval_every: int = 1,
    ):
        """
        eval_fn: agent → (loss, accuracy). Called every eval_every rounds.
        cross_eval_fn: (agent, loader) → scalar. Evaluates agent_i's model on
            agent_j's data loader for all pairs, building the n×n matrix.
        model_key: which registry entry to use for consensus distance.
        """
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.model_key = model_key
        self.cross_eval_fn = cross_eval_fn
        self.cross_eval_every = cross_eval_every
        self.history: List[RoundMetrics] = []

    def record(self, round_idx: int, agents: List["Agent"], layer_graphs: dict):
        m = RoundMetrics(round=round_idx)

        if self.eval_fn is not None and round_idx % self.eval_every == 0:
            losses, accs, agent_losses = [], [], {}
            for agent in agents:
                loss, acc = self.eval_fn(agent)
                losses.append(loss)
                accs.append(acc)
                agent_losses[agent.id] = loss
            m.mean_loss = float(np.mean(losses))
            m.mean_accuracy = float(np.mean(accs))
            m.agent_losses = agent_losses

        if self.cross_eval_fn is not None and round_idx % self.cross_eval_every == 0:
            matrix: Dict[Tuple[int, int], float] = {}
            for agent_i in agents:
                for agent_j in agents:
                    matrix[(agent_i.id, agent_j.id)] = self.cross_eval_fn(
                        agent_i, agent_j.data_loader
                    )
            m.cross_eval_matrix = matrix

        m.consensus_distance = self._consensus_distance(agents)
        m.spectral_gaps = self._spectral_gaps(layer_graphs)
        self.history.append(m)

    # ------------------------------------------------------------------
    # Derived series
    # ------------------------------------------------------------------

    def losses(self) -> List[float]:
        return [m.mean_loss for m in self.history if m.mean_loss is not None]

    def accuracies(self) -> List[float]:
        return [m.mean_accuracy for m in self.history if m.mean_accuracy is not None]

    def consensus_distances(self) -> List[float]:
        return [m.consensus_distance for m in self.history if m.consensus_distance is not None]

    def spectral_gaps_for(self, layer: str) -> List[float]:
        return [
            m.spectral_gaps[layer]
            for m in self.history
            if m.spectral_gaps and layer in m.spectral_gaps
        ]

    def cross_eval_matrices(self) -> List[Dict[Tuple[int, int], float]]:
        """All recorded cross-evaluation matrices in round order."""
        return [m.cross_eval_matrix for m in self.history if m.cross_eval_matrix is not None]

    def generalization_gaps(self) -> List[float]:
        """Mean off-diagonal minus mean diagonal of each cross-eval matrix.

        Sign depends on the metric returned by cross_eval_fn:
        accuracy → negative gap means models perform worse on foreign data (poor generalisation).
        loss     → positive gap means models perform worse on foreign data (poor generalisation).
        """
        gaps = []
        for mat in self.cross_eval_matrices():
            ids = sorted({i for i, _ in mat})
            diag = [mat[(i, i)] for i in ids if (i, i) in mat]
            off = [v for (i, j), v in mat.items() if i != j]
            if diag and off:
                gaps.append(float(np.mean(off)) - float(np.mean(diag)))
        return gaps

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _consensus_distance(self, agents: List["Agent"]) -> Optional[float]:
        flat_params = []
        for agent in agents:
            if self.model_key not in agent.registry:
                continue
            sd = agent.registry[self.model_key].model.state_dict()
            flat_params.append(torch.cat([p.float().flatten() for p in sd.values()]))
        if not flat_params:
            return None
        stacked = torch.stack(flat_params)
        mean = stacked.mean(0)
        return float(np.mean([(s - mean).norm().item() for s in stacked]))

    def _spectral_gaps(self, layer_graphs: dict) -> Dict[str, float]:
        gaps = {}
        for name, (G, W) in layer_graphs.items():
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
            gaps[name] = float(1.0 - eigvals[1]) if len(eigvals) > 1 else 1.0
        return gaps
