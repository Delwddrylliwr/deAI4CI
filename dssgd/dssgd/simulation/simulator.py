from dataclasses import dataclass
from typing import List, Optional

from ..nodes.agent import Agent
from ..compositor.compositors import LayerCompositor
from ..protocols.gossip import Protocol
from ..topology.multilayer import MultiLayerTopology
from ..metrics.tracker import MetricsTracker
from ..reputation.updaters import GlobalReputationUpdater
from ..reputation.store import ReputationStore


@dataclass
class SimulatorConfig:
    num_rounds: int = 100


class Simulator:
    def __init__(
        self,
        topology: MultiLayerTopology,
        compositor: LayerCompositor,
        protocol: Protocol,
        agents: List[Agent],
        metrics: MetricsTracker,
        global_updaters: Optional[List[GlobalReputationUpdater]] = None,
        global_reputation_store: Optional[ReputationStore] = None,
        config: Optional[SimulatorConfig] = None,
    ):
        self.topology = topology
        self.compositor = compositor
        self.protocol = protocol
        self.agents = agents
        self.metrics = metrics
        self.global_updaters = global_updaters or []
        self.global_reputation_store = global_reputation_store
        self.config = config or SimulatorConfig()

    def run(self) -> MetricsTracker:
        for round_idx in range(self.config.num_rounds):
            self._local_update_phase()
            layer_graphs = self._communication_phase(round_idx)
            self._global_reputation_phase()
            self.metrics.record(round_idx, self.agents, layer_graphs)
        return self.metrics

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _local_update_phase(self):
        for agent in self.agents:
            agent.local_step()

    def _communication_phase(self, round_idx: int) -> dict:
        layer_graphs = self.topology.step(round_idx)
        # All agents share the same registry key structure; use agents[0] as template
        plan = self.compositor.compose(layer_graphs, self.agents[0].registry)
        for comm_round in plan.rounds:
            self.protocol.execute(comm_round, self.agents)
        return layer_graphs

    def _global_reputation_phase(self):
        if self.global_reputation_store is not None:
            for updater in self.global_updaters:
                updater.update(self.agents, self.global_reputation_store)
