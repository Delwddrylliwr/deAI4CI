from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.data.partition import dirichlet_partition, iid_partition, make_loaders
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import AllReduce, GossipAveraging, PushSum
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import FullyConnectedTopology, GridTopology, RingTopology


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceTestConfig:
    """All degrees of freedom for a single consensus-convergence scenario."""

    name: str
    n_agents: int
    topology_factory: Callable              # () -> Topology
    protocol_factory: Callable              # () -> Protocol
    compositor_factory: Callable = CoupledCompositor
    layer_name: str = "social"
    model_dim: int = 4
    n_rounds: int = 50
    atol: float = 1e-4
    subsets: Dict[str, List[int]] = field(default_factory=dict)
    # --- Local training ---
    train_locally: bool = False
    lr: float = 0.0
    local_steps: int = 1
    loss_threshold: Optional[float] = None
    # --- Model / data overrides ---
    model_factory: Optional[Callable[[], nn.Module]] = None
    loss_fn: Optional[Callable] = None
    data_loaders_factory: Optional[Callable[[int], List[DataLoader]]] = None
    # --- Test-set accuracy evaluation ---
    # Shared (non-partitioned) loader used to evaluate all agents each eval_every rounds.
    eval_loader_factory: Optional[Callable[[], DataLoader]] = None
    accuracy_threshold: Optional[float] = None
    eval_every: int = 10


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceResult:
    """Per-round consensus distances, losses, and accuracy snapshots."""

    distances: List[float]
    subset_distances: Dict[str, List[float]]
    losses: List[float] = field(default_factory=list)
    # One entry per eval_every rounds; populated when eval_loader_factory is set.
    accuracies: List[float] = field(default_factory=list)

    def converged(self, atol: float) -> bool:
        return self.distances[-1] < atol

    def subset_converged(self, name: str, atol: float) -> bool:
        return self.subset_distances[name][-1] < atol

    def loss_decreased(self) -> bool:
        return len(self.losses) >= 2 and self.losses[-1] < self.losses[0]

    def half_life(self, name: Optional[str] = None) -> Optional[int]:
        """Round at which distance first dropped below half its initial value."""
        series = self.subset_distances[name] if name else self.distances
        if not series:
            return None
        threshold = series[0] / 2.0
        for i, d in enumerate(series):
            if d <= threshold:
                return i
        return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _make_dummy_loader(model_dim: int) -> DataLoader:
    x = torch.randn(16, model_dim)
    y = torch.zeros(16, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _make_mnist_loaders(
    n_agents: int,
    alpha: Optional[float] = None,
    batch_size: int = 64,
) -> List[DataLoader]:
    """MNIST training data partitioned across agents.

    alpha=None gives an IID split; a finite alpha uses a Dirichlet partition
    (lower alpha = more heterogeneous).
    """
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root=os.path.expanduser("~/.cache/dssgd/mnist"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    subsets = (
        iid_partition(dataset, n_agents)
        if alpha is None
        else dirichlet_partition(dataset, n_agents, alpha=alpha)
    )
    return make_loaders(subsets, batch_size=batch_size)


def _make_mnist_test_loader(batch_size: int = 256) -> DataLoader:
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root=os.path.expanduser("~/.cache/dssgd/mnist"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _make_mlp() -> nn.Module:
    """2-layer MLP from McMahan et al. (2017): 784 -> 200 -> 10."""
    return nn.Sequential(
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )


def _mnist_loss(model: nn.Module, batch) -> torch.Tensor:
    x, y = batch
    return F.cross_entropy(model(x.view(-1, 784)), y)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: int,
    config: ConvergenceTestConfig,
    loader: Optional[DataLoader] = None,
) -> Agent:
    torch.manual_seed(agent_id)
    model = config.model_factory() if config.model_factory else nn.Linear(config.model_dim, 1, bias=False)

    if config.loss_fn is not None:
        loss_fn = config.loss_fn
    elif config.train_locally:
        raise ValueError(f"config '{config.name}' has train_locally=True but no loss_fn")
    else:
        loss_fn = lambda m, batch: torch.tensor(0.0, requires_grad=True)

    entry = ModelEntry(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=config.lr),
        loss_fn=loss_fn,
        layers=frozenset({config.layer_name}),
        train_locally=config.train_locally,
        local_steps=config.local_steps,
    )
    registry = ModelRegistry({"model": entry})
    return Agent(agent_id, registry, loader or _make_dummy_loader(config.model_dim))


def _consensus_distance(agents: List[Agent], ids: Optional[List[int]] = None) -> float:
    subset = [a for a in agents if ids is None or a.id in ids]
    states = [
        torch.cat([p.flatten() for p in a.registry["model"].model.parameters()])
        for a in subset
    ]
    if len(states) < 2:
        return 0.0
    stacked = torch.stack(states)
    mean = stacked.mean(0)
    return float(torch.stack([(s - mean).norm() for s in stacked]).mean())


def _mean_loss(agents: List[Agent]) -> float:
    total = 0.0
    for agent in agents:
        entry = agent.registry["model"]
        batch = next(iter(agent.data_loader))
        with torch.no_grad():
            total += entry.loss_fn(entry.model, batch).item()
    return total / len(agents)


def _mean_accuracy(agents: List[Agent], eval_loader: DataLoader) -> float:
    """Mean test accuracy across all agents on the shared evaluation set."""
    total = 0.0
    for agent in agents:
        model = agent.registry["model"].model
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                pred = model(x.view(-1, 784)).argmax(dim=1)
                correct += (pred == y).sum().item()
                n += y.size(0)
        model.train()
        total += correct / n
    return total / len(agents)


def _run_convergence(config: ConvergenceTestConfig) -> ConvergenceResult:
    loaders = (
        config.data_loaders_factory(config.n_agents)
        if config.data_loaders_factory
        else [None] * config.n_agents
    )
    eval_loader = config.eval_loader_factory() if config.eval_loader_factory else None

    agents = [_make_agent(i, config, loaders[i]) for i in range(config.n_agents)]
    ml_topo = MultiLayerTopology({config.layer_name: config.topology_factory()})
    compositor = config.compositor_factory()
    protocol = config.protocol_factory()

    distances: List[float] = []
    subset_distances: Dict[str, List[float]] = {name: [] for name in config.subsets}
    losses: List[float] = []
    accuracies: List[float] = []

    for round_idx in range(config.n_rounds):
        for agent in agents:
            agent.local_step()

        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)

        distances.append(_consensus_distance(agents))
        for name, ids in config.subsets.items():
            subset_distances[name].append(_consensus_distance(agents, ids))
        if config.train_locally:
            losses.append(_mean_loss(agents))
        if eval_loader is not None and (round_idx + 1) % config.eval_every == 0:
            accuracies.append(_mean_accuracy(agents, eval_loader))

    return ConvergenceResult(distances, subset_distances, losses, accuracies)


# ---------------------------------------------------------------------------
# Pure-gossip test configurations  (fast — no local training)
# ---------------------------------------------------------------------------


CONVERGENCE_CONFIGS = [
    ConvergenceTestConfig(
        name="gossip/fully_connected/n=5",
        n_agents=5,
        topology_factory=lambda: FullyConnectedTopology(5),
        protocol_factory=GossipAveraging,
        n_rounds=30,
    ),
    ConvergenceTestConfig(
        name="gossip/ring/n=6",
        n_agents=6,
        topology_factory=lambda: RingTopology(6),
        protocol_factory=GossipAveraging,
        n_rounds=200,
        atol=1e-3,
        subsets={"first_half": [0, 1, 2], "second_half": [3, 4, 5]},
    ),
    ConvergenceTestConfig(
        name="gossip/grid_3x3",
        n_agents=9,
        topology_factory=lambda: GridTopology(3, 3),
        protocol_factory=GossipAveraging,
        n_rounds=100,
        subsets={"row_0": [0, 1, 2], "row_1": [3, 4, 5], "row_2": [6, 7, 8]},
    ),
    ConvergenceTestConfig(
        name="allreduce/fully_connected/n=5",
        n_agents=5,
        topology_factory=lambda: FullyConnectedTopology(5),
        protocol_factory=AllReduce,
        n_rounds=1,
        atol=1e-6,
    ),
    ConvergenceTestConfig(
        name="pushsum/ring/n=6",
        n_agents=6,
        topology_factory=lambda: RingTopology(6),
        protocol_factory=PushSum,
        n_rounds=200,
        atol=1e-3,
        subsets={"first_half": [0, 1, 2], "second_half": [3, 4, 5]},
    ),
]


# ---------------------------------------------------------------------------
# Simulation configurations  (slow — MNIST + local training)
#
# Accuracy thresholds are set conservatively below each paper's reported peak
# to allow for differences in exact hyperparameter tuning.
# ---------------------------------------------------------------------------


SIMULATION_CONFIGS = [
    # McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
    # from Decentralized Data", Table 1.
    # 2-NN MNIST, IID, full participation ≈ AllReduce.  Their B=10/E=5 setup
    # reaches 97 % in 35 rounds with 100 clients; we set 90 % after 100 rounds
    # with 10 agents all communicating (strictly easier).
    ConvergenceTestConfig(
        name="mcmahan2017/allreduce/mnist/n=10/iid",
        n_agents=10,
        topology_factory=lambda: FullyConnectedTopology(10),
        protocol_factory=AllReduce,
        n_rounds=100,
        atol=0.5,
        train_locally=True,
        lr=0.1,
        local_steps=5,
        model_factory=_make_mlp,
        loss_fn=_mnist_loss,
        data_loaders_factory=lambda n: _make_mnist_loaders(n, alpha=None),
        eval_loader_factory=_make_mnist_test_loader,
        accuracy_threshold=0.90,
    ),
    # Lian et al. (2017) "Can Decentralized Algorithms Outperform Centralized
    # Algorithms?", main claim: D-PSGD on a well-connected graph matches
    # centralised SGD.  Fully connected gossip should reach the same accuracy
    # as AllReduce above.
    ConvergenceTestConfig(
        name="lian2017/gossip/mnist/fully_connected/n=10/iid",
        n_agents=10,
        topology_factory=lambda: FullyConnectedTopology(10),
        protocol_factory=GossipAveraging,
        n_rounds=100,
        atol=0.5,
        train_locally=True,
        lr=0.1,
        local_steps=5,
        model_factory=_make_mlp,
        loss_fn=_mnist_loss,
        data_loaders_factory=lambda n: _make_mnist_loaders(n, alpha=None),
        eval_loader_factory=_make_mnist_test_loader,
        accuracy_threshold=0.88,
    ),
    # Lian et al. (2017): ring is the hardest connected topology; convergence
    # is slower due to small spectral gap O(1/n²) but the method still converges.
    ConvergenceTestConfig(
        name="lian2017/gossip/mnist/ring/n=8/iid",
        n_agents=8,
        topology_factory=lambda: RingTopology(8),
        protocol_factory=GossipAveraging,
        n_rounds=200,
        atol=1.0,
        train_locally=True,
        lr=0.05,
        local_steps=1,
        model_factory=_make_mlp,
        loss_fn=_mnist_loss,
        data_loaders_factory=lambda n: _make_mnist_loaders(n, alpha=None),
        eval_loader_factory=_make_mnist_test_loader,
        accuracy_threshold=0.85,
        eval_every=20,
        subsets={"left_half": [0, 1, 2, 3], "right_half": [4, 5, 6, 7]},
    ),
    # Li et al. (2020) "Federated Optimization in Heterogeneous Networks"
    # (FedProx), Table 2.  Dirichlet α=0.5 is their standard non-IID benchmark.
    # Heterogeneous data + ring topology means lower accuracy is expected.
    ConvergenceTestConfig(
        name="li2020/gossip/mnist/ring/n=6/noniid_alpha0.5",
        n_agents=6,
        topology_factory=lambda: RingTopology(6),
        protocol_factory=GossipAveraging,
        n_rounds=200,
        atol=1.0,
        train_locally=True,
        lr=0.05,
        local_steps=1,
        model_factory=_make_mlp,
        loss_fn=_mnist_loss,
        data_loaders_factory=lambda n: _make_mnist_loaders(n, alpha=0.5),
        eval_loader_factory=_make_mnist_test_loader,
        accuracy_threshold=0.72,
        eval_every=20,
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    CONVERGENCE_CONFIGS,
    ids=[c.name for c in CONVERGENCE_CONFIGS],
)
def test_consensus_converges(config: ConvergenceTestConfig) -> None:
    result = _run_convergence(config)

    assert result.converged(config.atol), (
        f"Global consensus not reached: final distance {result.distances[-1]:.2e} "
        f"(tol={config.atol:.2e})"
    )
    for name in config.subsets:
        assert result.subset_converged(name, config.atol), (
            f"Subset '{name}' did not converge: "
            f"final distance {result.subset_distances[name][-1]:.2e} "
            f"(tol={config.atol:.2e})"
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    SIMULATION_CONFIGS,
    ids=[c.name for c in SIMULATION_CONFIGS],
)
def test_gossip_sgd_accuracy(config: ConvergenceTestConfig) -> None:
    result = _run_convergence(config)

    if config.loss_threshold is not None:
        assert result.loss_decreased(), (
            f"Loss did not decrease: {result.losses[0]:.4f} -> {result.losses[-1]:.4f}"
        )
        assert result.losses[-1] < config.loss_threshold, (
            f"Final loss {result.losses[-1]:.4f} above threshold {config.loss_threshold}"
        )
    if config.accuracy_threshold is not None:
        assert result.accuracies, "No accuracy recorded — check eval_loader_factory"
        final_acc = result.accuracies[-1]
        assert final_acc >= config.accuracy_threshold, (
            f"Final accuracy {final_acc:.3f} below published threshold "
            f"{config.accuracy_threshold} after {config.n_rounds} rounds"
        )
