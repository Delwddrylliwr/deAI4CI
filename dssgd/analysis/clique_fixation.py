"""Within-clique fixation test (Experiments E7, E12(b)).

A minimal driver, independent of NaturalCascadeConfig's leaf/hierarchy
bookkeeping (this is a single fully-connected clique, no hierarchy at all):
m agents, j of them seeded at B, the rest at A; run Regime-C gossip-SGD and
record whether the clique fixes at all-B. The empirical fixation frequency
over many seeds is compared against theory.fixation_probability(j, m,
theory.fixation_bias(a, b, alpha, r=r)) -- Lemma 3.1's formula with a
*computed*, not fitted, bias (E7 at r=1; E12(b) sweeps r, Lemma 10.1).
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import AsynchronousGossip, GossipAveraging
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology

from .active_escape import (
    BistableParameterModule,
    _dummy_loader_bistable,
    basin_label,
    make_asymmetric_bistable_loss_fn,
    make_bistable_loss_fn,
    verify_bistable_loss,
)


@dataclass
class CliqueFixationConfig:
    """Degrees of freedom for one within-clique fixation trial."""

    name: str
    m: int = 8              # clique size
    j_seeds: int = 1        # number of agents seeded at B (Lemma 3.1's j)
    a: float = 0.5
    b: float = 0.042
    r: float = 1.0          # curvature ratio (Lemma 10.1); 1.0 = equal-curvature (E7)
    seed: int = 0
    lr: float = 0.1
    local_steps: int = 50
    n_rounds: int = 200
    epsilon: float = 0.2
    gossip_protocol: str = "async_poisson"
    gossip_rate: Optional[float] = None
    gossip_alpha: float = 0.5


@dataclass
class CliqueFixationRun:
    """Outcome of one within-clique fixation trial."""

    name: str
    m: int
    j_seeds: int
    a: float
    b: float
    r: float
    seed: int
    fixed_at_B: bool  # True iff every agent ended in basin B

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CliqueFixationRun":
        with open(path, "rb") as f:
            return pickle.load(f)


def run_clique_fixation_trial(config: CliqueFixationConfig) -> CliqueFixationRun:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    theta_A_t = torch.zeros(1)
    theta_B_t = torch.tensor([1.0])
    theta_A_np, theta_B_np = theta_A_t.numpy(), theta_B_t.numpy()

    if config.b != 0.0:
        verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b, require_barrier=False)

    loss_fn = (
        make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b) if config.r == 1.0
        else make_asymmetric_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b, r=config.r)
    )

    # branching=1, depth=0 gives exactly one clique of leaf_size=m and no
    # cross-module edges (NestedModularTopology's level loop is empty).
    topo = NestedModularTopology(branching=1, depth=0, leaf_size=config.m, p=2.0, seed=config.seed)
    loaders = [_dummy_loader_bistable() for _ in range(config.m)]
    seed_ids = set(range(config.j_seeds))

    def _make_agent(i: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + i)
        init = (theta_B_t.clone() if i in seed_ids else theta_A_t.clone()) + torch.randn(1) * 0.02
        model = BistableParameterModule(d_param=1, init_value=init)
        entry = ModelEntry(
            model=model, optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
            loss_fn=loss_fn, layers=frozenset({"social"}), train_locally=True, local_steps=1,
        )
        return Agent(i, ModelRegistry({"model": entry}), loaders[i])

    agents = [_make_agent(i) for i in range(config.m)]
    ml_topo = MultiLayerTopology({"social": topo})
    compositor = CoupledCompositor()

    _async = config.gossip_protocol == "async_poisson"
    if _async:
        rate = (config.gossip_rate or float(config.m)) / float(config.local_steps)
        protocol = AsynchronousGossip(
            rate=rate, mode="poisson", alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )
    else:
        protocol = GossipAveraging()

    for round_idx in range(config.n_rounds):
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            if _async:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
        if not _async:
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)

    final_params = [
        list(a.registry["model"].model.parameters())[0].detach().numpy() for a in agents
    ]
    fixed_at_B = all(
        basin_label(p, theta_A_np, theta_B_np, config.epsilon) == "B" for p in final_params
    )
    return CliqueFixationRun(
        name=config.name, m=config.m, j_seeds=config.j_seeds, a=config.a, b=config.b,
        r=config.r, seed=config.seed, fixed_at_B=fixed_at_B,
    )


# ---------------------------------------------------------------------------
# Experiment factories: E7 (fixation formula) and E12(b) (curvature ratchet)
# ---------------------------------------------------------------------------


def experiment_E7(
    m_list: List[int] = (4, 8, 16),
    a: float = 0.5,
    b_list: List[float] = (0.0, 0.01, 0.02, 0.03, 0.04),
    seeds: List[int] = tuple(range(50)),
    local_steps: int = 50,
    n_rounds: int = 200,
    gossip_protocol: str = "async_poisson",
) -> List[CliqueFixationConfig]:
    """E7: single-seed (j=1) within-clique fixation vs the derived rho(lambda).

    Sweeps m in {4,8,16} and b (hence lambda) at fixed a; theory.fixation_bias
    predicts a step function of lambda at the default gossip_alpha=0.5 (see
    that function's docstring) -- this is exactly the comparison E7 is
    designed to make between the idealised prediction and the repo's actual
    interleaved kick/local_steps dynamics.
    """
    prefix = "E7S" if gossip_protocol != "async_poisson" else "E7"
    configs = []
    for m in m_list:
        for b in b_list:
            for seed in seeds:
                configs.append(CliqueFixationConfig(
                    name=f"{prefix}/m={m}/b={b}/seed={seed}",
                    m=m, j_seeds=1, a=a, b=b, r=1.0, seed=seed,
                    local_steps=local_steps, n_rounds=n_rounds,
                    gossip_protocol=gossip_protocol,
                ))
    return configs


def experiment_E12b(
    r_list: List[float] = (1.0, 1.1, 1.2, 1.5, 2.0),
    m_list: List[int] = (4, 8, 16),
    a: float = 0.5,
    seeds: List[int] = tuple(range(50)),
    local_steps: int = 50,
    n_rounds: int = 200,
    gossip_protocol: str = "async_poisson",
) -> List[CliqueFixationConfig]:
    """E12(b): within-clique curvature ratchet (Lemma 10.1) at lambda=0 (b=0):
    single seed (j=1) placed in the SHARP basin (B, by curvature_epsilon's
    convention); fixation frequency on the FLAT basin (A) should approach
    1 - rho_curv(r) and grow with m, with r=1.0 recovering the symmetric
    (undirected) random walk of Lemma 3.1's rho=1 case exactly.
    """
    prefix = "E12bS" if gossip_protocol != "async_poisson" else "E12b"
    configs = []
    for r in r_list:
        for m in m_list:
            for seed in seeds:
                configs.append(CliqueFixationConfig(
                    name=f"{prefix}/r={r}/m={m}/seed={seed}",
                    m=m, j_seeds=1, a=a, b=0.0, r=r, seed=seed,
                    local_steps=local_steps, n_rounds=n_rounds,
                    gossip_protocol=gossip_protocol,
                ))
    return configs
