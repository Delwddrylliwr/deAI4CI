"""Within-clique fixation test (Experiments E7, E12(b), E15).

A minimal driver, independent of NaturalCascadeConfig's leaf/hierarchy
bookkeeping (this is a single fully-connected clique, no hierarchy at all):
m agents, j of them seeded at B, the rest at A; run Regime-C gossip-SGD and
record whether the clique fixes at all-B. The empirical fixation frequency
over many seeds is compared against theory.fixation_probability(j, m,
theory.fixation_bias(a, b, alpha, r=r)) -- Lemma 3.1's formula with a
*computed*, not fitted, bias (E7 at r=1; E12(b) sweeps r, Lemma 10.1).

E15 is E12b's counterpart under a genuinely continuous kick-weight law
(CliqueFixationConfig.kick_weight_law="uniform") rather than E12b's fixed
alpha -- compare against theory.fixation_bias_distributed, not
theory.fixation_bias, for that one. See experiment_E15_distributed_kick_
curvature_ratchet's docstring for why this needed its own experiment rather
than a parameter sweep on E12b.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import (
    AsynchronousGossip,
    GossipAveraging,
    HybridGossip,
    SynchronousPairwiseGossip,
    make_uniform_alpha_sampler,
    protocol_suffix,
)
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
    # "fixed" (default): every kick uses gossip_alpha exactly, matching
    # theory.fixation_bias's degenerate kick-weight law. "uniform": each
    # kick draws alpha ~ Uniform(0,1) fresh (gossip.make_uniform_alpha_
    # sampler), matching theory.fixation_bias_distributed's default CDF --
    # a genuinely continuous kick-weight law, testing Lemma 3.1/10.1's
    # general (non-degenerate) reduction rather than its point-mass special
    # case. A plain string (not a live Callable) so this config stays
    # JSON-serialisable for the HPC queue; run_clique_fixation_trial builds
    # the actual sampler from it at runtime. Only meaningful for
    # gossip_protocol in {"async_poisson", "sync_pairwise"} (the two that
    # use a scalar per-kick alpha at all) -- "uniform" with
    # "sync_neighbourhood" raises, rather than silently having no effect.
    kick_weight_law: str = "fixed"

    # Hybrid protocol only (gossip_protocol="hybrid"): round ratio K
    # (eq. 2.2c) and the Type-N period, exactly as NaturalCascadeConfig's
    # fields of the same name -- see HybridGossip.pairwise_rate_for_K.
    round_ratio: Optional[float] = None
    epsilon_n_rounds: int = 1

    # B.0.2 mandatory logging: within-clique occupancy n(t) (B-count),
    # sampled after every protocol.execute() call -- Type-P granularity,
    # guaranteed to include every round boundary since Type-N (when it
    # fires) runs inside the same execute() call, before that call's Type-P
    # sub-step. This is the direct, non-fitted measurement of T_hit
    # (eq. 3.2c) the calibration ladder (E14/H2) needs to score q_hyb.
    # Only meaningful -- and only allowed -- under gossip_protocol="hybrid".
    track_occupancy: bool = False


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
    # (round_idx, n_B) after every protocol.execute() call, oldest first;
    # None unless config.track_occupancy=True (see that field's docstring).
    occupancy_log: Optional[List[Tuple[int, int]]] = None

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

    if config.track_occupancy and config.gossip_protocol != "hybrid":
        raise ValueError(
            "track_occupancy requires gossip_protocol='hybrid' (n(t) at "
            f"Type-P/Type-N granularity is undefined for {config.gossip_protocol!r})"
        )

    if config.kick_weight_law not in ("fixed", "uniform"):
        raise ValueError(
            f"Unknown kick_weight_law {config.kick_weight_law!r}; expected "
            f"'fixed' or 'uniform'."
        )
    alpha_sampler = (
        make_uniform_alpha_sampler(np.random.default_rng(config.seed + 43))
        if config.kick_weight_law == "uniform" else None
    )

    _async = config.gossip_protocol == "async_poisson"
    if _async:
        rate = (config.gossip_rate or float(config.m)) / float(config.local_steps)
        protocol = AsynchronousGossip(
            rate=rate, mode="poisson", alpha=config.gossip_alpha,
            alpha_sampler=alpha_sampler,
            rng=np.random.default_rng(config.seed + 42),
        )
    elif config.gossip_protocol == "sync_pairwise":
        protocol = SynchronousPairwiseGossip(
            alpha=config.gossip_alpha,
            alpha_sampler=alpha_sampler,
            rng=np.random.default_rng(config.seed + 42),
        )
    elif config.gossip_protocol == "sync_neighbourhood":
        if config.kick_weight_law != "fixed":
            raise ValueError(
                "kick_weight_law='uniform' has no effect under "
                "gossip_protocol='sync_neighbourhood' (GossipAveraging has "
                "no scalar per-kick alpha to distribute -- it mixes with the "
                "whole neighbourhood via the topology's own weights); "
                "combining them would silently test nothing, so this is an "
                "error rather than a silent no-op."
            )
        protocol = GossipAveraging()
    elif config.gossip_protocol == "hybrid":
        _async = True
        rate = (
            HybridGossip.pairwise_rate_for_K(config.round_ratio, config.epsilon_n_rounds)
            if config.round_ratio is not None
            else (config.gossip_rate or float(config.m))
        )
        protocol = HybridGossip(
            pairwise_rate=rate / float(config.local_steps),
            epsilon_n_rounds=config.epsilon_n_rounds,
            alpha=config.gossip_alpha,
            alpha_sampler=alpha_sampler,
            rng=np.random.default_rng(config.seed + 42),
        )
    else:
        raise ValueError(
            f"Unknown gossip_protocol {config.gossip_protocol!r}; expected "
            f"'async_poisson', 'sync_pairwise', 'sync_neighbourhood', or 'hybrid'."
        )

    def _count_B() -> int:
        return sum(
            1 for a in agents
            if basin_label(
                list(a.registry["model"].model.parameters())[0].detach().numpy(),
                theta_A_np, theta_B_np, config.epsilon,
            ) == "B"
        )

    occupancy_log: Optional[List[Tuple[int, int]]] = [] if config.track_occupancy else None
    _has_round_idx = hasattr(protocol, "round_idx")

    for round_idx in range(config.n_rounds):
        if _has_round_idx:
            protocol.round_idx = round_idx
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            if _async:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
                if occupancy_log is not None:
                    occupancy_log.append((round_idx, _count_B()))
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
        r=config.r, seed=config.seed, fixed_at_B=fixed_at_B, occupancy_log=occupancy_log,
    )


# ---------------------------------------------------------------------------
# Experiment factories: E7 (fixation formula) and E12(b) (curvature ratchet)
# ---------------------------------------------------------------------------


def experiment_E14_round_ratio_sweep(
    K_list: List[float] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
    m_list: List[int] = (4, 8, 16),
    a: float = 0.5,
    b_list: List[float] = (0.02, 0.04),  # Annex B.1's "two lambda"
    epsilon_n_rounds: int = 5,
    seeds: List[int] = tuple(range(20)),
    local_steps: int = 50,
    n_rounds: int = 200,
) -> List[CliqueFixationConfig]:
    """paper1_computing_hybrid_gossip.md Annex B.1's revised E14
    ("round-ratio sweep. Sweep K over four decades... m in {4,8,16}, two
    lambda; per-round fixation frequency from a single seed, and T_hit
    directly from the logged n(t)") -- a DIFFERENT experiment from the
    original paper's E14 (which swept local_steps x G x protocol on the
    full NMH hierarchy testbed, natural_cascade_experiments.
    experiment_E14_meritocratic_filter_local_steps, still queued as-is
    under phase 6a). Given the deliberately unambiguous-naming discipline
    gossip_mechanisms.md establishes for exactly this kind of collision
    (two different things briefly sharing one label), this experiment uses
    prefix "E14RR" (round ratio), never bare "E14", so its task/pkl/review
    provenance can never be confused with phase 6a's E14 data regardless of
    when either was generated.

    K is swept by varying the aggregate Type-P rate at a FIXED
    epsilon_n_rounds, not (as the paper's prose suggests) by varying
    epsilon_n at fixed epsilon_p: q_hyb depends on K alone, not on
    epsilon_p/epsilon_n separately (Sec. 2.1's whole point in defining K as
    a single dimensionless group), and this discrete-round simulator cannot
    represent epsilon_n_rounds < 1, so sweeping via round_ratio (which
    derives pairwise_rate = K / epsilon_n_rounds, HybridGossip.
    pairwise_rate_for_K) realises the identical target K values that
    varying epsilon_n_rounds directly would, without hitting that floor.

    track_occupancy=True on every task: the direct, non-fitted measurement
    of T_hit (eq. 3.2c) and per-round fixation frequency check_phaseh2.py
    scores against theory.fixation_probability_hybrid / detection_floor.
    """
    prefix = "E14RR"
    configs = []
    for m in m_list:
        for b in b_list:
            for K in K_list:
                for seed in seeds:
                    configs.append(CliqueFixationConfig(
                        name=f"{prefix}/m={m}/b={b}/K={K}/seed={seed}",
                        m=m, j_seeds=1, a=a, b=b, r=1.0, seed=seed,
                        local_steps=local_steps, n_rounds=n_rounds,
                        gossip_protocol="hybrid", round_ratio=K,
                        epsilon_n_rounds=epsilon_n_rounds,
                        track_occupancy=True,
                    ))
    return configs


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
    prefix = f"E7{protocol_suffix(gossip_protocol)}"
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
    prefix = f"E12b{protocol_suffix(gossip_protocol)}"
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


def experiment_E15_distributed_kick_curvature_ratchet(
    r_list: List[float] = (1.0, 1.1, 1.2, 1.5, 2.0),
    m_list: List[int] = (4, 8, 16),
    a: float = 0.5,
    seeds: List[int] = tuple(range(50)),
    local_steps: int = 50,
    n_rounds: int = 200,
    gossip_protocol: str = "async_poisson",
) -> List[CliqueFixationConfig]:
    """E15: E12b's within-clique curvature ratchet (Lemma 10.1), but under a
    genuinely continuous kick-weight law (kick_weight_law="uniform": alpha ~
    Uniform(0,1) per kick, gossip.make_uniform_alpha_sampler) instead of
    E12b's fixed alpha=0.5 -- the fair test of Lemma 10.1's GENERAL
    reduction, not its degenerate point-mass special case.

    Why this is a separate experiment from E12b, not a parameter on it: E12b
    is deliberately kept literal to the degenerate law (theory.fixation_bias,
    "kept literal ... disagreement with measured fixation frequency is
    informative, not a bug to silently patch over" -- see that function's
    docstring). Under the degenerate law, at b=0 with the fixed alpha=0.5
    sitting exactly on the r=1 symmetric threshold, ANY r>1 flips
    predicted_q_fix from j/m to a hard 0 (theory.fixation_bias returns
    rho=inf) -- a discontinuity the session that added this experiment
    found does not match the smooth empirical decay E12b's own async data
    shows. theory.fixation_bias_distributed predicts, and this experiment
    tests, whether that smooth decay is recovered once the simulation's own
    kick weight is ALSO genuinely distributed (not just the prediction
    formula): compare empirical fixation frequency against
    theory.fixation_bias_distributed(a, 0.0, r=r) here, not
    theory.fixation_bias(a, 0.0, kick_weight=..., r=r) (E12b's comparison,
    still correct for what E12b actually simulates).

    Output prefix "E15", never "E12b" -- keeps this from being confused
    with or overwriting E12b's own (degenerate-law, still valid) results.
    """
    prefix = f"E15{protocol_suffix(gossip_protocol)}"
    configs = []
    for r in r_list:
        for m in m_list:
            for seed in seeds:
                configs.append(CliqueFixationConfig(
                    name=f"{prefix}/r={r}/m={m}/seed={seed}",
                    m=m, j_seeds=1, a=a, b=0.0, r=r, seed=seed,
                    local_steps=local_steps, n_rounds=n_rounds,
                    gossip_protocol=gossip_protocol, kick_weight_law="uniform",
                ))
    return configs
