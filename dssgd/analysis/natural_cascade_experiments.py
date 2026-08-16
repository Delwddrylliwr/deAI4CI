"""Experiment factory functions for NMH Regime C natural-cascade experiments.

Each function returns a list of NaturalCascadeConfig objects corresponding
to one of the NMH experiments specified in the timesep specification:

  NMH-1  — hierarchical mixing-time scaling (slope-1 test, natural cascade)
  NMH-1b — local_steps sensitivity (Regime C threshold identification)
  NMH-2  — nucleation probability q_l as function of bias b
  NMH-3  — three-regime phase structure with unclamped source
  NMH-4  — cascade size distribution and power-law exponent
  NMH-5  — geometric and temporal filter composition
  NMH-6  — stationary variance decomposition (weak bias b≈0.01)
  NMH-7  — detailed balance and Gibbs structure (b=0, symmetric)

Default Regime C operating point: local_steps=50, clamp_source=False (implicit),
b=0.042 (near bistability limit for a=0.5).

Basin-symmetry requirements (see plan):
  - NMH-1/3/4/5: b≈0.042 (asymmetric OK; topology drives 2^l scaling)
  - NMH-2: sweep b to vary saddle position θ_s
  - NMH-6: b≈0.01 (weak bias; quasi-stationary before absorption)
  - NMH-7: b=0 (symmetric; required by Theorem 2 / Gibbs derivation)

gossip_protocol parameter (all functions):
  "async_poisson"        — AsynchronousGossip (default; Phase xA), no suffix
  "sync_pairwise"        — SynchronousPairwiseGossip (Remark 4.3's H-sched
                           class), suffix "SP" (e.g. "NMH1SP/")
  "sync_neighbourhood"   — GossipAveraging (simultaneous m-way mean, Lemma
                           6.1's basin-destroying mechanism), suffix "SN"
  See dssgd.protocols.gossip.protocol_suffix / gossip_mechanisms.md: the bare
  "synchronous"/"S" label is retired -- it used to mean GossipAveraging, then
  briefly meant SynchronousPairwiseGossip, an ambiguity that made "S"-suffixed
  files impossible to interpret without knowing when they were generated.
"""

from typing import List, Optional, Tuple

import numpy as np

from dssgd.protocols.gossip import protocol_suffix

from . import theory
from .generality import per_leaf_loss_params_for_generality
from .natural_cascade import NaturalCascadeConfig


# ---------------------------------------------------------------------------
# NMH-1: Hierarchical mixing-time scaling
# ---------------------------------------------------------------------------


def experiment_NMH1(
    a_list: List[float] = (0.5, 1.0, 2.0, 4.0),
    seeds: List[int] = tuple(range(50)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Primary slope-1 test: log₂(T_flip(d)) linear in d with slope ≈ 1.

    Tests Proposition 4.2 (eq. 4.3): T_fp^(l) = Θ(2^{l+1} / (ε·p·M_0·q_l)).
    Sweeps a across Regime I (all levels flip); 50 seeds for CI estimation.

    Uses force_flip_source=True (one random leaf set to B post-warmup) so
    that cascade propagation is driven purely by gossip without thermal noise.
    """
    prefix = f"NMH1{protocol_suffix(gossip_protocol)}"
    configs = []
    for a in a_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/a={a}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                local_steps=local_steps,
                a=a,
                b=b,
                force_flip_source=True,
                flip_noise_scale=0.0,
                gossip_protocol=gossip_protocol,
            ))
    return configs


# ---------------------------------------------------------------------------
# NMH-1b: local_steps sensitivity
# ---------------------------------------------------------------------------


def experiment_NMH1b(
    a: float = 0.5,
    local_steps_list: List[int] = (10, 50, 200, 1000),
    seeds: List[int] = tuple(range(20)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    b: float = 0.042,
    lr: float = 0.1,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Regime C sensitivity: slope-1 accuracy as function of local_steps.

    Predicts: slope-1 becomes accurate only above a threshold local_steps;
    below it, Regime A/B contamination introduces drift.
    Force-flip source so that the cascade is gossip-driven across all local_steps values.
    """
    prefix = f"NMH1b{protocol_suffix(gossip_protocol)}"
    configs = []
    for ls in local_steps_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/ls={ls}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                local_steps=ls,
                a=a,
                b=b,
                force_flip_source=True,
                flip_noise_scale=0.0,
                gossip_protocol=gossip_protocol,
            ))
    return configs


# ---------------------------------------------------------------------------
# NMH-1sb: Transition-zone 2D sweep (a × local_steps, synchronous gossip)
# ---------------------------------------------------------------------------


def experiment_NMH1sb(
    a_list: List[float] = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    local_steps_list: List[int] = (5, 10, 20, 30, 50),
    seeds: List[int] = tuple(range(75)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    b: float = 0.042,
    lr: float = 0.1,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "sync_pairwise",
) -> List[NaturalCascadeConfig]:
    """Transition-zone phase diagram: cascade size distributions across a × local_steps.

    Targets the regime where synchronous gossip transitions from full cascade
    (a≈0.5) to complete arrest (a≥1.0), characterising whether an intermediate
    Griffiths power-law phase exists. Force-flip source decouples cascade
    propagation from nucleation statistics.
    """
    prefix = f"NMH1sb{protocol_suffix(gossip_protocol)}"
    configs = []
    for a in a_list:
        for ls in local_steps_list:
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"{prefix}/a={a}/ls={ls}/seed={seed}",
                    depth=depth,
                    leaf_size=leaf_size,
                    p=p,
                    seed=seed,
                    n_warmup=n_warmup,
                    n_meas_rounds=n_meas,
                    lr=lr,
                    local_steps=ls,
                    a=a,
                    b=b,
                    force_flip_source=True,
                    flip_noise_scale=0.0,
                    gossip_protocol=gossip_protocol,
                ))
    return configs


# ---------------------------------------------------------------------------
# NMH-2: Nucleation probability q_l as bias function
# ---------------------------------------------------------------------------


def experiment_NMH2(
    b_list: List[float] = (0.02, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050),
    seeds: List[int] = tuple(range(100)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    a: float = 1.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Nucleation probability q_l as function of basin bias b.

    Tests Lemma 3.1's fixation formula (eq. 3.2) at j=1: q_l = q_fix(1;
    M_{l-1}, rho) = (1 - rho) / (1 - rho^{M_{l-1}}), fit here in its
    voter-functional-form guise q_l = (1-e^{-2θ})/(1-e^{-2θ*M_{l-1}}) with θ
    a free parameter (Gate 2's ΔAIC criterion tests this functional form
    against a pairwise-constant alternative -- see check_phase2.py).
    Sweeps b to vary the effective saddle shift θ_s and empirical q_l.
    100 seeds per b value (q_l is a frequency estimate needing large N).
    Natural nucleation (no force-flip): small Langevin noise enables escape
    so that the rate varies measurably across b values.
    """
    prefix = f"NMH2{protocol_suffix(gossip_protocol)}"
    configs = []
    for b in b_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/b={b:.4f}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                local_steps=local_steps,
                a=a,
                b=b,
                flip_noise_scale=0.10,
                gossip_protocol=gossip_protocol,
            ))
    return configs


# ---------------------------------------------------------------------------
# NMH-3: Three-regime phase structure
# ---------------------------------------------------------------------------


def experiment_NMH3(
    a_list: Optional[List[float]] = None,
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Phase structure test: ferromagnetic / Griffiths / paramagnetic regimes.

    Tests Sec. 4.5's ordered/stratified/contained phase structure under
    force-flip cascade dynamics.
    Observable: cascade depth d_max(a) and fraction f_L(a) reaching root.

    Predicted boundaries at p=2:
      Regime I (ferromagnetic): a < 0.5
      Regime II (Griffiths):    0.5 ≤ a < 8
      Regime III (paramagnetic): a ≥ 8
    Force-flip source so that higher-a suppression reflects gossip attenuation,
    not a lack of nucleation events.
    """
    if a_list is None:
        a_list = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    prefix = f"NMH3{protocol_suffix(gossip_protocol)}"
    configs = []
    for a in a_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/a={a}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                local_steps=local_steps,
                a=a,
                b=b,
                force_flip_source=True,
                flip_noise_scale=0.0,
                gossip_protocol=gossip_protocol,
            ))
    return configs


# ---------------------------------------------------------------------------
# NMH-4: Cascade size distribution (power-law exponent)
# ---------------------------------------------------------------------------


def experiment_NMH4(
    seeds: List[int] = tuple(range(2000)),
    depth: int = 7,
    leaf_size: int = 4,
    p: float = 2.0,
    a: float = 2.0,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 800,
    t_horizon: int = 600,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Cascade size distribution: P(size=s) ~ s^{-τ} in the Griffiths phase.

    Tests Proposition 4.7 (eq. 4.6): the quenched-vs-annealed exponent gap
    tau_typ - tau_ann as the rare-region (Griffiths) signature.  Uses depth=7
    (N=512) for range
    over at least 1.5 decades.  2000 seeds for power-law fitting (MLE).
    a=2 is in the middle of the Griffiths regime (0.5 ≤ a < 8 for p=2).
    Force-flip source so cascade size reflects gossip propagation, not nucleation.
    """
    prefix = f"NMH4{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        configs.append(NaturalCascadeConfig(
            name=f"{prefix}/seed={seed}",
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a,
            b=b,
            t_horizon=t_horizon,
            force_flip_source=True,
            flip_noise_scale=0.0,
            gossip_protocol=gossip_protocol,
        ))
    return configs


# ---------------------------------------------------------------------------
# NMH-5: Geometric and temporal filter composition
# ---------------------------------------------------------------------------


def experiment_NMH5(
    a_list: List[float] = (1.0, 2.0, 4.0),
    b_on_a_list: List[float] = (0.02, 0.03, 0.04, 0.05, 0.06),
    seeds: List[int] = tuple(range(50)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Filter composition: propagation depth d_prop as function of b/a.

    Tests Theorem 4.5 (eq. 4.5): the level-matching filter's geometric gate
    (vartheta <= max C) and soft persistence gate composing via Prop. 3.3.
    Observable: d_prop(b/a) increases monotonically (flatter basins propagate
    further) with a sharp threshold at the deepest level geometric filter.
    Force-flip source so propagation depth reflects gossip filter attenuation.
    """
    prefix = f"NMH5{protocol_suffix(gossip_protocol)}"
    configs = []
    for a in a_list:
        for b_on_a in b_on_a_list:
            b = b_on_a * a
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"{prefix}/a={a}/b_on_a={b_on_a:.3f}/seed={seed}",
                    depth=depth,
                    leaf_size=leaf_size,
                    p=p,
                    seed=seed,
                    n_warmup=n_warmup,
                    n_meas_rounds=n_meas,
                    lr=lr,
                    local_steps=local_steps,
                    a=a,
                    b=b,
                    force_flip_source=True,
                    flip_noise_scale=0.0,
                    gossip_protocol=gossip_protocol,
                ))
    return configs


# ---------------------------------------------------------------------------
# NMH-6: Stationary variance decomposition (heterogeneous loss, weak bias)
# ---------------------------------------------------------------------------


def experiment_NMH6(
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    a_base: float = 1.0,
    b_base: float = 0.01,
    b_spread: float = 0.003,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 2000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Stationary variance decomposition with heterogeneous per-leaf loss.

    Tests Sec. 4.8's heuristic ANOVA identity V_L = Σ_l V_l^between (scaling
    V_l^between ~ 2^{-(L-l)*zeta} is a Heuristic, exponent zeta parametric --
    see nmh_observables.hierarchical_variance_decomposition for the
    nested-ANOVA estimator check_phase2.py's nmh6_decomp_ok scores).
    Per-leaf b_i ~ N(b_base, b_spread²), clipped to [0.005, 0.035] to keep
    all leaves in genuine bistable regime (bistability limit ≈ 0.074 for a=1)
    with non-negligible B→A back-transitions.

    b_base=0.01 (weak bias) ensures a quasi-stationary distribution exists
    before absorption.  Long n_meas=2000 for time-averaged variance.
    """
    branching = 2
    n_leaf_types = branching ** depth
    prefix = f"NMH6{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 99991)  # independent of simulation seed
        b_vals = rng.normal(b_base, b_spread, size=n_leaf_types)
        b_vals = np.clip(b_vals, 0.005, 0.035).tolist()
        per_leaf_loss_params = [(a_base, float(b_i)) for b_i in b_vals]
        configs.append(NaturalCascadeConfig(
            name=f"{prefix}/seed={seed}",
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a_base,
            b=b_base,
            per_leaf_loss_params=per_leaf_loss_params,
            flip_noise_scale=0.10,
            gossip_protocol=gossip_protocol,
        ))
    return configs


# ---------------------------------------------------------------------------
# NMH-7: Detailed balance and Gibbs structure (symmetric basins)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# E1: Provenance tracing + gossip-severed control (Cor. 3.5, Prop. 3.4)
# ---------------------------------------------------------------------------


def experiment_E1(
    a_list: List[float] = (0.5, 1.0, 2.0, 4.0),
    b: float = 0.042,
    sigma_list: List[float] = (0.0, 0.01),
    sever_min_distances: List[Optional[int]] = (None, 1, 2, 3),
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
) -> List[NaturalCascadeConfig]:
    """Provenance tracing (kick vs escape attribution) at matched (a,b,sigma)
    operating points, plus the gossip-severed control at several severing
    distances (sever_min_distances includes None = no severing, i.e. the
    ordinary provenance-tracked run) -- Proposition 3.4's slope-0 null is
    read off the severed variants, cascade-attributable statistics off the
    unsevered ones, both via the SAME track_provenance=True mechanism.
    async_poisson only (provenance requires it).
    """
    configs = []
    for a in a_list:
        for sigma in sigma_list:
            for sever_d in sever_min_distances:
                tag = "none" if sever_d is None else str(sever_d)
                for seed in seeds:
                    configs.append(NaturalCascadeConfig(
                        name=f"E1/a={a}/sigma={sigma}/sever={tag}/seed={seed}",
                        depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                        n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                        a=a, b=b, flip_noise_scale=sigma,
                        force_flip_source=(sever_d is None), track_provenance=True,
                        sever_min_distance=sever_d, gossip_protocol="async_poisson",
                    ))
    return configs


# ---------------------------------------------------------------------------
# E2: Fixed-lambda sweep (Remark 2.3)
# ---------------------------------------------------------------------------


def experiment_E2(
    a_list: List[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
    a_anchor: float = 0.5,
    b_anchor: float = 0.042,
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Fixed-lambda sweep: b co-varies with a to hold lambda constant at the
    validated a=0.5 anchor, while a ranges across the boundary observed
    under a fixed-b sweep (NMH-3). Discriminates the geometric criterion
    (transmission uniform across this sweep) from the energetic criterion
    (the boundary reappears at the same absolute a as NMH-3), per Remark 2.3.
    """
    lam = theory.dimensionless_tilt(a_anchor, b_anchor)
    prefix = f"E2{protocol_suffix(gossip_protocol)}"
    configs = []
    for a in a_list:
        b = lam * theory.KAPPA_PHI * a
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/a={a}/lambda={lam:.4f}/seed={seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                gossip_protocol=gossip_protocol,
            ))
    return configs


# ---------------------------------------------------------------------------
# E3: Fine interior sweep + hysteresis (Section 4.5, boundary-ratio constraint)
# ---------------------------------------------------------------------------


def experiment_E3(
    a_list: List[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    b: float = 0.042,
    seeds: List[int] = tuple(range(75)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    init_basins: List[str] = ("A", "B"),
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Fine interior sweep at high seed count (staircase in d_max vs cliff),
    run from BOTH all-A and all-B initialisation (hysteresis test): a single
    leaf is force-flipped against the bulk's basin in each case (init_basin=A
    forces one leaf to B; init_basin=B forces one leaf to A -- the runner
    picks the flip target as the opposite of init_basin automatically), and
    the question is whether that "wrong-direction" leaf is reabsorbed by the
    ratchet (expected, if the bulk's basin is genuinely favoured at this a)
    or persists/cascades (a hysteresis break -- the reachable phase would
    then depend on initial condition, not just (a,b)).
    """
    prefix = f"E3{protocol_suffix(gossip_protocol)}"
    configs = []
    for a in a_list:
        for init_basin in init_basins:
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"{prefix}/a={a}/init={init_basin}/seed={seed}",
                    depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b, init_basin=init_basin, force_flip_source=True,
                    flip_noise_scale=0.0, gossip_protocol=gossip_protocol,
                ))
    return configs


# ---------------------------------------------------------------------------
# E4: Quenched vs annealed cascade-size tails (Proposition 4.7)
# ---------------------------------------------------------------------------


def experiment_E4(
    a: float = 2.0,
    b: float = 0.042,
    n_graph_seeds: int = 30,
    n_dynamics_seeds_per_graph: int = 5,
    depth: int = 7,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Quenched-vs-annealed cascade tail comparison: decouples the topology
    draw (graph_seed) from the dynamics draw (seed) so the same quenched
    wiring realisation can be replayed against multiple dynamics seeds (the
    quenched ensemble: fixed graph_seed, varying seed) while the graph also
    varies across the outer loop (the annealed ensemble: pool over all
    graph_seed values) -- both are read off the same task list, grouped
    differently at analysis time (with nmh_observables.realized_cross_edge_count
    keyed on graph_seed, not seed). depth=7 matches NMH-4's deeper hierarchy.
    """
    prefix = f"E4{protocol_suffix(gossip_protocol)}"
    configs = []
    for g in range(n_graph_seeds):
        for d in range(n_dynamics_seeds_per_graph):
            dynamics_seed = g * n_dynamics_seeds_per_graph + d
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/graph_seed={g}/seed={dynamics_seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=dynamics_seed, graph_seed=g,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                gossip_protocol=gossip_protocol,
            ))
    return configs


# ---------------------------------------------------------------------------
# E5: sigma-sweep for the crossover stage l_c (eq. 3.6)
# ---------------------------------------------------------------------------


def experiment_E5(
    sigma_list: List[float] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05),
    a: float = 2.0,
    b: float = 0.042,
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    K_list: Tuple[float, ...] = (),
    epsilon_n_rounds: int = 5,
) -> List[NaturalCascadeConfig]:
    """sigma-sweep at fixed (a,b,epsilon): track_provenance=True so l_c
    (highest hierarchical distance still kick-attributed) can be computed
    post-hoc from the returned event log via provenance.crossover_stage.

    paper1_computing_hybrid_gossip.md Annex B.2's revised E5 ("sigma-sweep
    at fixed (a,b,epsilon_p), repeated at three K; measure l_c... and the
    widening of the attributable window at finite K, Rem. 3.7"): K_list=()
    (default) reproduces the ORIGINAL single-protocol E5 exactly --
    async_poisson only, phase 6a's existing call site is unaffected. Each K
    in K_list adds a matched "E5H" arm at that round ratio over the SAME
    (a,b,sigma,seed) grid, gossip_protocol="hybrid" with track_provenance=
    True -- HybridGossip's Type-P sub-step gets the identical
    ProvenanceAsyncGossip event logging the async_poisson arm uses (see
    gossip.py's HybridGossip.pairwise_protocol / natural_cascade.py's
    "hybrid" track_provenance branch), so l_c is measured identically
    across the K axis and the async_poisson arm IS the K -> infinity
    reference point Rem. 3.7's widening claim compares against -- not a
    separate mechanism requiring a separate null.
    """
    configs = []
    for sigma in sigma_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"E5/sigma={sigma}/seed={seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, flip_noise_scale=sigma, force_flip_source=False,
                track_provenance=True, gossip_protocol="async_poisson",
            ))
    for K in K_list:
        for sigma in sigma_list:
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"E5H/K={K}/sigma={sigma}/seed={seed}",
                    depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b, flip_noise_scale=sigma, force_flip_source=False,
                    track_provenance=True, gossip_protocol="hybrid",
                    round_ratio=K, epsilon_n_rounds=epsilon_n_rounds,
                ))
    return configs


def experiment_E5Hv2(
    sigma_list: List[float] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4),
    K_list: Tuple[float, ...] = (),
    a: float = 2.0,
    b: float = 0.042,
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    epsilon_n_rounds: int = 5,
) -> List[NaturalCascadeConfig]:
    """E5H, re-parameterised: H3's real run of experiment_E5's K_list arm
    ("E5H") came back with an EMPTY crossover table -- zero flips in 540/540
    runs, at every K and every sigma up to 0.05 (E5's original sigma_list
    ceiling). Confirmed this is not hybrid-specific: the pre-existing,
    unrelated original async_poisson E5 arm (results/phase6a/pkl/E5, 180
    runs) shows the SAME zero-flips-at-every-sigma result, and gate6_review
    .json's own e5_crossover has always been [] -- this predates the hybrid
    work and was never gated or flagged.

    Root cause, quantified: at a=2.0, b=0.042, chord_geometry gives a saddle
    distance of ~0.436 from theta_A (chord length 1.0). The noise mechanism
    (natural_cascade.py's flip_noise_scale) adds INDEPENDENT N(0, sigma^2)
    noise to each agent separately, once per measurement round; the leaf-
    level centroid basin classification (nmh_observables/active_escape's
    find_t_flip) tracks the MEAN across a leaf's leaf_size agents, whose
    effective noise std is sigma/sqrt(leaf_size) -- at leaf_size=4, sigma/2.
    Crossing the 0.436 saddle distance therefore needs roughly a
    (2*0.436/sigma)-effective-sigma event. At the original ceiling sigma=
    0.05 that's an ~17-effective-sigma event (astronomically improbable);
    empirically (local calibration at depth=2/n_meas=100/local_steps=10,
    3 seeds/point) sigma=0.05/0.1/0.15 gave 0/3 seeds with any flip while
    sigma=0.2/0.3 gave 3/3 -- the working threshold sits between 0.15 and
    0.2, consistent with the leaf-averaged distance estimate. sigma_list's
    new default spans that transition (0.1, 0.15 as below-threshold
    controls; 0.2 upward as the working range) rather than sitting entirely
    below it.

    K_list=() (default) reproduces the async_poisson-only sweep at the NEW
    sigma range; each K in K_list adds the matched hybrid ("E5Hv2") arm,
    identically to experiment_E5's K_list mechanism. a/b/p/depth/leaf_size
    are unchanged from experiment_E5's defaults deliberately: Lemma 2.1's
    scale invariance means the saddle's CHORD-FRACTION location is
    unaffected by curvature, so only sigma needed to move, not the
    landscape itself -- keeping (a,b) fixed means this remains the same
    physical comparison point E11H/E16v2 and the rest of the K-axis
    campaign share.
    """
    prefix_async = "E5v2"
    prefix_hybrid = "E5Hv2"
    configs = []
    for sigma in sigma_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix_async}/sigma={sigma}/seed={seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, flip_noise_scale=sigma, force_flip_source=False,
                track_provenance=True, gossip_protocol="async_poisson",
            ))
    for K in K_list:
        for sigma in sigma_list:
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"{prefix_hybrid}/K={K}/sigma={sigma}/seed={seed}",
                    depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b, flip_noise_scale=sigma, force_flip_source=False,
                    track_provenance=True, gossip_protocol="hybrid",
                    round_ratio=K, epsilon_n_rounds=epsilon_n_rounds,
                ))
    return configs


# ---------------------------------------------------------------------------
# E6 / E12(a): Level-matching filter and overfitting-as-low-generality
# ---------------------------------------------------------------------------
#
# Both experiments realise Definition 4.2's tree-structured heterogeneity
# model via the EXISTING per_leaf_loss_params mechanism (built for NMH-6):
# in-scope leaves (generality.in_scope, level=G) get a favourable per-leaf
# bias, out-of-scope leaves an unfavourable one. E12(a)'s "overfitting
# injected at scale l" is, by the paper's own reduction (Prop. 4.9's proof:
# "the containment half of Theorem 4.5 applies verbatim"), exactly this
# mechanism with generality_level=l -- so one factory serves both,
# parametrised by which generality level is under test.


def experiment_E6(
    generality_levels: List[int] = (1, 2, 3, 4, 5),
    a: float = 0.5,
    b_in: float = 0.042,
    b_out: float = 0.042,
    seeds: List[int] = tuple(range(50)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    source_leaf: int = 0,
    gossip_protocol: str = "async_poisson",
    hybrid_K_list: Tuple[float, ...] = (),
    epsilon_n_rounds: int = 5,
) -> List[NaturalCascadeConfig]:
    """E6: controlled generality level G(b) via per-leaf bias; measures
    max propagation depth d_max vs G (Theorem 4.5's level-matching filter).
    Use nmh_observables.cascade_depth(centroid_traj, source_leaf=source_leaf,
    ...) on the returned run to read off d_max, and check it equals G.

    force_flip_source=True (matching NMH-3/E11's convention): Theorem 4.5 is
    a claim about PROPAGATION (does an already-arisen innovation reach depth
    G?), not about spontaneous origination. Under deterministic gradient
    descent with lambda<1 (genuine bistability, per Lemma 2.1) and no peer
    kicks available to an isolated source at t=0, the source leaf has no
    mechanism to escape basin A on its own -- confirmed empirically: with
    force_flip_source=False (the prior default here), the source never
    nucleates at all (nucleation_leaf=None every seed), making d_max=0
    universally and by construction, regardless of the generality/tree-loss
    machinery under test. Force-flipping the source is exactly the "assume
    the innovation has just occurred" initial condition NMH-1/NMH-3/E11 all
    use to isolate propagation from origination.

    paper1_computing_hybrid_gossip.md Annex B.3's E6 ("... a (K, lambda)
    grid... testing the margin condition of Rem. 4.5'"), phase H4:
    hybrid_K_list=() (default) reproduces the ORIGINAL single-protocol E6
    exactly, so phase 5a's existing call site is unaffected. Each K in
    hybrid_K_list adds a matched "E6H" arm at that round ratio over the
    SAME (G, seed) grid, gossip_protocol="hybrid" -- testing whether K
    (filter *resolution*) and m/leaf_size (filter *reliability*) actually
    separate as Rem. 4.5' predicts, against this K -> infinity
    (gossip_protocol="async_poisson") arm as the reference point.
    """
    branching = 2
    n_leaf_types = branching ** depth
    prefix = f"E6{protocol_suffix(gossip_protocol)}"
    configs = []
    for G in generality_levels:
        plp = per_leaf_loss_params_for_generality(
            n_leaf_types=n_leaf_types, source_leaf=source_leaf, generality_level=G,
            branching=branching, a=a, b_in=b_in, b_out=b_out,
        )
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/G={G}/seed={seed}",
                branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b_in, per_leaf_loss_params=plp, force_flip_source=True,
                source_leaf=source_leaf, gossip_protocol=gossip_protocol,
            ))
    for K in hybrid_K_list:
        for G in generality_levels:
            plp = per_leaf_loss_params_for_generality(
                n_leaf_types=n_leaf_types, source_leaf=source_leaf, generality_level=G,
                branching=branching, a=a, b_in=b_in, b_out=b_out,
            )
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"E6H/K={K}/G={G}/seed={seed}",
                    branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b_in, per_leaf_loss_params=plp, force_flip_source=True,
                    source_leaf=source_leaf, gossip_protocol="hybrid",
                    round_ratio=K, epsilon_n_rounds=epsilon_n_rounds,
                ))
    return configs


def experiment_E6_positive_control(
    generality_levels: List[int] = (1, 2),
    a: float = 0.5,
    b_in: float = 0.042,
    b_out: float = 0.042,
    seeds: List[int] = tuple(range(20)),
    depth: int = 3,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    source_leaf: int = 0,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Positive control for E6/E12a's containment/attainment machinery
    (Theorem 4.5), assessment doc A.6: a small, fast (depth=3 instead of the
    production depth=5) variant of experiment_E6 at the SAME default bias, so
    check_phase6.py/check_phase5.py can verify the generality-scoping +
    force-flip + cascade_depth pipeline actually detects d_max==G before any
    real E6/E12a null (d_max==0 for every G) is trusted. Uses the identical
    mechanism as experiment_E6 (same per_leaf_loss_params_for_generality,
    same force_flip_source=True/source_leaf targeting) -- this is a scale
    reduction for cheap, frequent verification, not a different (easier)
    physical regime; confirmed to give d_max==G reliably at these defaults
    once source_leaf is correctly force-flipped (see natural_cascade.py's
    source_leaf field and experiment_E6's docstring for why the pre-fix
    version could never nucleate at all).
    """
    configs = experiment_E6(
        generality_levels=list(generality_levels), a=a, b_in=b_in, b_out=b_out,
        seeds=seeds, depth=depth, leaf_size=leaf_size, p=p, lr=lr,
        local_steps=local_steps, n_warmup=n_warmup, n_meas=n_meas,
        source_leaf=source_leaf, gossip_protocol=gossip_protocol,
    )
    prefix = f"E6ctrl{protocol_suffix(gossip_protocol)}"
    for cfg in configs:
        cfg.name = cfg.name.replace(f"E6{protocol_suffix(gossip_protocol)}/", f"{prefix}/")
    return configs


def experiment_E12a(
    overfit_scales: List[int] = (1, 2, 3),
    a: float = 0.5,
    b_in: float = 0.042,
    b_out: float = 0.042,
    seeds: List[int] = tuple(range(50)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    source_leaf: int = 0,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """E12(a): overfitting-as-low-generality (Prop. 4.9). Identical
    mechanism to E6 with generality_level = the injected overfitting scale
    l: containment (d_max <= l with probability -> 1 in m) is the same
    Theorem-4.5 bound, read on a basin relabelled "overfit at scale l"
    rather than "innovation with G(b)=l" -- the paper's point is that these
    are the same object.
    """
    configs = experiment_E6(
        generality_levels=list(overfit_scales), a=a, b_in=b_in, b_out=b_out, seeds=seeds,
        depth=depth, leaf_size=leaf_size, p=p, lr=lr, local_steps=local_steps,
        n_warmup=n_warmup, n_meas=n_meas, source_leaf=source_leaf,
        gossip_protocol=gossip_protocol,
    )
    prefix = f"E12a{protocol_suffix(gossip_protocol)}"
    for cfg in configs:
        cfg.name = cfg.name.replace(f"E6{protocol_suffix(gossip_protocol)}/", f"{prefix}/")
    return configs


# ---------------------------------------------------------------------------
# E11: Scheduling sweep (Remark 4.3)
# ---------------------------------------------------------------------------


def experiment_E11(
    schedulings: List[Tuple[str, int]] = (
        ("sync_pairwise", 0), ("async_poisson", 0), ("bounded_staleness", 2),
        ("bounded_staleness", 5), ("bounded_staleness", 10),
    ),
    a_list: List[float] = (0.5, 1.0, 2.0, 4.0),
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    hybrid_K_list: Tuple[float, ...] = (),
    epsilon_n_rounds: int = 5,
) -> List[NaturalCascadeConfig]:
    """Scheduling sweep at fixed (a,b,epsilon): (i) round-synchronous,
    (ii) free per-edge async, (iii) bounded-staleness at several staleness
    bounds. track_provenance=True wherever the protocol is async_poisson-
    based (bounded_staleness's event mechanics are the same class, but
    provenance tracking is only implemented for AsynchronousGossip and its
    subclasses via ProvenanceAsyncGossip -- bounded_staleness therefore runs
    WITHOUT provenance filtering here; the concavity/slope test on raw
    log2(T_flip) vs d is still meaningful, just not provenance-filtered).

    paper1_computing_hybrid_gossip.md Annex B.2's revised E11 additionally
    brackets (i)/(ii) with (iii) the two-jump protocol itself, "at three K
    spanning the H-round window" (H2's measured K_low/K_mid/K_high, per
    Remark 4.3's protocol paragraph: under the hybrid the immigration-driven
    degradation should appear at LOW d rather than high d, the reversal
    (3.4a) predicts). hybrid_K_list=() (default) reproduces the original
    scheduling-only sweep exactly, unaffected for phase 6a's existing call
    site. Each K in hybrid_K_list adds a gossip_protocol="hybrid" arm over
    the same (a,b,seed) grid, prefixed "E11H" so it can't be confused by
    name with the scheduling-only "E11" runs. track_provenance stays False
    here (unlike E5): E11's own concavity/slope test reads raw
    log2(T_flip) vs d from centroid_traj, exactly as its existing
    bounded_staleness entries already do "without provenance filtering."
    """
    configs = []
    for protocol, staleness in schedulings:
        for a in a_list:
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"E11/proto={protocol}/stale={staleness}/a={a}/seed={seed}",
                    depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                    gossip_protocol=protocol, staleness_bound=staleness,
                    track_provenance=(protocol == "async_poisson"),
                ))
    for K in hybrid_K_list:
        for a in a_list:
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"E11H/K={K}/a={a}/seed={seed}",
                    depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                    gossip_protocol="hybrid", round_ratio=K, epsilon_n_rounds=epsilon_n_rounds,
                ))
    return configs


# ---------------------------------------------------------------------------
# E16: Cross-module ceiling for the Type-N channel (Proposition 4.10)
# ---------------------------------------------------------------------------


def experiment_E16(
    p_list: List[float] = (0.5, 2.0, 8.0, 32.0),
    a: float = 0.5,
    b: float = 0.02,  # fixed lambda -- b is in both E0's and E7's shared grid
    seeds: List[int] = tuple(range(20)),
    depth: int = 5,
    leaf_size: int = 4,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 2000,  # "long horizons" (Annex B.2's E16 design sketch)
) -> List[NaturalCascadeConfig]:
    """Annex B.2's E16: ceiling location (Proposition 4.10, eq. 4.9).
    "Type-N rounds only, no Type-P channel; sweep p... at fixed
    vartheta_dagger (via fixed lambda from E0); measure the highest level
    ever crossed over long horizons." Realised as gossip_protocol="hybrid"
    with round_ratio=0.0 -- HybridGossip.pairwise_rate_for_K(0, ...) == 0.0
    exactly, so the Type-P sub-protocol's Poisson rate is genuinely zero
    (AsynchronousGossip.execute with rate=0 always draws 0 events; this is
    the SAME mechanism/logging path every other H-phase experiment uses,
    not a separate sync_neighbourhood special case -- deliberately, so a
    ceiling measured here is directly comparable to E11H/E5H's K axis
    rather than a structurally different run).

    force_flip_source=True (propagation test, matching E6/E11/E13's
    convention): Proposition 4.10's ceiling question is "how far does an
    ALREADY-ARISEN innovation get under Type-N alone", not origination.

    Annex B.2's own note on the default: "at p=2, vartheta_dagger=1/2,
    (4.9) gives l_theta=0 -- the ceiling is ABSENT rather than LOCATED."
    p_list's default already includes p=8,32 to move l_theta into {1,2} and
    make the prediction quantitative rather than a null; scoring
    (check_phaseh3.py) should read off the highest hierarchical distance
    any leaf ever flipped to, per seed, via nmh_observables.cascade_depth,
    and compare against theory's l_theta at each swept p.
    """
    prefix = "E16"
    configs = []
    for p in p_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/p={p}/seed={seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                gossip_protocol="hybrid", round_ratio=0.0, epsilon_n_rounds=1,
            ))
    return configs


def experiment_E16src(
    p_list: List[float] = (0.5, 2.0, 8.0, 32.0),
    a: float = 0.5,
    b: float = 0.02,
    seeds: List[int] = tuple(range(20)),
    depth: int = 5,
    leaf_size: int = 4,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 2000,
) -> List[NaturalCascadeConfig]:
    """Phase H4's source-commitment/boundary-worker diagnostic for E16's
    p-inversion. Configs are IDENTICAL to experiment_E16v2 (same
    connectivity-retry-fixed topology construction) -- the only thing that
    differs is the runtime code that produces the pkl: this is meant to be
    queued only after NaturalCascadeRun/hpc.worker's `source_leaf`
    persistence fix has landed, so the resulting pkls carry the TRUE
    force-flipped source_leaf (not just the post-hoc nucleation_leaf
    proxy), letting check_phaseh4.py's compute_e16src_table compute
    nmh_observables.source_module_boundary_fraction unambiguously in every
    cell, including the "source never committed" cells the proxy can't
    resolve. See check_phaseh3.py's re-scored gateh3_e16_ceiling.csv
    (H4 fix) for the preliminary, proxy-based version of this same
    analysis run against the ORIGINAL phase H3 E16 pkls with no new
    compute -- already striking: frac_source_committed fell 1.0 -> 0.25 ->
    0.0 -> 0.0 as p rose 0.5 -> 2 -> 8 -> 32, matching the boundary-worker
    dilution hypothesis's predicted direction.

    Coordination note: if phase H3fix's own experiment_E16v2 run happens to
    execute AFTER the source_leaf persistence fix has landed, its pkls
    already satisfy everything this function exists for -- read
    results/phaseh3fix/pkl/E16v2/ directly instead of queuing this as a
    separate "E16src" run, to avoid duplicating identical compute under a
    different name.
    """
    prefix = "E16src"
    configs = []
    for p in p_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/p={p}/seed={seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                gossip_protocol="hybrid", round_ratio=0.0, epsilon_n_rounds=1,
            ))
    return configs


def experiment_E16v2(
    p_list: List[float] = (0.5, 2.0, 8.0, 32.0),
    a: float = 0.5,
    b: float = 0.02,
    seeds: List[int] = tuple(range(20)),
    depth: int = 5,
    leaf_size: int = 4,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 2000,
) -> List[NaturalCascadeConfig]:
    """E16, re-issued under a new name rather than re-run under the same one
    -- H3's real run of `experiment_E16` found p=0.5 disconnects ~80% of the
    time for this (branching=2, depth=5, leaf_size=4) shape (16/20 seeds
    failed with `NestedModularTopology(...) produced a disconnected graph`),
    so its surviving 4-seed p=0.5 sample is survivorship-biased toward
    atypically well-connected draws, not a fair sample of "what happens at
    p=0.5." `build_nmh_topology_with_retry` (analysis/natural_cascade.py)
    now retries with a perturbed graph seed on exactly that failure, giving
    every seed here a fair shot at a full, unbiased 20-seed sample --- but
    reusing the bare "E16" name for data collected under a different (now
    bug-fixed) topology-construction policy would silently conflate two
    different samples under one label, exactly the kind of ambiguity
    gossip_mechanisms.md's "S" retirement and this campaign's own E14/E14RR,
    E15/E15MB splits were built to avoid. Identical to `experiment_E16` in
    every parameter (p=2/8/32 arms are bit-for-bit reproducible against the
    original run, since attempt=0 always succeeds immediately there with the
    unperturbed seed) -- only the topology-construction robustness differs.

    Scoring should ALSO compute nmh_observables.source_module_consensus per
    run (not just cascade_depth): H3's real data showed mean_d_max falling
    as p rose from 2 to 32, opposite theory.cross_module_ceiling's
    prediction (larger p -> more permissive ceiling). One live hypothesis is
    that this is source-side (the source module's own members become
    boundary workers at high p, diluting its OWN internal consensus before
    propagation is ever tested, per Lemma 2.4's boundary-worker exception) --
    source_module_consensus distinguishes "source never even committed"
    (source-side artifact) from "source committed but nothing propagated"
    (a genuine transport-ceiling finding).
    """
    prefix = "E16v2"
    configs = []
    for p in p_list:
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/p={p}/seed={seed}",
                depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, force_flip_source=True, flip_noise_scale=0.0,
                gossip_protocol="hybrid", round_ratio=0.0, epsilon_n_rounds=1,
            ))
    return configs


# ---------------------------------------------------------------------------
# E13: DAG-nested ideal-matching filter (Section 11, Theorem 11.3)
# ---------------------------------------------------------------------------


def experiment_E13(
    delta_in_list: List[int] = (1, 2, 3, 4),
    m_list: List[int] = (4, 8, 16, 32),
    overlap_level: int = 1,
    n_overlap: int = 4,
    a: float = 0.5,
    b_in: float = 0.042,
    b_out: float = 0.042,
    seeds: List[int] = tuple(range(30)),
    depth: int = 5,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    source_leaf: int = 0,
    gossip_protocol: str = "async_poisson",
    hybrid_K_list: Tuple[float, ...] = (),
    epsilon_n_rounds: int = 5,
) -> List[NaturalCascadeConfig]:
    """DAG nesting: overlapping module hierarchies with tunable boundary
    in-degree Delta_in (delta_in) at `overlap_level`; a favourable ideal
    I(b) set by generality_level = overlap_level + 1 (so the boundary being
    tested is exactly the overlap boundary), swept against module size m
    (leaf_size) to locate the containment boundary m*(Delta_in) predicted
    by Theorem 11.3 / Proposition 11.2. delta_in=1 is the tree (n_overlap
    irrelevant -- OverlappingModularTopology's OverlapInfo is empty).

    force_flip_source=True/source_leaf pinned to `source_leaf` (same fix as
    experiment_E6): this uses the same per_leaf_loss_params_for_generality
    origin-keyed heterogeneity mechanism E6/E12a use, and without a forced
    initial condition at the configured source leaf, the source has no
    mechanism to spontaneously nucleate under deterministic gradient descent
    (lambda<1) -- see experiment_E6's docstring for the confirmed failure
    mode this avoids.

    paper1_computing_hybrid_gossip.md Annex B.3's E13 ("... at two K...
    locate m*(Delta_in)"), phase H4: hybrid_K_list=() (default) reproduces
    the ORIGINAL single-protocol E13 exactly, so phase 5a's existing call
    site is unaffected. Each K in hybrid_K_list adds a matched "E13H" arm
    at that round ratio over the SAME (delta_in, m, seed) grid,
    gossip_protocol="hybrid" -- testing Remark 11.2''s clause switch: does
    overlap enter logarithmically (m > 1 + log(...)) under the hybrid's
    renewal-enforcing Type-N clock, rather than linearly (m > Delta_in) as
    under free asynchrony -- against this K -> infinity
    (gossip_protocol="async_poisson") arm as the reference point.
    """
    branching = 2
    n_leaf_types = branching ** depth
    prefix = f"E13{protocol_suffix(gossip_protocol)}"
    configs = []
    for delta_in in delta_in_list:
        for m in m_list:
            from .generality import OverlapInfo, per_leaf_loss_params_for_generality as _plp

            # Build the SAME overlap assignment the topology will draw (same
            # seed formula as OverlappingModularTopology's internal rng),
            # so per_leaf_loss_params matches the actual extra-parent map.
            for seed in seeds:
                from dssgd.topology.static import OverlappingModularTopology
                topo = OverlappingModularTopology(
                    branching=branching, depth=depth, leaf_size=m, p=p,
                    overlap_level=overlap_level, delta_in=delta_in, n_overlap=n_overlap,
                    seed=seed,
                )
                overlap = OverlapInfo(overlap_level=overlap_level, extra_parents=topo.extra_parents)
                plp = _plp(
                    n_leaf_types=n_leaf_types, source_leaf=source_leaf,
                    generality_level=overlap_level + 1, branching=branching,
                    a=a, b_in=b_in, b_out=b_out, overlap=overlap,
                )
                configs.append(NaturalCascadeConfig(
                    name=f"{prefix}/delta_in={delta_in}/m={m}/seed={seed}",
                    branching=branching, depth=depth, leaf_size=m, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b_in, per_leaf_loss_params=plp, force_flip_source=True,
                    source_leaf=source_leaf,
                    overlap_level=overlap_level, delta_in=delta_in, n_overlap=n_overlap,
                    gossip_protocol=gossip_protocol,
                ))
    for K in hybrid_K_list:
        for delta_in in delta_in_list:
            for m in m_list:
                from .generality import OverlapInfo, per_leaf_loss_params_for_generality as _plp

                for seed in seeds:
                    from dssgd.topology.static import OverlappingModularTopology
                    topo = OverlappingModularTopology(
                        branching=branching, depth=depth, leaf_size=m, p=p,
                        overlap_level=overlap_level, delta_in=delta_in, n_overlap=n_overlap,
                        seed=seed,
                    )
                    overlap = OverlapInfo(overlap_level=overlap_level, extra_parents=topo.extra_parents)
                    plp = _plp(
                        n_leaf_types=n_leaf_types, source_leaf=source_leaf,
                        generality_level=overlap_level + 1, branching=branching,
                        a=a, b_in=b_in, b_out=b_out, overlap=overlap,
                    )
                    configs.append(NaturalCascadeConfig(
                        name=f"E13H/K={K}/delta_in={delta_in}/m={m}/seed={seed}",
                        branching=branching, depth=depth, leaf_size=m, p=p, seed=seed,
                        n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                        a=a, b=b_in, per_leaf_loss_params=plp, force_flip_source=True,
                        source_leaf=source_leaf,
                        overlap_level=overlap_level, delta_in=delta_in, n_overlap=n_overlap,
                        gossip_protocol="hybrid", round_ratio=K, epsilon_n_rounds=epsilon_n_rounds,
                    ))
    return configs


# ---------------------------------------------------------------------------
# E14: local_steps sensitivity of the meritocratic filter (Theorem 4.5)
# ---------------------------------------------------------------------------


def experiment_E14_meritocratic_filter_local_steps(
    generality_levels: List[int] = (1, 3),
    local_steps_list: List[int] = (50, 100, 200, 400),
    protocols: List[str] = ("async_poisson", "sync_pairwise", "sync_neighbourhood"),
    a: float = 0.5,
    b_in: float = 0.042,
    b_out: float = 0.042,
    seeds: List[int] = tuple(range(20)),
    depth: int = 3,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    n_warmup: int = 400,
    n_meas: int = 1000,
    source_leaf: int = 0,
) -> List[NaturalCascadeConfig]:
    """E14: is Theorem 4.5's level-matching filter's fidelity a function of
    local_steps (Regime-C relaxation time), and does that dependence differ
    by scheduling protocol?

    A DEDICATED, SEPARATE experiment from E6/E12a -- same containment/
    attainment mechanism (per_leaf_loss_params_for_generality, force-flipped
    source at `source_leaf`), but sweeping local_steps x protocol instead of
    holding both fixed, so its outputs (prefix "E14", never "E12a"/"E6") can
    never be confused with or overwrite the production E6/E12a results this
    was designed to investigate.

    Motivation: a Phase 6 gate6 review found sync_pairwise's E12a containment
    completely G-independent (identical d_max distributions for G=1,2,3 at
    local_steps=50), unlike async_poisson's partial, G-trending containment
    -- the opposite of Remark 4.3's prediction that round-synchronous H-sched
    scheduling should PRESERVE the filter, not defeat it entirely. NMH-1b's
    clean local_steps=50 threshold (Regime C achieved for a single forced
    leaf's flip persisting against a UNIFORM neighbourhood) does not by
    itself rule out a *different* failure mode specific to this test: an
    out-of-scope leaf accumulating pressure from MULTIPLE, differently-biased
    neighbours across levels, rather than resisting one homogeneous pull.
    This experiment tests that directly by varying local_steps on the actual
    containment observable, across all three gossip mechanisms (see
    gossip_mechanisms.md) rather than inferring from a simpler proxy.

    Protocol is embedded in the config name (name=f"E14/G={G}/ls={ls}/proto=
    {proto}/seed={seed}"), following E11's convention for experiments that
    sweep protocol as one of their OWN dimensions, rather than using
    protocol_suffix's one-call-per-protocol directory-suffix convention --
    this keeps all three mechanisms' results in one place for direct,
    matched-seed comparison (see check_phase6.compute_e14_local_steps_table).

    depth=3 (not E6/E12a's production depth=5): this is a diagnostic
    sensitivity sweep across local_steps x G x protocol (already a large
    grid at low seed count), not a production-scale confirmatory run -- same
    scale-reduction rationale as experiment_E6_positive_control.
    """
    branching = 2
    n_leaf_types = branching ** depth
    configs = []
    for G in generality_levels:
        plp = per_leaf_loss_params_for_generality(
            n_leaf_types=n_leaf_types, source_leaf=source_leaf, generality_level=G,
            branching=branching, a=a, b_in=b_in, b_out=b_out,
        )
        for ls in local_steps_list:
            for proto in protocols:
                for seed in seeds:
                    configs.append(NaturalCascadeConfig(
                        name=f"E14/G={G}/ls={ls}/proto={proto}/seed={seed}",
                        branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                        n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=ls,
                        a=a, b=b_in, per_leaf_loss_params=plp, force_flip_source=True,
                        source_leaf=source_leaf, gossip_protocol=proto,
                    ))
    return configs


def experiment_NMH7(
    seeds: List[int] = tuple(range(20)),
    depth: int = 4,
    leaf_size: int = 4,
    p: float = 2.0,
    a: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 4000,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """Detailed balance and Gibbs measure test.  Requires b=0 (symmetric).

    Tests Proposition 4.6 (eq. 4.7), the symmetric reference case of Sec. 4.7's
    stationary structure.  With b=0 both A and B are equally stable; the
    system explores both basins ergodically over the long measurement window.
    Rate ratio R_l = rate(A→B) / rate(B→A) should equal e^{2·β_eff·J_l} under
    reversibility at this symmetric (lambda=0) point.

    n_meas=4000 (longer run) for accurate rate estimates.
    Smaller depth=4 to keep per-run cost manageable.
    """
    prefix = f"NMH7{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        configs.append(NaturalCascadeConfig(
            name=f"{prefix}/seed={seed}",
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a,
            b=0.0,           # symmetric basins: required by Theorem 2
            flip_noise_scale=0.10,
            gossip_protocol=gossip_protocol,
        ))
    return configs
