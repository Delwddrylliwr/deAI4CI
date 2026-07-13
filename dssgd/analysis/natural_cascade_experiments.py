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
  "async_poisson"  — AsynchronousGossip (default; Phase xA)
  "synchronous"    — GossipAveraging (Phase xS)
  When "synchronous", the experiment name prefix gains an "S" suffix
  (e.g. "NMH1S/", "NMH1bS/") so async and sync results live in separate dirs.
"""

from typing import List, Optional, Tuple

import numpy as np

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

    Tests Theorem 1: τ_mix^(l) = Θ(2^{l+1} / (ε·p·M_0·q_l)).
    Sweeps a across Regime I (all levels flip); 50 seeds for CI estimation.

    Uses force_flip_source=True (one random leaf set to B post-warmup) so
    that cascade propagation is driven purely by gossip without thermal noise.
    """
    prefix = "NMH1S" if gossip_protocol != "async_poisson" else "NMH1"
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
    prefix = "NMH1bS" if gossip_protocol != "async_poisson" else "NMH1b"
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

    Tests Equation (14): q_l = (1 - e^{-2θ}) / (1 - e^{-2θ M_{l-1}}).
    Sweeps b to vary the effective saddle shift θ_s and empirical q_l.
    100 seeds per b value (q_l is a frequency estimate needing large N).
    Natural nucleation (no force-flip): small Langevin noise enables escape
    so that the rate varies measurably across b values.
    """
    prefix = "NMH2S" if gossip_protocol != "async_poisson" else "NMH2"
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

    Tests §2.7 phase structure under force-flip cascade dynamics.
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
    prefix = "NMH3S" if gossip_protocol != "async_poisson" else "NMH3"
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

    Tests Proposition 1, Equation (18).  Uses depth=7 (N=512) for range
    over at least 1.5 decades.  2000 seeds for power-law fitting (MLE).
    a=2 is in the middle of the Griffiths regime (0.5 ≤ a < 8 for p=2).
    Force-flip source so cascade size reflects gossip propagation, not nucleation.
    """
    prefix = "NMH4S" if gossip_protocol != "async_poisson" else "NMH4"
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

    Tests Theorem 3, Equations (19)-(20).
    Observable: d_prop(b/a) increases monotonically (flatter basins propagate
    further) with a sharp threshold at the deepest level geometric filter.
    Force-flip source so propagation depth reflects gossip filter attenuation.
    """
    prefix = "NMH5S" if gossip_protocol != "async_poisson" else "NMH5"
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

    Tests Proposition 2, Equations (22)-(23): V_L = Σ_l V_l^between.
    Per-leaf b_i ~ N(b_base, b_spread²), clipped to [0.005, 0.035] to keep
    all leaves in genuine bistable regime (bistability limit ≈ 0.074 for a=1)
    with non-negligible B→A back-transitions.

    b_base=0.01 (weak bias) ensures a quasi-stationary distribution exists
    before absorption.  Long n_meas=2000 for time-averaged variance.
    """
    branching = 2
    n_leaf_types = branching ** depth
    prefix = "NMH6S" if gossip_protocol != "async_poisson" else "NMH6"
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

    Tests Theorem 2, Equations (15)-(17).  With b=0 both A and B are
    equally stable; the system explores both basins ergodically over the
    long measurement window.  Rate ratio R_l = rate(A→B) / rate(B→A)
    should equal e^{2·β_eff·J_l} under Theorem 2.

    n_meas=4000 (longer run) for accurate rate estimates.
    Smaller depth=4 to keep per-run cost manageable.
    """
    prefix = "NMH7S" if gossip_protocol != "async_poisson" else "NMH7"
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
