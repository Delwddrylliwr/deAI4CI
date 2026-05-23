"""Experiment factory functions for NMH active-escape dynamics experiments (v2).

Each function returns a list of ActiveEscapeSimConfig objects corresponding
to one of the four primary experiments in the v2 dynamics specification:

  A1 — primary slope-1 test (log₂ t_flip vs d, Regime I)
  A2 — regime sweep (locate ell_c by varying barrier curvature a)
  A3 — containment phase transition (fine a sweep, many seeds)
  B  — iso-γmp collapse check (fundamental vs finite-size)

Default parameters: γ=1, m=4, p=2, L=5, n=128, d_param=1,
  θ_A=0, θ_B=1, b=0.01 (small B-preference bias).

With p=2 (sparse Safari regime): γmp=8, γ_1=2, γ_5=0.125.
  Regime I:   a < 0.5  (κ_loc/2 < γ_5=0.125)
  Regime III: a ≥ 8    (κ_loc/2 ≥ γ_1=2)
  Regime II:  0.5 ≤ a < 8
  Critical curvature a_c = 8 (II→III transition).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from . import theory
from .active_escape import ActiveEscapeSimConfig


# ---------------------------------------------------------------------------
# Experiment A1 — primary slope-1 test
# ---------------------------------------------------------------------------


def experiment_A1(
    lr_list: List[float] = (0.005, 0.01, 0.05, 0.1),
    seeds: List[int] = tuple(range(10)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    a: float = 0.3,
    b: float = 0.01,
    n_warmup: int = 400,
    n_meas: int = 800,
    source_leaf: int = 0,
) -> List[ActiveEscapeSimConfig]:
    """Primary test: verify log₂(t_flip) vs d has slope ≈ 1 in Regime I.

    Uses p=2 (sparse Safari regime) with a=0.3 (κ_loc=0.15, safely in Regime I:
    κ_loc/2=0.075 < γ_5=0.125).  Sweeps lr to confirm slope is lr-independent
    (the doubling-per-level ratio is a topology property, not a temperature one).
    10 seeds for statistical power.
    """
    configs = []
    for lr in lr_list:
        for seed in seeds:
            configs.append(ActiveEscapeSimConfig(
                name=f"exp_A1/a={a}/lr={lr}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                a=a,
                b=b,
                source_leaf=source_leaf,
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment A1nat — natural cascade (no source clamping)
# ---------------------------------------------------------------------------


def experiment_A1_natural(
    lr_list: List[float] = (0.05, 0.1, 0.2),
    seeds: List[int] = tuple(range(5)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    a: float = 0.5,
    b: float = 0.042,
    n_warmup: int = 400,
    n_meas: int = 800,
    local_steps: int = 50,
    source_leaf: int = 0,
) -> List[ActiveEscapeSimConfig]:
    """Natural-cascade variant of A1: no source clamping.

    Uses b close to the bistability limit (b≈0.042, limit≈0.048) to shift the
    saddle from θ≈0.46 (standard) to θ≈0.30.  After one gossip round, the
    level-1 cousin lands at θ≈0.33 > 0.30 (above saddle), so its local gradient
    points toward B.  With local_steps=50, local SGD has enough time to drive
    agents from above-saddle to B before the next gossip exchange (~50×0.1×0.04=0.20
    per round, comparable to gossip decay ~0.20 per round).

    NOTE: local_steps>>1 is a two-timescale (FedAvg-style) regime, distinct from
    the single-timescale SDE model behind the T_flip(d)∝2^d theory.  This experiment
    tests whether B's lower energy is sufficient to sustain a cascade once local
    equilibration is fast relative to gossip.
    """
    configs = []
    for lr in lr_list:
        for seed in seeds:
            configs.append(ActiveEscapeSimConfig(
                name=f"exp_A1nat/a={a}/b={b}/lr={lr}/ls={local_steps}/seed={seed}",
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
                source_leaf=source_leaf,
                clamp_source=False,
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment A2 — regime sweep (locate ell_c)
# ---------------------------------------------------------------------------


def experiment_A2(
    a_list: List[float] = (0.2, 0.5, 2.0, 6.0, 9.0),
    seeds: List[int] = tuple(range(10)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.01,
    b: float = 0.01,
    n_warmup: int = 400,
    n_meas: int = 800,
    source_leaf: int = 0,
) -> List[ActiveEscapeSimConfig]:
    """Regime sweep: locate ell_c by varying barrier curvature a.

    For p=2, m=4, L=5 (γmp=8, γ_1=2, γ_5=0.125):
      a=0.2  → κ_loc=0.10, κ_loc/2=0.05  < γ_5=0.125 → Regime I  (ell_c=5)
      a=0.5  → κ_loc=0.25, κ_loc/2=0.125 = γ_5        → Regime II (ell_c=4)
      a=2.0  → κ_loc=1.00, κ_loc/2=0.5   < γ_1=2      → Regime II (ell_c=2)
      a=6.0  → κ_loc=3.00, κ_loc/2=1.5   < γ_1=2      → Regime II (ell_c=1)
      a=9.0  → κ_loc=4.50, κ_loc/2=2.25  ≥ γ_1=2      → Regime III

    Observable: flip_fraction(d) = fraction of target leaves at distance d
    that ever enter B.  Should be 1.0 for d ≤ ell_c and drop sharply above.
    """
    configs = []
    for a in a_list:
        for seed in seeds:
            configs.append(ActiveEscapeSimConfig(
                name=f"exp_A2/a={a}/lr={lr}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                a=a,
                b=b,
                source_leaf=source_leaf,
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment A3 — containment phase transition
# ---------------------------------------------------------------------------


def experiment_A3(
    a_list: Optional[List[float]] = None,
    seeds: List[int] = tuple(range(20)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.01,
    b: float = 0.01,
    n_warmup: int = 600,
    n_meas: int = 2000,
    source_leaf: int = 0,
    n_a_values: int = 20,
) -> List[ActiveEscapeSimConfig]:
    """Containment phase transition: fine a sweep, many seeds.

    Observable: final_basin_fraction = fraction of target leaves in basin B
    at the final measurement round.  Expects a sharp sigmoidal transition
    near the critical curvature a_c where γ_1 = κ_loc/2.

    With p=2: γ_1=2, so a_c = 4·γ_1 = 8.  The sweep covers 0.2 to 16
    (approximately 0.5·a_c to 2·a_c) to capture the full transition.

    Uses float formatting :.4f in names to avoid NTFS path-length issues
    with linspace-generated decimals.
    """
    if a_list is None:
        a_list = list(np.linspace(0.2, 16.0, n_a_values))

    configs = []
    for a in a_list:
        for seed in seeds:
            configs.append(ActiveEscapeSimConfig(
                name=f"exp_A3/a={a:.4f}/lr={lr}/seed={seed}",
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                a=a,
                b=b,
                source_leaf=source_leaf,
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment B — iso-γmp collapse check
# ---------------------------------------------------------------------------


def experiment_B_active(
    seeds: List[int] = (0, 1, 2),
    n_warmup: int = 400,
    n_meas: int = 800,
    lr: float = 0.01,
    a: float = 0.3,
    b: float = 0.01,
    source_leaf: int = 0,
) -> List[ActiveEscapeSimConfig]:
    """Iso-γmp collapse test with bistable loss.

    Same (m, p, L) variants — all give γmp = 16 and n = 128:
      (m=2,  p=8,  L=6)  — note: p=8 gives level-1 prob>1 (complete bipartite)
      (m=4,  p=4,  L=5)  — level-1 complete bipartite
      (m=8,  p=2,  L=4)  — level-1 prob=0.5 (sparse)
      (m=16, p=1,  L=3)  — level-1 prob=0.25 (sparse)

    Under the fundamental reading of T_flip(d) = 2^(d+1)/(γmp), all four
    configurations should produce identical log₂(t_flip) vs d curves.
    Finite-size effects (centroid fluctuation ~ σ²/m) would show systematic
    variation with m.

    Uses a=0.3 to stay in Regime I for the (m=8,p=2) and (m=16,p=1) variants
    where γ_1 is smaller.

    Note: source_leaf=0 is the leaf with the lowest index, valid for all depths.
    """
    iso_configs = [
        (2,  8.0, 6),
        (4,  4.0, 5),
        (8,  2.0, 4),
        (16, 1.0, 3),
    ]
    configs = []
    for leaf_size, p, depth in iso_configs:
        for seed in seeds:
            configs.append(ActiveEscapeSimConfig(
                name=f"exp_B_active/m={leaf_size}/p={p}/seed={seed}",
                branching=2,
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                a=a,
                b=b,
                source_leaf=source_leaf,
            ))
    return configs
