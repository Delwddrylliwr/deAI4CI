"""Experiment configuration suites for NMH simulation vs SDE theory comparison.

Each function returns a list of NMHSimConfig objects that explore one
dimension of the theory (depth, p, branching, temperature).

Run a single suite:
    from analysis.experiments import depth_sweep
    from analysis.simulation import run_nmh_simulation
    runs = [run_nmh_simulation(c) for c in depth_sweep()]
"""
from __future__ import annotations

import functools
from typing import List

import torch.nn as nn

from .simulation import NMHSimConfig, make_homogeneous_loaders, make_regression_loaders, regression_loss


# Default model factory for n_dim=8 regression experiments
def _linear_model() -> nn.Module:
    return nn.Linear(8, 1, bias=False)


def depth_sweep(
    branching: int = 2,
    leaf_size: int = 4,
    p: float = 4.0,
    n_rounds: int = 800,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Vary depth 1-5 with fixed branching, leaf_size, p (pure gossip, no training).

    Tests degree limit convergence and per-level consensus ordering.
    """
    return [
        NMHSimConfig(
            name=f"depth_sweep/depth={d}",
            branching=branching,
            depth=d,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
        )
        for d in range(1, 6)
    ]


def depth_sweep_trained(
    branching: int = 2,
    leaf_size: int = 4,
    p: float = 4.0,
    lr: float = 0.01,
    n_rounds: int = 800,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Depth sweep with heterogeneous regression training for coupling-ratio analysis.

    Each leaf module has a distinct true weight vector; within-module agents share
    the same target, creating the type-heterogeneous loss landscape the SDE theory
    analyses.  Depth 1 is omitted (no hierarchical levels to compare).
    """
    return [
        NMHSimConfig(
            name=f"depth_sweep_trained/depth={d}",
            branching=branching,
            depth=d,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
            lr=lr,
            model_factory=_linear_model,
            loss_fn=regression_loss,
            data_loaders_factory=functools.partial(
                make_regression_loaders,
                branching=branching,
                depth=d,
                leaf_size=leaf_size,
            ),
        )
        for d in range(2, 6)
    ]


def p_sweep(
    branching: int = 2,
    depth: int = 3,
    leaf_size: int = 4,
    n_rounds: int = 500,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Vary p across {1, 2, 4, 8, 16} with fixed topology shape.

    Tests whether topological dimension D scales linearly with p and
    whether the spectral gap grows with p (better mixing for larger p).
    """
    return [
        NMHSimConfig(
            name=f"p_sweep/p={p}",
            branching=branching,
            depth=depth,
            leaf_size=leaf_size,
            p=float(p),
            seed=seed,
            n_rounds=n_rounds,
        )
        for p in [1, 2, 4, 8, 16]
    ]


def branching_sweep(
    depth: int = 3,
    leaf_size: int = 4,
    p: float = 4.0,
    n_rounds: int = 500,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Vary branching across {2, 3} (must be < 4 for degree limit to exist).

    Tests degree-limit formula and within vs cross-module convergence structure.
    """
    return [
        NMHSimConfig(
            name=f"branching_sweep/b={b}",
            branching=b,
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
        )
        for b in [2, 3]
    ]


def temperature_sweep(
    branching: int = 2,
    depth: int = 3,
    leaf_size: int = 4,
    p: float = 4.0,
    n_rounds: int = 600,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Vary learning rate to probe T_eff = eta*sigma^2/2 at stationarity.

    Uses heterogeneous regression data so gradient noise is non-trivial.
    """
    return [
        NMHSimConfig(
            name=f"temperature_sweep/lr={lr}",
            branching=branching,
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
            lr=lr,
            model_factory=_linear_model,
            loss_fn=regression_loss,
            data_loaders_factory=functools.partial(
                make_regression_loaders,
                branching=branching,
                depth=depth,
                leaf_size=leaf_size,
            ),
        )
        for lr in [0.001, 0.005, 0.01, 0.05, 0.1]
    ]


def kramers_sweep(
    branching: int = 2,
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    n_rounds: int = 400,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Consensus-initialised training runs for testing Kramers escape-time predictions.

    All agents start from identical parameters (start_from_consensus=True), then
    training pulls modules toward their heterogeneous optima.  The time for
    cross_distances[level] to rise to 50% of its stationary value is the
    observable analogue of the Kramers escape time.

    Uses depth=5 so that levels 3-5 have well depths small enough to produce
    finite Kramers predictions at the chosen learning rates.  High lr (0.05, 0.1)
    gives T_eff ~ 0.05-0.09, making tau^(3) ~ 10-100 rounds and observable.
    """
    return [
        NMHSimConfig(
            name=f"kramers_sweep/lr={lr}",
            branching=branching,
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
            lr=lr,
            model_factory=_linear_model,
            loss_fn=regression_loss,
            data_loaders_factory=functools.partial(
                make_regression_loaders,
                branching=branching,
                depth=depth,
                leaf_size=leaf_size,
            ),
            start_from_consensus=True,
        )
        for lr in [0.01, 0.05, 0.1]
    ]


def escape_kramers_sweep(
    branching: int = 2,
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    n_warmup: int = 400,
    n_rounds: int = 800,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Two-phase Kramers test: heterogeneous warmup -> homogeneous escape phase.

    Phase 1 (rounds 0..n_warmup-1): heterogeneous regression training drives
    modules toward their distinct optima; run long enough to reach stationarity.
    Phase 2 (rounds n_warmup..n_rounds-1): switch to homogeneous data (shared
    zero true-weight), removing the module-specific gradient pinning.  The gossip
    coupling then tries to re-mix the established cluster structure.

    Metric (stats.time_to_half_escape_cross): rounds after phase-2 onset until
    cross_distances[level] drops to 50% of its warmup-stationary value.  This
    dissolution time tau_D should equal ~1/gamma_ell (spectral relaxation) if
    gossip shortcuts bypass the Kramers barrier, or ~exp(DV/T_eff) if the
    Kramers picture holds.
    """
    return [
        NMHSimConfig(
            name=f"escape_kramers/lr={lr}",
            branching=branching,
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
            lr=lr,
            model_factory=_linear_model,
            loss_fn=regression_loss,
            data_loaders_factory=functools.partial(
                make_regression_loaders,
                branching=branching,
                depth=depth,
                leaf_size=leaf_size,
            ),
            n_warmup=n_warmup,
            escape_data_loaders_factory=make_homogeneous_loaders,
            skip_spectral_gaps=True,
        )
        for lr in [0.01, 0.05, 0.1]
    ]


def leaf_size_kramers_sweep(
    branching: int = 2,
    depth: int = 5,
    p: float = 4.0,
    lr: float = 0.1,
    n_rounds: int = 400,
    seed: int = 0,
) -> List[NMHSimConfig]:
    """Consensus-initialised Kramers sweep varying leaf_size to test peer-pressure hypothesis.

    Each doubling of leaf_size shifts ΔV = gamma*m*p/4^ell up by 2×, moving the
    observable Kramers window (ΔV/T_eff ~ 1-5) to higher hierarchy levels.
    Equivalent comparison points across m values:
        m=4  level 3  <->  m=16 level 4  (both ΔV/T_eff ~ 2.9 at lr=0.1)
        m=8  level 4  <->  m=32 level 5  (both ΔV/T_eff ~ 1.5 at lr=0.1)
    If peer pressure within larger modules moderates the gossip shortcut, points
    for larger m should sit closer to the Kramers diagonal at matched ΔV/T_eff.
    skip_spectral_gaps=True avoids O(n^2) eigvalsh calls every round for large m.
    """
    return [
        NMHSimConfig(
            name=f"leaf_size_kramers/m={m}",
            branching=branching,
            depth=depth,
            leaf_size=m,
            p=p,
            seed=seed,
            n_rounds=n_rounds,
            lr=lr,
            model_factory=_linear_model,
            loss_fn=regression_loss,
            data_loaders_factory=functools.partial(
                make_regression_loaders,
                branching=branching,
                depth=depth,
                leaf_size=m,
            ),
            start_from_consensus=True,
            skip_spectral_gaps=True,
        )
        for m in [4, 8, 16, 32]
    ]
