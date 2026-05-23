"""Experiment factory functions for NMH catch-up dynamics experiments.

Each function returns a list of CatchupSimConfig objects corresponding to
one of the five experiments in the dynamics specification:

  A — catch-up curve (primary test, log₂ t_50 vs distance d)
  B — iso-γmp sweep  (fundamental vs finite-size discrimination)
  C — temperature crossover (Kramers branch)
  D — decomposition consistency (centroid/fluctuation autocorrelations)
  E — heterogeneity sensitivity (hierarchical / homogeneous / i.i.d. data)
"""
from __future__ import annotations

import functools
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import theory
from .catchup import CatchupSimConfig
from .simulation import make_homogeneous_loaders, make_regression_loaders, regression_loss


# Default model for 8-dim regression
def _linear_model() -> nn.Module:
    return nn.Linear(8, 1, bias=False)


# ---------------------------------------------------------------------------
# Hierarchically-structured regression loaders (Experiment E, mode 1)
# ---------------------------------------------------------------------------


def make_hierarchical_regression_loaders(
    n_agents: int,
    branching: int,
    depth: int,
    leaf_size: int,
    n_dim: int = 8,
    noise_std: float = 0.5,
    n_samples: int = 64,
    batch_size: int = 16,
    rng_seed: int = 42,
) -> List[DataLoader]:
    """Regression data with hierarchically-correlated module optima.

    θ*_leaf = sum over ancestors of level-ℓ random vector with std ∝ 2^{-ℓ/2}.
    This gives ||θ*_i - θ*_j|| ~ 2^{-d(i,j)/2}: leaves sharing more ancestors
    have more similar optima.  The catch-up curve should be identical to the
    i.i.d. regime (the topology, not the data, drives propagation speed).
    """
    n_leaf_types = branching ** depth
    rng = np.random.default_rng(rng_seed)

    # Assign a random vector to each internal node, scaled by depth level
    # Level 0 = leaves (no contribution), level ℓ ancestor contributes std 2^{-ℓ/2}
    # Represent the binary tree: leaf i has ancestors at positions i>>1, i>>2, ... (0-indexed)
    node_vecs: dict[int, np.ndarray] = {}
    for ell in range(1, depth + 1):
        n_nodes_at_level = n_leaf_types >> ell  # 2^(depth-ℓ) internal nodes at this level
        std = 2.0 ** (-ell / 2.0)
        for node_id in range(n_nodes_at_level):
            node_vecs[(ell, node_id)] = rng.standard_normal(n_dim) * std

    leaf_weights = np.zeros((n_leaf_types, n_dim))
    for tau in range(n_leaf_types):
        w = np.zeros(n_dim)
        for ell in range(1, depth + 1):
            ancestor_id = tau >> ell
            w += node_vecs[(ell, ancestor_id)]
        leaf_weights[tau] = w

    loaders: List[DataLoader] = []
    for agent_id in range(n_agents):
        tau = agent_id // leaf_size
        true_w = torch.from_numpy(leaf_weights[tau]).float()
        torch.manual_seed(rng_seed * 1000 + agent_id)
        x = torch.randn(n_samples, n_dim)
        y = x @ true_w + noise_std * torch.randn(n_samples)
        loaders.append(
            DataLoader(TensorDataset(x, y.unsqueeze(1)), batch_size=batch_size, shuffle=True)
        )
    return loaders


def make_shared_nonzero_loaders(
    n_agents: int,
    n_dim: int = 8,
    noise_std: float = 0.5,
    n_samples: int = 64,
    batch_size: int = 16,
    rng_seed: int = 42,
) -> List[DataLoader]:
    """Homogeneous data with a shared non-zero true weight (Experiment E mode 2).

    All agents share the same loss L_i ≡ L_0.  The gradient pinning is uniform,
    so there are no type-specific wells; only gradient noise drives fluctuations.
    Using a non-zero target weight (rather than zero) avoids degenerate loss landscapes.
    """
    rng = np.random.default_rng(rng_seed)
    true_w = torch.from_numpy(rng.standard_normal(n_dim)).float()
    loaders: List[DataLoader] = []
    for agent_id in range(n_agents):
        torch.manual_seed(rng_seed * 1000 + agent_id)
        x = torch.randn(n_samples, n_dim)
        y = x @ true_w + noise_std * torch.randn(n_samples)
        loaders.append(
            DataLoader(TensorDataset(x, y.unsqueeze(1)), batch_size=batch_size, shuffle=True)
        )
    return loaders


# ---------------------------------------------------------------------------
# Experiment A — catch-up curve (primary test)
# ---------------------------------------------------------------------------


def experiment_A(
    lr_list: List[float] = (0.01, 0.05, 0.1),
    seeds: List[int] = (0, 1, 2),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    n_warmup: int = 800,
    n_meas: int = 400,
    source_leaf: int = 0,
) -> List[CatchupSimConfig]:
    """Standard catch-up curve at multiple learning rates and seeds.

    Uses make_regression_loaders (i.i.d. heterogeneous per module, existing).
    """
    configs = []
    for lr in lr_list:
        for seed in seeds:
            n_agents = (2 ** depth) * leaf_size
            configs.append(CatchupSimConfig(
                name=f"exp_A/lr={lr}/seed={seed}",
                branching=2,
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                source_leaf=source_leaf,
                model_factory=_linear_model,
                loss_fn=regression_loss,
                data_loaders_factory=functools.partial(
                    make_regression_loaders,
                    branching=2,
                    depth=depth,
                    leaf_size=leaf_size,
                ),
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment B — iso-γmp sweep
# ---------------------------------------------------------------------------


def experiment_B(
    seeds: List[int] = (0, 1, 2),
    n_warmup: int = 800,
    n_meas: int = 400,
    lr: float = 0.05,
    source_leaf: int = 0,
) -> List[CatchupSimConfig]:
    """Iso-γmp sweep: (m, p, L) variants all giving γmp = 16 and n = 128.

    (m, p, L) ∈ {(2, 8, 6), (4, 4, 5), (8, 2, 4), (16, 1, 3)}
    Under the fundamental reading all catch-up curves should collapse;
    finite-size corrections would show systematic variation with m.
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
            configs.append(CatchupSimConfig(
                name=f"exp_B/m={leaf_size}/p={p}/seed={seed}",
                branching=2,
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                source_leaf=source_leaf,
                model_factory=_linear_model,
                loss_fn=regression_loss,
                data_loaders_factory=functools.partial(
                    make_regression_loaders,
                    branching=2,
                    depth=depth,
                    leaf_size=leaf_size,
                ),
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment C — temperature crossover (Kramers branch)
# ---------------------------------------------------------------------------


def experiment_C(
    lr_list: List[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    seeds: List[int] = (0, 1, 2),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    n_warmup: int = 800,
    n_meas: int = 10000,
    source_leaf: int = 0,
) -> List[CatchupSimConfig]:
    """Temperature crossover: sweep lr across 3 decades.

    Long measurement runs (10⁴ rounds) needed to observe Kramers escape at low lr.
    Focus observable: t_50(d=1) vs lr.
    """
    configs = []
    for lr in lr_list:
        for seed in seeds:
            configs.append(CatchupSimConfig(
                name=f"exp_C/lr={lr}/seed={seed}",
                branching=2,
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                source_leaf=source_leaf,
                model_factory=_linear_model,
                loss_fn=regression_loss,
                data_loaders_factory=functools.partial(
                    make_regression_loaders,
                    branching=2,
                    depth=depth,
                    leaf_size=leaf_size,
                ),
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment D — decomposition consistency (autocorrelations)
# ---------------------------------------------------------------------------


def experiment_D(
    seeds: List[int] = (0, 1, 2),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    n_warmup: int = 800,
    n_meas: int = 2000,
    lr: float = 0.01,
    source_leaf: int = 0,
) -> List[CatchupSimConfig]:
    """Decomposition autocorrelation check.

    Saves full per-agent trajectories (record_agent_traj=True) so that both
    C_δ(t) and C_mτ(t) can be computed post-hoc from the same run data.
    No perturbation analysis is done here; t_50_table is still computed but
    the primary output is centroid_traj and agent_traj.
    """
    configs = []
    for seed in seeds:
        configs.append(CatchupSimConfig(
            name=f"exp_D/seed={seed}",
            branching=2,
            depth=depth,
            leaf_size=leaf_size,
            p=p,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            source_leaf=source_leaf,
            model_factory=_linear_model,
            loss_fn=regression_loss,
            data_loaders_factory=functools.partial(
                make_regression_loaders,
                branching=2,
                depth=depth,
                leaf_size=leaf_size,
            ),
            record_agent_traj=True,
        ))
    return configs


# ---------------------------------------------------------------------------
# Experiment E — heterogeneity sensitivity
# ---------------------------------------------------------------------------


def experiment_E(
    seeds: List[int] = (0, 1, 2),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 4.0,
    lr: float = 0.05,
    n_warmup: int = 800,
    n_meas: int = 400,
    source_leaf: int = 0,
) -> List[CatchupSimConfig]:
    """Heterogeneity sensitivity: three matched data-generating regimes.

    mode='hierarchical' — module optima correlated by tree distance
    mode='homogeneous'  — shared non-zero true weight, no module structure
    mode='iid'          — i.i.d. random optima per module (existing default)

    Theory predicts identical catch-up curves across all three modes.
    """
    configs = []
    loaders_by_mode = {
        "hierarchical": functools.partial(
            make_hierarchical_regression_loaders,
            branching=2,
            depth=depth,
            leaf_size=leaf_size,
        ),
        "homogeneous": make_shared_nonzero_loaders,
        "iid": functools.partial(
            make_regression_loaders,
            branching=2,
            depth=depth,
            leaf_size=leaf_size,
        ),
    }
    for mode, loader_factory in loaders_by_mode.items():
        for seed in seeds:
            configs.append(CatchupSimConfig(
                name=f"exp_E/mode={mode}/seed={seed}",
                branching=2,
                depth=depth,
                leaf_size=leaf_size,
                p=p,
                seed=seed,
                n_warmup=n_warmup,
                n_meas_rounds=n_meas,
                lr=lr,
                source_leaf=source_leaf,
                model_factory=_linear_model,
                loss_fn=regression_loss,
                data_loaders_factory=loader_factory,
            ))
    return configs
