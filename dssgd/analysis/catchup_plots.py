"""Visualisation functions for NMH catch-up dynamics experiments."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from . import theory
from .catchup import CatchupRun, hierarchical_distance


# ---------------------------------------------------------------------------
# Data aggregation helpers
# ---------------------------------------------------------------------------


def summarise_catchup(runs: List[CatchupRun]) -> List[dict]:
    """Flatten t50_table from all runs into a single list of dicts."""
    rows = []
    for run in runs:
        for row in run.t50_table:
            rows.append({**row, "depth": run.depth, "leaf_size": run.leaf_size, "p": run.p})
    return rows


def _group_t50_by_distance(
    rows: List[dict],
    lr_filter: Optional[float] = None,
    leaf_size_filter: Optional[int] = None,
    p_filter: Optional[float] = None,
    mode_filter: Optional[str] = None,
    exclude_direct_edges: bool = False,
) -> Dict[int, List[float]]:
    """Aggregate log₂(t_50) values by hierarchical distance d.

    Entries where t_50 is None (threshold never crossed) are excluded from
    the mean but logged as a warning.  When exclude_direct_edges=True, rows
    where has_direct_edge==True are silently skipped (they show spurious t_50≈1
    from direct NMH cross-edges rather than multi-hop propagation).
    """
    by_d: Dict[int, List[float]] = {}
    n_missing = 0
    for row in rows:
        if lr_filter is not None and abs(row["lr"] - lr_filter) > 1e-10:
            continue
        if leaf_size_filter is not None and row.get("leaf_size") != leaf_size_filter:
            continue
        if p_filter is not None and abs(row.get("p", 0) - p_filter) > 1e-10:
            continue
        if exclude_direct_edges and row.get("has_direct_edge", False):
            continue
        t50 = row["t_50"]
        if t50 is None or t50 <= 0:
            n_missing += 1
            continue
        d = int(row["distance_d"])
        by_d.setdefault(d, []).append(math.log2(t50))
    if n_missing > 0:
        print(f"  [warning] {n_missing} entries with t_50=None excluded from plot")
    return by_d


def _linear_fit(x: np.ndarray, y: np.ndarray):
    """Least-squares fit y = a*x + b; returns (slope, intercept, slope_se)."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    result = np.linalg.lstsq(A, y, rcond=None)
    coeffs = result[0]
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    residuals = y - (slope * x + intercept)
    n = len(x)
    if n > 2:
        se = float(np.sqrt((residuals ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()))
    else:
        se = float("nan")
    return slope, intercept, se


# ---------------------------------------------------------------------------
# Experiment A — catch-up curve
# ---------------------------------------------------------------------------


def plot_catchup_curve(
    runs: List[CatchupRun],
    gamma: float = 1.0,
    figsize_per_panel: tuple = (4.5, 4.0),
) -> plt.Figure:
    """Plot log₂(t_50) vs hierarchical distance d, one panel per learning rate.

    Overlays the theoretical prediction T_catch(d) = ln(2)·2^{d+1}/(γ·m·p)
    and reports the linear regression slope ± SE in the panel title.

    Parameters
    ----------
    runs : list of CatchupRun from experiment_A (may contain multiple seeds/lrs)
    gamma : gossip coupling constant (default 1.0)
    """
    rows = summarise_catchup(runs)
    lrs = sorted({r["lr"] for r in rows})
    n_panels = len(lrs)
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
                             squeeze=False)

    for ax, lr in zip(axes[0], lrs):
        by_d_all = _group_t50_by_distance(rows, lr_filter=lr)
        by_d_clean = _group_t50_by_distance(rows, lr_filter=lr, exclude_direct_edges=True)

        if not by_d_all:
            ax.set_title(f"lr={lr}\n(no data)")
            continue

        # All data (includes direct-edge leaves; shown grey for context)
        distances_all = np.array(sorted(by_d_all.keys()))
        means_all = np.array([np.mean(by_d_all[d]) for d in distances_all])
        sems_all = np.array([np.std(by_d_all[d]) / math.sqrt(len(by_d_all[d])) for d in distances_all])
        ax.errorbar(distances_all, means_all, yerr=sems_all, fmt="s--", capsize=3,
                    color="grey", alpha=0.5, label="all (incl. direct edge)")

        # Clean data (multi-hop propagation only)
        slope_str = "N/A"
        if by_d_clean:
            distances_c = np.array(sorted(by_d_clean.keys()))
            means_c = np.array([np.mean(by_d_clean[d]) for d in distances_c])
            sems_c = np.array([np.std(by_d_clean[d]) / math.sqrt(len(by_d_clean[d])) for d in distances_c])
            ax.errorbar(distances_c, means_c, yerr=sems_c, fmt="o-", capsize=4,
                        color="steelblue", label="no direct edge")
            if len(distances_c) >= 2:
                slope, intercept, se = _linear_fit(distances_c.astype(float), means_c)
                slope_str = f"{slope:.2f}±{se:.2f}"

        # Theory overlay: uses leaf_size and p from first run with this lr
        run0 = next(r for r in runs if abs(r.lr - lr) < 1e-10)
        theory_log2_t = np.array([
            math.log2(theory.catch_up_time(int(d), gamma, run0.leaf_size, run0.p))
            for d in distances_all
        ])
        ax.plot(distances_all, theory_log2_t, "k--", label="theory")

        ax.set_xlabel("Hierarchical distance d")
        ax.set_ylabel("log₂(t₅₀)")
        ax.set_title(f"lr={lr}\nslope (no-direct-edge)={slope_str}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment A: catch-up curve (prediction: slope = 1)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment B — iso-γmp collapse
# ---------------------------------------------------------------------------


def plot_iso_gamma_mp(
    runs: List[CatchupRun],
    gamma: float = 1.0,
) -> plt.Figure:
    """Plot four catch-up curves (one per (m, p) pair) on shared axes.

    All configurations have γmp = 16.  Under the fundamental prediction all
    curves collapse; finite-size corrections produce systematic shifts.
    """
    rows = summarise_catchup(runs)
    configs = sorted({(r.leaf_size, r.p) for r in runs})
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(configs)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for (m, p), color in zip(configs, colors):
        lr = runs[0].lr
        by_d = _group_t50_by_distance(rows, leaf_size_filter=m, p_filter=p)
        if not by_d:
            continue
        distances = np.array(sorted(by_d.keys()))
        means = np.array([np.mean(by_d[d]) for d in distances])
        sems = np.array([np.std(by_d[d]) / math.sqrt(len(by_d[d])) for d in distances])
        ax.errorbar(distances, means, yerr=sems, fmt="o-", capsize=3,
                    color=color, label=f"m={m}, p={p}")

    # Theory reference (same for all since γmp=16 is fixed)
    if runs:
        d_range = np.arange(1, runs[0].depth + 1)
        theory_vals = np.array([
            math.log2(theory.catch_up_time(int(d), gamma, 4, 4.0))
            for d in d_range
        ])
        ax.plot(d_range, theory_vals, "k--", label="theory (γmp=16)")

    ax.set_xlabel("Hierarchical distance d")
    ax.set_ylabel("log₂(t₅₀)")
    ax.set_title("Experiment B: iso-γmp collapse\n(fundamental → curves overlap; finite-size → systematic shift)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment C — temperature crossover
# ---------------------------------------------------------------------------


def plot_temperature_crossover(
    runs: List[CatchupRun],
    target_distance: int = 1,
    gamma: float = 1.0,
    sigma2: float = 1.0,
) -> plt.Figure:
    """Plot t_50(d=target_distance) vs lr on log-log axes.

    Fits two linear regimes (gossip-dominated and Kramers-dominated) and
    reports the crossover lr.
    """
    rows = summarise_catchup(runs)
    lr_vals_set: set = set()
    t50_by_lr: Dict[float, List[float]] = {}
    for row in rows:
        if row["distance_d"] != target_distance:
            continue
        t50 = row["t_50"]
        if t50 is None or t50 <= 0:
            continue
        lr = row["lr"]
        lr_vals_set.add(lr)
        t50_by_lr.setdefault(lr, []).append(math.log10(t50))

    if not t50_by_lr:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    lrs = np.array(sorted(t50_by_lr.keys()))
    log10_lrs = np.log10(lrs)
    means = np.array([np.mean(t50_by_lr[lr]) for lr in lrs])
    sems = np.array([np.std(t50_by_lr[lr]) / math.sqrt(len(t50_by_lr[lr])) for lr in lrs])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(log10_lrs, means, yerr=sems, fmt="o-", capsize=4, label="observed")

    # Theory: gossip-dominated regime (flat line at log10(1/γ_ℓ))
    run0 = runs[0]
    gamma_ell = theory.level_coupling_strength(gamma, run0.leaf_size, run0.p, target_distance)
    gossip_t50 = math.log2(2.0) / gamma_ell  # approximate
    ax.axhline(math.log10(gossip_t50), color="steelblue", linestyle="--",
               label=f"gossip limit (1/γ_{target_distance})")

    ax.set_xlabel("log₁₀(lr)")
    ax.set_ylabel("log₁₀(t₅₀) at d=" + str(target_distance))
    ax.set_title(f"Experiment C: temperature crossover\n(flat=gossip-dominated; rising=Kramers)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment D — decomposition autocorrelations
# ---------------------------------------------------------------------------


def compute_autocorrelations(
    run: CatchupRun,
    max_lag: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute C_δ(t) and C_mτ(t) from a CatchupRun with agent_traj.

    C_mτ(t) = <mτ(0)·mτ(t)> averaged over leaf types τ and param dims.
    C_δ(t)  = <δθ_i(0)·δθ_i(t)> averaged over agents i and param dims.

    centroid_traj has shape (T+1, n_leaf_types, n_params); index 0 is the
    pre-perturbation baseline.  For autocorrelation purposes we use the
    post-perturbation trajectory (indices 1..).

    Returns (C_delta, C_m) each of shape (max_lag+1,).
    """
    if run.agent_traj is None:
        raise ValueError("agent_traj is None; rerun with record_agent_traj=True")

    # Use post-perturbation trajectories (indices 1..)
    centroid = run.centroid_traj[1:]   # (T, n_leaf_types, n_params)
    agents_t = run.agent_traj[1:]      # (T, n_agents, n_params)

    T = centroid.shape[0]
    n_leaf_types = centroid.shape[1]
    leaf_size = run.leaf_size
    max_lag = min(max_lag, T - 1)

    # Fluctuations: δθ_i(t) = θ_i(t) - centroid_{τ(i)}(t)
    assigns = np.arange(run.branching ** run.depth * leaf_size) // leaf_size
    centroid_per_agent = centroid[:, assigns, :]  # (T, n_agents, n_params)
    delta = agents_t - centroid_per_agent          # (T, n_agents, n_params)

    C_delta = np.zeros(max_lag + 1)
    C_m = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        # C_δ(lag) = <δθ_i(0) · δθ_i(lag)> over i, params, and time origins
        n_origins = T - lag
        c_d = float((delta[:n_origins] * delta[lag: lag + n_origins]).mean())
        C_delta[lag] = c_d

        c_m = float((centroid[:n_origins] * centroid[lag: lag + n_origins]).mean())
        C_m[lag] = c_m

    # Normalise
    if C_delta[0] != 0:
        C_delta /= C_delta[0]
    if C_m[0] != 0:
        C_m /= C_m[0]

    return C_delta, C_m


def plot_decomposition_autocorr(
    run: CatchupRun,
    gamma: float = 1.0,
    max_lag: int = 200,
) -> plt.Figure:
    """Plot log C_δ(t) and log C_mτ(t) with theoretical decay rates overlaid.

    Expected:
      C_δ single-exponential with rate γ_total
      C_mτ multi-exponential; slowest mode at rate γ_L (level-L coupling)
    """
    C_delta, C_m = compute_autocorrelations(run, max_lag=max_lag)
    lags = np.arange(len(C_delta))

    gamma_total = theory.total_coupling_strength(gamma, run.leaf_size, run.p, run.depth)
    gamma_L = theory.level_coupling_strength(gamma, run.leaf_size, run.p, run.depth)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Clip to positive for log plot
    mask_d = C_delta > 0
    mask_m = C_m > 0
    ax.semilogy(lags[mask_d], C_delta[mask_d], "b-", label="C_δ (fluctuations)")
    ax.semilogy(lags[mask_m], C_m[mask_m], "r-", label="C_mτ (centroids)")

    # Theory overlays
    t_theory = np.linspace(0, max_lag, 300)
    ax.semilogy(t_theory, np.exp(-gamma_total * t_theory), "b--",
                alpha=0.6, label=f"exp(-γ_total·t), γ_total={gamma_total:.2f}")
    ax.semilogy(t_theory, np.exp(-gamma_L * t_theory), "r--",
                alpha=0.6, label=f"exp(-γ_L·t), γ_L={gamma_L:.3f}")

    ax.set_xlabel("Lag (rounds)")
    ax.set_ylabel("Normalised autocorrelation")
    ax.set_title(
        f"Experiment D: decomposition autocorrelations\n"
        f"lr={run.lr}, m={run.leaf_size}, p={run.p}, L={run.depth}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment E — heterogeneity sensitivity
# ---------------------------------------------------------------------------


def plot_heterogeneity_comparison(
    runs_by_mode: Dict[str, List[CatchupRun]],
    gamma: float = 1.0,
) -> plt.Figure:
    """Plot catch-up curves for three heterogeneity regimes on shared axes.

    runs_by_mode should have keys 'hierarchical', 'homogeneous', 'iid'.
    Theory predicts identical curves across all three modes.
    """
    colors = {"hierarchical": "steelblue", "homogeneous": "darkorange", "iid": "forestgreen"}
    fig, ax = plt.subplots(figsize=(6, 5))

    for mode, runs in runs_by_mode.items():
        rows = summarise_catchup(runs)
        if not rows:
            continue
        lr = runs[0].lr
        by_d = _group_t50_by_distance(rows)
        if not by_d:
            continue
        distances = np.array(sorted(by_d.keys()))
        means = np.array([np.mean(by_d[d]) for d in distances])
        sems = np.array([np.std(by_d[d]) / math.sqrt(len(by_d[d])) for d in distances])
        ax.errorbar(distances, means, yerr=sems, fmt="o-", capsize=3,
                    color=colors.get(mode, "grey"), label=mode)

    # Theory reference
    if runs_by_mode:
        run0 = next(iter(runs_by_mode.values()))[0]
        d_range = np.arange(1, run0.depth + 1)
        theory_vals = np.array([
            math.log2(theory.catch_up_time(int(d), gamma, run0.leaf_size, run0.p))
            for d in d_range
        ])
        ax.plot(d_range, theory_vals, "k--", label="theory")

    ax.set_xlabel("Hierarchical distance d")
    ax.set_ylabel("log₂(t₅₀)")
    ax.set_title("Experiment E: heterogeneity sensitivity\n(theory: curves identical across modes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
