"""Visualisation functions for NMH active-escape dynamics experiments (v2)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import theory
from .active_escape import ActiveEscapeRun, basin_label, hierarchical_distance


# ---------------------------------------------------------------------------
# Data aggregation helpers
# ---------------------------------------------------------------------------


def _group_tflip_by_distance(
    rows: List[dict],
    lr_filter: Optional[float] = None,
    a_filter: Optional[float] = None,
    leaf_size_filter: Optional[int] = None,
    p_filter: Optional[float] = None,
) -> Tuple[Dict[int, List[float]], int]:
    """Aggregate log₂(t_flip) by hierarchical distance d.

    Rows where t_flip is None (leaf never flipped or warmup failed) are
    counted and excluded.  Returns (by_d dict, n_excluded count).
    """
    by_d: Dict[int, List[float]] = {}
    n_excluded = 0
    for row in rows:
        if not row.get("warmup_ok", True):
            continue
        if lr_filter is not None and abs(row["lr"] - lr_filter) > 1e-10:
            continue
        if a_filter is not None and abs(row["a"] - a_filter) > 1e-10:
            continue
        if leaf_size_filter is not None and row.get("leaf_size") != leaf_size_filter:
            continue
        if p_filter is not None and abs(row.get("p", 0) - p_filter) > 1e-10:
            continue
        t_flip = row["t_flip"]
        if t_flip is None or t_flip <= 0:
            n_excluded += 1
            continue
        d = int(row["distance_d"])
        by_d.setdefault(d, []).append(math.log2(t_flip))
    return by_d, n_excluded


def _linear_fit(x: np.ndarray, y: np.ndarray):
    """Least-squares fit y = slope*x + intercept; returns (slope, intercept, slope_se)."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    result = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(result[0][0]), float(result[0][1])
    residuals = y - (slope * x + intercept)
    n = len(x)
    se = (
        float(np.sqrt((residuals ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()))
        if n > 2
        else float("nan")
    )
    return slope, intercept, se


def _final_basin_fraction(run: ActiveEscapeRun) -> float:
    """Fraction of non-source target leaves in basin B at the last measurement round."""
    if not run.warmup_ok:
        return 0.0
    theta_A = run.theta_A
    theta_B = run.theta_B
    n_leaf_types = run.branching ** run.depth
    n_in_B = 0
    n_total = 0
    for tau in range(n_leaf_types):
        if tau == run.source_leaf:
            continue
        final_centroid = run.centroid_traj[-1, tau, :]
        if basin_label(final_centroid, theta_A, theta_B, epsilon=0.2) == 'B':
            n_in_B += 1
        n_total += 1
    return n_in_B / n_total if n_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Experiment A1: log₂(t_flip) vs d, one panel per lr
# ---------------------------------------------------------------------------


def plot_a1_flip_times(
    runs: List[ActiveEscapeRun],
    gamma: float = 1.0,
) -> plt.Figure:
    """log₂(t_flip) vs hierarchical distance d, one panel per learning rate.

    Theory overlay: T_flip(d) = 2^(d+1)/(γmp); log₂(T_flip) = (d+1) + const.
    Slope should be ≈ 1.0 in Regime I.
    """
    all_rows = [row for run in runs for row in run.flip_table]
    lr_values = sorted(set(row["lr"] for row in all_rows))
    n_panels = len(lr_values)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0), squeeze=False)
    axes = axes[0]

    for ax, lr in zip(axes, lr_values):
        by_d, n_excl = _group_tflip_by_distance(all_rows, lr_filter=lr)
        if not by_d:
            ax.set_title(f"lr={lr} (no data)")
            continue

        ds = sorted(by_d)
        means = np.array([np.mean(by_d[d]) for d in ds])
        stes = np.array([np.std(by_d[d]) / max(math.sqrt(len(by_d[d])), 1) for d in ds])

        ax.errorbar(ds, means, yerr=stes, fmt="o", color="steelblue",
                    capsize=4, label="simulation")

        # Theory reference line: slope=1 anchored at empirical d=1 mean.
        # Continuous-time T_flip = 2^(d+1)/γmp gives the correct slope but
        # a very different absolute scale (T_flip(1)≈0.25 rounds in CT vs ≈50
        # discrete rounds), so we anchor to the data rather than show the raw
        # theoretical intercept which would be off-screen.
        if 1 in by_d:
            anchor_log2 = float(np.mean(by_d[1]))
            d_theory = np.linspace(min(ds) - 0.3, max(ds) + 0.3, 50)
            # slope=1 reference anchored at d=1
            log2_ref = anchor_log2 + (d_theory - 1.0)
            ax.plot(d_theory, log2_ref, "k--", alpha=0.7, label="theory slope=1")

        # Linear fit
        if len(ds) >= 2:
            x_fit = np.array(ds, dtype=float)
            y_fit = means
            slope, intercept, se = _linear_fit(x_fit, y_fit)
            x_line = np.linspace(min(ds) - 0.3, max(ds) + 0.3, 50)
            ax.plot(x_line, slope * x_line + intercept, "r-", alpha=0.6,
                    label=f"fit slope={slope:.2f}±{se:.2f}")

        ax.set_xlabel("Hierarchical distance d")
        ax.set_ylabel("log₂(t_flip)")
        ax.set_title(f"lr = {lr}")
        ax.legend(fontsize=8)
        if n_excl > 0:
            ax.text(0.02, 0.98, f"{n_excl} None excluded",
                    transform=ax.transAxes, fontsize=7, va="top", color="gray")

    fig.suptitle("A1: Active-escape flip time vs hierarchical distance", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment A2: regime sweep — flip fraction over time per a value
# ---------------------------------------------------------------------------


def plot_a2_regime_sweep(
    runs: List[ActiveEscapeRun],
    gamma: float = 1.0,
) -> plt.Figure:
    """Five subplots (one per a value in A2); each shows flip_fraction(d) vs time.

    flip_fraction(d, t) = fraction of target leaves at distance d that have
    entered basin B by round t.  Theory predicts this reaches 1.0 for d ≤ ell_c
    and stays near 0 for d > ell_c in finite measurement windows.
    """
    a_values = sorted(set(run.a for run in runs))
    n_panels = len(a_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.0), squeeze=False)
    axes = axes[0]

    for ax, a_val in zip(axes, a_values):
        a_runs = [r for r in runs if abs(r.a - a_val) < 1e-10 and r.warmup_ok]
        if not a_runs:
            ax.set_title(f"a={a_val:.2f} (no valid runs)")
            continue

        r0 = a_runs[0]
        regime = r0.regime
        ell_c = r0.ell_c
        n_leaf_types = r0.branching ** r0.depth
        T = r0.n_meas_rounds + 1
        depth = r0.depth

        # Group target leaves by distance
        distances = sorted(set(
            hierarchical_distance(r0.source_leaf, j)
            for j in range(n_leaf_types) if j != r0.source_leaf
        ))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(distances)))

        for dist, color in zip(distances, colors):
            # For each time step, compute fraction of (run, leaf) pairs in B
            fractions = []
            for t in range(T):
                in_B = 0
                total = 0
                for run in a_runs:
                    for tau in range(n_leaf_types):
                        if tau == run.source_leaf:
                            continue
                        if hierarchical_distance(run.source_leaf, tau) != dist:
                            continue
                        c = run.centroid_traj[t, tau, :]
                        if basin_label(c, run.theta_A, run.theta_B) == 'B':
                            in_B += 1
                        total += 1
                fractions.append(in_B / total if total > 0 else 0.0)
            ax.plot(range(T), fractions, color=color, label=f"d={dist}")

        ax.set_xlabel("Round")
        ax.set_ylabel("Flip fraction")
        ax.set_title(f"a={a_val:.1f} (Regime {regime}, ell_c={ell_c})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("A2: Regime sweep — flip fraction per distance", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment A3: containment phase transition
# ---------------------------------------------------------------------------


def plot_a3_phase_transition(
    runs: List[ActiveEscapeRun],
    gamma: float = 1.0,
) -> plt.Figure:
    """final_basin_fraction vs a, with error bars across seeds.

    Expects a sharp sigmoidal drop near the critical curvature
    a_c = 2·γ_1 / ||ΔΘ*||² = 2·γmp/4 = γmp/2 (for δ_norm=1).
    """
    a_values = sorted(set(run.a for run in runs))
    means, stds = [], []
    for a_val in a_values:
        fracs = [_final_basin_fraction(r) for r in runs if abs(r.a - a_val) < 1e-10]
        means.append(np.mean(fracs) if fracs else 0.0)
        stds.append(np.std(fracs) / max(math.sqrt(len(fracs)), 1) if len(fracs) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(a_values, means, yerr=stds, fmt="o-", color="steelblue",
                capsize=4, label="simulation")

    # Theory critical a_c: γ_1 = κ_loc/2 → a_c = 2·γ_1 / δ_norm²
    if runs:
        r0 = runs[0]
        gamma_1 = theory.level_coupling_strength(gamma, r0.leaf_size, r0.p, 1)
        delta_norm = float(np.linalg.norm(r0.theta_B - r0.theta_A))
        a_c = 2.0 * gamma_1 / (delta_norm ** 2) if delta_norm > 0 else float("nan")
        if not math.isnan(a_c):
            ax.axvline(a_c, color="red", linestyle="--", alpha=0.7,
                       label=f"theory a_c = {a_c:.2f}")

    ax.set_xlabel("Barrier curvature a")
    ax.set_ylabel("Final basin fraction (B adoption)")
    ax.set_title("A3: Containment phase transition")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment B: iso-γmp collapse
# ---------------------------------------------------------------------------


def plot_b_iso_gmp(
    runs: List[ActiveEscapeRun],
    gamma: float = 1.0,
) -> plt.Figure:
    """log₂(t_flip) vs d for four iso-γmp configurations.

    Under the fundamental reading (T_flip depends only on γmp, not m or p
    separately), all four curves should collapse onto the same line.
    """
    all_rows = [row for run in runs for row in run.flip_table]

    # Group by (leaf_size, p) combos
    combos = sorted(set(
        (run.leaf_size, run.p, run.depth)
        for run in runs
    ))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(combos)))

    fig, ax = plt.subplots(figsize=(6, 4))

    for (m, p, depth), color in zip(combos, colors):
        combo_rows = [
            row for row in all_rows
            if row.get("warmup_ok", True)
        ]
        # Filter by matching leaf_size and p via the run objects
        combo_rows = []
        for run in runs:
            if run.leaf_size == m and abs(run.p - p) < 1e-10:
                for row in run.flip_table:
                    if row.get("warmup_ok", True):
                        combo_rows.append(row)

        by_d, _ = _group_tflip_by_distance(combo_rows)
        if not by_d:
            continue

        ds = sorted(by_d)
        means = np.array([np.mean(by_d[d]) for d in ds])
        stes = np.array([np.std(by_d[d]) / max(math.sqrt(len(by_d[d])), 1) for d in ds])
        ax.errorbar(ds, means, yerr=stes, fmt="o-", color=color,
                    capsize=4, label=f"m={m}, p={p}")

    # Theory line (same for all iso-γmp)
    if runs:
        r0 = runs[0]
        m_ref, p_ref = r0.leaf_size, r0.p
        d_vals = np.linspace(1, max(r0.depth, 5), 50)
        log2_T = [math.log2(theory.deterministic_flip_time(d, gamma, m_ref, p_ref)) for d in d_vals]
        ax.plot(d_vals, log2_T, "k--", alpha=0.7, label="theory (slope=1)")

    ax.set_xlabel("Hierarchical distance d")
    ax.set_ylabel("log₂(t_flip)")
    ax.set_title("B: iso-γmp collapse (γmp=16)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
