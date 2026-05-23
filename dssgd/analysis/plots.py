"""Visualization functions for NMH simulation vs SDE theory comparison.

Each function produces one matplotlib Figure and returns it; saving is the
caller's responsibility (see run_all.py).  Functions are organised to match
the paper's claims section by section.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .results import AnalysisRun
from . import stats, theory


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _level_colors(n: int):
    return cm.viridis(np.linspace(0.15, 0.85, max(n, 1)))


# ---------------------------------------------------------------------------
# Topology / graph-structure
# ---------------------------------------------------------------------------


def plot_degree_vs_depth(
    runs: Sequence[AnalysisRun],
    title: str = "Mean degree vs hierarchy depth",
) -> plt.Figure:
    """Observed mean degree against depth with geometric-series theory overlay."""
    fig, ax = plt.subplots()

    depths = [r.depth for r in runs]
    obs = [r.topology.mean_degree for r in runs]

    if runs:
        r0 = runs[0]
        theory_depths = list(range(1, max(depths) + 3))
        theory_degrees = [
            theory.expected_degree(r0.p, r0.branching, r0.leaf_size, d)
            for d in theory_depths
        ]
        limit = theory.degree_limit(r0.p, r0.branching, r0.leaf_size)
        ax.plot(theory_depths, theory_degrees, "k--", label="Theory (finite depth)")
        if limit is not None:
            ax.axhline(
                limit, color="k", linestyle=":", alpha=0.5,
                label=f"Depth → ∞ limit = {limit:.1f}",
            )

    ax.scatter(depths, obs, color="steelblue", zorder=5, label="Observed")
    ax.set_xlabel("Hierarchy depth")
    ax.set_ylabel("Mean degree")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_neighbourhood_fraction(
    run: AnalysisRun,
    title: str = "Cross-module neighbours: observed vs theory",
) -> plt.Figure:
    """Grouped bar chart of per-level expected vs observed cross-module neighbours."""
    observed, expected = stats.observed_vs_theory_neighbours(run)
    levels = list(range(1, len(observed) + 1))

    fig, ax = plt.subplots()
    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w / 2, expected, w, label="Theory", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, observed, w, label="Observed", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {lv}" for lv in levels])
    ax.set_ylabel("Mean cross-module neighbours per node")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_topological_dimension(
    runs_by_p: Dict[float, List[AnalysisRun]],
    title: str = "Topological dimension D vs p",
) -> plt.Figure:
    """Fit n(r) ~ r^D from BFS neighbourhood growth for each p; theory predicts D ~ p."""
    ps = sorted(runs_by_p.keys())
    mean_Ds, std_Ds = [], []
    for p in ps:
        Ds = [
            stats.topological_dimension(r.topology.mean_nbhd_by_radius)
            for r in runs_by_p[p]
            if r.topology.mean_nbhd_by_radius
        ]
        Ds = [d for d in Ds if d is not None]
        mean_Ds.append(float(np.mean(Ds)) if Ds else float("nan"))
        std_Ds.append(float(np.std(Ds)) if len(Ds) > 1 else 0.0)

    fig, ax = plt.subplots()
    ax.errorbar(
        ps, mean_Ds, yerr=std_Ds, fmt="o", color="steelblue",
        capsize=4, label="Fitted D (mean +/- std across seeds)",
    )
    ax.plot(ps, ps, "k--", label="D = p (theory)")
    ax.set_xlabel("p  (base cross-module edge probability)")
    ax.set_ylabel("Fitted topological dimension D")
    ax.set_title(title)
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def plot_level_convergence(
    run: AnalysisRun,
    title: Optional[str] = None,
) -> plt.Figure:
    """Within-module consensus distance vs round for each hierarchy level."""
    title = title or f"Level convergence — {run.name}"
    fig, ax = plt.subplots()
    colors = _level_colors(run.depth + 1)

    for level in range(run.depth + 1):
        trace = run.level_trace.within_distances.get(level, [])
        if trace:
            ax.semilogy(trace, color=colors[level], label=f"Within level {level}")

    if run.level_trace.global_distances:
        ax.semilogy(
            run.level_trace.global_distances, "k--",
            linewidth=1.5, label="Global",
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean consensus distance (log scale)")
    ax.set_title(title)
    ax.legend(fontsize="small")
    return fig


def plot_level_coupling_vs_theory(
    run: AnalysisRun,
    gamma: float = 1.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Per-level convergence half-lives versus Kramers escape-time predictions.

    NOTE: only meaningful for consensus-initialised runs (start_from_consensus=True)
    where cross_distances rise from zero.  For randomly-initialised runs the
    observed half-life measures gossip mixing speed, not Kramers escape time.
    """
    title = title or f"Level coupling: half-lives vs Kramers — {run.name}"
    T_eff = (
        theory.effective_temperature(run.lr, run.sigma2)
        if run.sigma2 > 0
        else 1.0
    )
    levels = list(range(1, run.depth + 1))
    obs_times = stats.time_to_half_stationary_cross(run)

    theory_times = {
        lv: theory.kramers_escape_time(T_eff, theory.well_depth(gamma, run.leaf_size, run.p, lv))
        for lv in levels
    }

    obs_pairs = [(lv, obs_times[lv]) for lv in levels if obs_times.get(lv) is not None]
    theory_pairs = [(lv, t) for lv, t in theory_times.items() if t < 1e5]

    fig, ax = plt.subplots()
    if obs_pairs:
        lvs, vals = zip(*obs_pairs)
        ax.scatter(lvs, vals, color="steelblue", zorder=5,
                   label="Observed (time to 50% stationary cross-dist)")
    if theory_pairs:
        lvs_t, vals_t = zip(*theory_pairs)
        ax.plot(lvs_t, vals_t, "k--o", label=f"Kramers prediction (T_eff={T_eff:.4f})")

    ax.set_xlabel("Hierarchy level")
    ax.set_ylabel("Rounds")
    ax.set_title(title)
    ax.legend(fontsize="small")
    return fig


def plot_kramers_sweep(
    runs_by_key: Dict,
    gamma: float = 1.0,
    title: str = "Kramers escape times: observed vs theory",
    label_fn=None,
) -> plt.Figure:
    """Multi-group Kramers comparison from consensus-initialised runs.

    runs_by_key: dict mapping any key (lr float, leaf_size int, etc.) to a list
    of AnalysisRun objects.  T_eff is computed from the runs' lr and sigma2.
    label_fn: optional callable(key, run_list) -> str for legend labels.
    Plots observed time-to-half-stationary-cross-dist vs Kramers tau on log-log axes.
    """
    fig, ax = plt.subplots()
    keys = sorted(runs_by_key.keys())
    colors = cm.viridis(np.linspace(0.1, 0.9, max(len(keys), 1)))

    for key, color in zip(keys, colors):
        run_list = runs_by_key[key]
        sigma2 = float(np.mean([r.sigma2 for r in run_list]))
        if sigma2 <= 0:
            continue
        lr_val = float(np.mean([r.lr for r in run_list]))
        T_eff = theory.effective_temperature(lr_val, sigma2)

        all_obs: List[float] = []
        all_tau: List[float] = []

        for run in run_list:
            obs_times = stats.time_to_half_stationary_cross(run)
            for lv in range(1, run.depth + 1):
                dV = theory.well_depth(gamma, run.leaf_size, run.p, lv)
                try:
                    tau = theory.kramers_escape_time(T_eff, dV)
                except OverflowError:
                    continue
                if tau > 1e5 or obs_times.get(lv) is None:
                    continue
                all_obs.append(float(obs_times[lv]))
                all_tau.append(tau)

        if all_obs:
            label = label_fn(key, run_list) if label_fn else f"{key} (T_eff={T_eff:.4f})"
            ax.scatter(all_tau, all_obs, color=color, s=50, alpha=0.8, label=label)

    lim_lo, lim_hi = 0.5, 1000.0
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4, label="Perfect agreement")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Kramers tau^(ell) = exp(dV / T_eff)")
    ax.set_ylabel("Observed time to 50% stationary cross-distance (rounds)")
    ax.set_title(title)
    ax.legend(fontsize="small")
    return fig


def plot_escape_dissolution_trace(
    run: AnalysisRun,
    gamma: float = 1.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Cross-module distance traces across both phases, with the warmup boundary marked.

    Phase 1 (0..warmup_rounds): heterogeneous training, modules separate.
    Phase 2 (warmup_rounds..n_rounds): homogeneous data, gossip dissolves clusters.
    Horizontal dashed lines show 100% and 50% of the warmup-stationary value per level.
    """
    title = title or f"Two-phase Kramers dissolution — {run.name}"
    n_warmup = run.warmup_rounds
    levels = list(range(1, run.depth + 1))
    colors = _level_colors(len(levels))

    fig, ax = plt.subplots(figsize=(10, 5))
    for lv, color in zip(levels, colors):
        trace = run.level_trace.cross_distances.get(lv, [])
        if not trace:
            continue
        ax.plot(trace, color=color, label=f"Level {lv}")

    if n_warmup > 0:
        ax.axvline(n_warmup, color="k", linestyle="--", linewidth=1.5,
                   label="Phase switch (homogeneous)")

    ax.set_xlabel("Round")
    ax.set_ylabel("Cross-module centroid distance")
    ax.set_title(title)
    ax.legend(fontsize="small")
    return fig


def plot_escape_kramers_sweep(
    runs_by_key: Dict,
    gamma: float = 1.0,
    title: str = "Two-phase Kramers: dissolution tau_D vs spectral and Kramers timescales",
    label_fn=None,
) -> plt.Figure:
    """Per-level comparison of observed dissolution time, spectral timescale, and Kramers time.

    Three timescales per (key, level):
      tau_D      = time_to_half_escape_cross — observed gossip dissolution speed
      tau_spec   = 1/gamma_ell — pure spectral relaxation (no barrier)
      tau_K      = exp(DV/T_eff) — Kramers prediction (often astronomically large)

    If tau_D ~ tau_spec << tau_K: gossip shortcut bypasses the Kramers barrier entirely.
    Plotted per-level on a log y-axis; tau_K is capped at a display ceiling with a
    marker to indicate clipped values.
    """
    keys = sorted(runs_by_key.keys())
    n_keys = max(len(keys), 1)
    colors = cm.viridis(np.linspace(0.1, 0.9, n_keys))

    TAU_K_CEIL = 1e6  # display ceiling for Kramers times

    # Collect (level, tau_D, tau_spec, tau_K) per key
    records: Dict = {}
    for key, color in zip(keys, colors):
        run_list = runs_by_key[key]
        sigma2 = float(np.mean([r.sigma2 for r in run_list]))
        if sigma2 <= 0:
            continue
        lr_val = float(np.mean([r.lr for r in run_list]))
        T_eff = theory.effective_temperature(lr_val, sigma2)

        per_level_D: Dict[int, List[float]] = {}
        per_level_spec: Dict[int, float] = {}
        per_level_K: Dict[int, float] = {}

        depth = run_list[0].depth
        for lv in range(1, depth + 1):
            dV = theory.well_depth(gamma, run_list[0].leaf_size, run_list[0].p, lv)
            gamma_lv = theory.level_coupling_strength(
                gamma, run_list[0].leaf_size, run_list[0].p, lv
            )
            per_level_spec[lv] = 1.0 / gamma_lv if gamma_lv > 0 else float("inf")
            try:
                per_level_K[lv] = theory.kramers_escape_time(T_eff, dV)
            except OverflowError:
                per_level_K[lv] = float("inf")

            tau_d_vals: List[float] = []
            for run in run_list:
                esc = stats.time_to_half_escape_cross(run)
                v = esc.get(lv)
                if v is not None:
                    tau_d_vals.append(float(v))
            per_level_D[lv] = tau_d_vals

        label = label_fn(key, run_list) if label_fn else f"lr={lr_val} (T_eff={T_eff:.4f})"
        records[key] = dict(
            color=color, label=label, T_eff=T_eff,
            per_level_D=per_level_D,
            per_level_spec=per_level_spec,
            per_level_K=per_level_K,
            depth=depth,
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    offset = np.linspace(-0.25, 0.25, n_keys)
    for (key, rec), off in zip(records.items(), offset):
        depth = rec["depth"]
        levels = list(range(1, depth + 1))
        color = rec["color"]

        # tau_D (observed dissolution)
        tau_d_means = []
        for lv in levels:
            vals = rec["per_level_D"].get(lv, [])
            tau_d_means.append(float(np.mean(vals)) if vals else float("nan"))

        ax.plot(
            [lv + off for lv in levels], tau_d_means,
            "o-", color=color, linewidth=2, markersize=7,
            label=f"{rec['label']} — observed tau_D",
        )

        # tau_spec (spectral relaxation, 1/gamma_ell)  — same color, dashed
        tau_spec_vals = [rec["per_level_spec"].get(lv, float("nan")) for lv in levels]
        ax.plot(
            levels, tau_spec_vals,
            "--", color=color, linewidth=1.2, alpha=0.6,
            label=f"lr={key} — tau_spec = 1/gamma_ell",
        )

        # tau_K (Kramers) — same color, dotted, capped
        tau_k_vals = [min(rec["per_level_K"].get(lv, float("inf")), TAU_K_CEIL)
                      for lv in levels]
        clipped = [rec["per_level_K"].get(lv, float("inf")) > TAU_K_CEIL for lv in levels]
        ax.plot(
            levels, tau_k_vals, ":", color=color, linewidth=1.5, alpha=0.5,
            label=f"lr={key} — tau_K (Kramers, >=1e6 clipped)",
        )
        for lv, clp, tv in zip(levels, clipped, tau_k_vals):
            if clp:
                ax.annotate(">", xy=(lv, tv), ha="center", va="bottom",
                            color=color, fontsize=9, alpha=0.7)

        # 1/lr reference — gradient-driven dissolution timescale
        lr_val = float(key)
        ax.axhline(
            1.0 / lr_val, color=color, linewidth=0.8, alpha=0.3,
            linestyle=(0, (3, 10)),
        )

    ax.set_xlabel("Hierarchy level")
    ax.set_ylabel("Timescale (rounds, log scale)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize="x-small", ncol=2)
    return fig


def plot_coupling_ratio_vs_depth(
    runs: Sequence[AnalysisRun],
    title: str = "Convergence-rate ratio λ_ℓ / λ_{ℓ+1} per level",
) -> plt.Figure:
    """Per-level convergence-rate ratios; theory predicts each ≈ 2 (gamma_l/gamma_{l+1})."""
    fig, ax = plt.subplots()

    for run in runs:
        ratios = stats.level_coupling_ratios(run)
        pairs = [(i, r) for i, r in enumerate(ratios) if r is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, marker="o", label=run.name)

    ax.axhline(2.0, color="k", linestyle="--", label="Theory SDE (= 2)")
    ax.set_xlabel("Level ℓ")
    ax.set_ylabel("λ_ℓ / λ_{ℓ+1}")
    ax.set_title(title)
    ax.legend(fontsize="small")
    return fig


# ---------------------------------------------------------------------------
# Parameter-space clustering
# ---------------------------------------------------------------------------


def plot_type_clustering_pca(
    run: AnalysisRun,
    round_idx: Optional[int] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """PCA of agent parameter vectors, coloured by leaf-module membership."""
    title = title or f"Parameter clustering (PCA) — {run.name}"

    if round_idx is None:
        round_idx = max(run.snapshots.keys()) if run.snapshots else 0

    if round_idx not in run.snapshots:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No snapshot available", ha="center", va="center")
        ax.set_title(title)
        return fig

    snap = run.snapshots[round_idx]
    agent_ids = sorted(snap.keys())
    X = np.stack([snap[i] for i in agent_ids])

    # PCA via truncated SVD
    X_c = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    coords = X_c @ Vt[:2].T

    leaf_modules = theory.nmh_modules_by_level(run.branching, run.depth, run.leaf_size)[0]
    n_mods = len(leaf_modules)
    node_to_mod = {nid: m for m, mod in enumerate(leaf_modules) for nid in mod}
    mod_ids = [node_to_mod[i] for i in agent_ids]
    cmap = cm.tab20(np.linspace(0, 1, max(n_mods, 1)))

    fig, ax = plt.subplots()
    for mod_idx in range(n_mods):
        mask = [j for j, m in enumerate(mod_ids) if m == mod_idx]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=cmap[mod_idx], s=40, alpha=0.8,
            label=f"Module {mod_idx}" if n_mods <= 8 else None,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    if n_mods <= 8:
        ax.legend(fontsize="x-small", ncol=2)
    return fig


def plot_within_cross_type_distance(
    run: AnalysisRun,
    title: Optional[str] = None,
) -> plt.Figure:
    """Within leaf-module vs cross-module centroid distance over training rounds."""
    title = title or f"Within vs cross-module distance — {run.name}"
    rounds = list(range(run.n_rounds))

    within = run.level_trace.within_distances.get(0, [])
    cross = run.level_trace.cross_distances.get(1, [])
    global_trace = run.level_trace.global_distances

    fig, ax = plt.subplots()
    if within:
        ax.semilogy(rounds[: len(within)], within, label="Within leaf-module")
    if cross:
        ax.semilogy(
            rounds[: len(cross)], cross, color="coral",
            label="Cross-module centroid (level 1)",
        )
    if global_trace:
        ax.semilogy(
            rounds[: len(global_trace)], global_trace,
            "k--", linewidth=1.0, label="Global",
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Distance (log scale)")
    ax.set_title(title)
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Temperature / stationary distribution
# ---------------------------------------------------------------------------


def plot_stationary_variance_per_level(
    run: AnalysisRun,
    tail_fraction: float = 0.2,
    title: Optional[str] = None,
) -> plt.Figure:
    """Stationary within-module variance by level; theory: ratio per level = 2.

    Left panel: mean within-module distance in the final tail_fraction of rounds.
    Right panel: ratio of consecutive levels' variances; theory predicts each = 2
    (from gamma_ell / gamma_{ell+1} = 2 in the McKean-Vlasov coupling strengths).
    """
    title = title or f"Stationary variance per level — {run.name}"
    variances = stats.level_stationary_variance(run, tail_fraction)

    if not variances:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data (run needs training enabled)", ha="center", va="center")
        ax.set_title(title)
        return fig

    levels = sorted(variances.keys())
    vals = [variances[lv] for lv in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = _level_colors(len(levels))
    ax1.bar(levels, vals, color=colors, alpha=0.85)
    ax1.set_xlabel("Hierarchy level")
    ax1.set_ylabel("Mean within-module distance (stationary)")
    ax1.set_yscale("log")
    ax1.set_title("Stationary variance")

    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1) if vals[i] > 0]
    ratio_levels = levels[:-1]
    if ratios:
        ax2.plot(ratio_levels, ratios, "o-", color="steelblue", label="Observed ratio")
        ax2.axhline(2.0, color="k", linestyle="--", label="Theory (= 2)")
        ax2.set_xlabel("Level")
        ax2.set_ylabel("Var(level+1) / Var(level)")
        ax2.set_title("Variance ratio (theory: 2)")
        ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_stationary_temperature(
    runs_by_lr: Dict[float, List[AnalysisRun]],
    title: str = "Stationary spread vs T_eff = η σ² / 2",
) -> plt.Figure:
    """Observed parameter-spread at stationarity vs theoretical effective temperature.

    Accepts the raw per-lr seed groups (not averaged runs), because averaged runs
    have snapshots cleared and stationary_spread would return None for all of them.
    """
    fig, ax = plt.subplots()
    T_effs, spreads, spread_errs = [], [], []

    for lr, run_list in sorted(runs_by_lr.items()):
        sigma2 = float(np.mean([r.sigma2 for r in run_list]))
        if sigma2 <= 0:
            continue
        T = theory.effective_temperature(lr, sigma2)
        seed_spreads = [stats.stationary_spread(r) for r in run_list]
        seed_spreads = [s for s in seed_spreads if s is not None]
        if not seed_spreads:
            continue
        T_effs.append(T)
        spreads.append(float(np.mean(seed_spreads)))
        spread_errs.append(float(np.std(seed_spreads)))

    if T_effs:
        ax.errorbar(T_effs, spreads, yerr=spread_errs, fmt="o", color="steelblue",
                    capsize=4, zorder=5, label="Observed spread (mean +/- std)")
        if len(T_effs) >= 2:
            log_T = np.log(T_effs)
            log_s = np.log(spreads)
            slope, intercept = np.polyfit(log_T, log_s, 1)
            T_plot = np.linspace(min(T_effs), max(T_effs), 100)
            ax.plot(
                T_plot, np.exp(intercept) * T_plot ** slope, "k--",
                label=f"Power-law fit (slope={slope:.2f})",
            )

    ax.set_xlabel("T_eff = η σ² / 2")
    ax.set_ylabel("Parameter spread (std)")
    ax.set_title(title)
    ax.legend()
    return fig
