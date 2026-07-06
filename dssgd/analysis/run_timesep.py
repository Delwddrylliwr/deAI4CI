"""Orchestrator for NMH + NCP Regime C (timescale-separation) experiments.

Run experiments with result caching, CSV export, and plots:

    python -m analysis.run_timesep [--output-dir DIR]
                                   [--exps NMH1 NMH1b NMH3 NCP1]
                                   [--n-seeds N] [--force] [--no-plots]

Priority order (spec §14): NMH-1 → NMH-1b → NMH-3 → NCP-1.
NCP-1 is a graph-only check (no simulation); all others run the full
natural-cascade runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .natural_cascade import NaturalCascadeConfig, NaturalCascadeRun, run_natural_cascade_simulation
from .natural_cascade_experiments import (
    experiment_NMH1,
    experiment_NMH1b,
    experiment_NMH3,
)
from .nmh_observables import cascade_depth as obs_cascade_depth
from .ncp_runner import NCPSimConfig, NCPRun, run_ncp_simulation
from .ncp_experiments import experiment_NCP1_graph_configs
from dssgd.topology.forest_fire import ForestFireTopology


# ---------------------------------------------------------------------------
# Natural-cascade caching helpers
# ---------------------------------------------------------------------------


def _nc_key(config: NaturalCascadeConfig) -> str:
    """Stable cache filename derived from config name (which includes seed)."""
    return config.name.replace("/", "__").replace("=", "") + ".pkl"


def _load_or_run_nc(
    config: NaturalCascadeConfig,
    results_dir: Path,
    force: bool,
) -> NaturalCascadeRun:
    cache_path = results_dir / _nc_key(config)
    if cache_path.exists() and not force:
        print(f"  [cache] {config.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  [run]   {config.name}")
    run = run_natural_cascade_simulation(config)
    run.save(cache_path)
    if not run.warmup_ok:
        print(f"  [warn]  warmup FAILED: {config.name}")
    return run


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _export_nc_flip_csv(runs: List[NaturalCascadeRun], csv_path: Path) -> None:
    """Write flip_table rows to CSV for all runs with warmup_ok=True."""
    rows = []
    for run in runs:
        if not run.warmup_ok:
            continue
        for row in run.flip_table:
            rows.append({
                **row,
                "branching": run.branching,
                "depth": run.depth,
                "leaf_size": run.leaf_size,
                "p": run.p,
                "d_param": run.d_param,
                "n_meas_rounds": run.n_meas_rounds,
                "nucleation_leaf": run.nucleation_leaf,
                "t_nucleation": run.t_nucleation,
            })
    if not rows:
        print(f"  [warn] No flip rows to export → {csv_path.name}")
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {csv_path}")


def _export_nc_summary_csv(runs: List[NaturalCascadeRun], csv_path: Path) -> None:
    """One row per run: nucleation_leaf, t_nucleation, cascade depth."""
    theta_A = np.array([0.0], dtype=np.float32)
    theta_B = np.array([1.0], dtype=np.float32)
    rows = []
    for run in runs:
        depth_obs = (
            obs_cascade_depth(
                run.centroid_traj, run.nucleation_leaf, theta_A, theta_B
            )
            if run.warmup_ok and run.nucleation_leaf is not None
            else None
        )
        rows.append({
            "name": run.name,
            "a": run.a,
            "b": run.b,
            "local_steps": run.local_steps,
            "seed": run.seed,
            "depth": run.depth,
            "warmup_ok": run.warmup_ok,
            "nucleation_leaf": run.nucleation_leaf,
            "t_nucleation": run.t_nucleation,
            "cascade_depth": depth_obs,
            "regime": run.regime,
            "ell_c": run.ell_c,
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _group_tflip_by_distance(
    runs: List[NaturalCascadeRun],
) -> Dict[Tuple, Dict[int, List[float]]]:
    """Group log₂(t_flip_relative) by (a, local_steps) and distance_from_source.

    Returns {(a, local_steps): {distance: [log2_t_flip, ...]}}.
    Only includes entries where warmup_ok=True and t_flip_relative is not None.
    """
    grouped: Dict[Tuple, Dict[int, List[float]]] = {}
    for run in runs:
        if not run.warmup_ok:
            continue
        key = (run.a, run.local_steps)
        if key not in grouped:
            grouped[key] = {}
        for row in run.flip_table:
            t = row["t_flip_relative"]
            d = row["distance_from_source"]
            if t is None or d is None or t <= 0:
                continue
            grouped[key].setdefault(d, []).append(math.log2(t))
    return grouped


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """OLS fit y = slope·x + intercept; returns (slope, intercept, slope_se)."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    res = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(res[0][0]), float(res[0][1])
    residuals = y - (slope * x + intercept)
    n = len(x)
    denom = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(residuals.var(ddof=2) / denom)) if denom > 0 and n > 2 else float("nan")
    return slope, intercept, se


def plot_slope1(
    runs: List[NaturalCascadeRun],
    group_key: str,          # 'a' or 'local_steps'
    title: str,
    out_path: Path,
) -> None:
    """Log₂(mean t_flip_relative) vs distance d — slope-1 test."""
    grouped = _group_tflip_by_distance(runs)
    if not grouped:
        print(f"  [warn] No flip data for slope-1 plot — skipping {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(grouped)))

    for (a_val, ls_val), by_d in sorted(grouped.items()):
        if not by_d:
            continue
        ds = sorted(by_d.keys())
        means = [np.mean(by_d[d]) for d in ds]
        stes = [np.std(by_d[d]) / math.sqrt(len(by_d[d])) if len(by_d[d]) > 1 else 0.0
                for d in ds]
        label = f"a={a_val}" if group_key == "a" else f"local_steps={ls_val}"
        color = colors[list(grouped.keys()).index((a_val, ls_val))]
        ax.errorbar(ds, means, yerr=stes, fmt="o-", capsize=4, label=label, color=color)

        # Fit slope on distances where we have ≥5 data points
        fit_ds = np.array([d for d in ds if len(by_d[d]) >= 5], dtype=float)
        fit_means = np.array([np.mean(by_d[d]) for d in fit_ds])
        if len(fit_ds) >= 2:
            slope, intercept, _ = _linear_fit(fit_ds, fit_means)
            x_line = np.array([fit_ds[0], fit_ds[-1]])
            ax.plot(x_line, slope * x_line + intercept, "--", color=color, alpha=0.5,
                    linewidth=0.8)

    # Reference slope-1 line anchored at (1, mean_at_d1)
    all_d1 = [v for g in grouped.values() for v in g.get(1, [])]
    if all_d1:
        ref_base = np.mean(all_d1)
        d_vals = sorted({d for g in grouped.values() for d in g})
        x_ref = np.array([d_vals[0], d_vals[-1]], dtype=float)
        ax.plot(x_ref, ref_base + (x_ref - 1.0), "k--", linewidth=1.5,
                label="slope = 1 (theory)", zorder=0)

    ax.set_xlabel("Hierarchical distance d")
    ax.set_ylabel("log₂(mean t_flip_relative)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


def plot_phase_structure(
    runs: List[NaturalCascadeRun],
    title: str,
    out_path: Path,
) -> None:
    """Mean cascade depth vs a — three-regime phase structure."""
    theta_A = np.array([0.0], dtype=np.float32)
    theta_B = np.array([1.0], dtype=np.float32)

    depth_by_a: Dict[float, List[int]] = {}
    for run in runs:
        if not run.warmup_ok or run.nucleation_leaf is None:
            continue
        d = obs_cascade_depth(run.centroid_traj, run.nucleation_leaf, theta_A, theta_B)
        depth_by_a.setdefault(run.a, []).append(d)

    if not depth_by_a:
        print(f"  [warn] No phase data — skipping {out_path.name}")
        return

    a_vals = sorted(depth_by_a.keys())
    means = [np.mean(depth_by_a[a]) for a in a_vals]
    stes = [np.std(depth_by_a[a]) / math.sqrt(len(depth_by_a[a]))
            if len(depth_by_a[a]) > 1 else 0.0 for a in a_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(a_vals, means, yerr=stes, fmt="o-", capsize=4, color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0, label="Reg I/II (a=0.5)")
    ax.axvline(8.0, color="orange", linestyle="--", linewidth=1.0, label="Reg II/III (a=8)")
    ax.set_xscale("log")
    ax.set_xlabel("Loss curvature a")
    ax.set_ylabel("Mean cascade depth (max hierarchical distance flipped)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


def plot_ncp1_shells(
    stats_rows: List[dict],
    out_path: Path,
) -> None:
    """Shell count vs graph size for each p_f value."""
    from collections import defaultdict
    by_pf: Dict[float, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for row in stats_rows:
        n_list, shell_list = by_pf[row["p_f"]]
        n_list.append(row["n_nodes"])
        shell_list.append(row["n_shells"])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(by_pf)))
    for (p_f, (ns, shells)), color in zip(sorted(by_pf.items()), colors):
        ns_arr = np.array(ns)
        sh_arr = np.array(shells)
        means = {n: np.mean(sh_arr[ns_arr == n]) for n in sorted(set(ns))}
        ax.plot(list(means.keys()), list(means.values()), "o-",
                label=f"p_f={p_f}", color=color)
    ax.set_xscale("log")
    ax.set_xlabel("n (graph size)")
    ax.set_ylabel("n_shells (max k-core number)")
    ax.set_title("NCP-1: Forest Fire shell depth vs graph size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ---------------------------------------------------------------------------
# Per-experiment runners
# ---------------------------------------------------------------------------


def run_nmh1(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
    n_meas: Optional[int] = None,
    a_list: Optional[List[float]] = None,
) -> List[NaturalCascadeRun]:
    print("\n=== NMH-1: hierarchical mixing-time scaling (slope-1 test) ===")
    # Default a=0.5 only: with p=2/b=0.042, larger a values (1.0, 2.0, 4.0) fail to
    # cascade because the gossip weight to d=1 cousins (~0.35) falls below the
    # saddle-crossing threshold (~0.36+).  a=0.5 is the Reg I/II boundary and
    # reliably cascades.  Pass --a-list explicitly for broader sweeps.
    if a_list is None:
        a_list = [0.5]
    kw: dict = {} if n_meas is None else {"n_meas": n_meas}
    configs = experiment_NMH1(seeds=list(range(n_seeds)), a_list=a_list, **kw)
    print(f"  {len(configs)} configs ({len(set(c.a for c in configs))} a-values × {n_seeds} seeds)")
    runs = [_load_or_run_nc(c, results_dir, force) for c in configs]

    _export_nc_flip_csv(runs, results_dir / "nmh1_flip.csv")
    _export_nc_summary_csv(runs, results_dir / "nmh1_summary.csv")

    n_no_flip = sum(1 for r in runs if r.warmup_ok and r.nucleation_leaf is None)
    n_failed = sum(1 for r in runs if not r.warmup_ok)
    print(f"  warmup failures: {n_failed}/{len(runs)}")
    print(f"  no natural flip detected: {n_no_flip}/{len(runs)}")

    if not no_plots:
        plot_slope1(
            runs, group_key="a",
            title="NMH-1: log₂(T_flip) vs hierarchical distance (slope-1 test)",
            out_path=plots_dir / "nmh1_slope1.png",
        )
    return runs


def run_nmh1b(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
    n_meas: Optional[int] = None,
    local_steps_list: Optional[List[int]] = None,
) -> List[NaturalCascadeRun]:
    print("\n=== NMH-1b: local_steps sensitivity ===")
    # Default omits 1000: 128 agents × 1000 local_steps × 600 rounds ≈ 13 hrs/seed.
    # Use --local-steps-list 10 50 200 1000 to include it explicitly.
    if local_steps_list is None:
        local_steps_list = [10, 50, 200]
    kw: dict = {} if n_meas is None else {"n_meas": n_meas}
    configs = experiment_NMH1b(seeds=list(range(n_seeds)), local_steps_list=local_steps_list, **kw)
    print(f"  {len(configs)} configs")
    runs = [_load_or_run_nc(c, results_dir, force) for c in configs]

    _export_nc_flip_csv(runs, results_dir / "nmh1b_flip.csv")
    _export_nc_summary_csv(runs, results_dir / "nmh1b_summary.csv")

    n_failed = sum(1 for r in runs if not r.warmup_ok)
    if n_failed:
        print(f"  [warn] {n_failed}/{len(runs)} warmup failures")

    if not no_plots:
        plot_slope1(
            runs, group_key="local_steps",
            title="NMH-1b: slope-1 accuracy vs local_steps (Regime C threshold)",
            out_path=plots_dir / "nmh1b_slope1.png",
        )
    return runs


def run_nmh3(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
    n_meas: Optional[int] = None,
) -> List[NaturalCascadeRun]:
    print("\n=== NMH-3: three-regime phase structure ===")
    kw = {} if n_meas is None else {"n_meas": n_meas}
    configs = experiment_NMH3(seeds=list(range(n_seeds)), **kw)
    print(f"  {len(configs)} configs ({len(set(c.a for c in configs))} a-values × {n_seeds} seeds)")
    runs = [_load_or_run_nc(c, results_dir, force) for c in configs]

    _export_nc_flip_csv(runs, results_dir / "nmh3_flip.csv")
    _export_nc_summary_csv(runs, results_dir / "nmh3_summary.csv")

    n_failed = sum(1 for r in runs if not r.warmup_ok)
    if n_failed:
        print(f"  [warn] {n_failed}/{len(runs)} warmup failures")

    if not no_plots:
        plot_phase_structure(
            runs,
            title="NMH-3: cascade depth vs a (regime boundaries at a=0.5 and a=8)",
            out_path=plots_dir / "nmh3_phase.png",
        )
    return runs


def run_ncp1(
    results_dir: Path,
    plots_dir: Path,
    no_plots: bool,
) -> List[dict]:
    """Graph-only: instantiate ForestFireTopology, record shell statistics."""
    print("\n=== NCP-1: Forest Fire shell structure (graph-only) ===")
    configs = experiment_NCP1_graph_configs()
    print(f"  {len(configs)} graph instances")

    stats_rows = []
    csv_path = results_dir / "ncp1_graph_stats.csv"

    # Load cached if it exists
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            stats_rows = list(reader)
        print(f"  [cache] {csv_path}")
    else:
        for cfg in configs:
            topo = ForestFireTopology(
                n=cfg.n_nodes, p_f=cfg.p_f, r=cfg.r,
                seed=cfg.seed, ensure_connected=True,
            )
            shells = topo.shell_assignment()
            n_shells = max(shells.values())
            shell_counts = {}
            for s in shells.values():
                shell_counts[s] = shell_counts.get(s, 0) + 1
            G, _ = topo.step(0)
            mean_degree = float(np.mean([d for _, d in G.degree()]))
            stats_rows.append({
                "name": cfg.name,
                "n_nodes": cfg.n_nodes,
                "p_f": cfg.p_f,
                "seed": cfg.seed,
                "n_shells": n_shells,
                "max_shell": max(shells.values()),
                "mean_degree": round(mean_degree, 3),
                "shell_sizes": json.dumps(shell_counts),
            })
            print(f"  {cfg.name}: n_shells={n_shells}, max_shell={max(shells.values())}, "
                  f"mean_degree={mean_degree:.2f}")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"  CSV: {csv_path}")

    if not no_plots and stats_rows:
        # Convert types for plotting
        plot_rows = [
            {
                "p_f": float(r["p_f"]),
                "n_nodes": int(r["n_nodes"]),
                "n_shells": int(r["n_shells"]),
            }
            for r in stats_rows
        ]
        plot_ncp1_shells(plot_rows, out_path=plots_dir / "ncp1_shells.png")

    return stats_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NMH + NCP Regime C timescale-separation experiments"
    )
    parser.add_argument(
        "--output-dir", default="analysis_output",
        help="Root directory for results and plots",
    )
    parser.add_argument(
        "--exps", nargs="+",
        choices=["NMH1", "NMH1b", "NMH3", "NCP1"],
        default=["NMH1", "NMH1b", "NMH3", "NCP1"],
        help="Which experiments to run (default: priority set from spec §14)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=50,
        help="Number of random seeds per config (default: 50 = full spec scale for NMH-1)",
    )
    parser.add_argument(
        "--n-meas", type=int, default=None,
        help="Override n_meas_rounds for NMH-1/1b/3 (default: factory default of 1000). "
             "Use 200 for force-flip pilots — cascades complete in ~32 rounds for depth=5.",
    )
    parser.add_argument(
        "--a-list", nargs="+", type=float, default=None,
        help="Override a-values for NMH-1 (default: [0.5]).  "
             "Example: --a-list 0.5 1.0 2.0 4.0",
    )
    parser.add_argument(
        "--local-steps-list", nargs="+", type=int, default=None,
        help="Override local_steps values for NMH-1b (default: [10, 50, 200]).  "
             "Example: --local-steps-list 10 50 200 1000",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "timesep_results"
    plots_dir = output_dir / "timesep_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir.resolve()}")
    print(f"Experiments: {args.exps}")
    print(f"Seeds: {args.n_seeds}")

    runners = {
        "NMH1":  lambda: run_nmh1(results_dir, plots_dir, args.force, args.n_seeds, args.no_plots, args.n_meas, args.a_list),
        "NMH1b": lambda: run_nmh1b(results_dir, plots_dir, args.force, args.n_seeds, args.no_plots, args.n_meas, args.local_steps_list),
        "NMH3":  lambda: run_nmh3(results_dir, plots_dir, args.force, args.n_seeds, args.no_plots, args.n_meas),
        "NCP1":  lambda: run_ncp1(results_dir, plots_dir, args.no_plots),
    }

    priority = ["NCP1", "NMH1", "NMH1b", "NMH3"]
    for exp in priority:
        if exp in args.exps:
            runners[exp]()

    print(f"\nDone.  Results: {results_dir}  Plots: {plots_dir}")


if __name__ == "__main__":
    main()
