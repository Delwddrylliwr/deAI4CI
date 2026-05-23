"""Orchestrator for NMH active-escape dynamics experiments (v2).

Run all experiments (or a subset) and save results and plots:

    python -m analysis.run_active_escape [--output-dir DIR]
                                         [--exps A1 A2 A3 B]
                                         [--n-seeds N] [--force]
                                         [--no-plots] [--skip-sanity]

Experiments run in priority order A1 → A2 → A3 → B.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .active_escape import (
    ActiveEscapeRun,
    ActiveEscapeSimConfig,
    run_active_escape_simulation,
    verify_bistable_loss,
)
from .active_escape_experiments import (
    experiment_A1,
    experiment_A1_natural,
    experiment_A2,
    experiment_A3,
    experiment_B_active,
)
from .active_escape_plots import (
    plot_a1_flip_times,
    plot_a2_regime_sweep,
    plot_a3_phase_transition,
    plot_b_iso_gmp,
)
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _run_key(config: ActiveEscapeSimConfig) -> str:
    return (
        f"{config.name.replace('/', '__')}"
        f"__w{config.n_warmup}_m{config.n_meas_rounds}"
        f"__s{config.seed}.pkl"
    )


def _load_or_run(
    config: ActiveEscapeSimConfig,
    results_dir: Path,
    force: bool,
) -> ActiveEscapeRun:
    cache_path = results_dir / _run_key(config)
    if cache_path.exists() and not force:
        print(f"  [cache] {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  [run]   {config.name}")
    run = run_active_escape_simulation(config)
    run.save(cache_path)
    if not run.warmup_ok:
        print(f"  [warn]  warmup FAILED for {config.name} — excluded from analysis")
    return run


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _export_flip_csv(runs: List[ActiveEscapeRun], csv_path: Path) -> None:
    """Write flip_table rows to CSV, skipping runs where warmup_ok=False."""
    rows = []
    for run in runs:
        if not run.warmup_ok:
            continue
        for row in run.flip_table:
            rows.append({
                **row,
                "depth": run.depth,
                "leaf_size": run.leaf_size,
                "p": run.p,
                "d_param": run.d_param,
            })
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Per-experiment runners
# ---------------------------------------------------------------------------


def run_experiment_A1(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
) -> List[ActiveEscapeRun]:
    print("\n=== Experiment A1: active escape — slope-1 test ===")
    configs = experiment_A1(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]

    _export_flip_csv(runs, results_dir / "exp_A1_flip_summary.csv")

    if not no_plots:
        fig = plot_a1_flip_times(runs)
        fpath = plots_dir / "exp_A1_flip_times.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {fpath}")

    n_failed = sum(1 for r in runs if not r.warmup_ok)
    if n_failed:
        print(f"  [warn] {n_failed}/{len(runs)} runs had warmup failures")

    return runs


def run_experiment_A1_natural(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
) -> List[ActiveEscapeRun]:
    print("\n=== Experiment A1nat: natural cascade (no clamping) ===")
    configs = experiment_A1_natural(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]

    _export_flip_csv(runs, results_dir / "exp_A1nat_flip_summary.csv")

    if not no_plots:
        fig = plot_a1_flip_times(runs)
        fpath = plots_dir / "exp_A1nat_flip_times.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {fpath}")

    n_failed = sum(1 for r in runs if not r.warmup_ok)
    if n_failed:
        print(f"  [warn] {n_failed}/{len(runs)} runs had warmup failures")

    return runs


def run_experiment_A2(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
) -> List[ActiveEscapeRun]:
    print("\n=== Experiment A2: regime sweep (locate ell_c) ===")
    configs = experiment_A2(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]

    _export_flip_csv(runs, results_dir / "exp_A2_flip_summary.csv")

    if not no_plots:
        fig = plot_a2_regime_sweep(runs)
        fpath = plots_dir / "exp_A2_regime_sweep.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {fpath}")

    return runs


def run_experiment_A3(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
) -> List[ActiveEscapeRun]:
    print("\n=== Experiment A3: containment phase transition ===")
    configs = experiment_A3(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]

    _export_flip_csv(runs, results_dir / "exp_A3_flip_summary.csv")

    if not no_plots:
        fig = plot_a3_phase_transition(runs)
        fpath = plots_dir / "exp_A3_phase_transition.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {fpath}")

    return runs


def run_experiment_B(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
    no_plots: bool,
) -> List[ActiveEscapeRun]:
    print("\n=== Experiment B: iso-γmp collapse check ===")
    configs = experiment_B_active(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]

    _export_flip_csv(runs, results_dir / "exp_B_active_flip_summary.csv")

    if not no_plots:
        fig = plot_b_iso_gmp(runs)
        fpath = plots_dir / "exp_B_active_iso_gmp.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {fpath}")

    return runs


# ---------------------------------------------------------------------------
# Sanity-check runner
# ---------------------------------------------------------------------------


def _run_sanity() -> None:
    print("\n=== Active-escape sanity checks ===")
    import torch
    import numpy as np
    from .active_escape import (
        make_bistable_loss_fn, verify_bistable_loss,
        basin_label, find_t_flip,
    )
    from . import theory

    # 1. Bistable loss verification
    theta_A = torch.tensor([0.0])
    theta_B = torch.tensor([1.0])
    try:
        verify_bistable_loss(theta_A, theta_B, a=0.5, b=0.01)
        print("  [ok] bistable loss landscape is correctly structured")
    except ValueError as e:
        print(f"  [FAIL] bistable loss: {e}")
        sys.exit(1)

    # 2. Regime classification at default parameters (p=2 sparse Safari regime)
    # With p=2: γ_1=2, γ_5=0.125; Regime I for a<0.5, Regime III for a≥8
    regime = theory.classify_regime(0.3, 1.0, 4, 2.0, 5)
    assert regime == 'I', f"Expected Regime I for a=0.3 with p=2, got {regime}"
    regime_iii = theory.classify_regime(20.0, 1.0, 4, 2.0, 5)
    assert regime_iii == 'III', f"Expected Regime III for a=20.0 with p=2, got {regime_iii}"
    print("  [ok] regime classification correct")

    # 3. Basin label at canonical points
    A_np = np.array([0.0])
    B_np = np.array([1.0])
    assert basin_label(np.array([0.05]), A_np, B_np) == 'A'
    assert basin_label(np.array([0.95]), A_np, B_np) == 'B'
    assert basin_label(np.array([0.50]), A_np, B_np) == 'X'
    print("  [ok] basin labeling correct")

    print("  All sanity checks passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NMH active-escape dynamics experiments (v2)"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Root directory for results and plots",
    )
    parser.add_argument(
        "--exps",
        nargs="+",
        choices=["A1", "A1nat", "A2", "A3", "B"],
        default=["A1", "A1nat", "A2", "A3", "B"],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds per configuration",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity checks")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "active_escape_results"
    plots_dir = output_dir / "active_escape_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sanity:
        _run_sanity()

    runners = {
        "A1":    run_experiment_A1,
        "A1nat": run_experiment_A1_natural,
        "A2":    run_experiment_A2,
        "A3":    run_experiment_A3,
        "B":     run_experiment_B,
    }

    priority_order = ["A1", "A1nat", "A2", "A3", "B"]
    for exp in priority_order:
        if exp in args.exps:
            runners[exp](results_dir, plots_dir, args.force, args.n_seeds, args.no_plots)

    print("\nDone.  Results in", results_dir, "— plots in", plots_dir)


if __name__ == "__main__":
    main()
