"""Orchestrator for NMH catch-up dynamics experiments.

Run all experiments (or a subset) and save results and plots:

    python -m analysis.run_catchup [--output-dir DIR] [--exps A B C D E]
                                   [--n-seeds N] [--force]

Experiments run in priority order A → D → B → E → C.
Experiment C is skipped by default (very long runs: 10⁴ measurement rounds).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .catchup import CatchupRun, CatchupSimConfig, compute_direct_edge_leaves, run_catchup_simulation, run_sanity_checks
from .catchup_experiments import experiment_A, experiment_B, experiment_C, experiment_D, experiment_E
from .catchup_plots import (
    compute_autocorrelations,
    plot_catchup_curve,
    plot_decomposition_autocorr,
    plot_heterogeneity_comparison,
    plot_iso_gamma_mp,
    plot_temperature_crossover,
    summarise_catchup,
)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _run_key(config: CatchupSimConfig) -> str:
    return (
        f"{config.name.replace('/', '__')}"
        f"__w{config.n_warmup}_m{config.n_meas_rounds}"
        f"__s{config.seed}.pkl"
    )


def _load_or_run(
    config: CatchupSimConfig,
    results_dir: Path,
    force: bool,
) -> CatchupRun:
    cache_path = results_dir / _run_key(config)
    if cache_path.exists() and not force:
        print(f"  [cache] {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  [run]   {config.name}")
    run = run_catchup_simulation(config)
    run.save(cache_path)
    return run


# ---------------------------------------------------------------------------
# Per-experiment runners
# ---------------------------------------------------------------------------


def run_experiment_A(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
) -> List[CatchupRun]:
    print("\n=== Experiment A: catch-up curve ===")
    configs = experiment_A(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]

    # Enrich t50_table rows with has_direct_edge for pickles that predate this field
    for run in runs:
        if any("has_direct_edge" not in row for row in run.t50_table):
            del_leaves = compute_direct_edge_leaves(run)
            for row in run.t50_table:
                if "has_direct_edge" not in row:
                    row["has_direct_edge"] = row["target_leaf"] in del_leaves

    fig = plot_catchup_curve(runs)
    fig.savefig(plots_dir / "exp_A_catchup_curve.png", dpi=150)
    plt.close(fig)

    # CSV summary
    import csv
    rows = summarise_catchup(runs)
    csv_path = results_dir / "exp_A_t50_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved CSV: {csv_path}")

    print(f"  Saved plot: {plots_dir / 'exp_A_catchup_curve.png'}")
    return runs


def run_experiment_D(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
) -> List[CatchupRun]:
    print("\n=== Experiment D: decomposition autocorrelations ===")
    configs = experiment_D(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]
    for run in runs:
        if run.agent_traj is None:
            print(f"  [skip] {run.name}: agent_traj not recorded")
            continue
        fig = plot_decomposition_autocorr(run)
        fname = f"exp_D_autocorr_seed{run.seed}.png"
        fig.savefig(plots_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved plot: {plots_dir / fname}")
    return runs


def run_experiment_B(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
) -> List[CatchupRun]:
    print("\n=== Experiment B: iso-γmp sweep ===")
    configs = experiment_B(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]
    fig = plot_iso_gamma_mp(runs)
    fig.savefig(plots_dir / "exp_B_iso_gamma_mp.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plots_dir / 'exp_B_iso_gamma_mp.png'}")
    return runs


def run_experiment_E(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
) -> List[CatchupRun]:
    print("\n=== Experiment E: heterogeneity sensitivity ===")
    configs = experiment_E(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]
    runs_by_mode: Dict[str, List[CatchupRun]] = {}
    for run in runs:
        # Extract mode from name: "exp_E/mode=hierarchical/seed=0" → "hierarchical"
        parts = {kv.split("=")[0]: kv.split("=")[1] for kv in run.name.split("/")[1:] if "=" in kv}
        mode = parts.get("mode", "unknown")
        runs_by_mode.setdefault(mode, []).append(run)
    fig = plot_heterogeneity_comparison(runs_by_mode)
    fig.savefig(plots_dir / "exp_E_heterogeneity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plots_dir / 'exp_E_heterogeneity.png'}")
    return runs


def run_experiment_C(
    results_dir: Path,
    plots_dir: Path,
    force: bool,
    n_seeds: int,
) -> List[CatchupRun]:
    print("\n=== Experiment C: temperature crossover (long runs) ===")
    configs = experiment_C(seeds=list(range(n_seeds)))
    runs = [_load_or_run(c, results_dir, force) for c in configs]
    fig = plot_temperature_crossover(runs)
    fig.savefig(plots_dir / "exp_C_temperature_crossover.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plots_dir / 'exp_C_temperature_crossover.png'}")
    return runs


# ---------------------------------------------------------------------------
# Sanity-check runner
# ---------------------------------------------------------------------------


def _run_sanity(depth: int = 5, leaf_size: int = 4, p: float = 4.0) -> None:
    print("\n=== Sanity checks ===")
    try:
        run_sanity_checks(
            branching=2,
            depth=depth,
            leaf_size=leaf_size,
            p=p,
        )
        print("  All sanity checks passed.")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NMH catch-up dynamics experiments")
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Root directory for results and plots",
    )
    parser.add_argument(
        "--exps",
        nargs="+",
        choices=["A", "B", "C", "D", "E"],
        default=["A", "D", "B", "E"],
        help="Which experiments to run (default: A D B E; C omitted due to long runtime)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds per configuration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cached results exist",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip sanity checks before running experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "catchup_results"
    plots_dir = output_dir / "catchup_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sanity:
        _run_sanity()

    runners = {
        "A": run_experiment_A,
        "D": run_experiment_D,
        "B": run_experiment_B,
        "E": run_experiment_E,
        "C": run_experiment_C,
    }

    # Run in priority order: A first, C last
    priority_order = ["A", "D", "B", "E", "C"]
    for exp in priority_order:
        if exp in args.exps:
            runners[exp](results_dir, plots_dir, args.force, args.n_seeds)

    print("\nDone.  Results in", results_dir, "— plots in", plots_dir)


if __name__ == "__main__":
    main()
