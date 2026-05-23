"""Orchestrator: run all NMH experiments, generate and save all plots.

Usage (from the dssgd/ directory):
    python -m analysis.run_all
    python -m analysis.run_all --output-dir results/ --force
    python -m analysis.run_all --no-plots
    python -m analysis.run_all --n-seeds 5

Options:
    --output-dir  Directory for plots and pickled results (default: analysis_output)
    --force       Re-run even if a cached pickle already exists
    --no-plots    Save results only; skip rendering figures
    --n-seeds     Number of random seeds per configuration (default: 3)
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .experiments import (
    branching_sweep,
    depth_sweep,
    depth_sweep_trained,
    escape_kramers_sweep,
    kramers_sweep,
    leaf_size_kramers_sweep,
    p_sweep,
    temperature_sweep,
)
from .plots import (
    plot_coupling_ratio_vs_depth,
    plot_degree_vs_depth,
    plot_escape_dissolution_trace,
    plot_escape_kramers_sweep,
    plot_kramers_sweep,
    plot_level_convergence,
    plot_level_coupling_vs_theory,
    plot_neighbourhood_fraction,
    plot_stationary_temperature,
    plot_stationary_variance_per_level,
    plot_topological_dimension,
    plot_type_clustering_pca,
    plot_within_cross_type_distance,
)
from .results import AnalysisRun
from .simulation import NMHSimConfig, run_nmh_simulation
from .stats import make_averaged_run


# ---------------------------------------------------------------------------
# Load-or-run  (cache key includes n_rounds so stale caches are ignored)
# ---------------------------------------------------------------------------


def load_or_run(
    config: NMHSimConfig,
    cache_dir: Path,
    force: bool = False,
) -> AnalysisRun:
    safe_name = config.name.replace("/", "__")
    cache_path = cache_dir / f"{safe_name}_r{config.n_rounds}.pkl"
    if not force and cache_path.exists():
        print(f"  [cache] {cache_path.name}")
        return AnalysisRun.load(cache_path)
    print(f"  [run]   {config.name}")
    run = run_nmh_simulation(config)
    run.save(cache_path)
    return run


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------


def run_with_seeds(
    base_configs: List[NMHSimConfig],
    cache_dir: Path,
    force: bool = False,
    n_seeds: int = 3,
) -> Dict[str, List[AnalysisRun]]:
    """Run each config n_seeds times; return {base_name: [run_seed0, run_seed1, ...]}."""
    groups: Dict[str, List[AnalysisRun]] = {}
    for cfg in base_configs:
        seed_runs = []
        for s in range(n_seeds):
            seeded = dataclasses.replace(cfg, seed=s, name=f"{cfg.name}/s{s}")
            seed_runs.append(load_or_run(seeded, cache_dir, force))
        groups[cfg.name] = seed_runs
    return groups


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [plot]  {path.name}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(
    output_dir: Path = Path("analysis_output"),
    force: bool = False,
    make_plots: bool = True,
    n_seeds: int = 3,
) -> None:
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"

    # -------------------------------------------------------------------
    # 1. Depth sweep (untrained) — degree limit + convergence ordering
    # -------------------------------------------------------------------
    print("\n=== Depth sweep (untrained) ===")
    depth_groups = run_with_seeds(depth_sweep(), results_dir, force, n_seeds)
    depth_avg = [
        make_averaged_run(runs, name=bname)
        for bname, runs in depth_groups.items()
    ]

    if make_plots:
        _save(plot_degree_vs_depth(depth_avg), plots_dir / "degree_vs_depth.png")
        for bname, runs in depth_groups.items():
            safe = bname.replace("/", "__")
            avg = make_averaged_run(runs, name=bname)
            _save(plot_level_convergence(avg), plots_dir / f"convergence__{safe}.png")
            _save(plot_type_clustering_pca(runs[0]), plots_dir / f"pca__{safe}.png")

    # -------------------------------------------------------------------
    # 2. Depth sweep (trained) — coupling ratios + stationary variance
    # -------------------------------------------------------------------
    print("\n=== Depth sweep (trained) ===")
    trained_groups = run_with_seeds(depth_sweep_trained(), results_dir, force, n_seeds)
    trained_avg = [
        make_averaged_run(runs, name=bname)
        for bname, runs in trained_groups.items()
    ]

    if make_plots:
        _save(
            plot_coupling_ratio_vs_depth(trained_avg),
            plots_dir / "coupling_ratio_trained.png",
        )
        for bname, runs in trained_groups.items():
            safe = bname.replace("/", "__")
            avg = make_averaged_run(runs, name=bname)
            _save(
                plot_stationary_variance_per_level(avg),
                plots_dir / f"stationary_variance__{safe}.png",
            )
            _save(
                plot_level_coupling_vs_theory(avg),
                plots_dir / f"coupling_theory__{safe}.png",
            )

    # -------------------------------------------------------------------
    # 3. p sweep — topological dimension D ~ p
    # -------------------------------------------------------------------
    print("\n=== p sweep ===")
    p_groups = run_with_seeds(p_sweep(), results_dir, force, n_seeds)

    # Build {p: all runs (all depths x all seeds)} for the D-vs-p fit
    print("  (building depth x p grid for D fit)")
    p_to_all_runs: Dict[float, List[AnalysisRun]] = {}
    for pv in [1.0, 2.0, 4.0, 8.0]:
        all_runs: List[AnalysisRun] = []
        for _, runs in run_with_seeds(depth_sweep(p=pv), results_dir, force, n_seeds).items():
            all_runs.extend(runs)
        p_to_all_runs[pv] = all_runs

    if make_plots:
        for bname, runs in p_groups.items():
            safe = bname.replace("/", "__")
            avg = make_averaged_run(runs, name=bname)
            _save(
                plot_neighbourhood_fraction(avg),
                plots_dir / f"neighbour_fraction__{safe}.png",
            )
        _save(
            plot_topological_dimension(p_to_all_runs),
            plots_dir / "topological_dimension.png",
        )

    # -------------------------------------------------------------------
    # 4. Branching sweep
    # -------------------------------------------------------------------
    print("\n=== Branching sweep ===")
    b_groups = run_with_seeds(branching_sweep(), results_dir, force, n_seeds)

    if make_plots:
        for bname, runs in b_groups.items():
            safe = bname.replace("/", "__")
            avg = make_averaged_run(runs, name=bname)
            _save(
                plot_within_cross_type_distance(avg),
                plots_dir / f"within_cross__{safe}.png",
            )

    # -------------------------------------------------------------------
    # 5. Temperature sweep — stationary spread vs T_eff
    # -------------------------------------------------------------------
    print("\n=== Temperature sweep ===")
    temp_groups = run_with_seeds(temperature_sweep(), results_dir, force, n_seeds)

    if make_plots:
        temp_runs_by_lr = {
            float(bname.split("lr=")[1]): runs
            for bname, runs in temp_groups.items()
        }
        _save(
            plot_stationary_temperature(temp_runs_by_lr),
            plots_dir / "stationary_temperature.png",
        )

    # -------------------------------------------------------------------
    # 6. Kramers sweep — consensus-init runs for escape-time comparison
    # -------------------------------------------------------------------
    print("\n=== Kramers sweep (consensus init) ===")
    kramers_groups = run_with_seeds(kramers_sweep(), results_dir, force, n_seeds)

    if make_plots:
        kramers_runs_by_lr = {
            float(bname.split("lr=")[1]): runs
            for bname, runs in kramers_groups.items()
        }
        _save(
            plot_kramers_sweep(
                kramers_runs_by_lr,
                label_fn=lambda lr, runs: f"lr={lr} (T_eff={float(np.mean([r.lr*r.sigma2/2 for r in runs])):.4f})",
            ),
            plots_dir / "kramers_escape_times.png",
        )
        for bname, runs in kramers_groups.items():
            safe = bname.replace("/", "__")
            avg = make_averaged_run(runs, name=bname)
            _save(
                plot_level_coupling_vs_theory(avg),
                plots_dir / f"kramers_coupling__{safe}.png",
            )

    # -------------------------------------------------------------------
    # 7. Leaf-size Kramers sweep — peer-pressure hypothesis
    # -------------------------------------------------------------------
    print("\n=== Leaf-size Kramers sweep (peer pressure) ===")
    leafsize_groups = run_with_seeds(leaf_size_kramers_sweep(), results_dir, force, n_seeds)

    if make_plots:
        leafsize_runs_by_m = {
            int(bname.split("m=")[1]): runs
            for bname, runs in leafsize_groups.items()
        }
        _save(
            plot_kramers_sweep(
                leafsize_runs_by_m,
                title="Kramers escape times: peer-pressure sweep (leaf_size)",
                label_fn=lambda m, runs: f"m={m} (n={runs[0].branching**runs[0].depth*m})",
            ),
            plots_dir / "kramers_leafsize.png",
        )

    # -------------------------------------------------------------------
    # 8. Two-phase Kramers test — dissolution after removing gradient pinning
    # -------------------------------------------------------------------
    print("\n=== Two-phase Kramers sweep (escape) ===")
    escape_groups = run_with_seeds(escape_kramers_sweep(), results_dir, force, n_seeds)

    if make_plots:
        escape_runs_by_lr = {
            float(bname.split("lr=")[1]): runs
            for bname, runs in escape_groups.items()
        }
        _save(
            plot_escape_kramers_sweep(
                escape_runs_by_lr,
                label_fn=lambda lr, runs: (
                    f"lr={lr} (T_eff="
                    f"{float(np.mean([r.lr * r.sigma2 / 2 for r in runs])):.4f})"
                ),
            ),
            plots_dir / "escape_kramers.png",
        )
        for bname, runs in escape_groups.items():
            safe = bname.replace("/", "__")
            avg = make_averaged_run(runs, name=bname)
            _save(
                plot_escape_dissolution_trace(avg),
                plots_dir / f"escape_trace__{safe}.png",
            )

    print(f"\nDone.  Results -> {results_dir.resolve()}")
    if make_plots:
        print(f"Plots  -> {plots_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--n-seeds", default=3, type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    run_all(
        output_dir=args.output_dir,
        force=args.force,
        make_plots=not args.no_plots,
        n_seeds=args.n_seeds,
    )
