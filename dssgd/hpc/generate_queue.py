"""Generate sharded task queue for HPC SLURM array jobs.

Usage:
  python -m hpc.generate_queue --phase 1 --queue-dir queue/phase1
  python -m hpc.generate_queue --phase 2 --queue-dir queue/phase2 \\
      --gate1-results review/gate1_review.json
  python -m hpc.generate_queue --phase 3 --queue-dir queue/phase3 \\
      --gate1-results review/gate1_review.json \\
      --gate2-results review/gate2_review.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as `python -m hpc.generate_queue` from dssgd/ directory.
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.natural_cascade_experiments import (
    experiment_NMH1,
    experiment_NMH1b,
    experiment_NMH2,
    experiment_NMH3,
    experiment_NMH4,
    experiment_NMH5,
    experiment_NMH6,
    experiment_NMH7,
)
from analysis.ncp_experiments import (
    experiment_NCP1_graph_configs,
    experiment_NCP2,
    experiment_NCP3,
    experiment_NCP4,
    experiment_NCP5,
)
from dssgd.topology.forest_fire import ForestFireTopology
from hpc.serialization import config_to_dict, task_id_from_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SHARDS = 100
CHECKPOINT_EVERY_DEFAULT = 50

# Estimated wall-clock hours per task (used for duration-balanced shard assignment).
EXPERIMENT_HOURS: Dict[str, float] = {
    "NMH1": 0.5,
    "NMH1b": 0.5,
    "NMH2": 0.6,
    "NMH2b": 0.6,
    "NMH3": 0.5,
    "NMH4_pilot": 4.6,
    "NMH4": 9.3,
    "NMH5": 0.5,
    "NMH6": 1.2,
    "NMH7_pilot": 1.0,
    "NMH7": 2.0,
    "NCP1": 0.02,
    "NCP2": 0.8,
    "NCP3": 0.8,
    "NCP4": 0.8,
    "NCP5": 68.0,
    "Comparative": 0.5,
}

# ---------------------------------------------------------------------------
# Task-dict construction helpers
# ---------------------------------------------------------------------------


def _nc_task(
    config,
    experiment: str,
    phase: int,
    results_root: Path,
    checkpoint_every: int = CHECKPOINT_EVERY_DEFAULT,
) -> Dict[str, Any]:
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "natural_cascade",
        "config": config_to_dict(config),
        "checkpoint_every": checkpoint_every,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _ncp_task(
    config,
    experiment: str,
    phase: int,
    results_root: Path,
    checkpoint_every: int = CHECKPOINT_EVERY_DEFAULT,
) -> Dict[str, Any]:
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "ncp_simulation",
        "config": config_to_dict(config),
        "checkpoint_every": checkpoint_every,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _ncp1_task(config, phase: int) -> Dict[str, Any]:
    """NCP-1 is graph-structure only — no simulation, no checkpoint, no pkl."""
    task_id = task_id_from_config(config)
    return {
        "task_id": task_id,
        "experiment": "NCP1",
        "phase": phase,
        "sim_type": "ncp_graph_only",
        "config": config_to_dict(config),
        "checkpoint_every": 0,
        "estimated_hours": EXPERIMENT_HOURS["NCP1"],
        "result_pkl_path": None,
    }


# ---------------------------------------------------------------------------
# Phase task builders
# ---------------------------------------------------------------------------


def _build_ncp2_tasks_with_clamping(
    seeds: List[int],
    phase: int,
    results_root: Path,
    n_nodes: int = 1000,
    p_f: float = 0.37,
) -> List[Dict[str, Any]]:
    """Build NCP-2 tasks with clamped_shell baked in from a reference graph.

    We instantiate one ForestFireTopology per (n_nodes, p_f, seed) to discover
    the actual innermost (max k-core) and outermost (min k-core) shell ids, then
    create two tasks per seed: one clamping the innermost shell (outward cascade)
    and one clamping the outermost shell (inward cascade).
    """
    tasks = []
    for seed in seeds:
        topo = ForestFireTopology(n=n_nodes, p_f=p_f, seed=seed)
        shells = topo.shell_assignment()
        shell_vals = list(shells.values())
        innermost = max(shell_vals)   # highest k-core number = densest core
        outermost = min(shell_vals)

        for direction, clamped in [("outward", innermost), ("inward", outermost)]:
            configs = experiment_NCP2(
                seeds=[seed],
                n_nodes=n_nodes,
                p_f=p_f,
                clamped_shell=clamped,
            )
            for cfg in configs:
                # Rename to include direction for distinguishable task_ids.
                cfg.name = cfg.name.replace(
                    f"clamp={clamped}", f"dir={direction}/clamp={clamped}"
                )
                tasks.append(_ncp_task(cfg, "NCP2", phase, results_root))
    return tasks


def build_phase_tasks(
    phase: int,
    gate_results: Dict[str, Any],
    results_root: Path,
    n_seeds_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return all task dicts for a given phase.

    gate_results is a merged dict from load_gate_results(); keys are prefixed
    with "gate1_", "gate2_", "gate3_" matching which review file they came from.

    n_seeds_override: if set, caps the number of seeds per config (useful for
    smoke-testing the queue generation without generating thousands of tasks).
    """
    tasks: List[Dict[str, Any]] = []

    def _seeds(default_count: int) -> List[int]:
        n = n_seeds_override if n_seeds_override is not None else default_count
        return list(range(n))

    if phase == 1:
        # NMH-1 extended (30 seeds, 4 a-values)
        for cfg in experiment_NMH1(seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "NMH1", phase, results_root))

        # NMH-1b extended (15 seeds × 4 local_steps values)
        for cfg in experiment_NMH1b(seeds=_seeds(15)):
            tasks.append(_nc_task(cfg, "NMH1b", phase, results_root))

        # NMH-3 extended (25 seeds × 8 a-values)
        for cfg in experiment_NMH3(seeds=_seeds(25)):
            tasks.append(_nc_task(cfg, "NMH3", phase, results_root))

        # NCP-1 (graph-structure only, 240 configs)
        for cfg in experiment_NCP1_graph_configs():
            tasks.append(_ncp1_task(cfg, phase))

        # NMH-7 pilot (5 seeds, depth=4)
        for cfg in experiment_NMH7(seeds=_seeds(5), depth=4, n_meas=2000):
            tasks.append(_nc_task(cfg, "NMH7_pilot", phase, results_root))

    elif phase == 2:
        b_list = gate_results.get(
            "gate1_recommended_b_values",
            [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050],
        )

        # NMH-2 (30 seeds × len(b_list) b-values)
        for cfg in experiment_NMH2(b_list=b_list, seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "NMH2", phase, results_root))

        # NMH-2b: same b sweep, different seeds for replication
        for cfg in experiment_NMH2(b_list=b_list, seeds=list(range(30, 30 + len(_seeds(30))))):
            # Rename to NMH2b to produce distinct task_ids and pkl paths.
            cfg.name = cfg.name.replace("NMH2/", "NMH2b/")
            tasks.append(_nc_task(cfg, "NMH2b", phase, results_root))

        # NMH-6 pilot (10 seeds)
        for cfg in experiment_NMH6(seeds=_seeds(10)):
            tasks.append(_nc_task(cfg, "NMH6", phase, results_root))

        # NCP-2 pilot (10 seeds, outward + inward directions)
        ncp2_seeds = _seeds(10)
        tasks.extend(_build_ncp2_tasks_with_clamping(ncp2_seeds, phase, results_root))

    elif phase == 3:
        b_on_a_list = gate_results.get(
            "gate2_recommended_b_values_phase3",
            [0.02, 0.03, 0.04, 0.05, 0.06],
        )

        # NMH-5 (50 seeds × 3 a-values × 5 b/a values)
        for cfg in experiment_NMH5(b_on_a_list=b_on_a_list, seeds=_seeds(50)):
            tasks.append(_nc_task(cfg, "NMH5", phase, results_root))

        # NMH-7 full (15 seeds, depth=4, n_meas=4000)
        for cfg in experiment_NMH7(seeds=_seeds(15), depth=4, n_meas=4000):
            tasks.append(_nc_task(cfg, "NMH7", phase, results_root))

        # NCP-3 (10 seeds)
        for cfg in experiment_NCP3(seeds=_seeds(10)):
            tasks.append(_ncp_task(cfg, "NCP3", phase, results_root))

        # NMH-4 pilot (200 seeds, depth=5)
        for cfg in experiment_NMH4(seeds=_seeds(200), depth=5, n_meas=500):
            tasks.append(_nc_task(cfg, "NMH4_pilot", phase, results_root))

    elif phase == 4:
        phase4_mods = gate_results.get("gate3_phase4_modifications", "")
        depth = 7 if "depth=7" in phase4_mods else 6

        # NMH-4 full (500 seeds × 3 a-values)
        for a in [1.0, 2.0, 4.0]:
            for cfg in experiment_NMH4(seeds=_seeds(500), depth=depth, a=a):
                tasks.append(_nc_task(cfg, "NMH4", phase, results_root))

        # NCP-4 (200 seeds)
        for cfg in experiment_NCP4(seeds=_seeds(200)):
            tasks.append(_ncp_task(cfg, "NCP4", phase, results_root))

        # NCP-5 (10 seeds; 68 hrs each — checkpointing critical)
        for cfg in experiment_NCP5(seeds=_seeds(10)):
            tasks.append(_ncp_task(cfg, "NCP5", phase, results_root,
                                    checkpoint_every=CHECKPOINT_EVERY_DEFAULT))

        # Comparative (100 seeds, NMH operating point)
        for cfg in experiment_NMH1(a_list=[0.5], seeds=_seeds(100)):
            cfg.name = cfg.name.replace("NMH1/", "Comparative/")
            tasks.append(_nc_task(cfg, "Comparative", phase, results_root))

    else:
        raise ValueError(f"Unknown phase: {phase}. Valid phases are 1–4.")

    return tasks


# ---------------------------------------------------------------------------
# Queue writing
# ---------------------------------------------------------------------------


def write_queue(
    tasks: List[Dict[str, Any]],
    queue_root: Path,
    n_shards: int = N_SHARDS,
    overwrite: bool = False,
) -> None:
    """Distribute tasks into pending shards and write one JSON file per task.

    Tasks are sorted by descending estimated_hours before shard assignment so
    each shard gets a mix of long and short tasks (prevents one shard holding
    all long tasks while others drain quickly).

    Directory structure created under queue_root/:
      pending/shard_000/ ... shard_NNN/
      claimed/shard_000/ ...
      completed/shard_000/ ...
      failed/shard_000/ ...
    """
    pending_root = queue_root / "pending"
    if pending_root.exists() and not overwrite:
        raise FileExistsError(
            f"{pending_root} already exists. Use --overwrite to regenerate."
        )

    # Sort by descending duration for balanced shard loading.
    tasks_sorted = sorted(tasks, key=lambda t: t["estimated_hours"], reverse=True)

    for subdir in ("pending", "claimed", "completed", "failed"):
        for shard_idx in range(n_shards):
            (queue_root / subdir / f"shard_{shard_idx:03d}").mkdir(
                parents=True, exist_ok=True
            )

    for rank, task in enumerate(tasks_sorted):
        shard_idx = rank % n_shards
        task_path = (
            queue_root / "pending" / f"shard_{shard_idx:03d}" / f"{task['task_id']}.json"
        )
        with open(task_path, "w") as fh:
            json.dump(task, fh)

    print(
        f"Wrote {len(tasks)} tasks across {n_shards} shards in {queue_root}/pending/"
    )


# ---------------------------------------------------------------------------
# Gate result loading
# ---------------------------------------------------------------------------


def load_gate_results(*paths: str) -> Dict[str, Any]:
    """Load and merge gate review JSON files.

    Each file's keys are prefixed with 'gate{N}_' where N is inferred from the
    presence of 'gate1', 'gate2', 'gate3' in the filename.  Falls back to the
    index order if the filename contains no such marker.
    """
    merged: Dict[str, Any] = {}
    for i, path in enumerate(paths, start=1):
        with open(path) as fh:
            data = json.load(fh)
        # Infer gate number from filename, e.g. "gate1_review.json" → prefix "gate1_"
        fname = Path(path).name.lower()
        for n in (1, 2, 3):
            if f"gate{n}" in fname:
                prefix = f"gate{n}_"
                break
        else:
            prefix = f"gate{i}_"
        for k, v in data.items():
            merged[f"{prefix}{k}"] = v
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sharded HPC task queue for timesep experiments."
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--queue-dir", type=Path, default=None,
        help="Root directory for the queue. Defaults to queue/phase<N>/.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Root directory for result pickles. Defaults to results/phase<N>/.",
    )
    parser.add_argument("--n-shards", type=int, default=N_SHARDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        help="Override seed count per config (useful for smoke tests).",
    )
    parser.add_argument("--gate1-results", type=str, default=None)
    parser.add_argument("--gate2-results", type=str, default=None)
    parser.add_argument("--gate3-results", type=str, default=None)
    args = parser.parse_args()

    queue_dir = args.queue_dir or Path(f"queue/phase{args.phase}")
    results_dir = args.results_dir or Path(f"results/phase{args.phase}")

    gate_paths = [
        p for p in [args.gate1_results, args.gate2_results, args.gate3_results]
        if p is not None
    ]
    gate_results = load_gate_results(*gate_paths) if gate_paths else {}

    tasks = build_phase_tasks(
        phase=args.phase,
        gate_results=gate_results,
        results_root=results_dir,
        n_seeds_override=args.n_seeds,
    )
    write_queue(tasks, queue_dir, n_shards=args.n_shards, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
