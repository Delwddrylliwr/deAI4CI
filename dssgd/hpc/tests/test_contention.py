"""Multi-process shard contention test (verification step 4 from plan).

Spins up N_WORKERS worker processes all pointing at the same small queue
and verifies:
  - Every task is completed exactly once (no double-execution)
  - No tasks left in claimed/ (no lost tasks)
  - JSONL output has exactly N_TASKS unique task_ids
"""
from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
from pathlib import Path

# Allow running as `python -m hpc.tests.test_contention` from dssgd/ directory.
_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.ncp_experiments import experiment_NCP1_graph_configs
from hpc.generate_queue import write_queue
from hpc.serialization import config_to_dict, task_id_from_config
from hpc.worker import worker_main


def _run_worker(args):
    queue_root, results_dir, ckpt_dir, worker_id, shard_id, n_shards = args
    worker_main(
        queue_root=Path(queue_root),
        results_dir=Path(results_dir),
        checkpoint_dir=Path(ckpt_dir),
        worker_id=worker_id,
        shard_id=shard_id,
        n_shards=n_shards,
        checkpoint_every=50,
        flush_every=5,
        max_empty_retries=2,
        retry_sleep=0.1,
    )


N_WORKERS = 4
N_SHARDS = 2


def main():
    configs = experiment_NCP1_graph_configs(
        p_f_list=[0.37], n_list=[100], n_instances=5
    )
    tasks = []
    for cfg in configs:
        tid = task_id_from_config(cfg)
        tasks.append({
            "task_id": tid,
            "experiment": "NCP1",
            "phase": 1,
            "sim_type": "ncp_graph_only",
            "config": config_to_dict(cfg),
            "checkpoint_every": 0,
            "estimated_hours": 0.01,
            "result_pkl_path": None,
        })
    n_tasks = len(tasks)

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_root = Path(tmpdir) / "queue"
        results_dir = Path(tmpdir) / "results"
        ckpt_dir = results_dir / "checkpoints"
        results_dir.mkdir()
        ckpt_dir.mkdir()

        write_queue(tasks, queue_root, n_shards=N_SHARDS)
        print(f"Generated {n_tasks} tasks in {N_SHARDS} shards")

        pool_args = [
            (str(queue_root), str(results_dir), str(ckpt_dir),
             wid, wid % N_SHARDS, N_SHARDS)
            for wid in range(N_WORKERS)
        ]
        with multiprocessing.Pool(N_WORKERS) as pool:
            pool.map(_run_worker, pool_args)

        completed = list((queue_root / "completed").rglob("*.json"))
        failed = list((queue_root / "failed").rglob("*.json"))
        pending = list((queue_root / "pending").rglob("*.json"))
        claimed = list((queue_root / "claimed").rglob("*.json"))
        print(f"completed={len(completed)}, failed={len(failed)}, "
              f"pending={len(pending)}, claimed={len(claimed)}")

        assert len(completed) == n_tasks, (
            f"Expected {n_tasks} completed, got {len(completed)}"
        )
        assert len(failed) == 0, f"Unexpected failures"
        assert len(pending) == 0 and len(claimed) == 0

        all_lines = []
        for jl in results_dir.glob("worker_*.jsonl"):
            for line in jl.read_text().splitlines():
                if line.strip():
                    all_lines.append(json.loads(line))
        ids_seen = [r["task"]["task_id"] for r in all_lines]
        assert len(ids_seen) == len(set(ids_seen)) == n_tasks, (
            f"{len(ids_seen)} lines, {len(set(ids_seen))} unique, {n_tasks} tasks"
        )
        print(f"Multi-process contention test: OK "
              f"({n_tasks} tasks, {N_WORKERS} workers, {N_SHARDS} shards)")


if __name__ == "__main__":
    main()
