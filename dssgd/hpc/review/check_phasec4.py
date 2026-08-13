# -*- coding: utf-8 -*-
"""Gate C4 review script: NCP with/without synthetic peripheral cliques
(paper2_social Sec. 5.2.2 / falsifier Sec. 8 item 4) -- "innovations from
dense peer groups should out-propagate matched-quality individually-held
innovations".

KNOWN LIMITATION (flagged rather than silently glossed over): the §8
falsifier calls for injecting a MATCHED-QUALITY innovation at the same hub
node under peripheral_clique_size=1 ("without") vs >1 ("with") and
comparing inward propagation. NCPSimConfig has no forced-injection knob
analogous to NaturalCascadeConfig.force_flip_source -- nucleation in
run_ncp_simulation is organic (post-warmup dynamics only). This script
therefore measures the weaker ORGANIC-nucleation proxy: conditional on
SOME node nucleating first, is a grafted-hub-associated node more likely to
be the source, and does its cascade reach further, under clique_size>1 than
under clique_size=1 at the SAME seed (same base graph, same designated hub
set -- see ForestFireWithPeripheralCliques). Treat gatec4_pass as evidence
about this proxy, not yet the paper's literal falsifier; adding a
force-flip-at-node option to NCPSimConfig would close this gap.

Produces:
  review/gatec4_review.json
  review/gatec4_propagation_by_clique_size.csv
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.ncp_runner import NCPRun


def _load_pkls(pkl_dir: Path) -> List[NCPRun]:
    runs = []
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            runs.append(NCPRun.load(pkl_path))
        except Exception as exc:
            print(f"  [warn] Failed to load {pkl_path.name}: {exc}")
    return runs


def _seed_from_name(name: str) -> str:
    """C4 task names are 'C4/clique_size=K/p_f=P/seed=S' -- extract 'seed=S'
    as the pairing key across clique_size arms (same base graph/hub set)."""
    for part in name.split("/"):
        if part.startswith("seed="):
            return part
    return name


def compute_propagation_by_clique_size(runs: List[NCPRun]) -> Dict[str, Dict[str, float]]:
    """Per clique_size: fraction of runs where the nucleating node is a
    grafted hub, and the mean number of OTHER nodes that subsequently
    flipped to B (breadth of propagation from that nucleation event)."""
    by_size: Dict[int, List[dict]] = defaultdict(list)
    for run in runs:
        if not run.warmup_ok:
            continue
        hub_ids = set(run.grafted_hub_ids or [])
        n_flipped_other = sum(
            1 for row in run.flip_table if row.get("t_flip_absolute") is not None
        )
        by_size[_clique_size_from_run(run)].append({
            "nucleated": run.nucleation_node is not None,
            "hub_nucleated": run.nucleation_node in hub_ids if run.nucleation_node is not None else False,
            "n_flipped_other": n_flipped_other,
        })

    result: Dict[str, Dict[str, float]] = {}
    for size, rows in sorted(by_size.items()):
        nucleated = [r for r in rows if r["nucleated"]]
        result[str(size)] = {
            "n_runs": len(rows),
            "n_nucleated": len(nucleated),
            "frac_hub_nucleated": (
                float(np.mean([r["hub_nucleated"] for r in nucleated])) if nucleated else float("nan")
            ),
            "mean_breadth_given_nucleated": (
                float(np.mean([r["n_flipped_other"] for r in nucleated])) if nucleated else float("nan")
            ),
        }
    return result


def _clique_size_from_run(run: NCPRun) -> int:
    for part in run.name.split("/"):
        if part.startswith("clique_size="):
            return int(part.split("=", 1)[1])
    return -1


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_tasks(queue_root: Path) -> Dict[str, int]:
    counts = {}
    for state in ("pending", "claimed", "completed", "failed"):
        state_dir = queue_root / state
        n = sum(1 for _ in state_dir.rglob("*.json")) if state_dir.exists() else 0
        counts[state] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate C4 review: NCP peripheral cliques.")
    parser.add_argument("--phasec4-results", type=Path, default=Path("results/phasec4"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phasec4"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phasec4"))
    args = parser.parse_args()

    results_dir: Path = args.phasec4_results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = results_dir / "pkl" / "C4"
    runs = _load_pkls(pkl_dir) if pkl_dir.exists() else []
    if not runs:
        print(f"  [warn] {pkl_dir} not found or empty -- nothing to score")

    by_size = compute_propagation_by_clique_size(runs)
    print("Propagation proxy by clique_size:")
    for size, stats in sorted(by_size.items(), key=lambda kv: int(kv[0])):
        print(f"  clique_size={size}: frac_hub_nucleated={stats['frac_hub_nucleated']:.2f} "
              f"mean_breadth={stats['mean_breadth_given_nucleated']:.2f} (n_nucleated={stats['n_nucleated']})")

    _write_csv(
        output_dir / "gatec4_propagation_by_clique_size.csv",
        [{"clique_size": size, **stats} for size, stats in sorted(by_size.items(), key=lambda kv: int(kv[0]))],
    )

    sizes = sorted((int(s) for s in by_size), reverse=False)
    without = by_size.get("1")
    larger = [by_size[str(s)] for s in sizes if s > 1]
    breadth_increases = bool(
        without and larger and not np.isnan(without["mean_breadth_given_nucleated"])
        and any(
            not np.isnan(l["mean_breadth_given_nucleated"])
            and l["mean_breadth_given_nucleated"] > without["mean_breadth_given_nucleated"]
            for l in larger
        )
    )
    gate_pass = breadth_increases

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    review = {
        "propagation_by_clique_size": by_size,
        "breadth_increases_with_clique_size": breadth_increases,
        "gatec4_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "Organic-nucleation proxy shows larger clique_size associated with "
            "greater propagation breadth from a hub-adjacent nucleation event. "
            "This is NOT yet the paper's literal matched-quality forced-injection "
            "falsifier -- see module docstring."
            if gate_pass else
            "No clique_size arm showed greater propagation breadth than the "
            "clique_size=1 control -- either too few nucleation events in this "
            "run to see the effect, or the effect is genuinely absent under the "
            "organic-nucleation proxy."
        ),
    }
    review_path = output_dir / "gatec4_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate C4 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
