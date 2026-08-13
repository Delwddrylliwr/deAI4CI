# -*- coding: utf-8 -*-
"""Gate H1 review script: E0, the chord-threshold calibration falsifier
(paper1_computing_hybrid_gossip.md Annex B.1's E0; Annex B.6 point 1: "a
cheap up-front falsifier ... on which most of Sec. 3 depends").

Checks the complementarity identity (2.7a),

    vartheta_{A->B}(lambda) + vartheta_{B->A}(lambda) == 1

across E0's (a, b) grid, using the two thresholds as INDEPENDENTLY computed
by theory.chord_geometry (not derived from each other) -- this is a genuine
test of Lemma 2.1'/H-chord, not a tautology. If it fails at large lambda
(the chord meeting more than one basin boundary, per B.6's own stated
risk), Lemma 2.4's clique-round projection and B.0.1's round-boundary-
sufficiency claim are only valid in whatever lambda range still passes, and
every H-phase experiment downstream (starting with H2's E7/E14, which
consume the vartheta_dagger(lambda) table this script produces) must be
confined to that range or not run at all.

Produces:
  review/phaseh1/gateh1_review.json   -- gateh1_pass, gateh1_vartheta_dagger_table
                                          (consumed by generate_queue.py --phase h2)
  review/phaseh1/gateh1_e0_complementarity.csv

Usage:
  python -m hpc.review.check_phaseh1 \\
      --phaseh1-results results/phaseh1 \\
      --queue-dir queue/phaseh1 \\
      --output-dir review/phaseh1
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Complementarity should hold near-exactly: E0's default grid uses r=1.0,
# the exact-cubic branch of theory.chord_geometry (theory.py's
# _saddle_node_roots), not the numeric root-finder used for r!=1. A looser
# tolerance would silently paper over a real regression in that solver.
COMPLEMENTARITY_TOLERANCE = 1e-6


def _load_e0_records(jsonl_paths: List[Path]) -> List[Dict[str, Any]]:
    """Parse worker JSONL summaries for E0 (chord_threshold_calibration) tasks."""
    records: List[Dict[str, Any]] = []
    for path in jsonl_paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            task = rec.get("task", {})
            if task.get("experiment") != "E0":
                continue
            result = rec.get("result", {})
            if not result:
                continue
            records.append(result)
    return records


def compute_vartheta_dagger_table(records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """{lambda, a, vartheta_dagger} rows, sorted by lambda -- the measured
    (not fitted) input E7/E14 (phase h2) consume in place of recomputing
    theory.chord_geometry themselves, exactly the role gate1's
    nmh3_regime_boundaries plays for Phase 7's a_list today."""
    rows = [
        {"lambda": rec["lambda"], "a": rec["a"], "b": rec["b"],
         "vartheta_dagger": rec["vartheta_A_to_B"]}
        for rec in records
    ]
    return sorted(rows, key=lambda r: r["lambda"])


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    import csv
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["a", "b", "r", "lambda", "vartheta_A_to_B", "vartheta_B_to_A",
                  "complementarity_error"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r["lambda"]):
            writer.writerow({k: rec.get(k) for k in fieldnames})


def audit_tasks(queue_root: Path) -> Dict[str, int]:
    counts = {}
    for state in ("pending", "claimed", "completed", "failed"):
        state_dir = queue_root / state
        n = sum(1 for _ in state_dir.rglob("*.json")) if state_dir.exists() else 0
        counts[state] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate H1 review: E0 chord-threshold calibration.")
    parser.add_argument("--phaseh1-results", type=Path, default=Path("results/phaseh1"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phaseh1"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phaseh1"))
    args = parser.parse_args()

    results_dir: Path = args.phaseh1_results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(results_dir.glob("worker_*.jsonl"))
    records = _load_e0_records(jsonl_files)
    if not records:
        print(f"  [warn] no E0 records found under {results_dir}/worker_*.jsonl -- nothing to score")

    errors = [rec["complementarity_error"] for rec in records]
    max_error = float(np.max(errors)) if errors else float("nan")
    mean_error = float(np.mean(errors)) if errors else float("nan")

    print(f"E0 complementarity: n={len(records)}  mean_error={mean_error:.3e}  max_error={max_error:.3e}")

    vartheta_dagger_table = compute_vartheta_dagger_table(records)
    _write_csv(output_dir / "gateh1_e0_complementarity.csv", records)

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    gate_pass = bool(records) and max_error < COMPLEMENTARITY_TOLERANCE

    review = {
        "n_points": len(records),
        "mean_complementarity_error": mean_error,
        "max_complementarity_error": max_error,
        "tolerance": COMPLEMENTARITY_TOLERANCE,
        "gateh1_pass": gate_pass,
        "gateh1_vartheta_dagger_table": vartheta_dagger_table,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "Complementarity holds within tolerance across the swept lambda "
            "range -- Lemma 2.4's clique-round projection is valid there, "
            "and H2's E7/E14 may be queued using this vartheta_dagger table."
            if gate_pass else
            "Complementarity FAILED (or no data) -- do NOT queue phase h2. "
            "If it failed only at large lambda, confine H-chord's validity "
            "to the passing sub-range (per Annex B.6 point 1) before "
            "regenerating E0's grid and re-running this gate; if it failed "
            "everywhere, this is a regression in theory.chord_geometry's "
            "root-finder, not a landscape-parameter issue."
        ),
    }
    review_path = output_dir / "gateh1_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate H1 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
