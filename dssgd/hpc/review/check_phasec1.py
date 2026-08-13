# -*- coding: utf-8 -*-
"""Gate C1 review script: capacity-limited ("compressed") gossip sweep
(paper2_social Sec. 6, Prediction 6.1) on the E6 heterogeneity testbed.

Passage is scored against each run's own generality level G (Def. 4.3,
Theorem 4.5's level-matching filter): passage = cascade_depth(run) >= G,
i.e. the innovation reached at least its own scope boundary. Selectivity at
a given capacity k is the spread (max - min) of passage rate across G --
a sharp filter passes low-G innovations and rejects/limits high-G ones
differently (wide spread); an unselective one treats every G alike
(narrow spread). Prediction 6.1 predicts this spread widens as k shrinks,
and that the uncompressed control (capacity_bits=None) recovers whatever
selectivity Paper I's own E6 sweep already shows at these same (a, b_in,
b_out, G) values.

Produces:
  review/gatec1_review.json   -- gatec1_pass, gatec1_basin_family (consumed
                                  by generate_queue.py --phase c3)
  review/gatec1_passage_by_k_G.csv
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.natural_cascade import NaturalCascadeRun
from analysis.nmh_observables import cascade_depth


def _load_pkls(pkl_dir: Path) -> List[NaturalCascadeRun]:
    runs = []
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            runs.append(NaturalCascadeRun.load(pkl_path))
        except Exception as exc:
            print(f"  [warn] Failed to load {pkl_path.name}: {exc}")
    return runs


def _parse_c1_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    """'C1/G={G}/{k_tag}/seed={seed}' -> (G, capacity_bits or None for the
    uncompressed control)."""
    g_val: Optional[int] = None
    k_val: Optional[int] = None
    for part in name.split("/"):
        if part.startswith("G="):
            g_val = int(part.split("=", 1)[1])
        elif part.startswith("k="):
            k_val = int(part.split("=", 1)[1])
        elif part == "uncompressed":
            k_val = None
    return g_val, k_val


def compute_passage_by_k_g(
    runs: List[NaturalCascadeRun],
) -> Dict[str, Dict[int, float]]:
    """{k_tag: {G: passage_rate}}, k_tag is 'uncompressed' or 'k=<bits>'."""
    by_k_g: Dict[Tuple[Optional[int], int], List[bool]] = defaultdict(list)
    for run in runs:
        if not run.warmup_ok or run.nucleation_leaf is None:
            continue
        g_val, k_val = _parse_c1_name(run.name)
        if g_val is None:
            continue
        depth = cascade_depth(
            run.centroid_traj, run.nucleation_leaf, run.theta_A, run.theta_B,
            epsilon=0.2, persistence=3,
        )
        by_k_g[(k_val, g_val)].append(depth >= g_val)

    result: Dict[str, Dict[int, float]] = defaultdict(dict)
    for (k_val, g_val), passed in by_k_g.items():
        k_tag = "uncompressed" if k_val is None else f"k={k_val}"
        result[k_tag][g_val] = float(np.mean(passed))
    return dict(result)


def compute_selectivity(passage_by_k_g: Dict[str, Dict[int, float]]) -> Dict[str, float]:
    """Selectivity(k) = spread (max - min) of passage rate across G."""
    return {
        k_tag: (max(by_g.values()) - min(by_g.values())) if by_g else float("nan")
        for k_tag, by_g in passage_by_k_g.items()
    }


def _k_sort_key(k_tag: str) -> float:
    return float("inf") if k_tag == "uncompressed" else float(k_tag.split("=", 1)[1])


def _write_csv(path: Path, passage_by_k_g: Dict[str, Dict[int, float]]) -> None:
    import csv
    rows = [
        {"k_tag": k_tag, "G": g_val, "passage_rate": p}
        for k_tag, by_g in passage_by_k_g.items()
        for g_val, p in sorted(by_g.items())
    ]
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
    parser = argparse.ArgumentParser(description="Gate C1 review: compressed-gossip capacity sweep.")
    parser.add_argument("--phasec1-results", type=Path, default=Path("results/phasec1"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phasec1"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phasec1"))
    args = parser.parse_args()

    results_dir: Path = args.phasec1_results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = results_dir / "pkl" / "C1"
    runs = _load_pkls(pkl_dir) if pkl_dir.exists() else []
    if not runs:
        print(f"  [warn] {pkl_dir} not found or empty -- nothing to score")

    passage_by_k_g = compute_passage_by_k_g(runs)
    selectivity = compute_selectivity(passage_by_k_g)

    print("Passage rate by k, G:")
    for k_tag in sorted(passage_by_k_g, key=_k_sort_key):
        print(f"  {k_tag}: {passage_by_k_g[k_tag]}  (selectivity={selectivity[k_tag]:.3f})")

    _write_csv(output_dir / "gatec1_passage_by_k_G.csv", passage_by_k_g)

    # -- Prediction 6.1 gate criterion --
    # Selectivity at the SMALLEST tested k should exceed the uncompressed
    # control's selectivity (compression sharpens the filter).
    k_tags_compressed = [k for k in selectivity if k != "uncompressed"]
    smallest_k_tag = min(k_tags_compressed, key=_k_sort_key) if k_tags_compressed else None
    uncompressed_sel = selectivity.get("uncompressed")
    smallest_k_sel = selectivity.get(smallest_k_tag) if smallest_k_tag else None

    gate_pass = bool(
        smallest_k_sel is not None and uncompressed_sel is not None
        and not np.isnan(smallest_k_sel) and not np.isnan(uncompressed_sel)
        and smallest_k_sel > uncompressed_sel
    )

    # -- Basin family for Phase c3 (same (a, b_in, b_out, G) combinations
    # this sweep already touched, so C3's operational_dl measurements are
    # directly comparable to this sweep's own passage-rate curve) --
    basin_family = []
    seen = set()
    for run in runs:
        g_val, _ = _parse_c1_name(run.name)
        if g_val is None:
            continue
        key = (run.a, run.b, g_val)
        if key in seen:
            continue
        seen.add(key)
        basin_family.append({
            "generality_level": g_val, "a": run.a, "b_in": run.b, "b_out": run.b,
            "n_leaf_types": run.branching ** run.depth, "source_leaf": run.nucleation_leaf or 0,
            "branching": run.branching,
        })

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    review = {
        "passage_by_k_G": passage_by_k_g,
        "selectivity_by_k": selectivity,
        "smallest_k_tag": smallest_k_tag,
        "gatec1_pass": gate_pass,
        "gatec1_basin_family": basin_family,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "Selectivity increases at the smallest tested capacity vs the "
            "uncompressed control, consistent with Prediction 6.1."
            if gate_pass else
            "Selectivity did NOT increase at the smallest tested capacity -- "
            "check whether the k sweep actually reaches small enough values, "
            "or whether passage rates are saturated (all 0 or all 1) across G."
        ),
    }
    review_path = output_dir / "gatec1_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate C1 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
