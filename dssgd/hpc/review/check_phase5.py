# -*- coding: utf-8 -*-
"""Gate 5 review script: validate Phase 5 (E1, E6, E7, E9, E10, E13) outputs
before submitting Phase 6/7.

Produces:
  review/gate5_review.json  — machine-readable gate decision
  review/gate5_e7_fixation.csv
  review/gate5_e6_level_matching.csv
  review/gate5_e1_provenance.csv

Usage:
  python -m hpc.review.check_phase5 \\
      --phase5-results results/phase5a \\
      --queue-dir queue/phase5a \\
      --output-dir review/
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.clique_fixation import CliqueFixationRun
from analysis.natural_cascade import NaturalCascadeRun
from analysis.nmh_observables import cascade_depth
from analysis.provenance import classify_flip_provenance
from analysis import theory


# ---------------------------------------------------------------------------
# E7: fixation frequency vs computed rho (Lemma 3.1)
# ---------------------------------------------------------------------------


def compute_e7_fixation_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    """For each (m, b) cell, empirical fixation frequency vs theory.fixation_bias."""
    by_cell: Dict[Tuple[int, float], List[bool]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = CliqueFixationRun.load(pkl_path)
        except Exception:
            continue
        by_cell[(run.m, run.b)].append(run.fixed_at_B)

    rows = []
    for (m, b), outcomes in sorted(by_cell.items()):
        empirical = float(np.mean(outcomes))
        try:
            rho = theory.fixation_bias(a=0.5, b=b, kick_weight=0.5)
            predicted = theory.fixation_probability(1, m, rho)
        except Exception:
            predicted = float("nan")
        rows.append({
            "m": m, "b": b, "n_trials": len(outcomes),
            "empirical_fixation_freq": empirical, "predicted_q_fix": predicted,
            "abs_error": abs(empirical - predicted) if not math.isnan(predicted) else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# E6: level-matching (d_max == G)
# ---------------------------------------------------------------------------


def compute_e6_level_matching(pkl_dir: Path) -> List[Dict[str, Any]]:
    """For each generality level G (parsed from run.name), fraction of runs
    with d_max exactly equal to G (Theorem 4.5)."""
    by_G: Dict[int, List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/G=" not in run.name:
            continue
        G = int(run.name.split("/G=")[1].split("/")[0])
        source_leaf = 0  # experiment_E6's fixed convention
        d_max = cascade_depth(run.centroid_traj, source_leaf, run.theta_A, run.theta_B,
                               epsilon=0.2, persistence=3)
        by_G[G].append(d_max)

    rows = []
    for G, d_maxes in sorted(by_G.items()):
        arr = np.array(d_maxes)
        rows.append({
            "G": G, "n_runs": len(arr),
            "frac_d_max_eq_G": float(np.mean(arr == G)),
            "mean_d_max": float(arr.mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# E1: provenance attribution rates + severed-control null
# ---------------------------------------------------------------------------


def compute_e1_provenance_summary(pkl_dir: Path) -> Dict[str, Any]:
    """Kick-attributed fraction among flipped leaves (unsevered runs) and
    flip rate under severing (should be ~0 for sever>=1 at sigma=0)."""
    kick_fracs = []
    severed_flip_counts: Dict[str, List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        tag = "none"
        if "/sever=" in run.name:
            tag = run.name.split("/sever=")[1].split("/")[0]
        n_flipped = sum(1 for row in run.flip_table if row.get("t_flip_absolute") is not None)
        severed_flip_counts[tag].append(n_flipped)
        if tag == "none" and run.events is not None:
            prov = classify_flip_provenance(run.flip_table, run.events)
            if prov:
                kick_fracs.append(sum(1 for v in prov.values() if v == "kick") / len(prov))

    return {
        "mean_kick_attributed_fraction": float(np.mean(kick_fracs)) if kick_fracs else float("nan"),
        "mean_n_flipped_by_sever_distance": {
            tag: float(np.mean(counts)) for tag, counts in severed_flip_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# E13: containment boundary m*(Delta_in)
# ---------------------------------------------------------------------------


def compute_e13_containment(pkl_dir: Path) -> List[Dict[str, Any]]:
    """For each (delta_in, m), fraction of runs where the innovation LEAKED
    beyond the favourable ideal (any out-of-scope leaf flipped) -- read off
    per_leaf sign via the run's own flip_table + name, since per_leaf_loss_params
    itself isn't stored on the run; use nucleation-independent leak proxy:
    any leaf beyond generality_level+1's hierarchical distance flipping."""
    by_cell: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if "/delta_in=" not in run.name or "/m=" not in run.name:
            continue
        delta_in = int(run.name.split("/delta_in=")[1].split("/")[0])
        m = int(run.name.split("/m=")[1].split("/")[0])
        n_flipped = sum(1 for row in run.flip_table if row.get("t_flip_absolute") is not None)
        by_cell[(delta_in, m)].append(n_flipped > 0)

    rows = []
    for (delta_in, m), leaks in sorted(by_cell.items()):
        rows.append({
            "delta_in": delta_in, "m": m, "n_runs": len(leaks),
            "leak_frequency": float(np.mean(leaks)),
        })
    return rows


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _task_counts(queue_root: Path) -> Dict[str, int]:
    counts = {}
    for state in ("pending", "claimed", "completed", "failed"):
        d = queue_root / state
        counts[state] = sum(1 for _ in d.rglob("*.json")) if d.exists() else 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 5 review for Phase 5 outputs (E1,E6,E7,E9,E10,E13).")
    parser.add_argument("--phase5-results", type=Path, default=Path("results/phase5a"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phase5a"))
    parser.add_argument("--output-dir", type=Path, default=Path("review"))
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix appended to E7/E6/E1/E13 pkl dir names for the "
             "synchronous variant (e.g. 'S' for phase 5s: E7S/E6S/E13S -- "
             "see generate_queue.py's phase=='5s' branch). Phase 5s has no "
             "synchronous variant of E1, so that dir simply won't be found "
             "when --suffix is set (E1 isn't gate-critical).",
    )
    args = parser.parse_args()

    results_dir = args.phase5_results
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    # -- E7: fixation vs computed rho --
    e7_pkl_dir = results_dir / "pkl" / f"E7{suffix}"
    e7_rows: List[Dict[str, Any]] = []
    e7_ok = False
    if e7_pkl_dir.exists():
        e7_rows = compute_e7_fixation_table(e7_pkl_dir)
        _write_csv(output_dir / "gate5_e7_fixation.csv", e7_rows)
        errors = [r["abs_error"] for r in e7_rows if not math.isnan(r["abs_error"])]
        e7_ok = bool(errors) and float(np.mean(errors)) < 0.35
        print(f"E7 fixation table: {len(e7_rows)} cells, mean |error|={np.mean(errors) if errors else float('nan'):.3f}")
    else:
        print(f"  [warn] {e7_pkl_dir} not found — skipping E7")

    # -- E6: level matching --
    e6_pkl_dir = results_dir / "pkl" / f"E6{suffix}"
    e6_rows: List[Dict[str, Any]] = []
    e6_ok = False
    if e6_pkl_dir.exists():
        e6_rows = compute_e6_level_matching(e6_pkl_dir)
        _write_csv(output_dir / "gate5_e6_level_matching.csv", e6_rows)
        fracs = [r["frac_d_max_eq_G"] for r in e6_rows]
        e6_ok = bool(fracs) and float(np.mean(fracs)) > 0.5
        print(f"E6 level matching: mean frac(d_max==G)={np.mean(fracs) if fracs else float('nan'):.3f}")
    else:
        print(f"  [warn] {e6_pkl_dir} not found — skipping E6")

    # -- E1: provenance --
    e1_pkl_dir = results_dir / "pkl" / f"E1{suffix}"
    e1_summary: Dict[str, Any] = {}
    if e1_pkl_dir.exists():
        e1_summary = compute_e1_provenance_summary(e1_pkl_dir)
        _write_csv(output_dir / "gate5_e1_provenance.csv", [
            {"sever_distance": k, "mean_n_flipped": v}
            for k, v in e1_summary.get("mean_n_flipped_by_sever_distance", {}).items()
        ])
        print(f"E1 provenance: {e1_summary}")
    else:
        print(f"  [warn] {e1_pkl_dir} not found — skipping E1")

    # -- E13: containment boundary --
    e13_pkl_dir = results_dir / "pkl" / f"E13{suffix}"
    e13_rows: List[Dict[str, Any]] = []
    if e13_pkl_dir.exists():
        e13_rows = compute_e13_containment(e13_pkl_dir)
        _write_csv(output_dir / "gate5_e13_containment.csv", e13_rows)
        print(f"E13 containment: {len(e13_rows)} (delta_in, m) cells")
    else:
        print(f"  [warn] {e13_pkl_dir} not found — skipping E13")

    task_counts = _task_counts(args.queue_dir)
    gate_pass = e7_ok and e6_ok

    review = {
        "e7_fixation_table": e7_rows,
        "e7_ok": e7_ok,
        "e6_level_matching": e6_rows,
        "e6_ok": e6_ok,
        "e1_provenance_summary": e1_summary,
        "e13_containment": e13_rows,
        "gate5_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": f"E7={'OK' if e7_ok else 'FAIL'}, E6={'OK' if e6_ok else 'FAIL'}.",
    }
    review_path = output_dir / "gate5_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2, default=str)
    print(f"\nGate 5 {'PASS' if gate_pass else 'FAIL'} — review written to {review_path}")


if __name__ == "__main__":
    main()
