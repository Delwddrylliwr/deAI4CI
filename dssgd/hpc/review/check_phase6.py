# -*- coding: utf-8 -*-
"""Gate 6 review script: validate Phase 6 (E5, E11, E12a, E12b) outputs.

Produces:
  review/gate6_review.json
  review/gate6_e11_slopes.csv
  review/gate6_e12b_curvature.csv

Usage:
  python -m hpc.review.check_phase6 \\
      --phase6-results results/phase6a \\
      --queue-dir queue/phase6a \\
      --output-dir review/
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.clique_fixation import CliqueFixationRun
from analysis.natural_cascade import NaturalCascadeRun
from analysis.provenance import crossover_stage
from analysis import theory
from hpc.review.check_phase5 import compute_e6_level_matching, _write_csv, _task_counts


# ---------------------------------------------------------------------------
# E5: crossover stage l_c vs sigma
# ---------------------------------------------------------------------------


def compute_e5_crossover(pkl_dir: Path) -> List[Dict[str, Any]]:
    by_sigma: Dict[float, List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if run.events is None or "/sigma=" not in run.name:
            continue
        sigma = float(run.name.split("/sigma=")[1].split("/")[0])
        l_c = crossover_stage(run.flip_table, run.events)
        if l_c is not None:
            by_sigma[sigma].append(l_c)

    rows = []
    for sigma, l_cs in sorted(by_sigma.items()):
        rows.append({"sigma": sigma, "n_runs": len(l_cs), "mean_l_c": float(np.mean(l_cs))})
    return rows


# ---------------------------------------------------------------------------
# E11: scheduling sweep — slope of log2(T_flip) vs hierarchical distance
# ---------------------------------------------------------------------------


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    res = np.linalg.lstsq(A, y, rcond=None)
    return float(res[0][0]), float(res[0][1])


def compute_e11_slopes(pkl_dir: Path) -> List[Dict[str, Any]]:
    """{(protocol, staleness): slope} of log2(mean t_flip_relative) vs distance."""
    by_key: Dict[Tuple[str, int], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/proto=" not in run.name:
            continue
        proto = run.name.split("/proto=")[1].split("/")[0]
        stale = int(run.name.split("/stale=")[1].split("/")[0])
        for row in run.flip_table:
            t, d = row.get("t_flip_relative"), row.get("distance_from_source")
            if t is not None and d is not None and t > 0:
                by_key[(proto, stale)][d].append(math.log2(t))

    rows = []
    for (proto, stale), by_d in sorted(by_key.items()):
        ds = np.array([d for d in sorted(by_d) if d >= 2 and len(by_d[d]) >= 2], dtype=float)
        means = np.array([np.mean(by_d[d]) for d in ds])
        slope, _ = _linear_fit(ds, means) if len(ds) >= 2 else (float("nan"), float("nan"))
        rows.append({"protocol": proto, "staleness_bound": stale, "slope": slope, "n_distances": len(ds)})
    return rows


# ---------------------------------------------------------------------------
# E12(b): curvature ratchet vs computed rho_curv(r)
# ---------------------------------------------------------------------------


def compute_e12b_curvature_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    by_cell: Dict[Tuple[int, float], List[bool]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = CliqueFixationRun.load(pkl_path)
        except Exception:
            continue
        by_cell[(run.m, run.r)].append(run.fixed_at_B)

    rows = []
    for (m, r), outcomes in sorted(by_cell.items()):
        empirical = float(np.mean(outcomes))  # frequency of fixing on the SHARP basin (seeded at B)
        try:
            rho = theory.fixation_bias(a=0.5, b=0.0, kick_weight=0.5, r=r)
            predicted = theory.fixation_probability(1, m, rho)
        except Exception:
            predicted = float("nan")
        rows.append({
            "m": m, "r": r, "n_trials": len(outcomes),
            "empirical_fixation_on_sharp_freq": empirical, "predicted_q_fix": predicted,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 6 review for Phase 6 outputs (E5,E11,E12a,E12b).")
    parser.add_argument("--phase6-results", type=Path, default=Path("results/phase6a"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phase6a"))
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to review/<phase6-results basename> (e.g. "
             "review/phase6s/ for --phase6-results results/phase6s), so "
             "async/sync reviews never collide or need a manually-labelled "
             "path. Pass explicitly to override.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix appended to E12a/E12b pkl dir names for the synchronous "
             "variant (e.g. 'S' for phase 6s: E12aS/E12bS -- see "
             "generate_queue.py's phase=='6s' branch). Phase 6s has no "
             "synchronous variant of E5/E11, so those are skipped entirely "
             "when --suffix is set.",
    )
    args = parser.parse_args()

    results_dir = args.phase6_results
    output_dir = args.output_dir or Path("review") / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    e5_rows: List[Dict[str, Any]] = []
    if suffix:
        print("  [skip] E5 has no synchronous variant (phase 6s) — skipping")
    else:
        e5_pkl_dir = results_dir / "pkl" / "E5"
        if e5_pkl_dir.exists():
            e5_rows = compute_e5_crossover(e5_pkl_dir)
            _write_csv(output_dir / "gate6_e5_crossover.csv", e5_rows)
            print(f"E5 crossover: {e5_rows}")
        else:
            print(f"  [warn] {e5_pkl_dir} not found — skipping E5")

    e11_rows: List[Dict[str, Any]] = []
    e11_ok = False
    if suffix:
        print("  [skip] E11 has no synchronous variant (phase 6s) — skipping")
    else:
        e11_pkl_dir = results_dir / "pkl" / "E11"
        if e11_pkl_dir.exists():
            e11_rows = compute_e11_slopes(e11_pkl_dir)
            _write_csv(output_dir / "gate6_e11_slopes.csv", e11_rows)
            sync_slopes = [r["slope"] for r in e11_rows if r["protocol"] == "synchronous" and not math.isnan(r["slope"])]
            e11_ok = bool(sync_slopes) and all(abs(s - 1.0) < 0.3 for s in sync_slopes)
            print(f"E11 slopes: {e11_rows}")
        else:
            print(f"  [warn] {e11_pkl_dir} not found — skipping E11")

    e12a_pkl_dir = results_dir / "pkl" / f"E12a{suffix}"
    e12a_rows: List[Dict[str, Any]] = []
    e12a_ok = False
    if e12a_pkl_dir.exists():
        e12a_rows = compute_e6_level_matching(e12a_pkl_dir)
        _write_csv(output_dir / "gate6_e12a_containment.csv", e12a_rows)
        fracs = [r["frac_d_max_eq_G"] for r in e12a_rows]
        e12a_ok = bool(fracs) and float(np.mean(fracs)) > 0.5
        print(f"E12a overfitting containment: mean frac(d_max==l)={np.mean(fracs) if fracs else float('nan'):.3f}")
    else:
        print(f"  [warn] {e12a_pkl_dir} not found — skipping E12a")

    e12b_pkl_dir = results_dir / "pkl" / f"E12b{suffix}"
    e12b_rows: List[Dict[str, Any]] = []
    e12b_ok = False
    if e12b_pkl_dir.exists():
        e12b_rows = compute_e12b_curvature_table(e12b_pkl_dir)
        _write_csv(output_dir / "gate6_e12b_curvature.csv", e12b_rows)
        # At r>1 (B is the sharp basin per curvature_epsilon's convention), the
        # ratchet should DISFAVOUR fixing there: empirical freq should DECREASE with r.
        by_m: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        for row in e12b_rows:
            by_m[row["m"]].append((row["r"], row["empirical_fixation_on_sharp_freq"]))
        monotone_ok = []
        for m, items in by_m.items():
            items.sort()
            freqs = [f for _, f in items]
            monotone_ok.append(all(freqs[i] >= freqs[i + 1] - 0.15 for i in range(len(freqs) - 1)))
        e12b_ok = bool(monotone_ok) and all(monotone_ok)
        print(f"E12b curvature ratchet monotone-decreasing-in-r: {e12b_ok}")
    else:
        print(f"  [warn] {e12b_pkl_dir} not found — skipping E12b")

    task_counts = _task_counts(args.queue_dir)
    gate_pass = (e12a_ok and e12b_ok) if suffix else (e11_ok and e12a_ok and e12b_ok)

    review = {
        "run_parametrization": {
            "phase6_results": str(results_dir),
            "queue_dir": str(args.queue_dir),
            "suffix": suffix,
            "gossip_protocol": "synchronous" if suffix else "asynchronous",
            "output_dir": str(output_dir),
        },
        "e5_crossover": e5_rows,
        "e11_slopes": e11_rows,
        "e11_ok": e11_ok,
        "e12a_containment": e12a_rows,
        "e12a_ok": e12a_ok,
        "e12b_curvature": e12b_rows,
        "e12b_ok": e12b_ok,
        "gate6_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            f"E12a={'OK' if e12a_ok else 'FAIL'}, E12b={'OK' if e12b_ok else 'FAIL'}."
            if suffix else
            f"E11={'OK' if e11_ok else 'FAIL'}, E12a={'OK' if e12a_ok else 'FAIL'}, E12b={'OK' if e12b_ok else 'FAIL'}."
        ),
    }
    review_path = output_dir / "gate6_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2, default=str)
    print(f"\nGate 6 {'PASS' if gate_pass else 'FAIL'} — review written to {review_path}")


if __name__ == "__main__":
    main()
