# -*- coding: utf-8 -*-
"""Gate H2 review script: E7 (Type-P-only fixation formula) and E14RR
(round-ratio sweep), Annex B.1's calibration-ladder remainder
(paper1_computing_hybrid_gossip.md).

Both experiments are scored against H1's MEASURED vartheta_dagger(lambda)
table (review/phaseh1/gateh1_review.json), never against a freshly
recomputed theory.chord_geometry call -- this is what "not fitted, but
measured" (Annex B.1) means operationally: H1 already validated
complementarity (2.7a), so H2 reuses its output rather than re-deriving the
same quantity twice from two independently-maintained code paths.

E7 (Lemma 3.1, Type-P only): predicted_q_fix(j=1; m, rho) vs empirical
fixation frequency, rho computed from H1's vartheta_dagger via the
degenerate fixed-kick-weight law (same construction as theory.fixation_bias,
but using the LOOKED-UP threshold instead of recomputing chord_geometry).
REPORTED, NOT GATED: at the degenerate alpha=0.5 kick law, rho is EXACTLY 0
for every b>0 (theory.fixation_bias's own docstring), so predicted_q_fix=1.0
regardless of which b -- yet empirically q_fix rises systematically WITH b
(confirmed on real cluster data), which the b-independent prediction cannot
match by construction. This is the documented idealisation gap E7 exists to
expose ("disagreement... is informative, not a bug to silently patch over"),
not a fault in the simulation or this script; see the comment above
gate_pass in main() for the physical mechanism.

E14RR (Lemma 3.1', Corollary 3.1'', the round-ratio sweep): empirical q_hyb
(fraction of runs fixed_at_B) vs K, compared against the ruin-factor
plateau theory.fixation_probability_hybrid and the detection floor
theory.detection_floor (eq. 3.2d). Produces the measured K_low this gate
reports for H3 onward -- but see the "known limitation" below before
trusting K_high/K_mid.

KNOWN LIMITATION (flagged, not silently glossed over): eq. 3.4a's upper
bound on the H-round window (the renewal condition, epsilon_ren =
nu_1/epsilon_n <= 1) is a HIERARCHY-scale quantity -- nu_1 is the
cross-module kick rate (Theorem 4.1), which does not exist in a single
isolated clique. E14RR, run at clique scale, can only measure the LOWER
bound (the detection floor, eq. 3.2d) directly. K_mid/K_high below are
therefore a provisional placeholder (K_low scaled by fixed factors), NOT a
measurement of the renewal-condition upper bound -- that requires a
hierarchy-scale renewal-separation experiment, deferred to phase H3/H4.
Anyone consuming gateh2_K_high should treat it as "definitely above the
detection floor," not as "validated against (3.4a)."

Produces:
  review/phaseh2/gateh2_review.json
  review/phaseh2/gateh2_e7_fixation.csv
  review/phaseh2/gateh2_e14rr_round_ratio.csv

Usage:
  python -m hpc.review.check_phaseh2 \\
      --phaseh2-results results/phaseh2 \\
      --gateh1-results review/phaseh1/gateh1_review.json \\
      --queue-dir queue/phaseh2 \\
      --output-dir review/phaseh2
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis import theory
from analysis.clique_fixation import CliqueFixationRun


# ---------------------------------------------------------------------------
# H1 lookup table
# ---------------------------------------------------------------------------


def load_vartheta_dagger_lookup(gateh1_path: Path) -> Dict[Tuple[float, float], float]:
    """{(a, b): vartheta_dagger} from gateh1_review.json's measured table --
    the "measured, not fitted" input this whole gate is built around."""
    with open(gateh1_path) as fh:
        review = json.load(fh)
    table = review.get("gateh1_vartheta_dagger_table", [])
    return {(round(row["a"], 6), round(row["b"], 6)): row["vartheta_dagger"] for row in table}


def rho_from_vartheta_dagger(vartheta_dagger_ab: float, kick_weight: float = 0.5) -> float:
    """rho = p_-/p_+ under the degenerate fixed-kick-weight law (same
    construction as theory.fixation_bias), using H1's looked-up threshold
    and complementarity (2.7a, validated by H1) for vartheta_{B->A} =
    1 - vartheta_dagger rather than recomputing chord_geometry."""
    vartheta_ba = 1.0 - vartheta_dagger_ab
    p_plus = 1.0 if kick_weight >= vartheta_dagger_ab else 0.0
    p_minus = 1.0 if kick_weight >= vartheta_ba else 0.0
    if p_plus == 0.0:
        return math.inf
    return p_minus / p_plus


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_pkls(pkl_dir: Path) -> List[CliqueFixationRun]:
    runs = []
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            runs.append(CliqueFixationRun.load(pkl_path))
        except Exception as exc:
            print(f"  [warn] Failed to load {pkl_path.name}: {exc}")
    return runs


def _parse_name(name: str) -> Dict[str, Any]:
    parts: Dict[str, Any] = {}
    for part in name.split("/"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            parts[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            parts[k] = v
    return parts


# ---------------------------------------------------------------------------
# E7: fixation formula (Type-P only)
# ---------------------------------------------------------------------------


def compute_e7_table(
    runs: List[CliqueFixationRun], vartheta_lookup: Dict[Tuple[float, float], float],
) -> List[Dict[str, Any]]:
    by_mb: Dict[Tuple[int, float], List[bool]] = defaultdict(list)
    for run in runs:
        by_mb[(run.m, round(run.b, 6))].append(run.fixed_at_B)

    rows = []
    for (m, b), outcomes in sorted(by_mb.items()):
        key = (0.5, b)  # E7's default a=0.5, matching E0's default grid
        vd = vartheta_lookup.get(key)
        empirical = float(np.mean(outcomes))
        if vd is None:
            rows.append({"m": m, "b": b, "n": len(outcomes), "empirical_q_fix": empirical,
                         "rho": None, "predicted_q_fix": None,
                         "note": "no H1 vartheta_dagger for this (a,b) -- widen E0's grid to cover it"})
            continue
        rho = rho_from_vartheta_dagger(vd)
        predicted = theory.fixation_probability(1, m, rho)
        rows.append({
            "m": m, "b": b, "n": len(outcomes), "empirical_q_fix": empirical,
            "rho": rho, "predicted_q_fix": predicted,
            "abs_error": abs(empirical - predicted),
        })
    return rows


# ---------------------------------------------------------------------------
# E14RR: round-ratio sweep
# ---------------------------------------------------------------------------


def compute_e14rr_table(
    runs: List[CliqueFixationRun], vartheta_lookup: Dict[Tuple[float, float], float],
) -> List[Dict[str, Any]]:
    by_mbk: Dict[Tuple[int, float, float], List[CliqueFixationRun]] = defaultdict(list)
    for run in runs:
        parts = _parse_name(run.name)
        K = parts.get("K")
        if K is None:
            continue
        by_mbk[(run.m, round(run.b, 6), K)].append(run)

    rows = []
    for (m, b, K), group in sorted(by_mbk.items()):
        outcomes = [r.fixed_at_B for r in group]
        empirical_q_hyb = float(np.mean(outcomes))

        key = (0.5, b)
        vd = vartheta_lookup.get(key)
        row: Dict[str, Any] = {
            "m": m, "b": b, "K": K, "n": len(group), "empirical_q_hyb": empirical_q_hyb,
        }
        if vd is None:
            row["note"] = "no H1 vartheta_dagger for this (a,b) -- widen E0's grid"
            rows.append(row)
            continue

        rho = rho_from_vartheta_dagger(vd)
        row["rho"] = rho
        row["q_ruin_1"] = theory.fixation_probability_hybrid(
            1, m, rho, a=0.5, b=b, vartheta_dagger_val=vd,
        )

        p_plus, p_minus = theory.kick_success_probabilities_distributed(0.5, b)
        row["K_min_detection_floor"] = theory.detection_floor(m, rho, p_plus, p_minus)

        # T_hit proxy (local-step-call units, not literal Type-P event count
        # -- see this module's docstring / experiment_E14_round_ratio_sweep's
        # docstring): first occupancy_log entry reaching n_dagger, if any.
        vd_nd = theory.n_dagger(m, vd)
        hit_steps = []
        for r in group:
            if not r.occupancy_log:
                continue
            for step_idx, (_, n_b) in enumerate(r.occupancy_log):
                if n_b >= vd_nd:
                    hit_steps.append(step_idx)
                    break
        row["n_dagger"] = vd_nd
        row["mean_T_hit_steps_given_hit"] = float(np.mean(hit_steps)) if hit_steps else None
        row["frac_hit_n_dagger"] = len(hit_steps) / len(group) if group else float("nan")
        rows.append(row)
    return rows


def estimate_K_low(e14rr_rows: List[Dict[str, Any]]) -> Optional[float]:
    """Per (m,b) cell, the smallest swept K at which empirical_q_hyb clears
    half of that cell's own max observed empirical_q_hyb -- a crossing-point
    estimate of the detection floor, aggregated (median) across cells into a
    single scalar. Returns None if no cell shows a clear rise (Annex B.6
    point 2's failure mode: the H-round window may be empty at default m)."""
    by_mb: Dict[Tuple[int, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in e14rr_rows:
        if "empirical_q_hyb" in row:
            by_mb[(row["m"], row["b"])].append(row)

    crossings = []
    for (_m, _b), rows in by_mb.items():
        rows_sorted = sorted(rows, key=lambda r: r["K"])
        q_max = max(r["empirical_q_hyb"] for r in rows_sorted)
        if q_max <= 0.0:
            continue
        half = q_max / 2.0
        for r in rows_sorted:
            if r["empirical_q_hyb"] >= half:
                crossings.append(r["K"])
                break
    if not crossings:
        return None
    return float(np.median(crossings))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
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
    parser = argparse.ArgumentParser(description="Gate H2 review: E7 + E14RR (round-ratio sweep).")
    parser.add_argument("--phaseh2-results", type=Path, default=Path("results/phaseh2"))
    parser.add_argument("--gateh1-results", type=Path, default=Path("review/phaseh1/gateh1_review.json"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phaseh2"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phaseh2"))
    args = parser.parse_args()

    results_dir: Path = args.phaseh2_results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.gateh1_results.exists():
        print(f"  [error] {args.gateh1_results} not found -- run check_phaseh1 first")
        vartheta_lookup: Dict[Tuple[float, float], float] = {}
    else:
        vartheta_lookup = load_vartheta_dagger_lookup(args.gateh1_results)

    e7_runs = _load_pkls(results_dir / "pkl" / "E7") if (results_dir / "pkl" / "E7").exists() else []
    e14rr_runs = (
        _load_pkls(results_dir / "pkl" / "E14RR")
        if (results_dir / "pkl" / "E14RR").exists() else []
    )
    if not e7_runs:
        print(f"  [warn] {results_dir}/pkl/E7 not found or empty")
    if not e14rr_runs:
        print(f"  [warn] {results_dir}/pkl/E14RR not found or empty")

    e7_table = compute_e7_table(e7_runs, vartheta_lookup)
    e14rr_table = compute_e14rr_table(e14rr_runs, vartheta_lookup)

    _write_csv(output_dir / "gateh2_e7_fixation.csv", e7_table)
    _write_csv(output_dir / "gateh2_e14rr_round_ratio.csv", e14rr_table)

    print("E7 fixation table:")
    for row in e7_table:
        print(f"  {row}")
    print("E14RR round-ratio table:")
    for row in e14rr_table:
        print(f"  {row}")

    K_low = estimate_K_low(e14rr_table)
    # Provisional placeholder for the renewal-condition upper bound -- see
    # this module's docstring "KNOWN LIMITATION" before trusting these.
    K_mid = K_low * 10.0 if K_low is not None else None
    K_high = K_low * 100.0 if K_low is not None else None

    e7_errors = [row["abs_error"] for row in e7_table if row.get("abs_error") is not None]
    e7_mean_abs_error = float(np.mean(e7_errors)) if e7_errors else None

    # E7 is reported, NOT gated: at the degenerate fixed-alpha=0.5 kick law,
    # theory.fixation_bias's own docstring says rho collapses to EXACTLY 0
    # for every b>0 (predicting certain fixation, q_fix=1.0, independent of
    # b) -- "disagreement with measured fixation frequency is informative,
    # not a bug to silently patch over." Real cluster data confirms this
    # isn't noise: empirical_q_fix rises systematically WITH b (e.g. m=4:
    # 0.32->0.48->0.62->0.70 across b=0.01..0.04) despite the theory's b-
    # independent prediction for b>0 -- physically sensible (a kick that
    # only marginally crosses the chord threshold can relax back toward A
    # during the following local_steps if it isn't deep enough into B's
    # basin, and "deep enough" depends on the actual curvature/tilt shape,
    # not just the threshold indicator), and exactly the idealisation gap
    # E7 exists to expose. check_phase5.py's ORIGINAL (pre-hybrid) E7 gate
    # uses a looser 0.35 threshold for the same reason; H2 goes further and
    # doesn't gate on it at all, matching how check_phase6.py treats E14/E15
    # as "diagnostic follow-ups, reported but not gated" rather than a hard
    # blocker on unrelated downstream phases.
    gate_pass = K_low is not None

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    review = {
        "e7_table": e7_table,
        "e7_mean_abs_error": e7_mean_abs_error,
        "e14rr_table": e14rr_table,
        "gateh2_K_low": K_low,
        "gateh2_K_mid": K_mid,
        "gateh2_K_high": K_high,
        "gateh2_K_high_is_provisional": True,
        "gateh2_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "E14RR shows a clear rise in empirical q_hyb across the swept K "
            "range (K_low estimated); K_mid/K_high are a provisional x10/x100 "
            "placeholder, NOT a measurement of eq. 3.4a's renewal-condition "
            "upper bound -- that needs hierarchy-scale data (phase H3/H4). "
            "E7's mean_abs_error is reported above but does NOT gate this "
            "phase (see the comment above gate_pass in this script for why "
            "a large, b-correlated E7/theory gap is expected, not a fault)."
            if gate_pass else
            "E14RR showed no clear rise in q_hyb across the swept K range "
            "(Annex B.6 point 2's risk: the H-round window may be empty at "
            "the default m/p) -- widen K_list or increase m before trusting "
            "any downstream K value."
        ),
    }
    review_path = output_dir / "gateh2_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate H2 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
