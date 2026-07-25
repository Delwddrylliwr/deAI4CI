# -*- coding: utf-8 -*-
"""Gate 1 review script: verify Phase 1 outputs before submitting Phase 2.

Scores NMH-1 (Proposition 4.2 slope-1 first-passage scaling), NMH-3 (Sec. 4.5
phase structure), and NCP-1 (Sec. 5.1 shell/bridge structural facts, Annex
A.4) against paper1_PDMP_wDAG_wData.md.

Produces:
  review/gate1_review.json   — machine-readable gate decision (input to generate_queue --phase 2)
  review/gate1_nmh1_slopes.csv
  review/gate1_nmh3_depths.csv
  review/gate1_ncp1_scaling.csv

Usage:
  python -m hpc.review.check_phase1 \\
      --phase1-results results/phase1 \\
      --queue-dir queue/phase1 \\
      --output-dir review/
"""
import argparse
import csv
import glob
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

from analysis.natural_cascade import NaturalCascadeRun
from analysis.nmh_observables import cascade_depth


# ---------------------------------------------------------------------------
# NMH-1: slope-1 test
# ---------------------------------------------------------------------------


def _group_tflip_by_distance(
    runs: List[NaturalCascadeRun],
) -> Dict[Tuple, Dict[int, List[float]]]:
    grouped: Dict[Tuple, Dict[int, List[float]]] = {}
    for run in runs:
        if not run.warmup_ok:
            continue
        key = (run.a, run.local_steps)
        if key not in grouped:
            grouped[key] = {}
        for row in run.flip_table:
            t = row.get("t_flip_relative")
            d = row.get("distance_from_source")
            if t is None or d is None or t <= 0:
                continue
            grouped[key].setdefault(d, []).append(math.log2(t))
    return grouped


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """OLS y = slope·x + intercept.  Returns (slope, intercept, slope_se)."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    res = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(res[0][0]), float(res[0][1])
    residuals = y - (slope * x + intercept)
    n = len(x)
    denom = float(np.sum((x - x.mean()) ** 2))
    se = (float(np.sqrt(residuals.var(ddof=2) / denom))
          if denom > 0 and n > 2 else float("nan"))
    return slope, intercept, se


def compute_nmh1_slopes(
    pkl_dir: Path,
) -> Dict[float, Tuple[float, float, int]]:
    """Load NMH-1 pickles and return {a: (slope, slope_se, n_seeds_with_flips)}."""
    runs = _load_pkls(pkl_dir)
    grouped = _group_tflip_by_distance(runs)
    results: Dict[float, Tuple[float, float, int]] = {}
    for (a_val, _ls), by_d in grouped.items():
        # Fit on d=2..5 as per spec (exclude d=1 floor effect)
        fit_ds = np.array([d for d in sorted(by_d) if d >= 2 and len(by_d[d]) >= 3],
                          dtype=float)
        fit_means = np.array([np.mean(by_d[d]) for d in fit_ds])
        if len(fit_ds) < 2:
            results[a_val] = (float("nan"), float("nan"), 0)
            continue
        slope, _intercept, se = _linear_fit(fit_ds, fit_means)
        n_seeds = len(set(r.seed for r in runs if r.a == a_val and r.warmup_ok))
        results[a_val] = (slope, se, n_seeds)
    return results


# ---------------------------------------------------------------------------
# NMH-3: phase structure
# ---------------------------------------------------------------------------


def compute_nmh3_depths(pkl_dir: Path) -> Dict[float, Tuple[float, float, int]]:
    """Load NMH-3 pickles and return {a: (mean_cascade_depth, std, n_runs)}."""
    runs = _load_pkls(pkl_dir)
    by_a: Dict[float, List[int]] = defaultdict(list)
    for run in runs:
        if not run.warmup_ok:
            continue
        theta_A = run.theta_A
        theta_B = run.theta_B
        if run.nucleation_leaf is not None:
            d = cascade_depth(run.centroid_traj, run.nucleation_leaf,
                              theta_A, theta_B, epsilon=0.2, persistence=3)
        else:
            d = 0
        by_a[run.a].append(d)
    result = {}
    for a, depths in sorted(by_a.items()):
        arr = np.array(depths)
        result[a] = (float(arr.mean()), float(arr.std()), len(arr))
    return result


def estimate_phase_boundaries(
    depths: Dict[float, Tuple[float, float, int]],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate (II->III boundary) by finding where mean cascade depth drops to ~0.

    Returns (a_boundary_I_II, a_boundary_II_III) — the boundary between Griffiths
    and paramagnetic phases is where depth -> 0.
    """
    sorted_a = sorted(depths)
    boundary_ii_iii = None
    for i in range(len(sorted_a) - 1):
        a0, a1 = sorted_a[i], sorted_a[i + 1]
        d0, d1 = depths[a0][0], depths[a1][0]
        if d0 > 0.5 and d1 < 0.5:
            boundary_ii_iii = (a0 + a1) / 2.0
            break
    # I->II boundary: where mean depth becomes large (all leaves flip)
    boundary_i_ii = None
    max_depth = max((d[0] for d in depths.values()), default=0)
    for a in sorted_a:
        if depths[a][0] >= max_depth * 0.9:
            boundary_i_ii = a
            break
    return boundary_i_ii, boundary_ii_iii


def recommend_b_values(boundary_ii_iii: Optional[float]) -> List[float]:
    """Suggest b sweep for NMH-2 based on Gate 1 phase boundary estimate."""
    # Default range around the operating point b=0.042
    return [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]


# ---------------------------------------------------------------------------
# NCP-1: shell scaling
# ---------------------------------------------------------------------------


def compute_ncp1_shell_scaling(jsonl_paths: List[Path]) -> Dict[float, Dict[int, float]]:
    """Parse JSONL summaries for NCP-1 graph-only tasks.

    Returns {p_f: {n_nodes: mean_n_shells}}.
    """
    by_pf_n: Dict[Tuple[float, int], List[int]] = defaultdict(list)
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
            if task.get("experiment") != "NCP1":
                continue
            result = rec.get("result", {})
            pf = task.get("config", {}).get("p_f", float("nan"))
            n = result.get("n_nodes", 0)
            n_shells = result.get("n_shells", 0)
            by_pf_n[(pf, n)].append(n_shells)

    result_dict: Dict[float, Dict[int, float]] = {}
    for (pf, n), vals in sorted(by_pf_n.items()):
        if pf not in result_dict:
            result_dict[pf] = {}
        result_dict[pf][n] = float(np.mean(vals))
    return result_dict


def compute_ncp1_bridge_exponent(jsonl_paths: List[Path]) -> Tuple[float, float, int]:
    """Fit the bridge exponent gamma (paper1_PDMP_wDAG_wData.md Sec. 5.1:
    B_{k,k+1} ~ |S_k|^gamma) from NCP-1's shell_sizes/bridge_counts records.

    Returns (gamma, r_squared, n_points) via OLS on log|S_k| vs log B_{k,k+1},
    pooling every (shell, seed, p_f) observation. (nan, nan, 0) if too few points.
    """
    log_sizes: List[float] = []
    log_bridges: List[float] = []
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
            if task.get("experiment") != "NCP1":
                continue
            result = rec.get("result", {})
            shell_sizes = result.get("shell_sizes", {})
            bridge_counts = result.get("bridge_counts", {})
            for key, b_count in bridge_counts.items():
                k_str, _ = key.split(",")
                s_k = shell_sizes.get(k_str)
                if s_k is None or s_k <= 0 or b_count <= 0:
                    continue
                log_sizes.append(math.log(s_k))
                log_bridges.append(math.log(b_count))

    if len(log_sizes) < 3:
        return float("nan"), float("nan"), len(log_sizes)

    x = np.array(log_sizes)
    y = np.array(log_bridges)
    gamma, _intercept, _se = _linear_fit(x, y)
    pred = gamma * x + _intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return gamma, r_squared, len(log_sizes)


# ---------------------------------------------------------------------------
# Task audit
# ---------------------------------------------------------------------------


def audit_tasks(queue_root: Path) -> Dict[str, int]:
    """Count tasks in each queue state."""
    counts = {}
    for state in ("pending", "claimed", "completed", "failed"):
        state_dir = queue_root / state
        n = sum(1 for _ in state_dir.rglob("*.json")) if state_dir.exists() else 0
        counts[state] = n
    return counts


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_pkls(pkl_dir: Path) -> List[NaturalCascadeRun]:
    runs = []
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            runs.append(NaturalCascadeRun.load(pkl_path))
        except Exception as exc:
            print(f"  [warn] Failed to load {pkl_path.name}: {exc}")
    return runs


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1 review for Phase 1 outputs.")
    parser.add_argument("--phase1-results", type=Path, default=Path("results/phase1"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phase1"))
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to review/<phase1-results basename> (e.g. "
             "review/phase1s/ for --phase1-results results/phase1s), so "
             "async/sync reviews never collide or need a manually-labelled "
             "path. Pass explicitly to override.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix appended to NMH1/NMH3 pkl dir names for the "
             "synchronous variant (e.g. 'S' for phase 1s: NMH1S/NMH3S -- "
             "see generate_queue.py's phase=='1s' branch). NCP1 is "
             "graph-only (no gossip protocol) and NMH7_pilot has no "
             "synchronous variant, so neither is affected by --suffix.",
    )
    args = parser.parse_args()

    results_dir = args.phase1_results
    output_dir = args.output_dir or Path("review") / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    # -- NMH-1 slopes --
    nmh1_pkl_dir = results_dir / "pkl" / f"NMH1{suffix}"
    nmh1_slopes: Dict[float, Tuple[float, float, int]] = {}
    if nmh1_pkl_dir.exists():
        nmh1_slopes = compute_nmh1_slopes(nmh1_pkl_dir)
        csv_rows = [
            {"a": a, "slope": s, "slope_se": se, "n_seeds_with_flips": n}
            for a, (s, se, n) in sorted(nmh1_slopes.items())
        ]
        _write_csv(output_dir / "gate1_nmh1_slopes.csv", csv_rows)
        print("NMH-1 slopes:")
        for a, (s, se, n) in sorted(nmh1_slopes.items()):
            flag = " OK" if not math.isnan(s) and abs(s - 1.0) < 0.15 else " FAIL"
            print(f"  a={a}: slope={s:.3f} ± {se:.3f} ({n} seeds){flag}")
    else:
        print(f"  [warn] {nmh1_pkl_dir} not found — skipping NMH-1 slopes")

    # -- NMH-3 phase structure --
    nmh3_pkl_dir = results_dir / "pkl" / f"NMH3{suffix}"
    depths: Dict[float, Tuple[float, float, int]] = {}
    boundary_i_ii: Optional[float] = None
    boundary_ii_iii: Optional[float] = None
    if nmh3_pkl_dir.exists():
        depths = compute_nmh3_depths(nmh3_pkl_dir)
        boundary_i_ii, boundary_ii_iii = estimate_phase_boundaries(depths)
        csv_rows = [
            {"a": a, "mean_depth": m, "std_depth": s, "n_runs": n}
            for a, (m, s, n) in sorted(depths.items())
        ]
        _write_csv(output_dir / "gate1_nmh3_depths.csv", csv_rows)
        print(f"NMH-3 phase boundaries: I->II ≈ {boundary_i_ii}, II->III ≈ {boundary_ii_iii}")
    else:
        print(f"  [warn] {nmh3_pkl_dir} not found — skipping NMH-3")

    # -- NCP-1 shell scaling --
    jsonl_files = sorted((results_dir).glob("worker_*.jsonl"))
    ncp1_scaling = compute_ncp1_shell_scaling(jsonl_files)
    if ncp1_scaling:
        csv_rows = [
            {"p_f": pf, "n_nodes": n, "mean_shells": v}
            for pf, by_n in sorted(ncp1_scaling.items())
            for n, v in sorted(by_n.items())
        ]
        _write_csv(output_dir / "gate1_ncp1_scaling.csv", csv_rows)
        print("NCP-1 shell scaling: done")
    else:
        print("  [warn] No NCP-1 JSONL data found")

    # -- NCP-1 bridge exponent gamma (B_{k,k+1} ~ |S_k|^gamma, Sec. 5.1) --
    ncp1_gamma, ncp1_gamma_r2, ncp1_gamma_n = compute_ncp1_bridge_exponent(jsonl_files)
    if not math.isnan(ncp1_gamma):
        print(f"NCP-1 bridge exponent: gamma={ncp1_gamma:.3f} (R^2={ncp1_gamma_r2:.3f}, n={ncp1_gamma_n})")
    else:
        print("  [warn] Not enough NCP-1 bridge-count data to fit gamma")

    # -- NMH-7 pilot warmup rate --
    nmh7_ok_rate: Optional[float] = None
    nmh7_runs = []
    nmh7_pilot_pkl_dir = results_dir / "pkl" / "NMH7_pilot"
    if nmh7_pilot_pkl_dir.exists():
        nmh7_runs = _load_pkls(nmh7_pilot_pkl_dir)
    if nmh7_runs:
        ok = sum(1 for r in nmh7_runs if r.warmup_ok)
        nmh7_ok_rate = ok / len(nmh7_runs)
        print(f"NMH-7 pilot warmup OK rate: {nmh7_ok_rate:.2f} ({ok}/{len(nmh7_runs)})")

    # -- Task audit --
    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    # -- Recommended b values for Phase 2 --
    rec_b = recommend_b_values(boundary_ii_iii)

    # -- Slope-1 gate criterion --
    slope_confirmed = any(
        not math.isnan(s) and abs(s - 1.0) <= 0.15
        for a, (s, se, n) in nmh1_slopes.items()
        if a == 0.5
    ) if nmh1_slopes else False

    gate_pass = slope_confirmed

    # -- Write gate review JSON --
    review = {
        "run_parametrization": {
            "phase1_results": str(results_dir),
            "queue_dir": str(args.queue_dir),
            "suffix": suffix,
            "gossip_protocol": "synchronous" if suffix else "asynchronous",
            "output_dir": str(output_dir),
        },
        "nmh1_slopes": {str(a): s for a, (s, se, n) in nmh1_slopes.items()},
        "nmh1_slope_ses": {str(a): se for a, (s, se, n) in nmh1_slopes.items()},
        "slope_confirmed": slope_confirmed,
        "nmh3_regime_boundaries": {
            "I_II": boundary_i_ii,
            "II_III": boundary_ii_iii,
        },
        "nmh7_pilot_warmup_ok_rate": nmh7_ok_rate,
        "ncp1_shell_scaling": {
            f"p_f={pf}": {str(n): v for n, v in by_n.items()}
            for pf, by_n in ncp1_scaling.items()
        },
        "ncp1_gamma_estimate": ncp1_gamma if not math.isnan(ncp1_gamma) else None,
        "ncp1_gamma_r_squared": ncp1_gamma_r2 if not math.isnan(ncp1_gamma_r2) else None,
        "recommended_b_values": rec_b,
        "gate1_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "Slope-1 confirmed." if slope_confirmed
            else "Slope-1 not confirmed — check local_steps or Regime C conditions."
        ),
    }

    review_path = output_dir / "gate1_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate 1 {'PASS' if gate_pass else 'FAIL'} — review written to {review_path}")


if __name__ == "__main__":
    main()
