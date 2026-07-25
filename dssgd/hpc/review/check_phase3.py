# -*- coding: utf-8 -*-
"""Gate 3 review script: filter composition, 1/f noise, NCP decoupling, power-law pilot.

Produces:
  review/gate3_review.json   — machine-readable gate decision (input to generate_queue --phase 4)
  review/gate3_nmh5_filter.csv
  review/gate3_nmh7_balance.csv
  review/gate3_ncp3_chi.csv
  review/gate3_nmh4_pilot.csv

Usage:
  python -m hpc.review.check_phase3 \\
      --phase3-results results/phase3 \\
      --queue-dir queue/phase3 \\
      --output-dir review/
"""
import argparse
import csv
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
from analysis.ncp_runner import NCPRun
from analysis.nmh_observables import cascade_depth, cascade_size, detailed_balance_ratio
from analysis.ncp_observables import decoupling_chi


# ---------------------------------------------------------------------------
# NMH-5: filter composition profile
# ---------------------------------------------------------------------------


def compute_nmh5_filter_profile(
    pkl_dir: Path,
) -> Dict[Tuple[float, float], Tuple[float, float, int]]:
    """Return {(a, b_on_a): (mean_cascade_depth, std, n_seeds)}."""
    by_key: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        b_on_a = round(run.b / run.a, 4) if run.a > 0 else float("nan")
        key = (run.a, b_on_a)
        if run.nucleation_leaf is not None:
            d = cascade_depth(run.centroid_traj, run.nucleation_leaf,
                              run.theta_A, run.theta_B, epsilon=0.2, persistence=3)
        else:
            d = 0
        by_key[key].append(d)

    result: Dict[Tuple[float, float], Tuple[float, float, int]] = {}
    for key, depths in sorted(by_key.items()):
        arr = np.array(depths)
        result[key] = (float(arr.mean()), float(arr.std()), len(arr))
    return result


def check_filter_monotone(
    profile: Dict[Tuple[float, float], Tuple[float, float, int]],
) -> bool:
    """Return True if, for each fixed a, cascade depth is non-decreasing in b/a."""
    by_a: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    for (a, b_on_a), (mean_d, _std, _n) in profile.items():
        by_a[a].append((b_on_a, mean_d))
    for a, items in by_a.items():
        items.sort(key=lambda x: x[0])
        depths = [d for _, d in items]
        if not all(depths[i] <= depths[i + 1] + 0.5 for i in range(len(depths) - 1)):
            return False
    return True


# ---------------------------------------------------------------------------
# NMH-7: detailed balance
# ---------------------------------------------------------------------------


def compute_nmh7_balance(
    pkl_dir: Path,
) -> Dict[int, Tuple[float, int]]:
    """Return {distance: (mean_empirical_ratio, n_runs)} from NMH-7 pickles.

    Uses detailed_balance_ratio() which computes fwd/bwd transition counts.
    """
    by_d: Dict[int, List[float]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        # For NMH-7 (b=0), nucleation leaf may be absent; use leaf 0 as reference
        src = run.nucleation_leaf if run.nucleation_leaf is not None else 0
        ratios = detailed_balance_ratio(run.centroid_traj, src,
                                        run.theta_A, run.theta_B, epsilon=0.2)
        for d, r in ratios.items():
            if not math.isnan(r):
                by_d[d].append(r)

    return {d: (float(np.mean(vals)), len(vals)) for d, vals in sorted(by_d.items())}


# ---------------------------------------------------------------------------
# NCP-3: decoupling profile
# ---------------------------------------------------------------------------


def compute_ncp3_chi_matrix(
    pkl_dir: Path,
) -> Dict[Tuple[int, int], Tuple[float, float, int]]:
    """Return {(inner_shell, outer_shell): (mean_chi, std_chi, n_runs)}."""
    by_pair: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NCPRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        shells = sorted(run.shell_assigns.values())
        if len(set(shells)) < 2:
            continue
        all_shells = sorted(set(shells))
        for i_idx in range(len(all_shells)):
            for o_idx in range(i_idx + 1, len(all_shells)):
                inner = all_shells[i_idx]
                outer = all_shells[o_idx]
                chi = decoupling_chi(run.shell_traj, inner, outer,
                                     run.theta_A, run.theta_B, epsilon=0.2)
                if not math.isnan(chi):
                    by_pair[(inner, outer)].append(chi)

    result: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
    for pair, vals in sorted(by_pair.items()):
        arr = np.array(vals)
        result[pair] = (float(arr.mean()), float(arr.std()), len(arr))
    return result


def check_chi_monotone(chi_matrix: Dict[Tuple[int, int], Tuple[float, float, int]]) -> bool:
    """Return True if chi(inner, outer) increases with (outer - inner)."""
    if not chi_matrix:
        return False
    items = [(outer - inner, mean_chi) for (inner, outer), (mean_chi, _, _) in chi_matrix.items()]
    items.sort()
    if len(items) < 2:
        return True
    means = [m for _, m in items]
    return all(means[i] <= means[i + 1] + 0.05 for i in range(len(means) - 1))


# ---------------------------------------------------------------------------
# NMH-4 pilot: power-law exponent
# ---------------------------------------------------------------------------


def fit_powerlaw_mle(cascade_sizes: List[int], s_min: int = 2) -> Tuple[float, float]:
    """MLE estimator for P(s) ~ s^{-tau} (Clauset et al. 2009).

    tau = 1 + n / (sum_i ln(s_i / (s_min - 0.5)))
    Returns (tau, tau_se).
    """
    data = [s for s in cascade_sizes if s >= s_min]
    n = len(data)
    if n < 5:
        return float("nan"), float("nan")
    log_terms = [math.log(s / (s_min - 0.5)) for s in data]
    tau = 1.0 + n / sum(log_terms)
    # Standard error: tau_se ~ (tau - 1) / sqrt(n)
    tau_se = (tau - 1.0) / math.sqrt(n)
    return tau, tau_se


def compute_nmh4_pilot_sizes(pkl_dir: Path) -> List[int]:
    """Return list of cascade_size values from NMH-4 pilot pickles."""
    sizes = []
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        s = cascade_size(run.centroid_traj, run.theta_A, run.theta_B,
                         t_horizon=run.t_horizon, epsilon=0.2)
        sizes.append(s)
    return sizes


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3 review for Phase 3 outputs.")
    parser.add_argument(
        "--phase3-results", type=Path, default=Path("results/phase3a"),
        help="Async by default (results/phase3a/, matching generate_queue.py's "
             "phase=='3a' output). Pass results/phase3s + --suffix S for the "
             "synchronous variant.",
    )
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phase3a"))
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to review/<phase3-results basename> (e.g. "
             "review/phase3s/ for --phase3-results results/phase3s), so "
             "async/sync reviews never collide or need a manually-labelled "
             "path. Pass explicitly to override.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix appended to NMH5/NMH7/NCP3 pkl dir names for the "
             "synchronous variant (e.g. 'S' for phase 3s: NMH5S/NMH7S/NCP3S "
             "-- see generate_queue.py's phase=='3s' branch). The pilot dir "
             "is a special case: generate_queue.py names it 'NMH4S_pilot' "
             "(suffix inserted before '_pilot'), not 'NMH4_pilotS'.",
    )
    args = parser.parse_args()

    results_dir = args.phase3_results
    output_dir = args.output_dir or Path("review") / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    # -- NMH-5: filter composition --
    nmh5_pkl_dir = results_dir / "pkl" / f"NMH5{suffix}"
    filter_profile: Dict[Tuple[float, float], Tuple[float, float, int]] = {}
    filter_ok = False
    if nmh5_pkl_dir.exists():
        filter_profile = compute_nmh5_filter_profile(nmh5_pkl_dir)
        filter_ok = check_filter_monotone(filter_profile)
        csv_rows = [
            {"a": a, "b_over_a": b_on_a, "mean_d_prop": m, "std_d_prop": s, "n_seeds": n}
            for (a, b_on_a), (m, s, n) in sorted(filter_profile.items())
        ]
        _write_csv(output_dir / "gate3_nmh5_filter.csv", csv_rows)
        print(f"NMH-5 filter monotone: {filter_ok}")
    else:
        print(f"  [warn] {nmh5_pkl_dir} not found — skipping NMH-5")

    # -- NMH-7: detailed balance --
    nmh7_pkl_dir = results_dir / "pkl" / f"NMH7{suffix}"
    balance: Dict[int, Tuple[float, int]] = {}
    balance_ok = False
    if nmh7_pkl_dir.exists():
        balance = compute_nmh7_balance(nmh7_pkl_dir)
        # For b=0 (symmetric), expect ratio ~ 1 at all distances
        ratios = [r for r, _n in balance.values()]
        balance_ok = bool(ratios) and all(0.5 <= r <= 2.0 for r in ratios)
        csv_rows = [
            {"distance": d, "mean_ratio": r, "n_runs": n}
            for d, (r, n) in sorted(balance.items())
        ]
        _write_csv(output_dir / "gate3_nmh7_balance.csv", csv_rows)
        print(f"NMH-7 detailed balance OK: {balance_ok}")
    else:
        print(f"  [warn] {nmh7_pkl_dir} not found — skipping NMH-7")

    # -- NCP-3: decoupling profile --
    ncp3_pkl_dir = results_dir / "pkl" / f"NCP3{suffix}"
    chi_matrix: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
    decoupling_ok = False
    if ncp3_pkl_dir.exists():
        chi_matrix = compute_ncp3_chi_matrix(ncp3_pkl_dir)
        decoupling_ok = check_chi_monotone(chi_matrix)
        csv_rows = [
            {"inner_shell": inner, "outer_shell": outer,
             "mean_chi": m, "std_chi": s, "n_runs": n}
            for (inner, outer), (m, s, n) in sorted(chi_matrix.items())
        ]
        _write_csv(output_dir / "gate3_ncp3_chi.csv", csv_rows)
        print(f"NCP-3 chi monotone: {decoupling_ok}")
    else:
        print(f"  [warn] {ncp3_pkl_dir} not found — skipping NCP-3")

    # -- NMH-4 pilot: power-law --
    nmh4_pkl_dir = results_dir / "pkl" / f"NMH4{suffix}_pilot"
    tau, tau_se = float("nan"), float("nan")
    cascade_sizes: List[int] = []
    proceed_to_full = False
    if nmh4_pkl_dir.exists():
        cascade_sizes = compute_nmh4_pilot_sizes(nmh4_pkl_dir)
        if cascade_sizes:
            tau, tau_se = fit_powerlaw_mle(cascade_sizes)
            proceed_to_full = not math.isnan(tau) and 1.5 <= tau <= 3.0
            from collections import Counter
            counts = Counter(cascade_sizes)
            csv_rows = [
                {"cascade_size": s, "count": c}
                for s, c in sorted(counts.items())
            ]
            _write_csv(output_dir / "gate3_nmh4_pilot.csv", csv_rows)
            tau_lo = tau - 2 * tau_se if not math.isnan(tau_se) else float("nan")
            tau_hi = tau + 2 * tau_se if not math.isnan(tau_se) else float("nan")
            print(f"NMH-4 pilot tau = {tau:.3f} +/- {tau_se:.3f} (95% CI: [{tau_lo:.2f}, {tau_hi:.2f}])")
    else:
        print(f"  [warn] {nmh4_pkl_dir} not found — skipping NMH-4 pilot")

    task_counts = _task_counts(args.queue_dir)

    gate_pass = filter_ok and balance_ok and decoupling_ok and proceed_to_full

    # Suggest depth for Phase 4 NMH-4 full run
    if not math.isnan(tau) and tau < 2.0:
        phase4_modifications = "Increase NMH-4 to depth=7 for better power-law tail."
        depth_note = "depth=7"
    else:
        phase4_modifications = "NMH-4 proceeds with default depth=6."
        depth_note = "depth=6"

    review = {
        "run_parametrization": {
            "phase3_results": str(results_dir),
            "queue_dir": str(args.queue_dir),
            "suffix": suffix,
            "gossip_protocol": "synchronous" if suffix else "asynchronous",
            "output_dir": str(output_dir),
        },
        "nmh5_filter_confirmed": filter_ok,
        "nmh5_d_prop_by_a_b_over_a": {
            f"a={a}_bova={b_on_a}": m
            for (a, b_on_a), (m, _s, _n) in filter_profile.items()
        },
        "nmh7_detailed_balance_ok": balance_ok,
        "nmh7_balance_ratios": {str(d): r for d, (r, _n) in balance.items()},
        "ncp3_decoupling_ok": decoupling_ok,
        "ncp3_chi_profile": {
            f"{inner}_{outer}": m
            for (inner, outer), (m, _s, _n) in chi_matrix.items()
        },
        "nmh4_pilot_tau": tau,
        "nmh4_pilot_tau_se": tau_se,
        "nmh4_pilot_tau_ci": [
            tau - 2 * tau_se if not math.isnan(tau_se) else float("nan"),
            tau + 2 * tau_se if not math.isnan(tau_se) else float("nan"),
        ],
        "nmh4_proceed_to_full": proceed_to_full,
        "phase4_modifications": phase4_modifications,
        "gate3_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            f"Filter={'OK' if filter_ok else 'FAIL'}, "
            f"Balance={'OK' if balance_ok else 'FAIL'}, "
            f"Decoupling={'OK' if decoupling_ok else 'FAIL'}, "
            f"tau={tau:.2f}. {depth_note}."
        ),
    }

    review_path = output_dir / "gate3_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate 3 {'PASS' if gate_pass else 'FAIL'} — review written to {review_path}")


if __name__ == "__main__":
    main()
