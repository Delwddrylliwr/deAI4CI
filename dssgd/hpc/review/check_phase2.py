# -*- coding: utf-8 -*-
"""Gate 2 review script: nucleation curve, variance decomposition, NCP-2 directionality.

Produces:
  review/gate2_review.json   — machine-readable gate decision (input to generate_queue --phase 3)
  review/gate2_nmh2_q_l.csv
  review/gate2_nmh6_variance.csv
  review/gate2_ncp2_cascade.csv

Usage:
  python -m hpc.review.check_phase2 \\
      --phase2-results results/phase2 \\
      --queue-dir queue/phase2 \\
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

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.natural_cascade import NaturalCascadeRun
from analysis.ncp_runner import NCPRun
from analysis.nmh_observables import hierarchical_variance_decomposition


# ---------------------------------------------------------------------------
# NMH-2: nucleation probability curve q_l(b)
# ---------------------------------------------------------------------------


def compute_nmh2_q_l(pkl_dir: Path) -> Dict[float, Tuple[float, float, int]]:
    """Return {b: (empirical_q_l, se, n_seeds)} from NMH-2 runs.

    q_l = fraction of runs where any leaf flipped (nucleation_leaf is not None).
    """
    by_b: Dict[float, List[bool]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        by_b[run.b].append(run.nucleation_leaf is not None)

    result: Dict[float, Tuple[float, float, int]] = {}
    for b, flags in sorted(by_b.items()):
        arr = np.array(flags, dtype=float)
        p = float(arr.mean())
        se = float(arr.std() / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        result[b] = (p, se, len(arr))
    return result


def _voter_q_l(b: float, theta: float, a: float, leaf_size: int) -> float:
    """Voter-theory nucleation probability (Eq. 14).

    q_l = (1 - exp(-2*θ*(b/a))) / (1 - exp(-2*θ*(b/a)*M))
    where M = leaf_size (level-1 module size = M_0).
    """
    h = theta * (b / max(a, 1e-12))
    M = float(leaf_size)
    num = 1.0 - math.exp(-2.0 * h)
    den = 1.0 - math.exp(-2.0 * h * M)
    if abs(den) < 1e-12:
        return 1.0 / M
    return num / den


def fit_voter_q_l(
    b_values: List[float],
    q_l_empirical: List[float],
    a: float,
    leaf_size: int,
) -> Tuple[float, float]:
    """Fit voter formula to empirical q_l values.  Returns (chi2, theta_fit).

    Falls back to (nan, nan) if scipy is not available or the fit fails.
    """
    if not _SCIPY or len(b_values) < 3:
        return float("nan"), float("nan")

    b_arr = np.array(b_values)
    q_arr = np.array(q_l_empirical)

    def model(b, theta):
        return np.array([_voter_q_l(float(bi), float(theta), a, leaf_size) for bi in b])

    try:
        popt, _ = curve_fit(model, b_arr, q_arr, p0=[1.0], bounds=(0.01, 100.0),
                            maxfev=5000)
        theta_fit = float(popt[0])
        q_pred = model(b_arr, theta_fit)
        residuals = q_arr - q_pred
        chi2 = float(np.sum((residuals ** 2) / np.maximum(q_pred * (1 - q_pred) / 30, 1e-6)))
        return chi2, theta_fit
    except Exception:
        return float("nan"), float("nan")


def recommend_b_values_phase3(
    q_l_data: Dict[float, Tuple[float, float, int]],
    theta_fit: float,
    a: float,
) -> List[float]:
    """Suggest b/a sweep for NMH-5 based on the calibrated nucleation curve.

    Targets the region where q_l transitions from ~0.1 to ~0.9 using 5 values.
    """
    if math.isnan(theta_fit) or not q_l_data:
        return [0.02, 0.03, 0.04, 0.05, 0.06]
    # Find b/a values giving q_l in [0.1, 0.9]
    b_on_a_candidates = np.linspace(0.01, 0.1, 200)
    from analysis.ncp_experiments import experiment_NCP1_graph_configs  # noqa (import check)
    import analysis.natural_cascade_experiments  # noqa
    leaf_size = 4
    q_vals = [_voter_q_l(float(b * a), theta_fit, a, leaf_size) for b in b_on_a_candidates]
    valid = [(b, q) for b, q in zip(b_on_a_candidates, q_vals) if 0.05 <= q <= 0.95]
    if not valid:
        return [0.02, 0.03, 0.04, 0.05, 0.06]
    lo = valid[0][0]
    hi = valid[-1][0]
    return [round(float(v), 4) for v in np.linspace(lo, hi, 5)]


# ---------------------------------------------------------------------------
# NMH-6: variance decomposition
# ---------------------------------------------------------------------------


def compute_nmh6_decomp(
    pkl_dir: Path,
) -> Dict[int, Tuple[float, float, int]]:
    """Return {level: (mean_V_ell, std_V_ell, n_runs)} from NMH-6 pickles."""
    by_level: Dict[int, List[float]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        decomp = hierarchical_variance_decomposition(
            run.centroid_traj, run.branching, run.depth
        )
        for level, v in decomp.items():
            by_level[level].append(v)

    result: Dict[int, Tuple[float, float, int]] = {}
    for level, vals in sorted(by_level.items()):
        arr = np.array(vals)
        result[level] = (float(arr.mean()), float(arr.std()), len(arr))
    return result


# ---------------------------------------------------------------------------
# NCP-2: directionality ratio
# ---------------------------------------------------------------------------


def compute_ncp2_directionality(pkl_dir: Path) -> Tuple[float, float, int, int]:
    """Return (p_outward_reach, p_inward_reach, n_outward, n_inward).

    Outward = clamped innermost shell (high k-core); inward = clamped outermost.
    'Reach' = cascade reached any node in the opposite extremal shell.
    """
    outward_flags: List[bool] = []
    inward_flags: List[bool] = []

    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NCPRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue

        max_shell = run.graph_stats.get("max_shell", 0)
        # Identify whether this was an outward (clamped innermost) or inward run
        clamped = getattr(run, "clamped_shell", None)
        if clamped is None:
            continue

        # A cascade "reaches the other end" if any node in the opposite extremal shell flipped
        shell_assigns = run.shell_assigns
        if clamped == max_shell:
            # Outward: check if outermost shell (min shell) has any flipped nodes
            target_shell = min(shell_assigns.values())
        else:
            # Inward: check if innermost shell (max shell) has any flipped nodes
            target_shell = max_shell

        target_nodes = {n for n, sh in shell_assigns.items() if sh == target_shell}
        reached = any(
            row["t_flip_absolute"] is not None and row["node"] in target_nodes
            for row in run.flip_table
        )

        if clamped == max_shell:
            outward_flags.append(reached)
        else:
            inward_flags.append(reached)

    p_out = float(np.mean(outward_flags)) if outward_flags else float("nan")
    p_in = float(np.mean(inward_flags)) if inward_flags else float("nan")
    return p_out, p_in, len(outward_flags), len(inward_flags)


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
    parser = argparse.ArgumentParser(description="Gate 2 review for Phase 2 outputs.")
    parser.add_argument(
        "--phase2-results", type=Path, default=Path("results/phase2a"),
        help="Async by default (results/phase2a/, matching generate_queue.py's "
             "phase=='2a' output). Pass results/phase2s + --suffix S for the "
             "synchronous variant.",
    )
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phase2a"))
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to review/<phase2-results basename> (e.g. "
             "review/phase2s/ for --phase2-results results/phase2s), so "
             "async/sync reviews never collide or need a manually-labelled "
             "path. Pass explicitly to override.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Suffix appended to NMH2/NMH6/NCP2 pkl dir names for the "
             "synchronous variant (e.g. 'S' for phase 2s: NMH2S/NMH6S/NCP2S "
             "-- see generate_queue.py's phase=='2s' branch).",
    )
    args = parser.parse_args()

    results_dir = args.phase2_results
    output_dir = args.output_dir or Path("review") / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    # -- NMH-2: q_l(b) curve --
    nmh2_pkl_dir = results_dir / "pkl" / f"NMH2{suffix}"
    q_l_data: Dict[float, Tuple[float, float, int]] = {}
    chi2 = float("nan")
    theta_fit = float("nan")
    a_nmh2 = 1.0  # fixed in experiment_NMH2
    leaf_size = 4
    if nmh2_pkl_dir.exists():
        q_l_data = compute_nmh2_q_l(nmh2_pkl_dir)
        b_vals = sorted(q_l_data)
        q_emp = [q_l_data[b][0] for b in b_vals]
        chi2, theta_fit = fit_voter_q_l(b_vals, q_emp, a_nmh2, leaf_size)
        # Theoretical prediction at each b
        csv_rows = [
            {
                "b": b,
                "q_l_empirical": q_l_data[b][0],
                "q_l_se": q_l_data[b][1],
                "q_l_theory": (
                    _voter_q_l(b, theta_fit, a_nmh2, leaf_size)
                    if not math.isnan(theta_fit) else float("nan")
                ),
                "n_seeds": q_l_data[b][2],
            }
            for b in b_vals
        ]
        _write_csv(output_dir / "gate2_nmh2_q_l.csv", csv_rows)
        print(f"NMH-2: chi^2={chi2:.2f}, theta_fit={theta_fit:.4f}")
    else:
        print(f"  [warn] {nmh2_pkl_dir} not found — skipping NMH-2")

    rec_b = recommend_b_values_phase3(q_l_data, theta_fit, a_nmh2)

    # -- NMH-6: variance decomposition --
    nmh6_pkl_dir = results_dir / "pkl" / f"NMH6{suffix}"
    decomp: Dict[int, Tuple[float, float, int]] = {}
    decomp_ok = False
    if nmh6_pkl_dir.exists():
        decomp = compute_nmh6_decomp(nmh6_pkl_dir)
        if decomp:
            levels = sorted(decomp)
            means = [decomp[l][0] for l in levels]
            decomp_ok = all(means[i] >= means[i + 1] - 1e-6 for i in range(len(means) - 1))
            csv_rows = [
                {"level": l, "mean_V": decomp[l][0], "std_V": decomp[l][1], "n_runs": decomp[l][2]}
                for l in levels
            ]
            _write_csv(output_dir / "gate2_nmh6_variance.csv", csv_rows)
            print(f"NMH-6 variance decomp OK: {decomp_ok}")
    else:
        print(f"  [warn] {nmh6_pkl_dir} not found — skipping NMH-6")

    # -- NCP-2: directionality --
    ncp2_pkl_dir = results_dir / "pkl" / f"NCP2{suffix}"
    p_out, p_in, n_out, n_in = float("nan"), float("nan"), 0, 0
    asymmetry_ratio = float("nan")
    if ncp2_pkl_dir.exists():
        p_out, p_in, n_out, n_in = compute_ncp2_directionality(ncp2_pkl_dir)
        asymmetry_ratio = p_out / max(p_in, 1e-9) if not math.isnan(p_in) and p_in > 0 else float("nan")
        csv_rows = [
            {"direction": "outward", "p_reach_far_shell": p_out, "n_seeds": n_out},
            {"direction": "inward",  "p_reach_far_shell": p_in,  "n_seeds": n_in},
        ]
        _write_csv(output_dir / "gate2_ncp2_cascade.csv", csv_rows)
        print(f"NCP-2 asymmetry ratio (outward/inward): {asymmetry_ratio:.2f}")
    else:
        print(f"  [warn] {ncp2_pkl_dir} not found — skipping NCP-2")

    task_counts = _task_counts(args.queue_dir)

    gate_pass = chi2 < 10 and not math.isnan(chi2) and (math.isnan(asymmetry_ratio) or asymmetry_ratio > 2.0)

    review = {
        "run_parametrization": {
            "phase2_results": str(results_dir),
            "queue_dir": str(args.queue_dir),
            "suffix": suffix,
            "gossip_protocol": "synchronous" if suffix else "asynchronous",
            "output_dir": str(output_dir),
        },
        "nmh2_q_l_by_b": {str(b): q_l_data[b][0] for b in sorted(q_l_data)},
        "nmh2_theory_fit_chi2": chi2,
        "q_l_calibration": {"theta_fit": theta_fit, "a": a_nmh2, "leaf_size": leaf_size},
        "nmh6_variance_decomp": {str(l): decomp[l][0] for l in sorted(decomp)},
        "nmh6_decomp_ok": decomp_ok,
        "ncp2_p_outward": p_out,
        "ncp2_p_inward": p_in,
        "ncp2_asymmetry_ratio": asymmetry_ratio,
        "recommended_b_values_phase3": rec_b,
        "gate2_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            f"Voter fit chi^2={chi2:.2f}. NCP-2 asymmetry ratio={asymmetry_ratio:.2f}."
        ),
    }

    review_path = output_dir / "gate2_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate 2 {'PASS' if gate_pass else 'FAIL'} — review written to {review_path}")


if __name__ == "__main__":
    main()
