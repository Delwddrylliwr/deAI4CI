# -*- coding: utf-8 -*-
"""Distributional analysis for Phase 1sB (sync gossip transition zone).

Tests whether cascade size distributions in the a × local_steps transition
zone follow a power law (Griffiths phase) or exponential decay (subcritical).

Uses the Vuong test (likelihood ratio for non-nested models) to compare
power-law and exponential fits. Standard KS goodness-of-fit tests are NOT
used for this comparison because p-values are invalid when parameters are
estimated from the same data.

Two-sample KS is used separately to locate phase boundaries by comparing
empirical distributions between adjacent (a, ls) conditions.

Usage:
  python -m hpc.review.check_phase1sb --phase 1sb
  python -m hpc.review.check_phase1sb --pkl-dir results/phase1sb/pkl/NMH1sbSP \\
      --out-dir review/phase1sb/
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.natural_cascade import NaturalCascadeRun
from hpc.review.check_phase1 import _load_pkls, _write_csv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sizes(pkl_dir: Path) -> Dict[Tuple[float, int], List[int]]:
    """Map (a, local_steps) → list of cascade sizes across seeds.

    Cascade size = number of non-source leaves with t_flip_relative not None.
    Zero-size entries (no propagation beyond source) are retained.
    """
    by_cell: Dict[Tuple[float, int], List[int]] = defaultdict(list)
    for run in _load_pkls(pkl_dir):
        if not run.warmup_ok:
            continue
        size = sum(
            1 for row in run.flip_table
            if row.get("t_flip_relative") is not None
        )
        by_cell[(run.a, run.local_steps)].append(size)
    return dict(by_cell)


# ---------------------------------------------------------------------------
# Distributional fitting and Vuong test
# ---------------------------------------------------------------------------


def _pl_loglik(sizes_nonzero: np.ndarray, alpha: float, s_min: float) -> float:
    """Log-likelihood of discrete power law P(s) ∝ s^{-alpha} for s >= s_min."""
    return float(np.sum(np.log(sizes_nonzero ** (-alpha) / _pl_normaliser(alpha, s_min, int(sizes_nonzero.max())))))


def _pl_normaliser(alpha: float, s_min: float, s_max: int) -> float:
    """Normalisation constant for discrete power law on [s_min, s_max]."""
    s_vals = np.arange(int(s_min), s_max + 1, dtype=float)
    return float(np.sum(s_vals ** (-alpha)))


def _exp_loglik(sizes_nonzero: np.ndarray, lam: float) -> float:
    """Log-likelihood of exponential P(s) = lam * exp(-lam * s) (continuous approx)."""
    return float(np.sum(stats.expon.logpdf(sizes_nonzero, scale=1.0 / lam)))


def fit_and_test(sizes: List[int]) -> Dict[str, Any]:
    """Fit power-law and exponential to non-zero cascade sizes; compare via Vuong test.

    Returns a dict with cascade_rate, fit parameters, per-observation log-likelihoods,
    and the Vuong z-statistic (positive → power law preferred, negative → exponential).
    Returns NaN for fit results when fewer than 5 non-zero cascades are available.
    """
    n_total = len(sizes)
    arr = np.array(sizes, dtype=float)
    nonzero = arr[arr > 0]
    n_cascade = len(nonzero)
    cascade_rate = n_cascade / n_total if n_total > 0 else float("nan")

    empty = {
        "cascade_rate": cascade_rate,
        "n_total": n_total,
        "n_cascade": n_cascade,
        "pl_alpha": float("nan"),
        "exp_lambda": float("nan"),
        "vuong_z": float("nan"),
        "vuong_p": float("nan"),
        "pl_loglik": float("nan"),
        "exp_loglik": float("nan"),
    }

    if n_cascade < 5:
        return empty

    s_min = 1.0
    # Power-law MLE: Clauset et al. 2009 Eq. (3.6) for discrete case
    pl_alpha = 1.0 + n_cascade / float(np.sum(np.log(nonzero / (s_min - 0.5))))
    # Exponential MLE
    exp_lam = 1.0 / float(nonzero.mean())

    s_max = int(nonzero.max())
    pl_norm = _pl_normaliser(pl_alpha, s_min, s_max)

    # Per-observation log-likelihoods for Vuong test
    pl_logliks = np.log(nonzero ** (-pl_alpha) / pl_norm)
    exp_logliks = stats.expon.logpdf(nonzero, scale=1.0 / exp_lam)

    diff = pl_logliks - exp_logliks
    vuong_z = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n_cascade))) if diff.std(ddof=1) > 0 else float("nan")
    vuong_p = float(2.0 * (1.0 - stats.norm.cdf(abs(vuong_z)))) if not math.isnan(vuong_z) else float("nan")

    return {
        "cascade_rate": cascade_rate,
        "n_total": n_total,
        "n_cascade": n_cascade,
        "pl_alpha": round(pl_alpha, 4),
        "exp_lambda": round(exp_lam, 4),
        "vuong_z": round(vuong_z, 3),
        "vuong_p": round(vuong_p, 4),
        "pl_loglik": round(float(pl_logliks.sum()), 2),
        "exp_loglik": round(float(exp_logliks.sum()), 2),
    }


# ---------------------------------------------------------------------------
# Boundary detection via two-sample KS
# ---------------------------------------------------------------------------


def run_boundary_tests(
    sizes_by_cell: Dict[Tuple[float, int], List[int]],
) -> List[Dict[str, Any]]:
    """Two-sample KS tests between adjacent a-values at each local_steps.

    Two-sample KS compares empirical distributions without fitting parameters,
    so standard p-values are valid here.
    """
    ls_vals = sorted({ls for _, ls in sizes_by_cell})
    a_vals = sorted({a for a, _ in sizes_by_cell})

    rows = []
    for ls in ls_vals:
        for i in range(len(a_vals) - 1):
            a1, a2 = a_vals[i], a_vals[i + 1]
            s1 = sizes_by_cell.get((a1, ls), [])
            s2 = sizes_by_cell.get((a2, ls), [])
            if len(s1) < 5 or len(s2) < 5:
                continue
            ks_stat, p_val = stats.ks_2samp(s1, s2)
            rows.append({
                "ls": ls,
                "a1": a1,
                "a2": a2,
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_val, 4),
                "n1": len(s1),
                "n2": len(s2),
            })
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(pkl_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes_by_cell = load_sizes(pkl_dir)
    if not sizes_by_cell:
        print(f"No pkl files found under {pkl_dir}")
        return

    a_vals = sorted({a for a, _ in sizes_by_cell})
    ls_vals = sorted({ls for _, ls in sizes_by_cell})

    # ── Fit results ──────────────────────────────────────────────────────────
    fit_rows = []
    print(f"\nCascade size distributional fits (Vuong test: z>0 → power law, z<0 → exponential)")
    print(f"{'a':>6}  {'ls':>4}  {'rate':>5}  {'n_cas':>5}  {'pl_α':>6}  {'exp_λ':>6}  {'vuong_z':>7}  {'vuong_p':>7}  verdict")

    for a in a_vals:
        for ls in ls_vals:
            sizes = sizes_by_cell.get((a, ls), [])
            fit = fit_and_test(sizes)
            fit["a"] = a
            fit["ls"] = ls
            fit_rows.append(fit)

            z = fit["vuong_z"]
            p = fit["vuong_p"]
            if math.isnan(z):
                verdict = "insufficient data"
            elif p > 0.05:
                verdict = "indeterminate"
            elif z > 0:
                verdict = "POWER LAW"
            else:
                verdict = "exponential"

            print(
                f"{a:>6.3f}  {ls:>4d}  {fit['cascade_rate']:>5.2f}  "
                f"{fit['n_cascade']:>5d}  {fit['pl_alpha']:>6.3f}  "
                f"{fit['exp_lambda']:>6.3f}  {z:>7.3f}  {p:>7.4f}  {verdict}"
            )

    _write_csv(out_dir / "fit_results.csv", fit_rows)

    # ── Boundary tests ───────────────────────────────────────────────────────
    boundary_rows = run_boundary_tests(sizes_by_cell)
    _write_csv(out_dir / "ks_boundary.csv", boundary_rows)

    print(f"\nPhase boundary (two-sample KS, adjacent a-values, p < 0.05 = significant shift):")
    for row in boundary_rows:
        sig = " *" if row["p_value"] < 0.05 else ""
        print(f"  ls={row['ls']:>3d}  a={row['a1']}→{row['a2']}: "
              f"KS={row['ks_stat']:.3f}  p={row['p_value']:.4f}{sig}")

    # ── Griffiths summary ────────────────────────────────────────────────────
    griffiths = [r for r in fit_rows
                 if not math.isnan(r.get("vuong_z", float("nan")))
                 and r.get("vuong_p", 1.0) < 0.05
                 and r.get("vuong_z", 0.0) > 0]
    if griffiths:
        print(f"\nCells where power law significantly preferred (Griffiths candidate):")
        for r in griffiths:
            print(f"  a={r['a']}, ls={r['ls']}: z={r['vuong_z']:.3f}, α={r['pl_alpha']:.3f}")
    else:
        print("\nNo cells show significant power-law preference over exponential.")

    print(f"\nOutputs written to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1sB distributional analysis: power law vs exponential cascade sizes."
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        help="Phase ID (e.g. '1sb'). Derives pkl-dir as results/phase{ID}/pkl/NMH1sbSP/.",
    )
    parser.add_argument("--pkl-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.phase is None and args.pkl_dir is None:
        parser.error("Provide --phase or --pkl-dir.")

    phase_id = args.phase.strip() if args.phase else None
    pkl_dir = args.pkl_dir or Path(f"results/phase{phase_id}/pkl/NMH1sbSP")
    out_dir = args.out_dir or Path(f"review/phase{phase_id}/")

    run_analysis(pkl_dir, out_dir)


if __name__ == "__main__":
    main()
