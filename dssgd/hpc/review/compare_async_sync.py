# -*- coding: utf-8 -*-
"""Cross-protocol comparison: async gossip (Phase xA) vs synchronous (Phase xS).

Loads NMH-1, NMH-1b, and NMH-3 results from both protocol variants and produces
side-by-side comparison CSVs and a console summary.

Usage:
  # Compare Phase 1A (results/phase1/) vs Phase 1S (results/phase1s/)
  python -m hpc.review.compare_async_sync --phase 1

  # Override directories explicitly
  python -m hpc.review.compare_async_sync \\
      --phaseA-dir results/phase1 \\
      --phaseS-dir results/phase1s \\
      --out-dir    results/comparison/phase1

Outputs written to --out-dir:
  comparison_slopes.csv         — protocol, a, slope, slope_se, n_seeds
  comparison_ls_threshold.csv   — protocol, ls, cascade_rate, mean_frac_flipped, bimodal
  comparison_phase_depths.csv   — protocol, a, mean_depth, std_depth, n_runs
"""
import argparse
import csv
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
from hpc.review.check_phase1 import (
    _group_tflip_by_distance,
    _linear_fit,
    _load_pkls,
    _write_csv,
    compute_nmh3_depths,
)


# ---------------------------------------------------------------------------
# Slope comparison (reuses check_phase1 internals)
# ---------------------------------------------------------------------------


def compute_slopes(runs: List[NaturalCascadeRun]) -> Dict[float, Tuple[float, float, int]]:
    """OLS slope of log₂(mean t_flip_relative) vs hierarchical distance d≥2.

    Returns {a: (slope, slope_se, n_seeds_with_flips)}.
    """
    grouped = _group_tflip_by_distance(runs)
    results: Dict[float, Tuple[float, float, int]] = {}
    for (a_val, _ls), by_d in grouped.items():
        fit_ds = np.array(
            [d for d in sorted(by_d) if d >= 2 and len(by_d[d]) >= 3], dtype=float
        )
        fit_means = np.array([np.mean(by_d[d]) for d in fit_ds])
        if len(fit_ds) < 2:
            results[a_val] = (float("nan"), float("nan"), 0)
            continue
        slope, _intercept, se = _linear_fit(fit_ds, fit_means)
        n_seeds = len(set(r.seed for r in runs if r.a == a_val and r.warmup_ok))
        results[a_val] = (slope, se, n_seeds)
    return results


# ---------------------------------------------------------------------------
# local_steps cascade-rate comparison
# ---------------------------------------------------------------------------


def compute_cascade_rates(
    runs: List[NaturalCascadeRun],
) -> Dict[int, Dict[str, Any]]:
    """Cascade rate and size distribution grouped by local_steps.

    Returns {ls: {cascade_rate, mean_frac_flipped, bimodal, sizes}}.
    bimodal=True when observed cascade sizes contain only 0 or the maximum
    possible leaf count (no intermediate values), with both extremes present.
    """
    by_ls: Dict[int, List[NaturalCascadeRun]] = defaultdict(list)
    for r in runs:
        if r.warmup_ok:
            by_ls[r.local_steps].append(r)

    result: Dict[int, Dict[str, Any]] = {}
    for ls, ls_runs in sorted(by_ls.items()):
        n_total = len(ls_runs)
        n_leaves = 2 ** ls_runs[0].depth * ls_runs[0].leaf_size - 1  # excludes source

        # Count leaves that flipped in each run
        sizes = []
        for r in ls_runs:
            flipped = sum(
                1 for row in r.flip_table
                if row.get("t_flip_relative") is not None
            )
            sizes.append(flipped)

        sizes_arr = np.array(sizes)
        n_cascade = int(np.sum(sizes_arr > 0))
        cascade_rate = n_cascade / n_total if n_total > 0 else float("nan")
        mean_frac = float(sizes_arr.mean()) / n_leaves if n_leaves > 0 else float("nan")

        unique_sizes = set(sizes)
        bimodal = (
            len(unique_sizes) <= 2
            and 0 in unique_sizes
            and n_cascade > 0
            and n_total - n_cascade > 0
        )

        result[ls] = {
            "cascade_rate": cascade_rate,
            "mean_frac_flipped": mean_frac,
            "bimodal": bimodal,
            "n_runs": n_total,
            "sizes": sizes,
        }
    return result


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def run_comparison(
    phaseA_dir: Path,
    phaseS_dir: Path,
    out_dir: Path,
    nmh1_subdir: str = "NMH1",
    nmh1s_subdir: str = "NMH1S",
    nmh1b_subdir: str = "NMH1b",
    nmh1bs_subdir: str = "NMH1bS",
    nmh3_subdir: str = "NMH3",
    nmh3s_subdir: str = "NMH3S",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── NMH-1 slopes ─────────────────────────────────────────────────────────
    slope_rows: List[Dict[str, Any]] = []
    for label, pkl_dir_rel, subdirname in [
        ("async",  phaseA_dir, nmh1_subdir),
        ("sync",   phaseS_dir, nmh1s_subdir),
    ]:
        pkl_dir = pkl_dir_rel / "pkl" / subdirname
        if not pkl_dir.exists():
            print(f"  [warn] {pkl_dir} not found — skipping {label} slopes")
            continue
        runs = _load_pkls(pkl_dir)
        slopes = compute_slopes(runs)
        for a_val, (slope, se, n) in sorted(slopes.items()):
            slope_rows.append({
                "protocol": label,
                "a": a_val,
                "slope": round(slope, 4) if not math.isnan(slope) else "nan",
                "slope_se": round(se, 4) if not math.isnan(se) else "nan",
                "n_seeds": n,
            })

    _write_csv(out_dir / "comparison_slopes.csv", slope_rows)
    print("\nSlope comparison (log₂(t_flip) vs hierarchical distance d):")
    for row in slope_rows:
        flag = ""
        if row["protocol"] == "async" and not isinstance(row["slope"], str):
            flag = " ✓" if abs(float(row["slope"]) - 1.0) < 0.15 else " ✗ (expected 1.0)"
        elif row["protocol"] == "sync" and not isinstance(row["slope"], str):
            flag = " ✓" if abs(float(row["slope"]) - 2.0) < 0.30 else " ✗ (expected ~2.0)"
        print(f"  {row['protocol']:5s}  a={row['a']}: slope={row['slope']} ± {row['slope_se']}{flag}")

    # Gate check: sync slope ≈ 2.0 at a=0.5
    sync_slope_05 = next(
        (float(r["slope"]) for r in slope_rows
         if r["protocol"] == "sync" and r["a"] == 0.5 and r["slope"] != "nan"),
        float("nan"),
    )
    sync_gate_ok = not math.isnan(sync_slope_05) and abs(sync_slope_05 - 2.0) < 0.30
    print(f"\n  Sync gate (|slope − 2.0| < 0.30 at a=0.5): "
          f"slope={sync_slope_05:.3f} → {'PASS' if sync_gate_ok else 'FAIL'}")

    # ── NMH-1b cascade rates ──────────────────────────────────────────────────
    ls_rows: List[Dict[str, Any]] = []
    for label, pkl_dir_rel, subdirname in [
        ("async", phaseA_dir, nmh1b_subdir),
        ("sync",  phaseS_dir, nmh1bs_subdir),
    ]:
        pkl_dir = pkl_dir_rel / "pkl" / subdirname
        if not pkl_dir.exists():
            print(f"  [warn] {pkl_dir} not found — skipping {label} ls threshold")
            continue
        runs = _load_pkls(pkl_dir)
        rates = compute_cascade_rates(runs)
        for ls, stats in rates.items():
            ls_rows.append({
                "protocol": label,
                "ls": ls,
                "cascade_rate": round(stats["cascade_rate"], 3),
                "mean_frac_flipped": round(stats["mean_frac_flipped"], 3),
                "bimodal": stats["bimodal"],
                "n_runs": stats["n_runs"],
            })

    _write_csv(out_dir / "comparison_ls_threshold.csv", ls_rows)
    print("\nCascade rate by local_steps:")
    for row in ls_rows:
        bm = " [BIMODAL]" if row["bimodal"] else ""
        print(f"  {row['protocol']:5s}  ls={row['ls']:4d}: "
              f"rate={row['cascade_rate']:.2f}  mean_frac={row['mean_frac_flipped']:.3f}{bm}")

    # ── NMH-3 phase depths ────────────────────────────────────────────────────
    depth_rows: List[Dict[str, Any]] = []
    for label, pkl_dir_rel, subdirname in [
        ("async", phaseA_dir, nmh3_subdir),
        ("sync",  phaseS_dir, nmh3s_subdir),
    ]:
        pkl_dir = pkl_dir_rel / "pkl" / subdirname
        if not pkl_dir.exists():
            print(f"  [warn] {pkl_dir} not found — skipping {label} phase depths")
            continue
        runs = _load_pkls(pkl_dir)
        depths = compute_nmh3_depths(pkl_dir)
        for a_val, (mean_d, std_d, n) in sorted(depths.items()):
            depth_rows.append({
                "protocol": label,
                "a": a_val,
                "mean_depth": round(mean_d, 3),
                "std_depth": round(std_d, 3),
                "n_runs": n,
            })

    _write_csv(out_dir / "comparison_phase_depths.csv", depth_rows)
    print("\nPhase structure (cascade depth by a-value):")
    for row in depth_rows:
        print(f"  {row['protocol']:5s}  a={row['a']}: "
              f"mean_depth={row['mean_depth']:.3f} ± {row['std_depth']:.3f} (n={row['n_runs']})")

    print(f"\nOutputs written to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare async gossip (Phase xA) vs synchronous (Phase xS) results."
    )
    parser.add_argument(
        "--phase", type=str, default=None,
        help=(
            "Phase number (e.g. '1'). Derives --phaseA-dir=results/phase{N}/ "
            "and --phaseS-dir=results/phase{N}s/ automatically."
        ),
    )
    parser.add_argument("--phaseA-dir", type=Path, default=None,
                        help="Override Phase A (async) results directory.")
    parser.add_argument("--phaseS-dir", type=Path, default=None,
                        help="Override Phase S (sync) results directory.")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Defaults to results/comparison/phase{N}/.",
    )
    args = parser.parse_args()

    if args.phase is None and (args.phaseA_dir is None or args.phaseS_dir is None):
        parser.error("Provide --phase N or both --phaseA-dir and --phaseS-dir.")

    phase_num = args.phase.strip() if args.phase else None

    phaseA_dir = args.phaseA_dir or Path(f"results/phase{phase_num}")
    phaseS_dir = args.phaseS_dir or Path(f"results/phase{phase_num}s")
    out_dir = args.out_dir or Path(f"results/comparison/phase{phase_num}")

    run_comparison(phaseA_dir, phaseS_dir, out_dir)


if __name__ == "__main__":
    main()
