# -*- coding: utf-8 -*-
"""Gate C2 review script: articulation ablation (paper2_social Claim 5.2 /
Sec. 5.2.1) -- m=1 "bare tree" vs m=4(+) NMH at matched total N.

Scores two things, at different levels of confidence:

1. STRUCTURAL sanity check (what gate_pass is keyed on): m>1 base modules
   show genuine within-module consensus formation under their own
   track_articulation_diagnostics window (chord barrier / parameter spread
   measurably contract from T_pre to T_post), while m=1 modules -- a single
   agent, no intra-module gossip possible at all -- show trivially zero
   spread throughout. This confirms the articulation mechanism (Claim 5.2's
   "tacit-to-explicit converter") is actually operating as designed before
   any weaker, harder-to-power comparison is trusted.

2. OBSERVATIONAL propagation comparison (reported, NOT gated): first-level
   stall rate (fraction of runs where cascade_depth==0 -- the innovation
   never reaches beyond its own source module) by leaf_size. This is the
   §5.2.1-motivated comparison, but per-run articulation_diagnostics only
   captures each module's PRE-CASCADE consensus tendency (the window runs
   once, between warmup and the force-flip/measurement phase -- see
   NaturalCascadeConfig's docstring), not the consensus state of the
   SPECIFIC module at the moment of its own stall. Classifying individual
   stall events as articulation- vs scope-limited (Claim 5.2's full
   ambition) would need per-stall-time snapshots this data doesn't carry;
   reported here as an aggregate stall-rate-by-m comparison instead, and
   left unpowered/ungated until a real campaign's seed count can support a
   proper test.

Produces:
  review/gatec2_review.json
  review/gatec2_consensus_by_m.csv
  review/gatec2_stall_rate_by_m.csv
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def compute_consensus_by_m(runs: List[NaturalCascadeRun]) -> Dict[int, Dict[str, float]]:
    """Per leaf_size: mean (spread_pre - spread_post) over every module in
    every run, and the fraction of modules whose diagnostic window
    converged (spread_post < articulation_consensus_eps) before exhausting
    max_intra_module_rounds."""
    by_m: Dict[int, List[dict]] = defaultdict(list)
    for run in runs:
        if not run.articulation_diagnostics:
            continue
        for module_diag in run.articulation_diagnostics.values():
            by_m[run.leaf_size].append(module_diag)

    result: Dict[int, Dict[str, float]] = {}
    for m, diags in sorted(by_m.items()):
        reductions = [d["spread_pre"] - d["spread_post"] for d in diags]
        result[m] = {
            "n_modules": len(diags),
            "mean_spread_reduction": float(np.mean(reductions)) if reductions else float("nan"),
            "mean_spread_pre": float(np.mean([d["spread_pre"] for d in diags])) if diags else float("nan"),
            "mean_spread_post": float(np.mean([d["spread_post"] for d in diags])) if diags else float("nan"),
            "frac_converged": float(np.mean([d["converged"] for d in diags])) if diags else float("nan"),
        }
    return result


def compute_stall_rate_by_m(runs: List[NaturalCascadeRun]) -> Dict[int, Dict[str, float]]:
    """Per leaf_size: fraction of (warmup_ok, nucleated) runs with
    cascade_depth==0 -- the force-flipped source module's innovation never
    reaches any other module (first-level stall, in the coarse
    per-run-not-per-event sense described in this module's docstring)."""
    by_m: Dict[int, List[int]] = defaultdict(list)
    for run in runs:
        if not run.warmup_ok or run.nucleation_leaf is None:
            continue
        depth = cascade_depth(
            run.centroid_traj, run.nucleation_leaf, run.theta_A, run.theta_B,
            epsilon=0.2, persistence=3,
        )
        by_m[run.leaf_size].append(depth)

    result: Dict[int, Dict[str, float]] = {}
    for m, depths in sorted(by_m.items()):
        arr = np.array(depths)
        result[m] = {
            "n_runs": len(arr),
            "stall_rate": float(np.mean(arr == 0)) if len(arr) else float("nan"),
            "mean_depth": float(arr.mean()) if len(arr) else float("nan"),
        }
    return result


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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
    parser = argparse.ArgumentParser(description="Gate C2 review: articulation ablation.")
    parser.add_argument("--phasec2-results", type=Path, default=Path("results/phasec2"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phasec2"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phasec2"))
    args = parser.parse_args()

    results_dir: Path = args.phasec2_results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = results_dir / "pkl" / "C2"
    runs = _load_pkls(pkl_dir) if pkl_dir.exists() else []
    if not runs:
        print(f"  [warn] {pkl_dir} not found or empty -- nothing to score")

    consensus_by_m = compute_consensus_by_m(runs)
    stall_by_m = compute_stall_rate_by_m(runs)

    print("Consensus formation by m:")
    for m, stats in sorted(consensus_by_m.items()):
        print(f"  m={m}: mean_spread_reduction={stats['mean_spread_reduction']:.4f} "
              f"frac_converged={stats['frac_converged']:.2f} (n={stats['n_modules']})")
    print("Stall rate by m:")
    for m, stats in sorted(stall_by_m.items()):
        print(f"  m={m}: stall_rate={stats['stall_rate']:.2f} mean_depth={stats['mean_depth']:.2f} (n={stats['n_runs']})")

    _write_csv(
        output_dir / "gatec2_consensus_by_m.csv",
        [{"leaf_size": m, **stats} for m, stats in sorted(consensus_by_m.items())],
    )
    _write_csv(
        output_dir / "gatec2_stall_rate_by_m.csv",
        [{"leaf_size": m, **stats} for m, stats in sorted(stall_by_m.items())],
    )

    # -- Structural gate criterion (Claim 5.2's mechanism is present) --
    # m=1 has no intra-module gossip at all, so spread is trivially 0 at
    # both snapshots (mean_spread_reduction ~ 0). Any m>1 arm should show a
    # STRICTLY positive mean spread reduction -- consensus is actually
    # forming -- for the gate to pass.
    m1_ok = (1 not in consensus_by_m) or math.isclose(
        consensus_by_m[1]["mean_spread_reduction"], 0.0, abs_tol=1e-9
    )
    larger_m = [m for m in consensus_by_m if m > 1]
    consensus_forms = any(
        not math.isnan(consensus_by_m[m]["mean_spread_reduction"])
        and consensus_by_m[m]["mean_spread_reduction"] > 0.0
        for m in larger_m
    )
    gate_pass = bool(m1_ok and consensus_forms and larger_m)

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    review = {
        "consensus_by_m": consensus_by_m,
        "stall_rate_by_m": stall_by_m,
        "m1_trivial_spread_confirmed": m1_ok,
        "consensus_formation_confirmed": consensus_forms,
        "gatec2_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "Articulation mechanism confirmed structurally: m>1 modules show "
            "measurable consensus formation, m=1 does not (by construction). "
            "Stall-rate-by-m is reported observationally, not gated -- see "
            "module docstring for why per-event articulation-vs-scope "
            "classification needs denser diagnostics than are captured here."
            if gate_pass else
            "Structural check failed -- either no m>1 data, or m>1 modules "
            "did not show measurable consensus formation. Check "
            "track_articulation_diagnostics was actually set and "
            "max_intra_module_rounds was large enough to see contraction."
        ),
    }
    review_path = output_dir / "gatec2_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate C2 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
