# -*- coding: utf-8 -*-
"""Gate H4 review script: E16src, E6H, E13H, E8H -- the two remaining gaps
in the hybrid-protocol campaign (paper1_computing_hybrid_gossip.md):

E16src (Prop. 4.10's p-inversion, source-commitment diagnostic): per run,
whether the source module itself ever consolidated on B
(nmh_observables.source_module_consensus) and, when the true source_leaf
is known, the fraction of the source module's own members that are
"boundary workers" in Lemma 2.4(3)'s sense
(nmh_observables.source_module_boundary_fraction) -- the direct test of
the leading untested hypothesis for why mean_d_max FELL as p rose from 2
to 32 in phase H3's real data, opposite theory.cross_module_ceiling's
predicted direction. Also re-scores phase H3's ORIGINAL E16 pkls (and,
once run, H3fix's E16v2 pkls) with the corrected source anchor -- both
check_phaseh3.py and check_phaseh3fix.py previously hardcoded
source_leaf=0, but the actual force-flipped leaf is drawn uniformly at
random (n_leaf_types=32 for the default sweep), so the pre-fix
mean_d_max/frac_source_committed numbers in review/phaseh3/gateh3_review
.json and dssgd/review/review/phaseh3/ANALYSIS.md are anchored correctly
in only ~1/32 of runs.

E6H (Thm. 4.5 / Rem. 4.5', margin condition): level-matching filter under
the hybrid protocol at the H-round window's K_low/K_mid/K_high -- does K
set the filter's *resolution* (smallest resolvable quality difference)
separately from leaf_size/m's *reliability* (containment exponent), as
Rem. 4.5' predicts?

E13H (Thm. 11.3 / Rem. 11.2', DAG ideal-matching): does overlap enter
LOGARITHMICALLY under the hybrid's renewal-enforcing Type-N clock
(m > 1 + log(...)) rather than LINEARLY via m > Delta_in, as under free
asynchrony?

E8H (Prop. 5.2 / Rem. 5.5, NCP positional filter, "protocol as repair
lever"): does a live Type-P channel at finite K measurably widen the
polynomial inward route relative to the K -> 0 limit? First-pass scoring
only -- there is no pre-existing async-side E8 scorer in this repo to
mirror (no check_phase4.py/check_phase7.py exist), so this is also the
first scorer for E8 itself, not just its hybrid arm.

Produces:
  review/phaseh4/gateh4_review.json
  review/phaseh4/gateh4_e16src_ceiling.csv
  review/phaseh4/gateh4_e6h_level_matching.csv
  review/phaseh4/gateh4_e13h_containment.csv
  review/phaseh4/gateh4_e8h_entrainment.csv

Usage:
  python -m hpc.review.check_phaseh4 \\
      --phaseh4-results results/phaseh4 \\
      --gateh1-results review/phaseh1/gateh1_review.json \\
      --gateh2-results review/phaseh2/gateh2_review.json \\
      --queue-dir queue/phaseh4 --output-dir review/phaseh4
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
from analysis.natural_cascade import NaturalCascadeRun
from analysis.ncp_runner import NCPRun
from analysis.nmh_observables import (
    cascade_depth,
    source_module_boundary_fraction,
    source_module_consensus,
)
from hpc.review.check_phaseh2 import load_vartheta_dagger_lookup


# ---------------------------------------------------------------------------
# E16src: source-commitment / boundary-worker diagnostic
# ---------------------------------------------------------------------------


def compute_e16src_table(
    pkl_dir: Path, vartheta_lookup: Dict[Tuple[float, float], float],
) -> List[Dict[str, Any]]:
    """Per p: frac_source_committed (did the source module's own centroid
    ever reach and hold B?) and, wherever the true source_leaf is known
    (new-format pkls; old pkls only when nucleation_leaf successfully
    anchors it -- see check_phaseh3.py's compute_e16_ceiling_table
    docstring), mean_boundary_fraction (source_module_boundary_fraction).
    The direct correlation test: under the dilution hypothesis,
    boundary_fraction should RISE with p while frac_source_committed
    FALLS; if the source commits reliably at every p yet mean_d_max still
    falls, the transport ceiling itself is implicated instead.
    """
    by_p: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    b_used = None
    graph_params = None
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/p=" not in run.name:
            continue
        p = float(run.name.split("/p=")[1].split("/")[0])
        seed = int(run.name.split("/seed=")[1].split("/")[0]) if "/seed=" in run.name else run.seed

        true_source = getattr(run, "source_leaf", None)
        source_leaf_is_proxy = true_source is None
        anchor = true_source if true_source is not None else run.nucleation_leaf

        if anchor is not None:
            d_max = cascade_depth(run.centroid_traj, anchor, run.theta_A, run.theta_B, epsilon=0.2, persistence=3)
            source_committed = source_module_consensus(
                run.centroid_traj, anchor, run.theta_A, run.theta_B, epsilon=0.2, persistence=3,
            ) is not None
        else:
            d_max = 0
            source_committed = False

        boundary_fraction = None
        # Only compute the graph-structural diagnostic when the anchor is
        # unambiguous: true_source (always trustworthy) or, for old pkls,
        # nucleation_leaf when the source DID commit (in a Type-N-only,
        # zero-noise run, nucleation_leaf can only be the source itself --
        # nothing else can originate B). Skip the "old pkl + source never
        # committed" cell: we don't know which leaf was forced, so the
        # boundary-fraction of the WRONG module would be meaningless.
        if anchor is not None and (true_source is not None or source_committed):
            graph_seed = run.seed  # NaturalCascadeConfig.graph_seed defaults to config.seed
            boundary_fraction = source_module_boundary_fraction(
                branching=run.branching, depth=run.depth, leaf_size=run.leaf_size,
                p=run.p, graph_seed=graph_seed, source_leaf=anchor,
            )

        by_p[p].append({
            "d_max": d_max,
            "source_committed": source_committed,
            "source_leaf_is_proxy": source_leaf_is_proxy,
            "boundary_fraction": boundary_fraction,
        })
        b_used = run.b

    vd = vartheta_lookup.get((0.5, round(b_used, 6))) if b_used is not None else None

    rows = []
    for p, records in sorted(by_p.items()):
        d_maxes = np.array([r["d_max"] for r in records])
        bfracs = [r["boundary_fraction"] for r in records if r["boundary_fraction"] is not None]
        predicted_l_theta = theory.cross_module_ceiling(p, vd) if vd is not None else None
        rows.append({
            "p": p, "n_runs": len(records), "mean_d_max": float(d_maxes.mean()),
            "max_d_max_observed": int(d_maxes.max()) if len(d_maxes) else None,
            "frac_source_committed": float(np.mean([r["source_committed"] for r in records])),
            "frac_source_leaf_is_proxy": float(np.mean([r["source_leaf_is_proxy"] for r in records])),
            "mean_boundary_fraction": float(np.mean(bfracs)) if bfracs else None,
            "n_boundary_fraction_samples": len(bfracs),
            "predicted_l_theta": predicted_l_theta,
            "predicted_floor_l_theta": math.floor(predicted_l_theta) if predicted_l_theta is not None else None,
        })
    return rows


# ---------------------------------------------------------------------------
# E6H: level-matching filter under hybrid, keyed by K
# ---------------------------------------------------------------------------


def compute_e6h_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    """For each (K, G), fraction of runs with d_max exactly equal to G
    (Theorem 4.5), mirroring check_phase5.py's compute_e6_level_matching
    with K as an added grouping dimension (K=None -> the non-hybrid
    reference arm, if this directory also contains one)."""
    by_KG: Dict[Tuple[Optional[float], int], List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/G=" not in run.name:
            continue
        G = int(run.name.split("/G=")[1].split("/")[0])
        K = float(run.name.split("/K=")[1].split("/")[0]) if "/K=" in run.name else None
        source_leaf = getattr(run, "source_leaf", None)
        if source_leaf is None:
            source_leaf = 0  # experiment_E6's fixed convention (pre-fix pkls)
        d_max = cascade_depth(run.centroid_traj, source_leaf, run.theta_A, run.theta_B,
                               epsilon=0.2, persistence=3)
        by_KG[(K, G)].append(d_max)

    rows = []
    for (K, G), d_maxes in sorted(by_KG.items(), key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, kv[0][1])):
        arr = np.array(d_maxes)
        rows.append({
            "K": K, "G": G, "n_runs": len(arr),
            "frac_d_max_eq_G": float(np.mean(arr == G)),
            "mean_d_max": float(arr.mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# E13H: DAG ideal-matching containment under hybrid, keyed by K
# ---------------------------------------------------------------------------


def compute_e13h_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    """For each (K, delta_in, m), leak frequency beyond the favourable
    ideal, mirroring check_phase5.py's compute_e13_containment with K as
    an added grouping dimension. Rem. 11.2' predicts the containment
    boundary in m should track delta_in LOGARITHMICALLY under the hybrid
    (K finite) rather than LINEARLY (K -> infinity, free async) -- compare
    the K != None rows' leak_frequency-vs-m falloff against the K=None
    rows' at matched delta_in."""
    by_cell: Dict[Tuple[Optional[float], int, int], List[bool]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if "/delta_in=" not in run.name or "/m=" not in run.name:
            continue
        delta_in = int(run.name.split("/delta_in=")[1].split("/")[0])
        m = int(run.name.split("/m=")[1].split("/")[0])
        K = float(run.name.split("/K=")[1].split("/")[0]) if "/K=" in run.name else None
        n_flipped = sum(1 for row in run.flip_table if row.get("t_flip_absolute") is not None)
        by_cell[(K, delta_in, m)].append(n_flipped > 0)

    rows = []
    for (K, delta_in, m), leaks in sorted(
        by_cell.items(), key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, kv[0][1], kv[0][2])
    ):
        rows.append({
            "K": K, "delta_in": delta_in, "m": m, "n_runs": len(leaks),
            "leak_frequency": float(np.mean(leaks)),
        })
    return rows


# ---------------------------------------------------------------------------
# E8H: directional shell entrainment under hybrid, keyed by K
# ---------------------------------------------------------------------------


def compute_e8h_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    """For each (K, clamped_shell k), the fraction of shell (k-1)'s nodes
    that flip within the run (outward entrainment FROM k, Prop. 5.2's
    "outward" direction) vs the fraction of shell (k+1)'s nodes that flip
    (inward entrainment INTO k, Prop. 5.2's "inward" direction) --
    Remark 5.5's "protocol as repair lever" claim is that mean_inward_frac
    should rise as K departs from 0 (K=None here is the K -> infinity,
    async_poisson reference arm, not K -> 0 -- there is no K=0 arm in
    experiment_E8; the K -> 0 comparison point is E16's Prop. 4.10 result
    instead). First-pass scoring: no pre-existing async-side E8 scorer
    exists in this repo (no check_phase4.py/check_phase7.py) to mirror, so
    this is also the first scorer for E8 itself, not just its hybrid arm.
    """
    by_Kk: Dict[Tuple[Optional[float], int], List[NCPRun]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NCPRun.load(pkl_path)
        except Exception:
            continue
        if run.clamped_shell is None:
            continue
        K = float(run.name.split("/K=")[1].split("/")[0]) if "/K=" in run.name else None
        by_Kk[(K, run.clamped_shell)].append(run)

    rows = []
    for (K, k), runs in sorted(by_Kk.items(), key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, kv[0][1])):
        outward_fracs = []
        inward_fracs = []
        for run in runs:
            shells = sorted(set(run.shell_assigns.values()))
            lower = max((s for s in shells if s < k), default=None)
            upper = min((s for s in shells if s > k), default=None)
            flipped_nodes = {row["node"] for row in run.flip_table if row.get("t_flip_absolute") is not None}
            if lower is not None:
                lower_nodes = [n for n, s in run.shell_assigns.items() if s == lower]
                if lower_nodes:
                    outward_fracs.append(sum(1 for n in lower_nodes if n in flipped_nodes) / len(lower_nodes))
            if upper is not None:
                upper_nodes = [n for n, s in run.shell_assigns.items() if s == upper]
                if upper_nodes:
                    inward_fracs.append(sum(1 for n in upper_nodes if n in flipped_nodes) / len(upper_nodes))
        rows.append({
            "K": K, "clamped_shell": k, "n_runs": len(runs),
            "mean_outward_frac": float(np.mean(outward_fracs)) if outward_fracs else None,
            "mean_inward_frac": float(np.mean(inward_fracs)) if inward_fracs else None,
        })
    return rows


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
    parser = argparse.ArgumentParser(description="Gate H4 review: E16src, E6H, E13H, E8H.")
    parser.add_argument("--phaseh4-results", type=Path, default=Path("results/phaseh4"))
    # E16src can also/instead be scored directly off H3's original pkls
    # (the anchoring-bug fix requires no new compute) or H3fix's E16v2
    # pkls (once run post-fix) -- both accepted, in addition to h4's own.
    parser.add_argument("--phaseh3-results", type=Path, default=None)
    parser.add_argument("--phaseh3fix-results", type=Path, default=None)
    parser.add_argument("--gateh1-results", type=Path, default=Path("review/phaseh1/gateh1_review.json"))
    parser.add_argument("--gateh2-results", type=Path, default=Path("review/phaseh2/gateh2_review.json"))
    parser.add_argument("--gateh3-results", type=Path, default=Path("review/phaseh3/gateh3_review.json"))
    parser.add_argument("--gateh3fix-results", type=Path, default=Path("review/phaseh3fix/gateh3fix_review.json"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phaseh4"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phaseh4"))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vartheta_lookup = (
        load_vartheta_dagger_lookup(args.gateh1_results) if args.gateh1_results.exists() else {}
    )
    if not vartheta_lookup:
        print(f"  [warn] {args.gateh1_results} not found or empty -- E16src scoring will be limited")

    for gate_path, label in ((args.gateh3_results, "gateh3_pass"), (args.gateh3fix_results, "gateh3fix_pass")):
        if gate_path is not None and gate_path.exists():
            with open(gate_path) as fh:
                upstream = json.load(fh)
            if not upstream.get(label, True):
                print(f"  [warn] {label} is False in {gate_path} -- upstream phase may not be trustworthy yet")

    # -- E16src: reads h4's own pkls plus, if given, H3's original E16 and
    # H3fix's E16v2 pkls (all three share the same scoring function; the
    # source_leaf/nucleation_leaf fallback logic in compute_e16src_table
    # handles pre-fix vs post-fix pkls transparently). --
    e16src_dirs = [args.phaseh4_results / "pkl" / "E16src"]
    if args.phaseh3_results is not None:
        e16src_dirs.append(args.phaseh3_results / "pkl" / "E16")
    if args.phaseh3fix_results is not None:
        e16src_dirs.append(args.phaseh3fix_results / "pkl" / "E16v2")
    e16src_table: List[Dict[str, Any]] = []
    for d in e16src_dirs:
        if d.exists():
            e16src_table.extend(compute_e16src_table(d, vartheta_lookup))
    print("E16src source-commitment/boundary table:")
    for row in e16src_table:
        print(f"  {row}")

    # -- E6H --
    e6h_dir = args.phaseh4_results / "pkl" / "E6H"
    e6h_table = compute_e6h_table(e6h_dir) if e6h_dir.exists() else []
    print("E6H level-matching table:")
    for row in e6h_table:
        print(f"  {row}")

    # -- E13H --
    e13h_dir = args.phaseh4_results / "pkl" / "E13H"
    e13h_table = compute_e13h_table(e13h_dir) if e13h_dir.exists() else []
    print("E13H containment table:")
    for row in e13h_table:
        print(f"  {row}")

    # -- E8H --
    e8h_dir = args.phaseh4_results / "pkl" / "E8H"
    e8h_table = compute_e8h_table(e8h_dir) if e8h_dir.exists() else []
    print("E8H entrainment table:")
    for row in e8h_table:
        print(f"  {row}")

    _write_csv(output_dir / "gateh4_e16src_ceiling.csv", e16src_table)
    _write_csv(output_dir / "gateh4_e6h_level_matching.csv", e6h_table)
    _write_csv(output_dir / "gateh4_e13h_containment.csv", e13h_table)
    _write_csv(output_dir / "gateh4_e8h_entrainment.csv", e8h_table)

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    # All four tables are exploratory/diagnostic on first pass (matching
    # H3's treatment of E11H/E5H as "reported, not gated") -- gate_pass
    # only asserts that the phase actually produced data, not a specific
    # pass/fail criterion on any one theory prediction.
    gate_pass = bool(e16src_table or e6h_table or e13h_table or e8h_table)

    review = {
        "e16src_table": e16src_table,
        "e6h_table": e6h_table,
        "e13h_table": e13h_table,
        "e8h_table": e8h_table,
        "gateh4_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "All four tables are reported, not gated on a specific "
            "pass/fail criterion (matching check_phaseh3.py's treatment of "
            "E11H/E5H as diagnostic follow-ups) -- gateh4_pass only asserts "
            "at least one table produced data. E16src's "
            "frac_source_committed / mean_boundary_fraction correlation "
            "across p is the direct test of the boundary-worker dilution "
            "hypothesis; E6H/E13H/E8H test whether the topology-level "
            "filtering claims (Thm 4.5, Thm 11.3, Prop 5.2/Rem 5.5) survive "
            "under the hybrid protocol, not just the K -> infinity limit "
            "phases 5a/6a/7a already covered."
            if gate_pass else
            "No table produced any data -- check that phase h4's queue "
            "actually ran, or that --phaseh3-results/--phaseh3fix-results "
            "were passed if relying on those pkls for E16src."
        ),
    }
    review_path = output_dir / "gateh4_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate H4 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
