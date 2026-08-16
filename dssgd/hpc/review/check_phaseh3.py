# -*- coding: utf-8 -*-
"""Gate H3 review script: E17, E16, E11H, E5H, E15MB -- Annex B.2's protocol
axis (paper1_computing_hybrid_gossip.md).

E17 (Lem. 3.1'/2.4, nucleation-and-growth signature): ANALYSIS ONLY, no new
simulations -- reclassifies H2's own E14RR occupancy logs (results/phaseh2/
pkl/E14RR/*.pkl) by whether the n_dagger crossing happened at a QUALIFYING
Type-N round boundary or during a plain Type-P step, per B.0.1's design
principle ("round boundaries are a sufficient statistic").

E16 (Prop. 4.10, eq. 4.9): Type-N-only ceiling l_theta, scored against
theory.cross_module_ceiling using nmh_observables.cascade_depth.

E11H (Rem. 4.3 / eq. 3.4a): slope of log2(T_flip) vs hierarchical distance
under the hybrid protocol at the H-round window's K_low/K_mid/K_high,
predicted to show LOW-d degradation rather than free-async's high-d
degradation (the reversal (3.4a) predicts).

E5H (Sec. 3.4, Rem. 3.7): crossover stage l_c vs sigma at each K, testing
the attributable-window-widening claim (l_c should not shrink, and may
grow, as K decreases toward the H-round window from K -> infinity).

E15MB (Remark 6.4, Lemma 6.1): barycentre vs label-plurality Type-N update
on a multi-basin clique -- scored on mean post-round loss (the reliable
signal; see analysis/multi_basin_destruction.py's KNOWN LIMITATION note on
why "landed in an unoccupied basin" is not scored here).

Produces:
  review/phaseh3/gateh3_review.json
  review/phaseh3/gateh3_e17_provenance.csv
  review/phaseh3/gateh3_e16_ceiling.csv
  review/phaseh3/gateh3_e11h_slopes.csv
  review/phaseh3/gateh3_e5h_crossover.csv
  review/phaseh3/gateh3_e15mb_destruction.csv

Usage:
  python -m hpc.review.check_phaseh3 \\
      --phaseh2-results results/phaseh2 --phaseh3-results results/phaseh3 \\
      --gateh1-results review/phaseh1/gateh1_review.json \\
      --gateh2-results review/phaseh2/gateh2_review.json \\
      --queue-dir queue/phaseh3 --output-dir review/phaseh3
"""
import argparse
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

from analysis import theory
from analysis.clique_fixation import CliqueFixationRun
from analysis.multi_basin_destruction import MultiBasinRun
from analysis.natural_cascade import NaturalCascadeRun
from analysis.nmh_observables import cascade_depth, source_module_consensus
from analysis.provenance import crossover_stage
from hpc.review.check_phaseh2 import load_vartheta_dagger_lookup, _parse_name


# ---------------------------------------------------------------------------
# E17: analysis-only, reads H2's own E14RR pkls
# ---------------------------------------------------------------------------

E17_EPSILON_N_ROUNDS = 5  # experiment_E14_round_ratio_sweep's fixed default


def analyze_e17_provenance(
    e14rr_pkl_dir: Path,
    vartheta_lookup: Dict[Tuple[float, float], float],
    epsilon_n_rounds: int = E17_EPSILON_N_ROUNDS,
) -> List[Dict[str, Any]]:
    """Per E14RR run that crossed n_dagger: was the crossing entry the
    FIRST occupancy_log entry of a Type-N-QUALIFYING round (could include
    Type-N's own contribution -- "type_n_round"), or anything else
    (definitely a pure Type-P kick -- "type_p_only")? Every occupancy_log
    entry after the first for a given round_idx is necessarily Type-P-only,
    since HybridGossip fires Type-N at most once, on the first execute()
    call of a qualifying round (see gossip.py's HybridGossip.execute)."""
    rows = []
    for pkl_path in sorted(e14rr_pkl_dir.glob("*.pkl")):
        try:
            run = CliqueFixationRun.load(pkl_path)
        except Exception:
            continue
        if not run.occupancy_log:
            continue
        vd = vartheta_lookup.get((0.5, round(run.b, 6)))
        if vd is None:
            continue
        nd = theory.n_dagger(run.m, vd)

        hit_idx = None
        hit_round = None
        for i, (round_idx, n_b) in enumerate(run.occupancy_log):
            if n_b >= nd:
                hit_idx, hit_round = i, round_idx
                break
        if hit_idx is None:
            continue

        is_first_of_round = hit_idx == 0 or run.occupancy_log[hit_idx - 1][0] != hit_round
        is_qualifying_round = (hit_round + 1) % epsilon_n_rounds == 0
        classification = (
            "type_n_round" if (is_first_of_round and is_qualifying_round) else "type_p_only"
        )
        parts = _parse_name(run.name)
        rows.append({
            "name": run.name, "m": run.m, "b": run.b, "K": parts.get("K"),
            "hit_round": hit_round, "classification": classification,
        })
    return rows


def summarize_e17(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_K: Dict[Any, List[str]] = defaultdict(list)
    for row in rows:
        by_K[row["K"]].append(row["classification"])
    out = []
    for K, classes in sorted(by_K.items(), key=lambda kv: (kv[0] is None, kv[0])):
        n = len(classes)
        out.append({
            "K": K, "n_crossings": n,
            "frac_type_n_round": classes.count("type_n_round") / n if n else float("nan"),
            "frac_type_p_only": classes.count("type_p_only") / n if n else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------
# E16: cross-module ceiling
# ---------------------------------------------------------------------------


def compute_e16_ceiling_table(
    pkl_dir: Path, vartheta_lookup: Dict[Tuple[float, float], float],
) -> List[Dict[str, Any]]:
    """H4 fix: earlier versions of this table anchored cascade_depth at a
    hardcoded source_leaf=0, but the actual force-flipped leaf is drawn
    uniformly at random (n_leaf_types=32 for the default depth=5 sweep), so
    the pre-fix mean_d_max reported in review/phaseh3/gateh3_review.json and
    dssgd/review/review/phaseh3/ANALYSIS.md is anchored correctly in only
    ~1/32 of runs and should not be trusted for the p-inversion question.

    Anchor is run.source_leaf when present (pkls produced after the
    source_leaf persistence fix). Older pkls fall back to
    run.nucleation_leaf as a proxy -- exact whenever the source leaf is the
    one that nucleates (the common case, since the source is force-flipped
    to B at t=0), and safe even when unknown in the "no leaf ever flipped"
    case: nucleation_leaf is None only when NO leaf (including the true
    source, whichever it was) ever reached B, so d_max=0 and
    source_committed=False are correct regardless of the true source's
    identity.
    """
    by_p: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    b_used = None
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/p=" not in run.name:
            continue
        p = float(run.name.split("/p=")[1].split("/")[0])
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
        by_p[p].append({
            "d_max": d_max,
            "source_committed": source_committed,
            "source_leaf_is_proxy": source_leaf_is_proxy,
        })
        b_used = run.b

    vd = vartheta_lookup.get((0.5, round(b_used, 6))) if b_used is not None else None

    rows = []
    for p, records in sorted(by_p.items()):
        d_maxes = np.array([r["d_max"] for r in records])
        predicted_l_theta = theory.cross_module_ceiling(p, vd) if vd is not None else None
        rows.append({
            "p": p, "n_runs": len(records), "mean_d_max": float(d_maxes.mean()),
            "max_d_max_observed": int(d_maxes.max()) if len(d_maxes) else None,
            "frac_source_committed": float(np.mean([r["source_committed"] for r in records])),
            "frac_source_leaf_is_proxy": float(np.mean([r["source_leaf_is_proxy"] for r in records])),
            "predicted_l_theta": predicted_l_theta,
            "predicted_floor_l_theta": math.floor(predicted_l_theta) if predicted_l_theta is not None else None,
        })
    return rows


# ---------------------------------------------------------------------------
# E11H: scheduling slope under hybrid, keyed by K
# ---------------------------------------------------------------------------


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    res = np.linalg.lstsq(A, y, rcond=None)
    return float(res[0][0]), float(res[0][1])


def compute_e11h_slopes(pkl_dir: Path) -> List[Dict[str, Any]]:
    """log2(T_flip) vs distance slope per K -- Rem. 4.3's reversal (3.4a)
    predicts degradation (sub-slope-1, concave) should appear at LOW d
    under the hybrid, unlike free async's high-d degradation."""
    by_K: Dict[float, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/K=" not in run.name:
            continue
        K = float(run.name.split("/K=")[1].split("/")[0])
        for row in run.flip_table:
            t, d = row.get("t_flip_relative"), row.get("distance_from_source")
            if t is not None and d is not None and t > 0:
                by_K[K][d].append(math.log2(t))

    rows = []
    for K, by_d in sorted(by_K.items()):
        ds_all = sorted(d for d in by_d if len(by_d[d]) >= 2)
        ds_low = np.array([d for d in ds_all if d < 3], dtype=float)
        ds_high = np.array([d for d in ds_all if d >= 3], dtype=float)
        means_low = np.array([np.mean(by_d[d]) for d in ds_low])
        means_high = np.array([np.mean(by_d[d]) for d in ds_high])
        slope_low, _ = _linear_fit(ds_low, means_low)
        slope_high, _ = _linear_fit(ds_high, means_high)
        rows.append({
            "K": K, "n_distances": len(ds_all),
            "slope_low_d": slope_low, "slope_high_d": slope_high,
        })
    return rows


# ---------------------------------------------------------------------------
# E5H: crossover stage l_c vs sigma, keyed by K
# ---------------------------------------------------------------------------


def compute_e5h_crossover(pkl_dir: Path) -> List[Dict[str, Any]]:
    by_Ksigma: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if run.events is None or "/sigma=" not in run.name or "/K=" not in run.name:
            continue
        K = float(run.name.split("/K=")[1].split("/")[0])
        sigma = float(run.name.split("/sigma=")[1].split("/")[0])
        l_c = crossover_stage(run.flip_table, run.events)
        if l_c is not None:
            by_Ksigma[(K, sigma)].append(l_c)

    rows = []
    for (K, sigma), l_cs in sorted(by_Ksigma.items()):
        rows.append({"K": K, "sigma": sigma, "n_runs": len(l_cs), "mean_l_c": float(np.mean(l_cs))})
    return rows


# ---------------------------------------------------------------------------
# E15MB: multi-basin destruction
# ---------------------------------------------------------------------------


def compute_e15mb_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    by_cell: Dict[Tuple[str, int, int], List[MultiBasinRun]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = MultiBasinRun.load(pkl_path)
        except Exception:
            continue
        by_cell[(run.update_rule, run.n_occupied, run.m)].append(run)

    rows = []
    for (rule, n_occupied, m), runs in sorted(by_cell.items()):
        final_losses = [r.round_log[-1][2] for r in runs if r.round_log]
        unoccupied_frac = float(np.mean([r.round_log[-1][3] for r in runs if r.round_log]))
        rows.append({
            "update_rule": rule, "n_occupied": n_occupied, "m": m, "n_runs": len(runs),
            "mean_final_loss": float(np.mean(final_losses)) if final_losses else None,
            "frac_landed_unoccupied": unoccupied_frac,
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
    parser = argparse.ArgumentParser(description="Gate H3 review: E17, E16, E11H, E5H, E15MB.")
    parser.add_argument("--phaseh2-results", type=Path, default=Path("results/phaseh2"))
    parser.add_argument("--phaseh3-results", type=Path, default=Path("results/phaseh3"))
    parser.add_argument("--gateh1-results", type=Path, default=Path("review/phaseh1/gateh1_review.json"))
    parser.add_argument("--gateh2-results", type=Path, default=Path("review/phaseh2/gateh2_review.json"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phaseh3"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phaseh3"))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vartheta_lookup = (
        load_vartheta_dagger_lookup(args.gateh1_results) if args.gateh1_results.exists() else {}
    )
    if not vartheta_lookup:
        print(f"  [warn] {args.gateh1_results} not found or empty -- E17/E16 scoring will be limited")

    # -- E17 (analysis-only, H2's own pkls) --
    e14rr_dir = args.phaseh2_results / "pkl" / "E14RR"
    e17_rows = analyze_e17_provenance(e14rr_dir, vartheta_lookup) if e14rr_dir.exists() else []
    e17_summary = summarize_e17(e17_rows)
    print("E17 provenance summary (by K):")
    for row in e17_summary:
        print(f"  {row}")

    # -- E16 --
    e16_dir = args.phaseh3_results / "pkl" / "E16"
    e16_table = compute_e16_ceiling_table(e16_dir, vartheta_lookup) if e16_dir.exists() else []
    print("E16 ceiling table:")
    for row in e16_table:
        print(f"  {row}")

    # -- E11H --
    e11h_dir = args.phaseh3_results / "pkl" / "E11H"
    e11h_table = compute_e11h_slopes(e11h_dir) if e11h_dir.exists() else []
    print("E11H slope table:")
    for row in e11h_table:
        print(f"  {row}")

    # -- E5H --
    e5h_dir = args.phaseh3_results / "pkl" / "E5H"
    e5h_table = compute_e5h_crossover(e5h_dir) if e5h_dir.exists() else []
    print("E5H crossover table:")
    for row in e5h_table:
        print(f"  {row}")

    # -- E15MB --
    e15mb_dir = args.phaseh3_results / "pkl" / "E15MB"
    e15mb_table = compute_e15mb_table(e15mb_dir) if e15mb_dir.exists() else []
    print("E15MB destruction table:")
    for row in e15mb_table:
        print(f"  {row}")

    _write_csv(output_dir / "gateh3_e17_provenance.csv", e17_rows)
    _write_csv(output_dir / "gateh3_e16_ceiling.csv", e16_table)
    _write_csv(output_dir / "gateh3_e11h_slopes.csv", e11h_table)
    _write_csv(output_dir / "gateh3_e5h_crossover.csv", e5h_table)
    _write_csv(output_dir / "gateh3_e15mb_destruction.csv", e15mb_table)

    # E15MB pass condition: for every (n_occupied>=3, m) cell, barycentre's
    # mean_final_loss is clearly above plurality's (plurality should be ~0
    # by construction -- it always snaps exactly to a well centre).
    by_key: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(dict)
    for row in e15mb_table:
        if row["mean_final_loss"] is not None:
            by_key[(row["n_occupied"], row["m"])][row["update_rule"]] = row["mean_final_loss"]
    e15mb_destruction_confirmed = all(
        vals.get("barycentre", 0.0) > vals.get("plurality", 0.0) + 1e-6
        for (n_occ, _m), vals in by_key.items()
        if n_occ >= 3 and "barycentre" in vals and "plurality" in vals
    ) if by_key else False

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    gate_pass = bool(e15mb_destruction_confirmed and e16_table and e17_summary)

    review = {
        "e17_summary": e17_summary,
        "e16_table": e16_table,
        "e11h_table": e11h_table,
        "e5h_table": e5h_table,
        "e15mb_table": e15mb_table,
        "e15mb_destruction_confirmed": e15mb_destruction_confirmed,
        "gateh3_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "E15MB confirms barycentre >> plurality post-round loss at "
            "n_occupied>=3 (Remark 6.4's local-destruction claim); E16/E17 "
            "produced data. E11H/E5H are reported but not gated (diagnostic "
            "follow-ups, matching check_phase6.py's treatment of the "
            "original E14/E15 as diagnostic rather than gating)."
            if gate_pass else
            "Either E15MB did not show the predicted barycentre/plurality "
            "loss separation, or E16/E17 produced no data -- check that "
            "phase h3's queue actually ran before trusting downstream phases."
        ),
    }
    review_path = output_dir / "gateh3_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate H3 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
