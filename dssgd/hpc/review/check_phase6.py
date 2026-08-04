# -*- coding: utf-8 -*-
"""Gate 6 review script: validate Phase 6 (E5, E11, E12a, E12b, E14, E15) outputs.

Scores E5 (Sec. 3.4 crossover stage l_c), E11 (Remark 4.3's scheduling
dependence of the mixing exponent), E12a (Proposition 4.9 / Theorem 4.5's
level-matching filter, gated on the E6ctrl positive control -- see
natural_cascade_experiments.experiment_E6_positive_control), and E12b
(Lemma 10.1's curvature ratchet, gated on the r=1 neutral-point control)
against paper1_PDMP_wDAG_wData.md. E14 (local_steps sensitivity of the
meritocratic filter) and E15 (E12b's curvature ratchet under a distributed,
not fixed, kick weight -- see theory.fixation_bias_distributed) are
diagnostic follow-ups, reported but not gated into gate6_pass.

Produces:
  review/gate6_review.json
  review/gate6_e11_slopes.csv
  review/gate6_e12b_curvature.csv
  review/gate6_e14_local_steps.csv
  review/gate6_e15_distributed_curvature.csv

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import binomtest
    _SCIPY = True
except ImportError:
    _SCIPY = False

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.clique_fixation import CliqueFixationRun
from analysis.natural_cascade import NaturalCascadeRun
from analysis.nmh_observables import cascade_depth
from analysis.provenance import crossover_stage
from analysis import theory
from dssgd.protocols.gossip import protocol_from_suffix
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
# E14: local_steps sensitivity of the meritocratic filter (Theorem 4.5)
# ---------------------------------------------------------------------------


def compute_e14_local_steps_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    """Per (G, local_steps, protocol) cell: frac_d_max_eq_G and mean_d_max,
    the same containment observable compute_e6_level_matching uses, but
    resolved across local_steps and protocol -- direct test of whether the
    G-independence found in real E12a data under sync_pairwise is a
    Regime-C relaxation-time artifact (frac should rise with local_steps if
    so) or persists regardless (frac stays flat/low at every local_steps).
    """
    by_cell: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok or "/G=" not in run.name:
            continue
        G = int(run.name.split("/G=")[1].split("/")[0])
        ls = int(run.name.split("/ls=")[1].split("/")[0])
        proto = run.name.split("/proto=")[1].split("/")[0]
        d_max = cascade_depth(run.centroid_traj, 0, run.theta_A, run.theta_B, epsilon=0.2, persistence=3)
        by_cell[(G, ls, proto)].append(d_max)

    rows = []
    for (G, ls, proto), d_maxes in sorted(by_cell.items()):
        arr = np.array(d_maxes)
        rows.append({
            "G": G, "local_steps": ls, "protocol": proto, "n_runs": len(arr),
            "frac_d_max_eq_G": float(np.mean(arr == G)),
            "mean_d_max": float(arr.mean()),
            "std_d_max": float(arr.std()),
        })
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


# ---------------------------------------------------------------------------
# E15: curvature ratchet under a genuinely distributed kick weight
# ---------------------------------------------------------------------------


def compute_e15_curvature_table(pkl_dir: Path) -> List[Dict[str, Any]]:
    """E12b's curvature-ratchet table, but scored against
    theory.fixation_bias_distributed (Uniform(0,1) kick weight) rather than
    the degenerate theory.fixation_bias -- the fair comparison for E15's
    actual simulation, which draws kick weight from that same distribution
    (kick_weight_law="uniform") rather than E12b's fixed alpha=0.5.
    """
    by_cell: Dict[Tuple[int, float], List[bool]] = defaultdict(list)
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = CliqueFixationRun.load(pkl_path)
        except Exception:
            continue
        by_cell[(run.m, run.r)].append(run.fixed_at_B)

    rows = []
    for (m, r), outcomes in sorted(by_cell.items()):
        empirical = float(np.mean(outcomes))
        try:
            rho = theory.fixation_bias_distributed(a=0.5, b=0.0, r=r)
            predicted = theory.fixation_probability(1, m, rho)
        except Exception:
            predicted = float("nan")
        rows.append({
            "m": m, "r": r, "n_trials": len(outcomes),
            "empirical_fixation_on_sharp_freq": empirical, "predicted_q_fix": predicted,
        })
    return rows


def r1_neutral_point_consistent(k: int, n: int, m: int, alpha: float = 0.05) -> Optional[bool]:
    """Positive-control precondition for e12b_ok (Lemma 3.1's parameter-free
    point): at r=1 (symmetric well) a single seed fixes at rate exactly 1/m
    (the fair-random-walk case q_fix(1;m,1)=1/m). This is the one E12b cell
    with no free/computed parameter to get wrong, so the r=1 row must itself
    be statistically consistent with 1/m before any r>1 monotonicity claim is
    trustworthy -- otherwise an all-zero row (predicted_q_fix indistinguishable
    from observed only because both are ~0) trivially satisfies monotonicity
    without the mechanism actually being observed to work at all.

    Returns True/False from a two-sided binomial test against p=1/m, or None
    if scipy is unavailable (in which case the precondition cannot be checked
    and callers should treat it as failing open, i.e. not satisfied).
    """
    if not _SCIPY or n == 0:
        return None
    result = binomtest(k, n, p=1.0 / m, alternative="two-sided")
    return result.pvalue > alpha


def compute_e12b_r1_controls(rows: List[Dict[str, Any]]) -> Dict[int, Optional[bool]]:
    """{m: r=1 neutral-point consistency} for every m present at r=1.0."""
    result: Dict[int, Optional[bool]] = {}
    for row in rows:
        if row["r"] != 1.0:
            continue
        m = row["m"]
        n = row["n_trials"]
        k = int(round(row["empirical_fixation_on_sharp_freq"] * n))
        result[m] = r1_neutral_point_consistent(k, n, m)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 6 review for Phase 6 outputs (E5,E11,E12a,E12b).")
    parser.add_argument(
        "--phase6-results", type=Path, default=None,
        help="Defaults from --suffix: results/phase6a for '' (async), "
             "results/phase6sp for 'SP', results/phase6sn for 'SN' -- "
             "matching generate_queue.py's phase ID convention (see "
             "gossip_mechanisms.md). Pass explicitly to override; if you "
             "do, make sure it actually matches --suffix, or E12a/E12b/"
             "E15 pkl dirs silently won't be found (see "
             "gossip_mechanisms.md's phase-ID-ambiguity note for why this "
             "bit people before).",
    )
    parser.add_argument(
        "--queue-dir", type=Path, default=None,
        help="Defaults from --suffix the same way as --phase6-results "
             "(queue/phase6a, queue/phase6sp, queue/phase6sn).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to review/<phase6-results basename> (e.g. "
             "review/phase6sp/ for --phase6-results results/phase6sp), so "
             "async/sync reviews never collide or need a manually-labelled "
             "path. Pass explicitly to override.",
    )
    parser.add_argument(
        "--suffix", type=str, default="", choices=["", "SP", "SN"],
        help="Suffix appended to E12a/E12b pkl dir names, naming which "
             "gossip mechanism: '' = async_poisson (default), 'SP' = "
             "sync_pairwise (phase 6sp: E12aSP/E12bSP -- see generate_queue"
             ".py's phase=='6sp' branch), 'SN' = sync_neighbourhood. The "
             "bare 'S' suffix from before these were distinguished is "
             "retired and no longer accepted -- see gossip_mechanisms.md. "
             "Phase 6sp has no synchronous variant of E5/E11, so those are "
             "skipped entirely when --suffix is set. --suffix also drives "
             "--phase6-results/--queue-dir defaults, see those.",
    )
    args = parser.parse_args()

    suffix = args.suffix
    _phase_tag = {"": "6a", "SP": "6sp", "SN": "6sn"}[suffix]
    results_dir = args.phase6_results or Path(f"results/phase{_phase_tag}")
    queue_dir = args.queue_dir or Path(f"queue/phase{_phase_tag}")
    output_dir = args.output_dir or Path("review") / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

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
            sync_slopes = [r["slope"] for r in e11_rows if r["protocol"] == "sync_pairwise" and not math.isnan(r["slope"])]
            e11_ok = bool(sync_slopes) and all(abs(s - 1.0) < 0.3 for s in sync_slopes)
            print(f"E11 slopes: {e11_rows}")
        else:
            print(f"  [warn] {e11_pkl_dir} not found — skipping E11")

    # Positive control (assessment doc A.6): E6ctrl/E6ctrlS are a small, fast,
    # same-bias variant of E6 verifying the containment/attainment pipeline
    # (per_leaf_loss_params_for_generality + force-flip targeting +
    # cascade_depth) can detect d_max==G at all. Required precondition for
    # e12a_ok -- without it, a real d_max==0-for-every-G null is
    # indistinguishable from a broken pipeline (exactly what happened before
    # the force_flip_source/source_leaf fix: see natural_cascade_experiments.
    # experiment_E6's docstring).
    e12a_ctrl_pkl_dir = results_dir / "pkl" / f"E6ctrl{suffix}"
    e12a_ctrl_ok = False
    if e12a_ctrl_pkl_dir.exists():
        ctrl_rows = compute_e6_level_matching(e12a_ctrl_pkl_dir)
        ctrl_fracs = [r["frac_d_max_eq_G"] for r in ctrl_rows]
        e12a_ctrl_ok = bool(ctrl_fracs) and float(np.mean(ctrl_fracs)) > 0.7
        print(f"E12a positive control (should detect d_max==G): mean frac={np.mean(ctrl_fracs) if ctrl_fracs else float('nan'):.3f}, ok={e12a_ctrl_ok}")
    else:
        print(f"  [warn] {e12a_ctrl_pkl_dir} not found — E12a positive control cannot run, e12a_ok forced False")

    e12a_pkl_dir = results_dir / "pkl" / f"E12a{suffix}"
    e12a_rows: List[Dict[str, Any]] = []
    e12a_ok = False
    if e12a_pkl_dir.exists():
        e12a_rows = compute_e6_level_matching(e12a_pkl_dir)
        _write_csv(output_dir / "gate6_e12a_containment.csv", e12a_rows)
        fracs = [r["frac_d_max_eq_G"] for r in e12a_rows]
        e12a_ok = e12a_ctrl_ok and bool(fracs) and float(np.mean(fracs)) > 0.5
        print(f"E12a overfitting containment: mean frac(d_max==l)={np.mean(fracs) if fracs else float('nan'):.3f}")
    else:
        print(f"  [warn] {e12a_pkl_dir} not found — skipping E12a")

    e12b_pkl_dir = results_dir / "pkl" / f"E12b{suffix}"
    e12b_rows: List[Dict[str, Any]] = []
    e12b_ok = False
    e12b_r1_controls: Dict[int, Optional[bool]] = {}
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
        # Positive-control precondition (Lemma 3.1's parameter-free r=1 point,
        # q_fix(1;m,1)=1/m): monotonicity alone is satisfiable by an all-zero
        # row, so require every m's r=1 cell to be statistically consistent
        # with the neutral prediction before trusting the monotonicity check.
        e12b_r1_controls = compute_e12b_r1_controls(e12b_rows)
        r1_ok = bool(e12b_r1_controls) and all(v is True for v in e12b_r1_controls.values())
        e12b_ok = bool(monotone_ok) and all(monotone_ok) and r1_ok
        print(f"E12b r=1 neutral-point controls: {e12b_r1_controls}")
        print(f"E12b curvature ratchet monotone-decreasing-in-r AND r=1 control passing: {e12b_ok}")
    else:
        print(f"  [warn] {e12b_pkl_dir} not found — skipping E12b")

    # E14: local_steps sensitivity of the meritocratic filter -- diagnostic
    # only (not gated into gate6_pass), queued once under phase 6a's results
    # tree regardless of which suffix this review invocation is for (E14
    # sweeps protocol as its own dimension, so it naturally won't be found
    # when reviewing a phase6sp-style results_dir where it was never queued).
    e14_pkl_dir = results_dir / "pkl" / "E14"
    e14_rows: List[Dict[str, Any]] = []
    if e14_pkl_dir.exists():
        e14_rows = compute_e14_local_steps_table(e14_pkl_dir)
        _write_csv(output_dir / "gate6_e14_local_steps.csv", e14_rows)
        print(f"E14 local_steps sensitivity: {len(e14_rows)} (G, local_steps, protocol) cells")
    else:
        print(f"  [warn] {e14_pkl_dir} not found — skipping E14 (diagnostic, not gate-critical)")

    # E15: E12b's curvature ratchet under a distributed (not fixed) kick
    # weight -- diagnostic only (not gated into gate6_pass), scored against
    # theory.fixation_bias_distributed rather than E12b's theory.fixation_bias.
    e15_pkl_dir = results_dir / "pkl" / f"E15{suffix}"
    e15_rows: List[Dict[str, Any]] = []
    if e15_pkl_dir.exists():
        e15_rows = compute_e15_curvature_table(e15_pkl_dir)
        _write_csv(output_dir / "gate6_e15_distributed_curvature.csv", e15_rows)
        print(f"E15 distributed-kick curvature ratchet: {len(e15_rows)} (m, r) cells")
    else:
        print(f"  [warn] {e15_pkl_dir} not found — skipping E15 (diagnostic, not gate-critical)")

    task_counts = _task_counts(queue_dir)
    gate_pass = (e12a_ok and e12b_ok) if suffix else (e11_ok and e12a_ok and e12b_ok)

    review = {
        "run_parametrization": {
            "phase6_results": str(results_dir),
            "queue_dir": str(queue_dir),
            "suffix": suffix,
            "gossip_protocol": protocol_from_suffix(suffix),
            "output_dir": str(output_dir),
        },
        "e5_crossover": e5_rows,
        "e11_slopes": e11_rows,
        "e11_ok": e11_ok,
        "e12a_positive_control_ok": e12a_ctrl_ok,
        "e12a_containment": e12a_rows,
        "e12a_ok": e12a_ok,
        "e12b_curvature": e12b_rows,
        "e12b_r1_neutral_point_controls": {str(m): v for m, v in e12b_r1_controls.items()},
        "e12b_ok": e12b_ok,
        "e14_local_steps_sensitivity": e14_rows,
        "e15_distributed_curvature": e15_rows,
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
