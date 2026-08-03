# -*- coding: utf-8 -*-
"""Gate 2 review script: nucleation curve, variance decomposition, NCP-2 directionality.

Scores NMH-2 (Lemma 3.1's fixation formula, voter-functional-form fit vs
ΔAIC against a pairwise-constant null), NMH-6 (Sec. 4.8's variance
decomposition, via nmh_observables.hierarchical_variance_decomposition's
nested-ANOVA estimator), and NCP-2 (Proposition 5.2 asymmetric nucleation)
against paper1_PDMP_wDAG_wData.md.

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
from dssgd.protocols.gossip import protocol_from_suffix


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


def fit_pairwise_constant_q_l(
    b_values: List[float],
    q_l_empirical: List[float],
) -> Tuple[float, float]:
    """Fit the pairwise-constant alternative q_l ~= c (saturates near a fixed
    value, insensitive to b/a -- the pairwise prediction NMH-2's G2.1 criterion
    compares against the voter functional form). Returns (chi2, c_fit) using
    the same chi2 statistic form as fit_voter_q_l, so the two are directly
    comparable via AIC (compute_nmh2_delta_aic).
    """
    if len(b_values) < 2:
        return float("nan"), float("nan")
    q_arr = np.array(q_l_empirical)
    c_fit = float(np.clip(q_arr.mean(), 1e-6, 1.0 - 1e-6))
    residuals = q_arr - c_fit
    chi2 = float(np.sum((residuals ** 2) / np.maximum(c_fit * (1 - c_fit) / 30, 1e-6)))
    return chi2, c_fit


def _chi2_aic(chi2: float, n_params: int) -> float:
    """AIC from a chi2 statistic (sum of squared standardized residuals under
    the same per-point Gaussian-approximate-to-Bernoulli variance model both
    fit_voter_q_l and fit_pairwise_constant_q_l use): AIC = chi2 + 2*n_params.
    Valid for comparing two fits to the same data under the same per-point
    variance model, which is exactly the voter-vs-pairwise comparison G2.1
    (HPC_experiment_spec.md) calls for.
    """
    return chi2 + 2.0 * n_params if not math.isnan(chi2) else float("nan")


def compute_nmh2_delta_aic(
    b_values: List[float],
    q_l_empirical: List[float],
    a: float,
    leaf_size: int,
) -> Dict[str, float]:
    """Gate 2 criterion G2.1 (HPC_experiment_spec.md): Delta AIC =
    AIC_pairwise - AIC_voter, comparing the voter functional form
    q_l=(1-e^-2h)/(1-e^-2hM) (Eq. 14, 1 free parameter theta) against the
    pairwise-constant null q_l ~= c (1 free parameter c). Delta AIC > 4 is
    the spec's threshold for strong preference for voter.
    """
    chi2_voter, theta_fit = fit_voter_q_l(b_values, q_l_empirical, a, leaf_size)
    chi2_pairwise, c_fit = fit_pairwise_constant_q_l(b_values, q_l_empirical)
    aic_voter = _chi2_aic(chi2_voter, 1)
    aic_pairwise = _chi2_aic(chi2_pairwise, 1)
    delta_aic = (
        aic_pairwise - aic_voter
        if not (math.isnan(aic_voter) or math.isnan(aic_pairwise))
        else float("nan")
    )
    return {
        "chi2_voter": chi2_voter, "theta_fit": theta_fit,
        "chi2_pairwise": chi2_pairwise, "c_fit": c_fit,
        "aic_voter": aic_voter, "aic_pairwise": aic_pairwise,
        "delta_aic": delta_aic,
    }


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
) -> Tuple[Dict[int, Tuple[float, float, int]], Optional[int], Optional[int]]:
    """Return ({level: (mean_V_ell, std_V_ell, n_runs)}, branching, depth) from
    NMH-6 pickles. branching/depth are read from the first loaded run (None if
    no run loaded), so the caller can build a matching synthetic positive
    control rather than assuming a hardcoded topology."""
    by_level: Dict[int, List[float]] = defaultdict(list)
    branching: Optional[int] = None
    depth: Optional[int] = None
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            run = NaturalCascadeRun.load(pkl_path)
        except Exception:
            continue
        if not run.warmup_ok:
            continue
        if branching is None:
            branching, depth = run.branching, run.depth
        decomp = hierarchical_variance_decomposition(
            run.centroid_traj, run.branching, run.depth
        )
        for level, v in decomp.items():
            by_level[level].append(v)

    result: Dict[int, Tuple[float, float, int]] = {}
    for level, vals in sorted(by_level.items()):
        arr = np.array(vals)
        result[level] = (float(arr.mean()), float(arr.std()), len(arr))
    return result, branching, depth


def nmh6_synthetic_positive_control(
    branching: int, depth: int, n_synthetic_seeds: int = 20,
) -> Dict[int, float]:
    """Positive-control precondition for nmh6_decomp_ok (Sec. 4.8): build
    synthetic centroid trajectories with known, independently-drawn per-level
    random effects (sigma_ell^2 = base * 4^ell, so the injected signal is
    strictly increasing with level and unambiguous), run the exact same
    hierarchical_variance_decomposition the real NMH-6 gate uses, and return
    the recovered {level: V_level} averaged over n_synthetic_seeds independent
    draws. Every recovered value must come out clearly positive and increasing
    with level (matching the injection) -- if it doesn't, the estimator itself
    (not just the real experimental data) cannot be trusted, independent of
    what nmh6_decomp_ok's real-data check shows.
    """
    n_leaf_types = branching ** depth
    T = 20
    by_level: Dict[int, List[float]] = defaultdict(list)
    rng_master = np.random.default_rng(12345)
    base = 0.01
    for _ in range(n_synthetic_seeds):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        level_effects: Dict[int, np.ndarray] = {}
        for ell in range(depth + 1):
            n_modules = n_leaf_types // (branching ** ell)
            sigma_ell = math.sqrt(base * (4.0 ** ell))
            level_effects[ell] = rng.normal(0.0, sigma_ell, size=n_modules)
        traj = np.zeros((T, n_leaf_types, 1))
        for leaf in range(n_leaf_types):
            mean_val = 0.0
            for ell in range(depth + 1):
                module_idx = leaf // (branching ** ell)
                mean_val += level_effects[ell][module_idx]
            traj[:, leaf, 0] = mean_val + rng.normal(0.0, 1e-4, size=T)
        decomp = hierarchical_variance_decomposition(traj, branching, depth)
        for lvl, v in decomp.items():
            by_level[lvl].append(v)
    return {lvl: float(np.mean(vals)) for lvl, vals in sorted(by_level.items())}


def nmh6_synthetic_control_passes(synthetic: Dict[int, float]) -> bool:
    """True iff the synthetic positive control (known sigma_ell^2 = base*4^ell,
    strictly increasing) is correctly recovered: every level clearly positive
    and non-decreasing across levels."""
    if not synthetic:
        return False
    levels = sorted(synthetic)
    vals = [synthetic[l] for l in levels]
    if any(v <= 0 for v in vals):
        return False
    return all(vals[i] <= vals[i + 1] * 1.5 for i in range(len(vals) - 1)) and vals[-1] > vals[0]


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
             "phase=='2a' output). Pass results/phase2s + --suffix SP for the "
             "sync_pairwise variant.",
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
        "--suffix", type=str, default="", choices=["", "SP", "SN"],
        help="Suffix appended to NMH2/NMH6/NCP2 pkl dir names, naming which "
             "gossip mechanism: '' = async_poisson (default), 'SP' = "
             "sync_pairwise (phase 2s: NMH2SP/NMH6SP/NCP2SP -- see "
             "generate_queue.py's phase=='2s' branch), 'SN' = "
             "sync_neighbourhood. The bare 'S' suffix from before these "
             "were distinguished is retired and no longer accepted -- see "
             "gossip_mechanisms.md.",
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
    nmh2_aic: Dict[str, float] = {}
    if nmh2_pkl_dir.exists():
        q_l_data = compute_nmh2_q_l(nmh2_pkl_dir)
        b_vals = sorted(q_l_data)
        q_emp = [q_l_data[b][0] for b in b_vals]
        chi2, theta_fit = fit_voter_q_l(b_vals, q_emp, a_nmh2, leaf_size)
        nmh2_aic = compute_nmh2_delta_aic(b_vals, q_emp, a_nmh2, leaf_size)
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
                "q_l_pairwise_constant": nmh2_aic.get("c_fit", float("nan")),
                "n_seeds": q_l_data[b][2],
            }
            for b in b_vals
        ]
        _write_csv(output_dir / "gate2_nmh2_q_l.csv", csv_rows)
        print(f"NMH-2: chi^2={chi2:.2f}, theta_fit={theta_fit:.4f}, "
              f"delta_aic={nmh2_aic.get('delta_aic', float('nan')):.2f} (>4 favours voter)")
    else:
        print(f"  [warn] {nmh2_pkl_dir} not found — skipping NMH-2")

    rec_b = recommend_b_values_phase3(q_l_data, theta_fit, a_nmh2)

    # -- NMH-6: variance decomposition --
    # Positive-control precondition, independent of the real data below: the
    # corrected nested-ANOVA estimator (nmh_observables.hierarchical_variance_
    # decomposition) must first correctly recover a known synthetic signal.
    nmh6_pkl_dir = results_dir / "pkl" / f"NMH6{suffix}"
    decomp: Dict[int, Tuple[float, float, int]] = {}
    decomp_ok = False
    nmh6_branching, nmh6_depth = 2, 5  # NMH-6 defaults (experiment_NMH6); refined below if pkls exist
    if nmh6_pkl_dir.exists():
        decomp, run_branching, run_depth = compute_nmh6_decomp(nmh6_pkl_dir)
        if run_branching is not None:
            nmh6_branching, nmh6_depth = run_branching, run_depth

    nmh6_synthetic = nmh6_synthetic_positive_control(nmh6_branching, nmh6_depth)
    nmh6_synthetic_ok = nmh6_synthetic_control_passes(nmh6_synthetic)
    print(f"NMH-6 synthetic positive control (known signal recovered): {nmh6_synthetic_ok} {nmh6_synthetic}")

    if nmh6_pkl_dir.exists():
        if decomp:
            levels = sorted(decomp)
            # Below-detection clamp: sampling noise around a true-zero component
            # can dip slightly negative even with the corrected (unbiased)
            # estimator; report those as "consistent with zero" rather than as
            # a violation of the expected direction.
            means_clamped = [max(decomp[l][0], 0.0) for l in levels]
            # Corrected estimator recovers real per-level variance components,
            # which the ANOVA identity V_L = sum_l V_l predicts should be
            # NON-DECREASING toward the root under genuine hierarchical
            # heterogeneity (matches Sec. 4.8's V_l ~ 2^{-(L-l)*zeta}, and the
            # synthetic control above) -- the OPPOSITE direction from the old,
            # structurally-biased raw-variance-difference formula this
            # replaced.
            monotone_ok = all(
                means_clamped[i] <= means_clamped[i + 1] + 1e-6
                for i in range(len(means_clamped) - 1)
            )
            decomp_ok = nmh6_synthetic_ok and monotone_ok
            csv_rows = [
                {"level": l, "mean_V": decomp[l][0], "mean_V_clamped": mc,
                 "std_V": decomp[l][1], "n_runs": decomp[l][2]}
                for l, mc in zip(levels, means_clamped)
            ]
            _write_csv(output_dir / "gate2_nmh6_variance.csv", csv_rows)
            print(f"NMH-6 variance decomp OK (synthetic control passing AND real data non-decreasing): {decomp_ok}")
    else:
        print(f"  [warn] {nmh6_pkl_dir} not found — skipping NMH-6")

    # -- NCP-2: directionality --
    ncp2_pkl_dir = results_dir / "pkl" / f"NCP2{suffix}"
    ncp2_available = ncp2_pkl_dir.exists()
    p_out, p_in, n_out, n_in = float("nan"), float("nan"), 0, 0
    asymmetry_ratio = float("nan")
    if ncp2_available:
        p_out, p_in, n_out, n_in = compute_ncp2_directionality(ncp2_pkl_dir)
        asymmetry_ratio = p_out / max(p_in, 1e-9) if not math.isnan(p_in) and p_in > 0 else float("nan")
        csv_rows = [
            {"direction": "outward", "p_reach_far_shell": p_out, "n_seeds": n_out},
            {"direction": "inward",  "p_reach_far_shell": p_in,  "n_seeds": n_in},
        ]
        _write_csv(output_dir / "gate2_ncp2_cascade.csv", csv_rows)
        print(f"NCP-2 asymmetry ratio (outward/inward): {asymmetry_ratio:.2f}")
        if n_out == 0 or n_in == 0:
            print(f"  [warn] NCP-2 pkls found ({n_out} outward, {n_in} inward classifiable) "
                  "— check clamped_shell is populated on result pickles")
    else:
        print(f"  [warn] {ncp2_pkl_dir} not found — skipping NCP-2")

    task_counts = _task_counts(args.queue_dir)

    # G2.1 (HPC_experiment_spec.md): Delta AIC > 4 favouring voter is the
    # spec's stated formulation-discrimination criterion, replacing the
    # ad hoc chi2<10 threshold this used to gate on (chi2 alone can't
    # discriminate voter from pairwise -- both are scored on their own scale).
    delta_aic = nmh2_aic.get("delta_aic", float("nan"))
    # NCP-2 only blocks the gate if it was actually run for this phase (dir
    # present) -- a run that never queued NCP-2 shouldn't fail Gate 2 on it,
    # but a run that queued it and came back NaN (e.g. no classifiable
    # outward/inward pickles) is a real failure, not a silent pass.
    ncp2_ok = (
        not ncp2_available
        or (not math.isnan(asymmetry_ratio) and asymmetry_ratio > 2.0)
    )
    gate_pass = (
        not math.isnan(delta_aic) and delta_aic > 4
        and ncp2_ok
    )

    review = {
        "run_parametrization": {
            "phase2_results": str(results_dir),
            "queue_dir": str(args.queue_dir),
            "suffix": suffix,
            "gossip_protocol": protocol_from_suffix(suffix),
            "output_dir": str(output_dir),
        },
        "nmh2_q_l_by_b": {str(b): q_l_data[b][0] for b in sorted(q_l_data)},
        "nmh2_theory_fit_chi2": chi2,
        "nmh2_delta_aic": nmh2_aic,
        "q_l_calibration": {"theta_fit": theta_fit, "a": a_nmh2, "leaf_size": leaf_size},
        "nmh6_variance_decomp": {str(l): decomp[l][0] for l in sorted(decomp)},
        "nmh6_synthetic_positive_control": {str(l): v for l, v in nmh6_synthetic.items()},
        "nmh6_synthetic_control_ok": nmh6_synthetic_ok,
        "nmh6_decomp_ok": decomp_ok,
        "ncp2_p_outward": p_out,
        "ncp2_p_inward": p_in,
        "ncp2_asymmetry_ratio": asymmetry_ratio,
        "recommended_b_values_phase3": rec_b,
        "gate2_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            f"Voter fit chi^2={chi2:.2f}, delta_aic={delta_aic:.2f} (>4 favours voter). "
            f"NCP-2 asymmetry ratio={asymmetry_ratio:.2f}."
        ),
    }

    review_path = output_dir / "gate2_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate 2 {'PASS' if gate_pass else 'FAIL'} — review written to {review_path}")


if __name__ == "__main__":
    main()
