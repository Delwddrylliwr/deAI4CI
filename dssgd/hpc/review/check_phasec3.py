# -*- coding: utf-8 -*-
"""Gate C3 review script: operational description length vs vartheta vs
generality (paper2_social Lemma 4.2 / Remark 4.4). Terminal analysis gate
for this track -- does not parametrize any further phase.

Gated: Lemma 4.2's chord-tolerance <-> description-length duality. vartheta
comes from theory.chord_geometry(a, b)["vartheta_A_to_B"] -- the SAME
closed-form geometric threshold the rest of the codebase uses (not
re-derived here) -- and Lemma 4.2 predicts dl_bits increases with vartheta
(small vartheta = large landing tolerance = low description length).
Reported, NOT gated: correlation between dl_bits and generality_level G
(Def. 4.3) -- G(b) is the paper's own "generalisable vs localised" referent
(see Sec. 2.1 of the mapping table), used here as the nearest in-toy-model
proxy to the generalisation-gap axis Remark 4.4 wants (this codebase's
MetricsTracker.generalization_gaps() needs a real cross_eval_fn/data
loaders this bistable-toy setup doesn't have -- see that method's
docstring -- so it is not literally applicable here; using G directly
avoids inventing a second, less-grounded proxy). No directional claim is
asserted for this correlation -- the paper does not commit to one -- so it
is not part of gatec3_pass, only reported.

Produces:
  review/gatec3_review.json
  review/gatec3_dl_vs_vartheta.csv
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis import theory
from analysis.description_length import OperationalDLResult


def _load_pkls(pkl_dir: Path) -> List[OperationalDLResult]:
    results = []
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        try:
            results.append(OperationalDLResult.load(pkl_path))
        except Exception as exc:
            print(f"  [warn] Failed to load {pkl_path.name}: {exc}")
    return results


def compute_dl_vartheta_pairs(
    results: List[OperationalDLResult],
) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        if r.dl_bits is None:
            continue  # DL_delta(b) not achieved anywhere in the search range
        spec = r.spec
        try:
            vartheta = theory.chord_geometry(spec["a"], spec["b_in"])["vartheta_A_to_B"]
        except ValueError as exc:
            print(f"  [warn] {r.name}: outside the bistable range, skipping ({exc})")
            continue
        rows.append({
            "name": r.name,
            "generality_level": spec.get("generality_level"),
            "a": spec["a"], "b_in": spec["b_in"],
            "vartheta_A_to_B": vartheta,
            "dl_bits": r.dl_bits,
            "passage_probability": r.passage_probability,
        })
    return rows


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
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
    parser = argparse.ArgumentParser(description="Gate C3 review: operational DL correlations.")
    parser.add_argument("--phasec3-results", type=Path, default=Path("results/phasec3"))
    parser.add_argument("--queue-dir", type=Path, default=Path("queue/phasec3"))
    parser.add_argument("--output-dir", type=Path, default=Path("review/phasec3"))
    args = parser.parse_args()

    results_dir: Path = args.phasec3_results
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_dir = results_dir / "pkl" / "C3"
    dl_results = _load_pkls(pkl_dir) if pkl_dir.exists() else []
    if not dl_results:
        print(f"  [warn] {pkl_dir} not found or empty -- nothing to score")

    rows = compute_dl_vartheta_pairs(dl_results)
    _write_csv(output_dir / "gatec3_dl_vs_vartheta.csv", rows)

    dl_bits = [row["dl_bits"] for row in rows]
    vartheta = [row["vartheta_A_to_B"] for row in rows]
    generality = [row["generality_level"] for row in rows if row["generality_level"] is not None]
    dl_for_generality = [
        row["dl_bits"] for row in rows if row["generality_level"] is not None
    ]

    corr_dl_vartheta = _pearson(dl_bits, vartheta)
    corr_dl_generality = _pearson(dl_for_generality, generality)

    print(f"n basins with achieved DL: {len(rows)} / {len(dl_results)}")
    print(f"corr(dl_bits, vartheta_A_to_B) = {corr_dl_vartheta:.3f}  (Lemma 4.2 predicts > 0)")
    print(f"corr(dl_bits, generality_level) = {corr_dl_generality:.3f}  (reported, not gated)")

    # -- Lemma 4.2 duality gate criterion --
    duality_confirmed = not np.isnan(corr_dl_vartheta) and corr_dl_vartheta > 0.0
    n_achieved_rate = (len(rows) / len(dl_results)) if dl_results else 0.0
    gate_pass = bool(duality_confirmed and len(rows) >= 3)

    task_counts = audit_tasks(args.queue_dir)
    print(f"Task audit: {task_counts}")

    review = {
        "n_dl_results": len(dl_results),
        "n_dl_achieved": len(rows),
        "dl_achieved_rate": n_achieved_rate,
        "corr_dl_vartheta": corr_dl_vartheta,
        "corr_dl_generality_level": corr_dl_generality,
        "duality_confirmed": duality_confirmed,
        "gatec3_pass": gate_pass,
        "n_tasks_completed": task_counts.get("completed", 0),
        "n_tasks_failed": task_counts.get("failed", 0),
        "notes": (
            "Lemma 4.2 chord-tolerance/description-length duality confirmed "
            "(positive dl_bits-vartheta correlation). H-isotropy is assumed, "
            "not tested, in this 1-D toy model (d_param=1 -- see Remark 4.4's "
            "own caveat: the isotropy gap is a d>1 phenomenon)."
            if gate_pass else
            "Duality not confirmed -- either too few basins achieved a DL "
            "within the search range, or the correlation was flat/negative. "
            "Check k_search_range covers the range where passage probability "
            "actually transitions from 0 to 1 for these basins."
        ),
    }
    review_path = output_dir / "gatec3_review.json"
    with open(review_path, "w") as fh:
        json.dump(review, fh, indent=2)
    print(f"\nGate C3 {'PASS' if gate_pass else 'FAIL'} -- review written to {review_path}")


if __name__ == "__main__":
    main()
