"""Statistical estimators for NMH simulation outputs.

Pure numpy — no external statistics libraries required.  Estimates:
  - Topological dimension D via BFS neighbourhood-growth log-log regression
  - Per-level convergence rates and coupling ratios
  - Stationary within-module variance (relevant in the training / SDE regime)
  - Multi-seed averaging utilities
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .results import AnalysisRun, LevelTrace, TopologyStats


# ---------------------------------------------------------------------------
# Topological dimension (BFS-based)
# ---------------------------------------------------------------------------


def topological_dimension(mean_nbhd_by_radius: List[float]) -> Optional[float]:
    """Fit n(r) ~ r^D over the growth regime of BFS neighbourhood sizes; return D.

    Restricts the fit to radii where n(r) < 90% of its maximum (i.e. the graph
    has not yet saturated), which is where the power-law holds.
    """
    arr = np.array(mean_nbhd_by_radius, dtype=float)
    if len(arr) < 3:
        return None
    max_n = arr[-1]
    # Find where saturation begins
    saturate_idx = int(np.argmax(arr >= 0.9 * max_n))
    if saturate_idx < 2:
        saturate_idx = max(3, len(arr) // 2)
    r = np.arange(1, saturate_idx + 1, dtype=float)
    nbhd = arr[:saturate_idx]
    valid = nbhd > 0
    if valid.sum() < 2:
        return None
    slope, _ = np.polyfit(np.log(r[valid]), np.log(nbhd[valid]), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Convergence rate estimators
# ---------------------------------------------------------------------------


def convergence_rate(distances: List[float]) -> Optional[float]:
    """Fit d(t) ~ A exp(-λt) to a distance trace; return the rate λ > 0.

    Truncates the trace at the first value that has converged to near-zero
    (< 1e-7) to avoid the numerical-floor distorting the fit.
    """
    if len(distances) < 4:
        return None
    arr = np.array(distances, dtype=float)
    # Truncate at numerical convergence floor
    arr = arr[arr > 1e-7]
    if len(arr) < 4:
        return None
    t = np.arange(len(arr), dtype=float)
    slope, _ = np.polyfit(t, np.log(arr), 1)
    return float(-slope)


def level_coupling_ratios(run: AnalysisRun) -> List[Optional[float]]:
    """Convergence-rate ratios λ_ℓ / λ_{ℓ+1} for successive levels.

    In the pure-gossip regime these are ~1 (all levels share the global spectral gap).
    In the SDE / training regime the ratios should approach gamma_ℓ/gamma_{ℓ+1} = 2.
    Returns a list of length depth; each entry may be None if a rate is undefined.
    """
    rates = [
        convergence_rate(run.level_trace.within_distances.get(lv, []))
        for lv in range(run.depth + 1)
    ]
    ratios: List[Optional[float]] = []
    for lv in range(len(rates) - 1):
        r0, r1 = rates[lv], rates[lv + 1]
        if r0 is not None and r1 is not None and r1 > 0:
            ratios.append(r0 / r1)
        else:
            ratios.append(None)
    return ratios


def half_life(distances: List[float]) -> Optional[int]:
    """Round index at which the distance first drops below half its initial value."""
    if not distances:
        return None
    threshold = distances[0] / 2.0
    for i, d in enumerate(distances):
        if d <= threshold:
            return i
    return None


def per_level_half_lives(run: AnalysisRun) -> Dict[int, Optional[int]]:
    """Map level -> half-life (rounds) for within-module consensus distances."""
    return {
        level: half_life(trace)
        for level, trace in run.level_trace.within_distances.items()
    }


def time_to_half_stationary_cross(
    run: AnalysisRun,
    tail_fraction: float = 0.25,
) -> Dict[int, Optional[int]]:
    """Round at which cross_distances[level] first reaches 50% of its stationary value.

    Designed for consensus-initialised runs: distances start near zero and rise
    toward a stationary separation driven by module heterogeneity.  The time to
    reach half the stationary value is the observable analogue of the Kramers
    escape time (by detailed balance, rise ≈ escape timescale).
    """
    result: Dict[int, Optional[int]] = {}
    for level, trace in run.level_trace.cross_distances.items():
        if len(trace) < 10:
            result[level] = None
            continue
        tail_start = max(0, int(len(trace) * (1.0 - tail_fraction)))
        stationary = float(np.mean(trace[tail_start:]))
        if stationary <= 0:
            result[level] = None
            continue
        threshold = 0.5 * stationary
        result[level] = next(
            (i for i, v in enumerate(trace) if v >= threshold),
            None,
        )
    return result


def time_to_half_escape_cross(
    run: AnalysisRun,
    warmup_tail_fraction: float = 0.25,
) -> Dict[int, Optional[int]]:
    """Rounds after warmup end until cross_distances drops to 50% of warmup-stationary level.

    Designed for two-phase runs (run.warmup_rounds > 0): Phase 1 establishes
    module separation with heterogeneous training data; Phase 2 switches to
    homogeneous data, removing the module-specific gradient pinning.  The
    dissolution time τ_D is the time for gossip to re-mix the established
    cluster structure.  If τ_D << τ_K = exp(ΔV/T_eff), gossip bypasses the
    Kramers barrier; if τ_D ~ τ_K, the barrier is respected.
    Returns None for a level if no dissolution is observed within the run.
    """
    n_warmup = run.warmup_rounds
    if n_warmup <= 0:
        return {}
    result: Dict[int, Optional[int]] = {}
    for level, trace in run.level_trace.cross_distances.items():
        arr = np.array(trace, dtype=float)
        if len(arr) <= n_warmup + 5:
            result[level] = None
            continue
        tail_start = max(0, int(n_warmup * (1.0 - warmup_tail_fraction)))
        warmup_stationary = float(np.mean(arr[tail_start:n_warmup]))
        if warmup_stationary <= 0:
            result[level] = None
            continue
        threshold = 0.5 * warmup_stationary
        escape_arr = arr[n_warmup:]
        result[level] = next(
            (i for i, v in enumerate(escape_arr) if v <= threshold),
            None,
        )
    return result


# ---------------------------------------------------------------------------
# Stationary variance (SDE / training regime)
# ---------------------------------------------------------------------------


def level_stationary_variance(
    run: AnalysisRun,
    tail_fraction: float = 0.2,
) -> Dict[int, float]:
    """Mean within-module consensus distance in the stationary tail of training.

    Averages within_distances[level] over the last tail_fraction of rounds.
    Higher levels (weaker coupling) should have larger stationary variance;
    theory predicts variance_ℓ / variance_{ℓ-1} = gamma_{ℓ-1} / gamma_ℓ = 2.
    """
    result: Dict[int, float] = {}
    for level, trace in run.level_trace.within_distances.items():
        if len(trace) < 10:
            continue
        tail_start = max(0, int(len(trace) * (1.0 - tail_fraction)))
        tail = trace[tail_start:]
        if tail:
            result[level] = float(np.mean(tail))
    return result


# ---------------------------------------------------------------------------
# Stationary spread (temperature test)
# ---------------------------------------------------------------------------


def stationary_spread(run: AnalysisRun) -> Optional[float]:
    """Std deviation of flattened parameter vectors in the last recorded snapshot.

    Proxy for the spread of the stationary distribution; theory predicts this
    scales as sqrt(T_eff) = sqrt(η σ² / 2).
    """
    if not run.snapshots:
        return None
    last_round = max(run.snapshots.keys())
    vectors = list(run.snapshots[last_round].values())
    if not vectors:
        return None
    return float(np.stack(vectors).std())


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def observed_vs_theory_neighbours(
    run: AnalysisRun,
) -> Tuple[List[float], List[float]]:
    """Return (observed_per_level, expected_per_level) cross-module neighbour counts."""
    return (
        run.topology.level_observed_neighbours,
        run.topology.level_expected_neighbours,
    )


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical 1-D Wasserstein-1 distance between two sample arrays."""
    n = min(len(a), len(b))
    return float(np.mean(np.abs(np.sort(a)[:n] - np.sort(b)[:n])))


# ---------------------------------------------------------------------------
# Multi-seed averaging
# ---------------------------------------------------------------------------


def average_topology(runs: List[AnalysisRun]) -> TopologyStats:
    """Average topology stats across multiple seeds (runs must share the same n_nodes)."""
    base = runs[0].topology
    n = base.n_nodes

    deg_arrs = [r.topology.degrees for r in runs if len(r.topology.degrees) == n]
    avg_degrees = np.mean(deg_arrs, axis=0) if deg_arrs else np.array([])

    exp_lists = [r.topology.level_expected_neighbours for r in runs if r.topology.level_expected_neighbours]
    avg_expected = list(np.mean(exp_lists, axis=0)) if exp_lists else []

    obs_lists = [r.topology.level_observed_neighbours for r in runs if r.topology.level_observed_neighbours]
    avg_observed = list(np.mean(obs_lists, axis=0)) if obs_lists else []

    nbhd_lists = [r.topology.mean_nbhd_by_radius for r in runs if r.topology.mean_nbhd_by_radius]
    if nbhd_lists:
        min_len = min(len(x) for x in nbhd_lists)
        avg_nbhd = list(np.mean([x[:min_len] for x in nbhd_lists], axis=0))
    else:
        avg_nbhd = []

    return TopologyStats(
        n_nodes=base.n_nodes,
        branching=base.branching,
        depth=base.depth,
        leaf_size=base.leaf_size,
        p=base.p,
        seed=-1,
        degrees=avg_degrees,
        level_expected_neighbours=avg_expected,
        level_observed_neighbours=avg_observed,
        spectral_gap=float(np.mean([r.topology.spectral_gap for r in runs])),
        mean_nbhd_by_radius=avg_nbhd,
    )


def average_level_trace(runs: List[AnalysisRun]) -> LevelTrace:
    """Average level_trace distances across multiple seeds."""

    def _avg_dict(key: str) -> Dict[int, List[float]]:
        all_levels: set = set()
        for r in runs:
            all_levels.update(getattr(r.level_trace, key).keys())
        result = {}
        for lv in all_levels:
            traces = [getattr(r.level_trace, key).get(lv, []) for r in runs]
            valid = [t for t in traces if len(t) > 0]
            if valid:
                min_len = min(len(t) for t in valid)
                result[lv] = list(np.mean([t[:min_len] for t in valid], axis=0))
        return result

    global_traces = [r.level_trace.global_distances for r in runs if r.level_trace.global_distances]
    if global_traces:
        min_len = min(len(t) for t in global_traces)
        avg_global = list(np.mean([t[:min_len] for t in global_traces], axis=0))
    else:
        avg_global = []

    return LevelTrace(
        within_distances=_avg_dict("within_distances"),
        cross_distances=_avg_dict("cross_distances"),
        global_distances=avg_global,
    )


def make_averaged_run(runs: List[AnalysisRun], name: Optional[str] = None) -> AnalysisRun:
    """Return a pseudo-AnalysisRun with topology and traces averaged over seeds."""
    base = runs[0]
    return AnalysisRun(
        name=name or base.name,
        branching=base.branching,
        depth=base.depth,
        leaf_size=base.leaf_size,
        p=base.p,
        seed=-1,
        n_rounds=base.n_rounds,
        lr=base.lr,
        sigma2=float(np.mean([r.sigma2 for r in runs])),
        topology=average_topology(runs),
        level_trace=average_level_trace(runs),
        snapshots={},   # not averaged; use seed[0] for PCA
        spectral_gaps=[],
    )
