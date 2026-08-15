# -*- coding: utf-8 -*-
"""Generate sharded task queue for HPC SLURM array jobs.

Phase ID convention:
  "1a" / 1  — Phase 1A: async gossip (existing Phase 1 queue)
  "1sp"     — Phase 1SP: synchronous gossip mirror of Phase 1A, using
              gossip_protocol="sync_pairwise" (SynchronousPairwiseGossip,
              suffix "SP" -- see dssgd.protocols.gossip.protocol_suffix /
              gossip_mechanisms.md).
  "2a" / 2  — Phase 2A: async gossip
  "2sp"     — Phase 2SP: synchronous gossip mirror
  (same pattern for 3a/3sp, 4a/4sp, ... 7a/7sp)

NOTE on phase-ID history: this phase-ID suffix used to be the bare "s"
(e.g. "1s", "2s"), on the assumption that a single phase-level letter was
enough to say "the sync track" -- unlike the experiment-name suffix
("SP"/"SN"), which was deliberately made 3-way because the same "S" string
had silently meant two different mechanisms (GossipAveraging vs
SynchronousPairwiseGossip) at different points in the repo's history (see
gossip_mechanisms.md). It turned out the phase-level label had exactly the
same problem: local pkl/review archives for the old "1s"/"2s"/"6s" phases
were generated at different points relative to the SynchronousPairwiseGossip
fix, so "the sync run of phase N" could not be trusted to mean one
mechanism either, and in Phase 2's and Phase 6's case turned out to mean
different mechanisms for different experiments *within the same phase*.
The phase ID is now "Xsp" throughout (matching what every phase branch
below actually, and exclusively, constructs: gossip_protocol="sync_pairwise").
There is currently no "Xsn" phase branch -- sync_neighbourhood was never an
intentional target of any phase, only a historical bug artifact -- but the
ID space is reserved (see gossip_mechanisms.md's phase-nomenclature section)
for anyone who deliberately wants to queue it as a comparison arm. Old
archived data generated under the bare "Xs" convention has been relabelled
to "Xsn" or "Xsp" (or quarantined, where it was a genuine sub-experiment
mix) to match what each archive actually contains -- see
gossip_mechanisms.md.

Integer phase IDs 1–4 are accepted as aliases for "1a"–"4a" (backwards compat).
Default queue/results directories are queue/phase{id}/ and results/phase{id}/,
so --phase 1sp automatically uses queue/phase1sp/ and results/phase1sp/.

Usage:
  python -m hpc.generate_queue --phase 1   --queue-dir queue/phase1
  python -m hpc.generate_queue --phase 1sp --queue-dir queue/phase1sp
  python -m hpc.generate_queue --phase 2a  --gate1-results review/gate1_review.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Allow running as `python -m hpc.generate_queue` from dssgd/ directory.
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.natural_cascade_experiments import (
    experiment_E1,
    experiment_E2,
    experiment_E3,
    experiment_E4,
    experiment_E5,
    experiment_E6,
    experiment_E6_positive_control,
    experiment_E11,
    experiment_E12a,
    experiment_E13,
    experiment_E14_meritocratic_filter_local_steps,
    experiment_E16,
    experiment_E16v2,
    experiment_E5Hv2,
    experiment_NMH1,
    experiment_NMH1b,
    experiment_NMH1sb,
    experiment_NMH2,
    experiment_NMH3,
    experiment_NMH4,
    experiment_NMH5,
    experiment_NMH6,
    experiment_NMH7,
)
from analysis.clique_fixation import (
    experiment_E7,
    experiment_E12b,
    experiment_E14_round_ratio_sweep,
    experiment_E15_distributed_kick_curvature_ratchet,
)
from analysis.chord_calibration import experiment_E0
from analysis.multi_basin_destruction import experiment_E15MB
from analysis.generic_topology_experiments import experiment_E9, experiment_E10
from analysis.ncp_experiments import (
    experiment_E8,
    experiment_NCP1_graph_configs,
    experiment_NCP2,
    experiment_NCP3,
    experiment_NCP4,
    experiment_NCP5,
)
from dssgd.protocols.gossip import protocol_suffix
from dssgd.topology.forest_fire import ForestFireTopology
from hpc.serialization import config_to_dict, task_id_from_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SHARDS = 100
CHECKPOINT_EVERY_DEFAULT = 50

_VALID_PHASES = {
    "1a", "1sp", "1sb", "2a", "2sp", "3a", "3sp", "4a", "4sp",
    "5a", "5sp", "6a", "6sp", "7a", "7sp",
    "h1",  # paper1_computing_hybrid_gossip.md Annex B.1's E0 gate (see plan doc)
    "h2",  # Annex B.1's E7 (reused as-is) + E14RR (round-ratio sweep)
    "h3",  # Annex B.2 protocol axis: E17 (analysis-only), E16, E11H, E5H, E15MB
    "h3fix",  # Interim re-run: E16v2 (connectivity-retry fix) + E5Hv2 (corrected sigma range)
}
_INT_ALIAS = {1: "1a", 2: "2a", 3: "3a", 4: "4a", 5: "5a", 6: "6a", 7: "7a"}

# Estimated wall-clock hours per task (used for duration-balanced shard assignment).
# SP-variant (sync_pairwise) estimates are approximately 2x their A counterparts,
# carried over from when these were measured against GossipAveraging
# (sync_neighbourhood, all-N-agent averaging per gradient step) under the old,
# now-retired bare "synchronous" label; SynchronousPairwiseGossip's round-
# synchronous maximal matching has comparable per-round cost to async's ~2.56
# events, so these estimates have not been re-measured for the pairwise
# mechanism specifically and may overstate the SP/A ratio. sync_neighbourhood
# is not currently queued by any phase branch below (see gossip_mechanisms.md);
# if it is added, its cost is expected to resemble these SP estimates (same
# order of per-round work, different mixing rule) until measured directly.
EXPERIMENT_HOURS: Dict[str, float] = {
    # Phase 1A / 2A / 3A / 4A
    "NMH1": 0.5,
    "NMH1b": 0.5,
    "NMH2": 0.6,
    "NMH2b": 0.6,
    "NMH3": 0.5,
    "NMH4_pilot": 4.6,
    "NMH4": 9.3,
    "NMH5": 0.5,
    "NMH6": 1.2,
    "NMH7_pilot": 1.0,
    "NMH7": 2.0,
    "NCP1": 0.02,
    "NCP2": 0.8,
    "NCP3": 0.8,
    "NCP4": 0.8,
    "NCP5": 68.0,
    "Comparative": 0.5,
    # Phase 1SP / 1sB / 2SP / 3SP / 4SP (sync_pairwise mirrors and transition sweep)
    "NMH1sb": 0.5,
    "NMH1sbSP": 1.0,
    "NMH1SP": 1.0,
    "NMH1bSP": 1.0,
    "NMH3SP": 1.0,
    "NMH2SP": 1.2,
    "NMH2bSP": 1.2,
    "NMH6SP": 2.4,
    "NCP2SP": 1.6,
    "NMH5SP": 1.0,
    "NMH7SP": 4.0,
    "NCP3SP": 1.6,
    "NMH4SP_pilot": 9.3,
    "NMH4SP": 18.6,
    "NCP4SP": 1.6,
    "NCP5SP": 136.0,
    "ComparativeSP": 1.0,
    # Phase 5A/5S, 6A/6S, 7A/7S (Annex B: E1-E13). Estimates below are
    # calibrated from a two-part local benchmark (see the project plan):
    # (1) a structural unit-count ratio to NMH1 (agents x total_rounds x
    #     local_steps, per task, at each experiment's OWN full-default
    #     parameters), scaled by NMH1's trusted 0.5h; (2) cross-validated by
    #     measuring LOCAL wall-clock time for E4/E8/E10/E13 (their full
    #     topology/agent-count, rounds reduced by the SAME fraction as a
    #     matched NMH1 reference run) and confirming the ratio-of-measured-
    #     times lands within ~15-20% of the structural prediction -- this
    #     cross-validation is what justifies using the structural ratio
    #     (rather than raw local seconds, which do not transfer to HPC
    #     hardware) for the experiments not separately measured.
    # E7/E9/E12b were measured directly at full scale (cheap enough to run
    # outright) and converted via the SAME local-to-HPC correction factor
    # implied by the NMH1 reference's reduced-vs-full-scale ratio.
    # NOTE: this exercise found E8's structural/measured cost (~3.0h) is
    # ~3.75x the EXISTING NCP2 entry (0.8h) despite identical per-task
    # config shape (same n_nodes/local_steps/rounds) -- flagging this as a
    # likely-stale NCP2 estimate for whoever maintains that entry, rather
    # than silently matching it.
    "E1": 0.5,       # NMH1-shaped (~2.5% provenance overhead, Phase 5 benchmark)
    "E2": 0.5,       # NMH1-shaped (fixed-lambda sweep)
    "E2SP": 1.0,
    "E3": 0.5,       # NMH1-shaped per TASK (the init=A/B split is separate tasks, not 2x cost each)
    "E3SP": 1.0,
    "E4": 1.8,       # depth=7 (512 agents): structural ratio 4.0x NMH1, measured ratio 3.47x
    "E4SP": 3.6,
    "E5": 0.5,       # NMH1-shaped sigma-sweep with provenance tracking
    "E6": 0.5,       # NMH1-shaped (per-leaf heterogeneity, no hierarchy change)
    "E6SP": 1.0,
    "E7": 0.01,      # single clique (worst case m=16), no hierarchy — measured at full scale
    "E7SP": 0.02,
    "E8": 3.0,       # measured: n_nodes=1000 is ~6-8x NMH1's 128 agents per round
    "E8SP": 6.0,
    "E9": 0.001,     # single-shot aggregation + relax, no gossip loop, no per-round Topology overhead
    "E10": 0.5,      # measured ratio 0.86x NMH1 (dumbbell at matched N) — structural estimate confirmed
    "E10SP": 1.0,
    "E11": 0.5,      # NMH1-shaped scheduling sweep
    "E12a": 0.5,     # == E6
    "E12aSP": 1.0,
    "E12b": 0.01,    # == E7
    "E12bSP": 0.02,
    "E13": 4.0,      # worst case m=32 (1024 agents): structural ratio 8.0x, measured ratio 7.15x
    "E13SP": 8.0,
    "E14": 0.3,      # UNMEASURED rough estimate: depth=3 (8 leaf-types, smaller than E6/E12a's
                      # depth=5) but local_steps up to 400 (8x the usual 50) in the same sweep;
                      # protocol is embedded in the config name (all 3 mechanisms share this one
                      # entry, not split into E14SP/E14SN) -- refine once actually run.
    "E15": 0.01,     # == E12b (identical mechanism, just a distributed vs fixed kick weight --
                      # no reason to expect a different per-task cost)
    "E15SP": 0.02,
    # Phase H1 (Annex B.1's E0 gate -- see the hybrid-campaign plan doc).
    # Pure root-finding on the loss landscape, no agents/gossip/RNG at all
    # -- cheaper even than NCP1 (0.02h), analogous treatment.
    "E0": 0.01,
    # Phase H2 (Annex B.1 remainder). E14RR is clique-scale like E7/E12b --
    # same order of cost, no reason to expect otherwise until measured.
    "E14RR": 0.01,
    # Phase H3 (Annex B.2 protocol axis). E17 is analysis-only (reads H2's
    # own pkls, no new tasks, no EXPERIMENT_HOURS entry needed). E16/E11H/
    # E5H are NMH1-shaped hierarchy sweeps like their un-suffixed
    # counterparts (E11/E5 above); E16's n_meas=2000 is ~2x NMH1's default
    # 1000, so estimated at ~2x E11/E5's 0.5h pending an actual measurement.
    # E15MB is clique-scale with NO local gradient steps at all (Type-N
    # rounds only, <=10 rounds, m<=16) -- cheaper even than E7/E12b/E14RR,
    # closer to E9's single-shot-aggregation cost.
    "E16": 1.0,
    "E11H": 0.5,
    "E5H": 0.5,
    "E15MB": 0.001,
    # Phase H3fix (interim re-run of E16/E5H, see review/review/phaseh3/
    # ANALYSIS.md for why): same per-task shape as their originals, no
    # reason to expect a different cost -- the fixes are a topology-
    # construction retry (cheap; verified locally, <1s for 20 seeds even
    # at the worst-case p=0.5) and a sigma value change (no effect on
    # per-task cost).
    "E16v2": 1.0,
    "E5v2": 0.5,
    "E5Hv2": 0.5,
}

# ---------------------------------------------------------------------------
# Task-dict construction helpers
# ---------------------------------------------------------------------------


def _nc_task(
    config,
    experiment: str,
    phase: Union[str, int],
    results_root: Path,
    checkpoint_every: int = CHECKPOINT_EVERY_DEFAULT,
) -> Dict[str, Any]:
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "natural_cascade",
        "config": config_to_dict(config),
        "checkpoint_every": checkpoint_every,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _ncp_task(
    config,
    experiment: str,
    phase: Union[str, int],
    results_root: Path,
    checkpoint_every: int = CHECKPOINT_EVERY_DEFAULT,
) -> Dict[str, Any]:
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "ncp_simulation",
        "config": config_to_dict(config),
        "checkpoint_every": checkpoint_every,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _clique_task(
    config,
    experiment: str,
    phase: Union[str, int],
    results_root: Path,
) -> Dict[str, Any]:
    """CliqueFixationConfig tasks (E7, E12b) — no checkpointing (each trial
    is a single small clique, seconds to run; see EXPERIMENT_HOURS)."""
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "clique_fixation",
        "config": config_to_dict(config),
        "checkpoint_every": 0,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _multi_basin_task(
    config,
    experiment: str,
    phase: Union[str, int],
    results_root: Path,
) -> Dict[str, Any]:
    """MultiBasinConfig tasks (E15MB) -- no checkpointing: Type-N rounds
    only, no local gradient steps, a handful of rounds on one clique (same
    "seconds to run" class as _clique_task's E7/E12b, if not cheaper)."""
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "multi_basin_destruction",
        "config": config_to_dict(config),
        "checkpoint_every": 0,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _generic_task(
    config,
    experiment: str,
    phase: Union[str, int],
    results_root: Path,
    checkpoint_every: int = CHECKPOINT_EVERY_DEFAULT,
) -> Dict[str, Any]:
    """GenericTopologyConfig tasks (E9 collapse, E10 cascade). Checkpointing
    for cascade-mode runs is handled inside generic_topology_runner itself
    (see that module's docstring) via the same checkpoint_every convention."""
    task_id = task_id_from_config(config)
    pkl_path = str(results_root / "pkl" / experiment / f"{task_id}.pkl")
    return {
        "task_id": task_id,
        "experiment": experiment,
        "phase": phase,
        "sim_type": "generic_topology",
        "config": config_to_dict(config),
        "checkpoint_every": checkpoint_every if config.mode == "cascade" else 0,
        "estimated_hours": EXPERIMENT_HOURS.get(experiment, 1.0),
        "result_pkl_path": pkl_path,
    }


def _chord_task(config, phase: Union[str, int]) -> Dict[str, Any]:
    """E0 is chord geometry only -- no simulation, no checkpoint, no pkl,
    same treatment as _ncp1_task below."""
    task_id = task_id_from_config(config)
    return {
        "task_id": task_id,
        "experiment": "E0",
        "phase": phase,
        "sim_type": "chord_threshold_calibration",
        "config": config_to_dict(config),
        "checkpoint_every": 0,
        "estimated_hours": EXPERIMENT_HOURS["E0"],
        "result_pkl_path": None,
    }


def _ncp1_task(config, phase: Union[str, int]) -> Dict[str, Any]:
    """NCP-1 is graph-structure only — no simulation, no checkpoint, no pkl."""
    task_id = task_id_from_config(config)
    return {
        "task_id": task_id,
        "experiment": "NCP1",
        "phase": phase,
        "sim_type": "ncp_graph_only",
        "config": config_to_dict(config),
        "checkpoint_every": 0,
        "estimated_hours": EXPERIMENT_HOURS["NCP1"],
        "result_pkl_path": None,
    }


# ---------------------------------------------------------------------------
# Phase task builders
# ---------------------------------------------------------------------------


def _build_ncp2_tasks_with_clamping(
    seeds: List[int],
    phase: Union[str, int],
    results_root: Path,
    n_nodes: int = 1000,
    p_f: float = 0.37,
    gossip_protocol: str = "async_poisson",
) -> List[Dict[str, Any]]:
    """Build NCP-2 tasks with clamped_shell baked in from a reference graph.

    We instantiate one ForestFireTopology per (n_nodes, p_f, seed) to discover
    the actual innermost (max k-core) and outermost (min k-core) shell ids, then
    create two tasks per seed: one clamping the innermost shell (outward cascade)
    and one clamping the outermost shell (inward cascade).
    """
    exp_name = f"NCP2{protocol_suffix(gossip_protocol)}"
    tasks = []
    for seed in seeds:
        topo = ForestFireTopology(n=n_nodes, p_f=p_f, seed=seed)
        shells = topo.shell_assignment()
        shell_vals = list(shells.values())
        innermost = max(shell_vals)   # highest k-core number = densest core
        outermost = min(shell_vals)

        for direction, clamped in [("outward", innermost), ("inward", outermost)]:
            configs = experiment_NCP2(
                seeds=[seed],
                n_nodes=n_nodes,
                p_f=p_f,
                clamped_shell=clamped,
                gossip_protocol=gossip_protocol,
            )
            for cfg in configs:
                # Rename to include direction for distinguishable task_ids.
                cfg.name = cfg.name.replace(
                    f"clamp={clamped}", f"dir={direction}/clamp={clamped}"
                )
                tasks.append(_ncp_task(cfg, exp_name, phase, results_root))
    return tasks


def _normalize_phase(phase: Union[str, int]) -> str:
    """Convert integer phase aliases (1–4) to canonical string IDs."""
    if isinstance(phase, int):
        if phase not in _INT_ALIAS:
            raise ValueError(f"Integer phase must be 1–4, got {phase}")
        return _INT_ALIAS[phase]
    phase = str(phase).lower()
    if phase not in _VALID_PHASES:
        raise ValueError(
            f"Unknown phase {phase!r}. Valid values: {sorted(_VALID_PHASES)} "
            f"or integers 1–4 (aliases for 1a–4a)."
        )
    return phase


def build_phase_tasks(
    phase: Union[str, int],
    gate_results: Dict[str, Any],
    results_root: Path,
    n_seeds_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return all task dicts for a given phase.

    gate_results is a merged dict from load_gate_results(); keys are prefixed
    with "gate1_", "gate2_", "gate3_" matching which review file they came from.

    n_seeds_override: if set, caps the number of seeds per config (useful for
    smoke-testing the queue generation without generating thousands of tasks).
    """
    phase = _normalize_phase(phase)
    tasks: List[Dict[str, Any]] = []

    def _seeds(default_count: int) -> List[int]:
        n = n_seeds_override if n_seeds_override is not None else default_count
        return list(range(n))

    # ── Phase 1A (async) ────────────────────────────────────────────────────
    if phase == "1a":
        for cfg in experiment_NMH1(seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "NMH1", phase, results_root))

        for cfg in experiment_NMH1b(seeds=_seeds(15)):
            tasks.append(_nc_task(cfg, "NMH1b", phase, results_root))

        for cfg in experiment_NMH3(seeds=_seeds(25)):
            tasks.append(_nc_task(cfg, "NMH3", phase, results_root))

        for cfg in experiment_NCP1_graph_configs():
            tasks.append(_ncp1_task(cfg, phase))

        for cfg in experiment_NMH7(seeds=_seeds(5), depth=4, n_meas=2000):
            tasks.append(_nc_task(cfg, "NMH7_pilot", phase, results_root))

    # ── Phase 1SP (sync_pairwise) ────────────────────────────────────────────
    elif phase == "1sp":
        for cfg in experiment_NMH1(gossip_protocol="sync_pairwise", seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "NMH1SP", phase, results_root))

        # Extended ls sweep: [1,2,5] probe the synchronous threshold;
        # [10,50,200] are direct comparison points with Phase 1A.
        for cfg in experiment_NMH1b(
            gossip_protocol="sync_pairwise",
            local_steps_list=[1, 2, 5, 10, 50, 200],
            seeds=_seeds(15),
        ):
            tasks.append(_nc_task(cfg, "NMH1bSP", phase, results_root))

        for cfg in experiment_NMH3(gossip_protocol="sync_pairwise", seeds=_seeds(25)):
            tasks.append(_nc_task(cfg, "NMH3SP", phase, results_root))

        # NCP1 excluded — graph-only, no gossip protocol.

    # ── Phase 1sB (sync transition zone: 2D a × local_steps sweep) ──────────
    elif phase == "1sb":
        for cfg in experiment_NMH1sb(gossip_protocol="sync_pairwise", seeds=_seeds(75)):
            tasks.append(_nc_task(cfg, "NMH1sbSP", phase, results_root))

    # ── Phase 2A (async) ────────────────────────────────────────────────────
    elif phase == "2a":
        b_list = gate_results.get(
            "gate1_recommended_b_values",
            [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050],
        )

        for cfg in experiment_NMH2(b_list=b_list, seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "NMH2", phase, results_root))

        for cfg in experiment_NMH2(b_list=b_list, seeds=list(range(30, 30 + len(_seeds(30))))):
            cfg.name = cfg.name.replace("NMH2/", "NMH2b/")
            tasks.append(_nc_task(cfg, "NMH2b", phase, results_root))

        for cfg in experiment_NMH6(seeds=_seeds(10)):
            tasks.append(_nc_task(cfg, "NMH6", phase, results_root))

        tasks.extend(_build_ncp2_tasks_with_clamping(_seeds(10), phase, results_root))

    # ── Phase 2SP (sync_pairwise) ────────────────────────────────────────────
    elif phase == "2sp":
        b_list = gate_results.get(
            "gate1_recommended_b_values",
            [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050],
        )

        for cfg in experiment_NMH2(
            b_list=b_list, seeds=_seeds(30), gossip_protocol="sync_pairwise"
        ):
            tasks.append(_nc_task(cfg, "NMH2SP", phase, results_root))

        for cfg in experiment_NMH2(
            b_list=b_list,
            seeds=list(range(30, 30 + len(_seeds(30)))),
            gossip_protocol="sync_pairwise",
        ):
            cfg.name = cfg.name.replace("NMH2S/", "NMH2bS/")
            tasks.append(_nc_task(cfg, "NMH2bSP", phase, results_root))

        for cfg in experiment_NMH6(seeds=_seeds(10), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "NMH6SP", phase, results_root))

        tasks.extend(
            _build_ncp2_tasks_with_clamping(
                _seeds(10), phase, results_root, gossip_protocol="sync_pairwise"
            )
        )

    # ── Phase 3A (async) ────────────────────────────────────────────────────
    elif phase == "3a":
        b_on_a_list = gate_results.get(
            "gate2_recommended_b_values_phase3",
            [0.02, 0.03, 0.04, 0.05, 0.06],
        )

        for cfg in experiment_NMH5(b_on_a_list=b_on_a_list, seeds=_seeds(50)):
            tasks.append(_nc_task(cfg, "NMH5", phase, results_root))

        for cfg in experiment_NMH7(seeds=_seeds(15), depth=4, n_meas=4000):
            tasks.append(_nc_task(cfg, "NMH7", phase, results_root))

        for cfg in experiment_NCP3(seeds=_seeds(10)):
            tasks.append(_ncp_task(cfg, "NCP3", phase, results_root))

        for cfg in experiment_NMH4(seeds=_seeds(200), depth=5, n_meas=500):
            tasks.append(_nc_task(cfg, "NMH4_pilot", phase, results_root))

    # ── Phase 3SP (sync_pairwise) ────────────────────────────────────────────
    elif phase == "3sp":
        b_on_a_list = gate_results.get(
            "gate2_recommended_b_values_phase3",
            [0.02, 0.03, 0.04, 0.05, 0.06],
        )

        for cfg in experiment_NMH5(
            b_on_a_list=b_on_a_list, seeds=_seeds(50), gossip_protocol="sync_pairwise"
        ):
            tasks.append(_nc_task(cfg, "NMH5SP", phase, results_root))

        for cfg in experiment_NMH7(
            seeds=_seeds(15), depth=4, n_meas=4000, gossip_protocol="sync_pairwise"
        ):
            tasks.append(_nc_task(cfg, "NMH7SP", phase, results_root))

        for cfg in experiment_NCP3(seeds=_seeds(10), gossip_protocol="sync_pairwise"):
            tasks.append(_ncp_task(cfg, "NCP3SP", phase, results_root))

        for cfg in experiment_NMH4(
            seeds=_seeds(200), depth=5, n_meas=500, gossip_protocol="sync_pairwise"
        ):
            tasks.append(_nc_task(cfg, "NMH4SP_pilot", phase, results_root))

    # ── Phase 4A (async) ────────────────────────────────────────────────────
    elif phase == "4a":
        phase4_mods = gate_results.get("gate3_phase4_modifications", "")
        depth = 7 if "depth=7" in phase4_mods else 6

        for a in [1.0, 2.0, 4.0]:
            for cfg in experiment_NMH4(seeds=_seeds(500), depth=depth, a=a):
                tasks.append(_nc_task(cfg, "NMH4", phase, results_root))

        for cfg in experiment_NCP4(seeds=_seeds(200)):
            tasks.append(_ncp_task(cfg, "NCP4", phase, results_root))

        for cfg in experiment_NCP5(seeds=_seeds(10)):
            tasks.append(_ncp_task(cfg, "NCP5", phase, results_root,
                                    checkpoint_every=CHECKPOINT_EVERY_DEFAULT))

        for cfg in experiment_NMH1(a_list=[0.5], seeds=_seeds(100)):
            cfg.name = cfg.name.replace("NMH1/", "Comparative/")
            tasks.append(_nc_task(cfg, "Comparative", phase, results_root))

    # ── Phase 4SP (sync_pairwise) ────────────────────────────────────────────
    elif phase == "4sp":
        phase4_mods = gate_results.get("gate3_phase4_modifications", "")
        depth = 7 if "depth=7" in phase4_mods else 6

        for a in [1.0, 2.0, 4.0]:
            for cfg in experiment_NMH4(
                seeds=_seeds(500), depth=depth, a=a, gossip_protocol="sync_pairwise"
            ):
                tasks.append(_nc_task(cfg, "NMH4SP", phase, results_root))

        for cfg in experiment_NCP4(seeds=_seeds(200), gossip_protocol="sync_pairwise"):
            tasks.append(_ncp_task(cfg, "NCP4SP", phase, results_root))

        for cfg in experiment_NCP5(seeds=_seeds(10), gossip_protocol="sync_pairwise"):
            tasks.append(_ncp_task(cfg, "NCP5SP", phase, results_root,
                                    checkpoint_every=CHECKPOINT_EVERY_DEFAULT))

        for cfg in experiment_NMH1(
            a_list=[0.5], seeds=_seeds(100), gossip_protocol="sync_pairwise"
        ):
            cfg.name = cfg.name.replace("NMH1S/", "ComparativeS/")
            tasks.append(_nc_task(cfg, "ComparativeSP", phase, results_root))

    # ── Phase 5A (Track A: independent new infra, no gate dependency) ───────
    # E1, E6, E7, E9, E10, E13 need only their own new subsystem (built this
    # phase), not any earlier phase's fitted results -- schedulable in
    # parallel with, or before, phases 1-4 completing.
    elif phase == "5a":
        for cfg in experiment_E1(seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "E1", phase, results_root))

        for cfg in experiment_E6(seeds=_seeds(50)):
            tasks.append(_nc_task(cfg, "E6", phase, results_root))

        for cfg in experiment_E7(seeds=_seeds(50)):
            tasks.append(_clique_task(cfg, "E7", phase, results_root))

        for cfg in experiment_E9(seeds=_seeds(50)):
            tasks.append(_generic_task(cfg, "E9", phase, results_root))

        for cfg in experiment_E10(seeds=_seeds(50)):
            tasks.append(_generic_task(cfg, "E10", phase, results_root))

        for cfg in experiment_E13(seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "E13", phase, results_root))

    # ── Phase 5SP (sync_pairwise mirrors, where meaningful) ─────────────────
    elif phase == "5sp":
        for cfg in experiment_E6(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "E6SP", phase, results_root))

        for cfg in experiment_E7(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_clique_task(cfg, "E7SP", phase, results_root))

        for cfg in experiment_E10(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_generic_task(cfg, "E10SP", phase, results_root))
        # E9 is single-shot (no gossip protocol) -- no sync variant.

        for cfg in experiment_E13(seeds=_seeds(30), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "E13SP", phase, results_root))

    # ── Phase 6A (Track C: second-order on Phase 5's OWN new infra) ────────
    # E5 needs E1's provenance mechanism; E11 needs both provenance and
    # bounded-staleness; E12(a) needs E6's generality machinery; E12(b)
    # needs E7's clique-fixation machinery. None of these are gated on
    # phase 1-4 RESULTS, only on phase 5's CODE existing (which it does).
    elif phase == "6a":
        for cfg in experiment_E5(seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "E5", phase, results_root))

        for cfg in experiment_E11(seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "E11", phase, results_root))

        for cfg in experiment_E12a(seeds=_seeds(50)):
            tasks.append(_nc_task(cfg, "E12a", phase, results_root))

        # Positive control (assessment doc A.6): small/fast, same-bias variant
        # verifying the containment/attainment pipeline detects d_max==G
        # before the real E12a null above is trusted -- queued here (not
        # Phase 5, where the underlying experiment_E6 mechanism lives) since
        # this is what check_phase6.py's e12a_ok scoring actually consumes,
        # from THIS phase's results tree.
        for cfg in experiment_E6_positive_control(seeds=_seeds(20)):
            tasks.append(_nc_task(cfg, "E6ctrl", phase, results_root))

        for cfg in experiment_E12b(seeds=_seeds(50)):
            tasks.append(_clique_task(cfg, "E12b", phase, results_root))

        # E14: local_steps sensitivity of the meritocratic filter, sweeping
        # all three gossip mechanisms internally (like E11) -- queued once
        # under 6a only, never under 6s (there is no separate E14 sync
        # variant; protocol is one of E14's own swept dimensions, embedded
        # in the config name, not a phase-level split).
        for cfg in experiment_E14_meritocratic_filter_local_steps(seeds=_seeds(20)):
            tasks.append(_nc_task(cfg, "E14", phase, results_root))

        # E15: E12b's curvature ratchet under a genuinely distributed kick
        # weight (kick_weight_law="uniform") instead of E12b's fixed alpha
        # -- compare against theory.fixation_bias_distributed, not
        # theory.fixation_bias. Separate output prefix "E15", never "E12b".
        for cfg in experiment_E15_distributed_kick_curvature_ratchet(seeds=_seeds(50)):
            tasks.append(_clique_task(cfg, "E15", phase, results_root))

    elif phase == "6sp":
        for cfg in experiment_E12a(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "E12aSP", phase, results_root))

        for cfg in experiment_E6_positive_control(seeds=_seeds(20), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "E6ctrlSP", phase, results_root))

        for cfg in experiment_E12b(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_clique_task(cfg, "E12bSP", phase, results_root))

        for cfg in experiment_E15_distributed_kick_curvature_ratchet(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_clique_task(cfg, "E15SP", phase, results_root))
        # E5 requires async_poisson (provenance); E11 already sweeps sync
        # internally via its own `schedulings` list -- no separate 6S variant.

    # ── Phase 7A (Track B: gated continuations of phases 1-4) ──────────────
    # E2 needs NMH1/NMH3's observed a-boundary; E3 needs NMH3 + E2; E4 needs
    # NMH4's Griffitts operating point; E8 needs NCP2's asymmetry machinery
    # to already exist (it does). Gate keys below fall back to the same
    # defaults NMH1/NMH3/NMH4 already use when no gate file is supplied --
    # the review scripts that would COMPUTE gate3_e2_a_anchor etc. from
    # phase 1-4's actual results are not yet written (see plan note).
    elif phase == "7a":
        e2_a_list, e3_a_list, e4_a = _gate1_derived_a_values(gate_results)

        for cfg in experiment_E2(a_list=e2_a_list, seeds=_seeds(30)):
            tasks.append(_nc_task(cfg, "E2", phase, results_root))

        for cfg in experiment_E3(a_list=e3_a_list, seeds=_seeds(75)):
            tasks.append(_nc_task(cfg, "E3", phase, results_root))

        for cfg in experiment_E4(a=e4_a, n_graph_seeds=30, n_dynamics_seeds_per_graph=5):
            tasks.append(_nc_task(cfg, "E4", phase, results_root))

        # E8 has no numeric gate dependency: it needs NCP-2's asymmetric-
        # nucleation *machinery* to exist (it does), not a fitted value from
        # NCP-2's results, so it runs at its own defaults unconditionally.
        for cfg in experiment_E8(seeds=_seeds(50)):
            tasks.append(_ncp_task(cfg, "E8", phase, results_root))

    elif phase == "7sp":
        e2_a_list, e3_a_list, e4_a = _gate1_derived_a_values(gate_results)

        for cfg in experiment_E2(a_list=e2_a_list, seeds=_seeds(30), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "E2SP", phase, results_root))

        for cfg in experiment_E3(a_list=e3_a_list, seeds=_seeds(75), gossip_protocol="sync_pairwise"):
            tasks.append(_nc_task(cfg, "E3SP", phase, results_root))

        for cfg in experiment_E4(
            a=e4_a, n_graph_seeds=30, n_dynamics_seeds_per_graph=5, gossip_protocol="sync_pairwise",
        ):
            tasks.append(_nc_task(cfg, "E4SP", phase, results_root))

        for cfg in experiment_E8(seeds=_seeds(50), gossip_protocol="sync_pairwise"):
            tasks.append(_ncp_task(cfg, "E8SP", phase, results_root))

    # ── Phase H1 (Annex B.1's E0: the up-front falsifier) ───────────────────
    # Gates every later H-phase (E7/E14 in H2 onward): see check_phaseh1.py /
    # gateh1_review.json and the hybrid-campaign plan doc's discussion of why
    # E0 is its own phase rather than folded into H2 alongside E7/E14.
    elif phase == "h1":
        for cfg in experiment_E0():
            tasks.append(_chord_task(cfg, phase))

    # ── Phase H2 (Annex B.1 remainder: E7 + E14RR) ──────────────────────────
    # E7 is REUSED as-is (async_poisson, Type-P only per Annex B.1's own
    # table -- E7 is not K-critical, it targets Lemma 3.1's ordinary
    # depth-only fixation formula), scored against H1's measured
    # vartheta_dagger table rather than recomputing chord_geometry
    # (check_phaseh2.py). E14RR is the new round-ratio sweep (see its
    # docstring for why it isn't named bare "E14"). gateh1_pass is checked
    # but does not hard-block queue generation -- consistent with every
    # other gate-chained phase in this file (e.g. gate1_pass never blocks
    # Phase 2 either); it's a printed warning, with the hard-stop being a
    # human/process check per check_phaseh1.py's own exit message.
    elif phase == "h2":
        if gate_results and not gate_results.get("gateh1_pass", True):
            print(
                "[warn] gateh1_pass is False -- H1's complementarity check "
                "failed or produced no data; H2's E7/E14RR scoring against "
                "H1's vartheta_dagger table will not be trustworthy until "
                "that's resolved. Queuing anyway (see check_phaseh1.py)."
            )

        for cfg in experiment_E7(seeds=_seeds(50)):
            tasks.append(_clique_task(cfg, "E7", phase, results_root))

        for cfg in experiment_E14_round_ratio_sweep(seeds=_seeds(20)):
            tasks.append(_clique_task(cfg, "E14RR", phase, results_root))

    # ── Phase H3 (Annex B.2 protocol axis: E17, E16, E11H, E5H, E15MB) ──────
    # Reads gateh2_review.json for K_low/K_mid/K_high (H2's measured H-round
    # window) -- E11H/E5H sweep across that window rather than re-deriving
    # K themselves. E17 is analysis-only and has NO task here: it reclassifies
    # H2's own E14RR pkls post-hoc (see check_phaseh3.py). E15MB needs no K
    # at all -- a pure Type-N-only mechanism comparison (barycentre vs
    # label-plurality), independent of the Type-P channel.
    elif phase == "h3":
        if gate_results and gate_results.get("gateh2_K_low") is None:
            print(
                "[warn] gateh2_K_low is missing -- H2's E14RR round-ratio "
                "sweep may have found no H-round window (Annex B.6's "
                "falsification risk #2: 'the H-round window may be empty "
                "at the default m'). Falling back to hardcoded K defaults; "
                "widen H2's m/K_list and re-run before trusting H3's results."
            )
        K_low = gate_results.get("gateh2_K_low") or 1.0
        K_mid = gate_results.get("gateh2_K_mid") or 10.0
        K_high = gate_results.get("gateh2_K_high") or 100.0
        K_window = [K_low, K_mid, K_high]

        for cfg in experiment_E16(seeds=_seeds(20)):
            tasks.append(_nc_task(cfg, "E16", phase, results_root))

        # experiment_E11/E5 also regenerate their ORIGINAL (already queued
        # in phase 6a) scheduling-only / sigma-only configs when called with
        # no schedulings/sigma_list override -- filter to just the new
        # hybrid arm so this phase doesn't duplicate phase 6a's queue.
        for cfg in experiment_E11(hybrid_K_list=K_window, seeds=_seeds(30)):
            if cfg.gossip_protocol == "hybrid":
                tasks.append(_nc_task(cfg, "E11H", phase, results_root))

        for cfg in experiment_E5(K_list=K_window, seeds=_seeds(30)):
            if cfg.gossip_protocol == "hybrid":
                tasks.append(_nc_task(cfg, "E5H", phase, results_root))

        for cfg in experiment_E15MB(seeds=_seeds(20)):
            tasks.append(_multi_basin_task(cfg, "E15MB", phase, results_root))

    # ── Phase H3fix (interim re-run: E16v2, E5Hv2) ───────────────────────────
    # See review/review/phaseh3/ANALYSIS.md for the full diagnosis of both
    # bugs this phase re-runs against. E16v2 differs from E16 only in that
    # its topology construction now retries a disconnected graph with a
    # perturbed seed (build_nmh_topology_with_retry, analysis/
    # natural_cascade.py) instead of failing the task outright -- p=0.5
    # disconnected ~80% of the time for E16's shape, so the original run's
    # surviving 4/20 p=0.5 seeds were a survivorship-biased sample, not a
    # fair one. E5Hv2 differs from E5H only in sigma_list: E5H's original
    # range (max 0.05) was ~17 effective-sigma short of the actual saddle
    # distance at a=2.0/b=0.042, giving zero flips in 540/540 runs (and,
    # confirmed separately, in the original non-hybrid E5 arm too, predating
    # this campaign entirely) -- E5Hv2's range (0.1-0.4) was picked from a
    # local calibration run showing the working threshold sits between 0.15
    # and 0.2. Reads the SAME gateh2_review.json K window as phase h3 (not
    # re-derived) so E5Hv2's hybrid arm stays comparable to E11H/E15MB.
    elif phase == "h3fix":
        if gate_results and gate_results.get("gateh2_K_low") is None:
            print(
                "[warn] gateh2_K_low is missing -- falling back to hardcoded "
                "K defaults for E5Hv2's hybrid arm (see phase h3's identical warning)."
            )
        K_low = gate_results.get("gateh2_K_low") or 1.0
        K_mid = gate_results.get("gateh2_K_mid") or 10.0
        K_high = gate_results.get("gateh2_K_high") or 100.0
        K_window = [K_low, K_mid, K_high]

        for cfg in experiment_E16v2(seeds=_seeds(20)):
            tasks.append(_nc_task(cfg, "E16v2", phase, results_root))

        for cfg in experiment_E5Hv2(K_list=K_window, seeds=_seeds(30)):
            experiment = "E5Hv2" if cfg.gossip_protocol == "hybrid" else "E5v2"
            tasks.append(_nc_task(cfg, experiment, phase, results_root))

    return tasks


def _gate1_derived_a_values(
    gate_results: Dict[str, Any],
) -> Tuple[List[float], List[float], float]:
    """Derive E2's a_list, E3's interior a_list, and E4's operating a from
    gate1_review.json's ALREADY-COMPUTED NMH-3 phase boundaries (the
    "nmh3_regime_boundaries" key, prefixed "gate1_" by load_gate_results),
    rather than a separately-fitted quantity: E2 needs the range of a "across
    the observed boundaries" (Remark 2.3) -- exactly boundary_i_ii to just
    past boundary_ii_iii; E3's "fine interior sweep" targets exactly the
    ambiguous region gate1 itself flagged as unresolved (Annex A.3: "interior
    ... statistically consistent with both a heavy-tailed stratified interior
    and containment; high-statistics discrimination outstanding (E3)"), i.e.
    a fine grid centred on boundary_ii_iii. Falls back to NMH-1/3's own
    literature defaults when no gate1 file is supplied (first-run behaviour,
    matching every other phase's fallback pattern).
    """
    boundaries = gate_results.get("gate1_nmh3_regime_boundaries", {})
    b_i_ii = boundaries.get("I_II") or 0.5
    b_ii_iii = boundaries.get("II_III") or 8.0

    e2_a_list = sorted({round(b_i_ii, 4), round((b_i_ii + b_ii_iii) / 2, 4), round(b_ii_iii, 4), round(b_ii_iii * 1.5, 4)})
    e3_a_list = sorted({round(b_ii_iii * f, 4) for f in (0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)})
    e4_a = round((b_i_ii + b_ii_iii) / 2, 4)  # midpoint of the Griffiths interior, matching NMH4's a=2 default at the literature boundaries
    return e2_a_list, e3_a_list, e4_a


# ---------------------------------------------------------------------------
# Queue writing
# ---------------------------------------------------------------------------


def write_queue(
    tasks: List[Dict[str, Any]],
    queue_root: Path,
    n_shards: int = N_SHARDS,
    overwrite: bool = False,
) -> None:
    """Distribute tasks into pending shards and write one JSON file per task.

    Tasks are sorted by descending estimated_hours before shard assignment so
    each shard gets a mix of long and short tasks (prevents one shard holding
    all long tasks while others drain quickly).

    Directory structure created under queue_root/:
      pending/shard_000/ ... shard_NNN/
      claimed/shard_000/ ...
      completed/shard_000/ ...
      failed/shard_000/ ...
    """
    pending_root = queue_root / "pending"
    if pending_root.exists() and not overwrite:
        raise FileExistsError(
            f"{pending_root} already exists. Use --overwrite to regenerate."
        )

    # Sort by descending duration for balanced shard loading.
    tasks_sorted = sorted(tasks, key=lambda t: t["estimated_hours"], reverse=True)

    for subdir in ("pending", "claimed", "completed", "failed"):
        for shard_idx in range(n_shards):
            (queue_root / subdir / f"shard_{shard_idx:03d}").mkdir(
                parents=True, exist_ok=True
            )

    for rank, task in enumerate(tasks_sorted):
        shard_idx = rank % n_shards
        task_path = (
            queue_root / "pending" / f"shard_{shard_idx:03d}" / f"{task['task_id']}.json"
        )
        with open(task_path, "w") as fh:
            json.dump(task, fh)

    print(
        f"Wrote {len(tasks)} tasks across {n_shards} shards in {queue_root}/pending/"
    )


# ---------------------------------------------------------------------------
# Gate result loading
# ---------------------------------------------------------------------------


def load_gate_results(*paths: str) -> Dict[str, Any]:
    """Load and merge gate review JSON files.

    Each file's keys are prefixed with 'gate{N}_' where N is inferred from the
    presence of 'gate1', 'gate2', 'gate3' in the filename (checked first, for
    exact backward compatibility with the original phase-1-7 convention).
    Anything else -- e.g. the letter-prefixed H-phase gates ("gateh1_review.
    json", "gateh2_review.json") -- is matched by a general "gate<label>_"
    regex instead of falling through to a purely positional "gate{i}_", so
    passing --gateh1-results doesn't silently collide with --gate1-results'
    "gate1_" prefix just because of argument order.

    A key already starting with the target prefix is left as-is rather than
    prefixed again: the H-phase gate scripts (check_phaseh1.py, check_
    phaseh2.py, ...) write their OWN keys already self-prefixed
    ("gateh1_pass", "gateh2_K_low", ...), unlike the original gate1-3
    convention where only "gate{n}_pass" is self-prefixed and everything
    else (e.g. "nmh3_regime_boundaries") is raw. Blindly prepending would
    double the prefix on every self-prefixed key ("gateh2_gateh2_K_low"),
    silently breaking every gate_results.get("gateh2_K_low", ...) lookup
    downstream with no error -- just a quiet fall-back to the default.
    """
    merged: Dict[str, Any] = {}
    for i, path in enumerate(paths, start=1):
        with open(path) as fh:
            data = json.load(fh)
        fname = Path(path).name.lower()
        prefix = None
        for n in (1, 2, 3):
            if f"gate{n}" in fname:
                prefix = f"gate{n}_"
                break
        if prefix is None:
            m = re.search(r"(gate[a-z0-9]+)_", fname)
            prefix = f"{m.group(1)}_" if m else f"gate{i}_"
        for k, v in data.items():
            merged[k if k.startswith(prefix) else f"{prefix}{k}"] = v
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sharded HPC task queue for timesep experiments."
    )
    parser.add_argument(
        "--phase", type=str, required=True,
        help=(
            "Phase ID: '1a'/'1sp', '2a'/'2sp', '3a'/'3sp', '4a'/'4sp', "
            "or integers 1–4 (aliases for 1a–4a)."
        ),
    )
    parser.add_argument(
        "--queue-dir", type=Path, default=None,
        help="Root directory for the queue. Defaults to queue/phase<ID>/.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Root directory for result pickles. Defaults to results/phase<ID>/.",
    )
    parser.add_argument("--n-shards", type=int, default=N_SHARDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        help="Override seed count per config (useful for smoke tests).",
    )
    parser.add_argument("--gate1-results", type=str, default=None)
    parser.add_argument("--gate2-results", type=str, default=None)
    parser.add_argument("--gate3-results", type=str, default=None)
    parser.add_argument(
        "--gateh1-results", type=str, default=None,
        help="Path to review/phaseh1/gateh1_review.json (phase h2's gate dependency).",
    )
    parser.add_argument(
        "--gateh2-results", type=str, default=None,
        help="Path to review/phaseh2/gateh2_review.json (phase h3's gate dependency).",
    )
    args = parser.parse_args()

    # Normalise phase: "1" → "1a", "1sp" → "1sp", 1 → "1a"
    try:
        phase_int = int(args.phase)
        phase_id = _normalize_phase(phase_int)
    except ValueError:
        phase_id = _normalize_phase(args.phase)

    queue_dir = args.queue_dir or Path(f"queue/phase{phase_id}")
    results_dir = args.results_dir or Path(f"results/phase{phase_id}")

    gate_paths = [
        p for p in [
            args.gate1_results, args.gate2_results, args.gate3_results,
            args.gateh1_results, args.gateh2_results,
        ]
        if p is not None
    ]
    gate_results = load_gate_results(*gate_paths) if gate_paths else {}

    tasks = build_phase_tasks(
        phase=phase_id,
        gate_results=gate_results,
        results_root=results_dir,
        n_seeds_override=args.n_seeds,
    )
    write_queue(tasks, queue_dir, n_shards=args.n_shards, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
