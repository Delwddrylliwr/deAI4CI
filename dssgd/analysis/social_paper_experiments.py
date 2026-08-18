"""Experiment factories for paper2_social's Annex A campaign (C1-C4).

C5 (observational analysis on real platform datasets) has no simulation
config and lives outside this module entirely -- see the plan's separate
non-SLURM track.

Naming mirrors natural_cascade_experiments.py / ncp_experiments.py /
clique_fixation.py: one `experiment_C*` factory per Annex A row, returning a
list of already-fully-specified configs for `hpc.generate_queue.py` to turn
into a task queue.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from dssgd.protocols.gossip import protocol_suffix

from .generality import per_leaf_loss_params_for_generality
from .natural_cascade import NaturalCascadeConfig
from .ncp_runner import NCPSimConfig

# ---------------------------------------------------------------------------
# C1 -- capacity-limited gossip sweep (Prediction 6.1)
# ---------------------------------------------------------------------------


def experiment_C1_capacity_sweep(
    capacity_bits_list: List[Optional[int]] = (None, 1, 2, 3, 4, 6, 8, 12),
    generality_levels: List[int] = (1, 2, 3, 4, 5),
    a: float = 0.5,
    b_in: float = 0.042,
    b_out: float = 0.042,
    seeds: List[int] = tuple(range(50)),
    depth: int = 5,
    leaf_size: int = 4,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    source_leaf: int = 0,
    capacity_lattice_ratio: float = 3.0,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """C1: capacity-k gossip on the Paper I E6 heterogeneity testbed
    (per-leaf generality-level bias, exactly experiment_E6's construction)
    -- sweep gossip_capacity_bits (None = uncompressed control) x
    generality_levels; check_phasec1.py measures passage probability vs
    measured DL (dssgd.analysis.description_length) and vs G(b), and locates
    the stratified propagation window as a function of k (Prediction 6.1).

    capacity_bits_list includes None as the uncompressed control arm --
    Prediction 6.1 explicitly requires "at large k the compressed model
    recovers Paper I's tug dynamics", and the only way to check that
    exactly (not just approximately, by using a very large k) is to also run
    the genuinely-uncompressed config.
    """
    branching = 2
    n_leaf_types = branching ** depth
    prefix = f"C1{protocol_suffix(gossip_protocol)}"
    configs = []
    for G in generality_levels:
        plp = per_leaf_loss_params_for_generality(
            n_leaf_types=n_leaf_types, source_leaf=source_leaf, generality_level=G,
            branching=branching, a=a, b_in=b_in, b_out=b_out,
        )
        for capacity_bits in capacity_bits_list:
            k_tag = "uncompressed" if capacity_bits is None else f"k={capacity_bits}"
            for seed in seeds:
                configs.append(NaturalCascadeConfig(
                    name=f"{prefix}/G={G}/{k_tag}/seed={seed}",
                    branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b_in, per_leaf_loss_params=plp, force_flip_source=True,
                    source_leaf=source_leaf, gossip_protocol=gossip_protocol,
                    gossip_capacity_bits=capacity_bits,
                    capacity_lattice_ratio=capacity_lattice_ratio,
                ))
    return configs


# ---------------------------------------------------------------------------
# C2 -- m=1 tree vs m=4 NMH articulation ablation (Claim 5.2 / Sec. 5.2.1)
# ---------------------------------------------------------------------------


def experiment_C2_articulation_ablation(
    leaf_sizes: List[int] = (1, 4, 8, 16),
    total_n_target: int = 128,
    branching: int = 2,
    a: float = 0.5,
    b: float = 0.042,
    p: float = 2.0,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    seeds: List[int] = tuple(range(50)),
    articulation_consensus_eps: float = 0.05,
    max_intra_module_rounds: int = 50,
    gossip_protocol: str = "async_poisson",
) -> List[NaturalCascadeConfig]:
    """C2: m in leaf_sizes at MATCHED total N (depth chosen per m so
    leaf_size * branching**depth ~= total_n_target -- gossip_rate auto-sets
    to n_agents, so matched N also matches total tug rate). leaf_size=1 is
    the "bare tree of individuals" (Sec. 5.2.1); larger m are NMH's normal
    peer-clique base module. track_articulation_diagnostics=True records
    each base module's chord barrier pre/post intra-module consensus (see
    NaturalCascadeConfig's docstring); check_phasec2.py composes this with
    nmh_observables.cascade_depth's existing first-level-stall detection to
    classify stalls as articulation-limited vs scope-limited.

    leaf_size=1 has no intra-module edges at all (a lone individual has no
    peers to gossip with before reaching level 1), so it needs a higher
    inter-module edge probability `p` than the leaf_size>=4 arms to stay
    connected (NestedModularTopology raises on a disconnected graph, rather
    than retrying with a different seed, so this is a hard requirement, not
    a tuning nicety) -- p is bumped automatically for leaf_size=1 rather
    than left for the caller to discover via a topology-construction crash.
    p=8.0 was chosen empirically (p=4.0 still produced disconnected graphs
    at some seeds/depths in this hierarchy shape); callers whose own
    leaf_sizes/total_n_target combination still hits a disconnected-graph
    error should raise p_eff further, or drop the affected seed.
    """
    prefix = f"C2{protocol_suffix(gossip_protocol)}"
    configs = []
    for leaf_size in leaf_sizes:
        depth = max(1, round(np.log(total_n_target / leaf_size) / np.log(branching)))
        p_eff = max(p, 8.0) if leaf_size == 1 else p
        for seed in seeds:
            configs.append(NaturalCascadeConfig(
                name=f"{prefix}/m={leaf_size}/depth={depth}/seed={seed}",
                branching=branching, depth=depth, leaf_size=leaf_size, p=p_eff, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, force_flip_source=True, source_leaf=0,
                gossip_protocol=gossip_protocol,
                track_articulation_diagnostics=True,
                articulation_consensus_eps=articulation_consensus_eps,
                max_intra_module_rounds=max_intra_module_rounds,
            ))
    return configs


# ---------------------------------------------------------------------------
# C4 -- NCP with/without synthetic peripheral cliques (Sec. 5.2.2)
# ---------------------------------------------------------------------------


def experiment_C4_peripheral_cliques(
    clique_sizes: List[int] = (1, 4, 8),
    seeds: List[int] = tuple(range(200)),
    n_nodes: int = 1000,
    p_f: float = 0.37,
    r: float = 0.5,
    a: float = 0.5,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 800,
    clique_fraction: float = 1.0,
    gossip_protocol: str = "async_poisson",
) -> List[NCPSimConfig]:
    """C4: peripheral_clique_size in clique_sizes (1 = paired "without"
    control -- see ForestFireWithPeripheralCliques's docstring for why this,
    not plain ForestFireTopology, is the correct control arm) x seed,
    modeled directly on experiment_NCP4's matched-quality-innovation
    propagation-matrix measurement. Every clique_size at a given seed shares
    the SAME grafted_hub_ids (same seed -> same periphery selection), so
    check_phasec4.py's inward-propagation comparison across clique_size is
    genuinely paired, not just matched-in-distribution.
    """
    prefix = f"C4{protocol_suffix(gossip_protocol)}"
    configs = []
    for clique_size in clique_sizes:
        for seed in seeds:
            configs.append(NCPSimConfig(
                name=f"{prefix}/clique_size={clique_size}/p_f={p_f}/seed={seed}",
                n_nodes=n_nodes, p_f=p_f, r=r, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, gossip_protocol=gossip_protocol,
                peripheral_clique_size=clique_size,
                peripheral_clique_fraction=clique_fraction,
            ))
    return configs


# ---------------------------------------------------------------------------
# C3 -- operational DL / vartheta / generalization-gap correlation
# (Lemma 4.2 / Remark 4.4)
# ---------------------------------------------------------------------------


def basin_specs_for_C3(
    gatec1_basin_family: Optional[List[Dict]] = None,
    generality_levels: List[int] = (1, 2, 3, 4, 5),
    a: float = 0.5,
    # Must stay inside the bistable range |b/a| < 0.096225 (theory.
    # _saddle_node_roots's KAPPA_PHI-derived limit) or theory.chord_geometry
    # raises -- at a=0.5 that's b < ~0.0481; 0.044 is the largest round
    # value comfortably inside it. Values at or above the limit correspond
    # to a single merged basin (no vartheta to speak of, and receiver
    # relaxation alone would reach B with no tug at all, which is exactly
    # the degenerate case Definition 4.1's bit-budget question is meant to
    # exclude).
    b_in_list: List[float] = (0.020, 0.026, 0.032, 0.038, 0.044),
    b_out: float = 0.042,
    depth: int = 5,
    branching: int = 2,
    source_leaf: int = 0,
) -> List[Dict]:
    """C3 has no agent-based simulation config -- description_length.
    operational_dl() runs a lightweight sender/receiver relax directly on a
    loss_fn, not a full NMH cascade -- so this returns basin SPECS (loss
    params + the vartheta each one nominally sits at), not NaturalCascadeConfig
    objects. check_phasec3.py turns each spec into a loss_fn via
    active_escape.make_bistable_loss_fn and calls operational_dl() on it.

    Sweeps b_in_list x generality_levels: b_in is what actually moves
    vartheta_A_to_B (theory.chord_geometry) -- Lemma 4.2's duality test
    needs basins that genuinely SPAN a vartheta range, not just a family
    tagged with different generality labels at one fixed (a, b) (which
    would give every basin the identical vartheta and make the correlation
    undefined). generality_level rides along as a second, independent axis
    for the (unGATED, reported-only) DL-vs-generality correlation in
    check_phasec3.py.

    gatec1_basin_family (from gatec1_review.json, via generate_queue.py's
    load_gate_results): if C1's own sweep already touched a family of
    (a, b_in, b_out, generality_level) combinations, reuse THOSE exactly so
    a DL measured here is directly comparable to C1's own passage-probability
    curve at the same basin, rather than a separately-chosen family that
    happens to look similar. Falls back to the (generality_levels, a,
    b_in_list, b_out) arguments when no gate file is supplied, matching
    every other phase's gate-optional fallback pattern.
    """
    if gatec1_basin_family:
        return list(gatec1_basin_family)

    n_leaf_types = branching ** depth
    specs = []
    for b_in in b_in_list:
        for G in generality_levels:
            specs.append({
                "generality_level": G,
                "a": a, "b_in": b_in, "b_out": b_out,
                "n_leaf_types": n_leaf_types, "source_leaf": source_leaf,
                "branching": branching,
            })
    return specs
