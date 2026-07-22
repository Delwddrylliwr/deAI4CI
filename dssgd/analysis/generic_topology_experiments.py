"""Experiment factory functions for generic (non-modular) topology experiments.

E9  -- centralised star: basin destruction (Lemma 6.1) + the permutation-
       alignment ablation (Remark 6.2).
E10 -- matched-spectral-gap dumbbell vs the NMH (Remark 7.1).
"""

from typing import List, Tuple

import numpy as np

from .generic_topology_runner import GenericTopologyConfig


def experiment_E9(
    n_nodes: int = 128,
    a: float = 0.5,
    b: float = 0.042,
    fraction_B_list: List[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    d_param: int = 4,
    align_groups_list: List[int] = (1, 2, 4),
    seeds: List[int] = tuple(range(50)),
    local_steps: int = 100,
    lr: float = 0.1,
) -> List[GenericTopologyConfig]:
    """One-round collapse statistics (Lemma 6.1) across a range of non-
    consensus fractions, plus the alignment ablation (Remark 6.2):
    align_groups=1 is raw (unaligned) averaging -- the ablation's baseline;
    align_groups>1 enables the simplified permutation-alignment proxy
    (generic_topology_runner._aligned_mean) on a d_param>1 parameter vector.
    """
    theta_A = np.zeros(d_param, dtype=np.float32)
    theta_B = np.zeros(d_param, dtype=np.float32)
    theta_B[0] = 1.0
    configs = []
    for frac in fraction_B_list:
        for n_groups in align_groups_list:
            for seed in seeds:
                configs.append(GenericTopologyConfig(
                    name=f"E9/frac_B={frac}/align_groups={n_groups}/seed={seed}",
                    mode="collapse", topology="star", n_nodes=n_nodes,
                    seed=seed, a=a, b=b, d_param=d_param,
                    theta_A=theta_A.copy(), theta_B=theta_B.copy(),
                    lr=lr, local_steps=local_steps,
                    collapse_fraction_B=frac, collapse_broadcast_weight=1.0,
                    collapse_align=(n_groups > 1), collapse_align_groups=n_groups,
                ))
    return configs


def nmh_reference_spectral_gap(
    branching: int = 2, depth: int = 5, leaf_size: int = 4, p: float = 2.0, seed: int = 0,
) -> float:
    """The NMH's own second-largest-eigenvalue gap at these parameters --
    the target Experiment E10's dumbbell bridge_p sweep should be matched
    against (pick whichever bridge_p in the swept grid gives the closest
    dumbbell gap, post-hoc, rather than solving for it analytically: the
    match point depends on the realised graph, not just its parameters).
    """
    from dssgd.topology.base import spectral_gap
    from dssgd.topology.static import NestedModularTopology

    topo = NestedModularTopology(branching=branching, depth=depth, leaf_size=leaf_size, p=p, seed=seed)
    _, W = topo.step(0)
    return spectral_gap(W)


def dumbbell_spectral_gap(block_size: int, bridge_p: float, seed: int = 0) -> float:
    """The dumbbell's own spectral gap at a given bridge_p -- call this
    across bridge_p_list (matching experiment_E10's sweep) to find the
    bridge_p whose gap is closest to nmh_reference_spectral_gap's result.
    """
    from dssgd.topology.base import spectral_gap
    from dssgd.topology.static import DumbbellTopology

    topo = DumbbellTopology(block_size=block_size, bridge_p=bridge_p, seed=seed)
    _, W = topo.step(0)
    return spectral_gap(W)


def experiment_E10(
    nmh_branching: int = 2,
    nmh_depth: int = 5,
    nmh_leaf_size: int = 4,
    nmh_p: float = 2.0,
    bridge_p_list: List[float] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    a_list: List[float] = (0.5, 1.0, 2.0),
    b: float = 0.042,
    seeds: List[int] = tuple(range(50)),
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 1000,
    gossip_protocol: str = "async_poisson",
) -> List[GenericTopologyConfig]:
    """Matched-spectral-gap pair (Remark 7.1): dumbbell at the same total N
    as the NMH, sweeping bridge_p across a grid. Use nmh_reference_spectral_gap
    and dumbbell_spectral_gap (both in this module) post-hoc to identify
    which bridge_p in bridge_p_list is the actual match at this N, then
    compare that slice's cascade statistics against the NMH's own (e.g.
    experiment_NMH1 at the same nmh_* parameters) -- identical spectral gap,
    the paper predicts categorically different stratification/filtering.
    """
    n_nodes = nmh_branching ** nmh_depth * nmh_leaf_size
    prefix = "E10S" if gossip_protocol != "async_poisson" else "E10"
    configs = []
    for bridge_p in bridge_p_list:
        for a in a_list:
            for seed in seeds:
                configs.append(GenericTopologyConfig(
                    name=f"{prefix}/bridge_p={bridge_p}/a={a}/seed={seed}",
                    mode="cascade", topology="dumbbell", n_nodes=n_nodes,
                    block_size=n_nodes // 2, bridge_p=bridge_p, seed=seed,
                    a=a, b=b, lr=lr, local_steps=local_steps,
                    n_warmup=n_warmup, n_meas_rounds=n_meas,
                    force_flip_source=True, gossip_protocol=gossip_protocol,
                ))
    return configs
