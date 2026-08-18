# -*- coding: utf-8 -*-
"""Multi-basin destruction test (Experiment E15MB).

paper1_computing_hybrid_gossip.md Annex B.2's E15: "Three or more occupied
basins in a single clique, Type-N rounds only; compare the parameter
barycentre (2.2b) against the label plurality (6.2), sweeping m and basin
count... barycentre predicted to show post-round loss spikes and landings
in unoccupied basins, frequency increasing in m and in basin count;
plurality predicted to show neither."

NAMED "E15MB" (multi-basin), never bare "E15": clique_fixation.py's
`experiment_E15_distributed_tug_curvature_ratchet` already occupies "E15"
for a DIFFERENT experiment (the original paper's distributed-tug-strength
curvature ratchet, Lemma 10.1's counterpart under a continuous tug-strength
law) -- an unrelated result that happens to share the revised paper's
Annex B numbering by coincidence. Reusing "E15" for this would be exactly
the naming collision gossip_mechanisms.md's "S"/"synchronous" retirement
and this campaign's own E14/E14RR split were both built to avoid.

This is a genuinely new landscape, not the two-well family of Sec. 2.4:
M >= 3 basin centres placed at `radius * e_i` (standard basis vectors) in
R^M, one well per centre, L(w) = min_i a * ||w - theta_i||^2. This is the
natural, geometry-agnostic generalisation Remark 6.4 needs (nearest-centre
classification, no chord/threshold structure, which is specific to M=2)
and keeps the destruction property automatic: the barycentre of >= 3
well-separated, symmetric centres lies equidistant from more than one
centre (or nearest to none of the originally-occupied ones) whenever the
occupied set doesn't have a majority centre, which is exactly Lemma 6.1's
"convex combinations of barrier-separated minima lie in high-loss regions"
specialised to this landscape.

No local gradient steps are taken (Annex B.2's own design: "Type-N rounds
only") -- this tests the ROUND MAP itself (LabelPluralityGossip vs
GossipAveraging), not gossip-SGD dynamics, so agents never call
`local_step()` and their loss_fn/optimizer exist only to satisfy the
Agent/ModelEntry interface.

KNOWN LIMITATION (flagged, not silently glossed over): Annex B.2 predicts
TWO barycentre signatures, "post-round loss spikes AND landings in
unoccupied basins." Empirically (smoke-tested at m=9, 3 occupied wells + 1
empty), only the first is reliably observed with this landscape's basis-
vector placement (theta_i = radius * e_i, pairwise-equidistant): the
barycentre of a symmetric occupied subset stays nearest-classified to the
occupied cluster it came from (2/3 vs 4/3 squared-distance at the balanced
3-of-4 example) rather than drifting toward an unrelated empty well
elsewhere in R^n_wells -- correct on reflection, since equidistant
placement gives the empty well no privileged position "between" the
occupied ones for the barycentre to fall into. `landed_in_unoccupied_basin`
is still computed and logged (harmless, and may fire under asymmetric
occupancy or with enough basins that ties become likely), but the
GATE SHOULD SCORE ON mean_loss (which robustly and clearly separates the
two update rules: ~0.66 for barycentre vs exactly 0.0 for plurality in the
smoke test), not on this flag. A landscape where empty wells sit
geometrically between occupied ones (e.g. minima along edges of a
simplex's occupied face) would likely reproduce the unoccupied-landing
signature too, but that is future work, not implemented here.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import GossipAveraging, LabelPluralityGossip
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology

from .active_escape import BistableParameterModule, _dummy_loader_bistable


# ---------------------------------------------------------------------------
# Landscape: M isotropic quadratic wells at simplex-like basis-vector centres
# ---------------------------------------------------------------------------


def basin_centers(n_basins: int, radius: float = 1.0) -> List[torch.Tensor]:
    """theta_i = radius * e_i in R^n_basins -- pairwise distance
    radius*sqrt(2) between every pair, by construction (a regular simplex
    up to scale), so no basin is privileged by geometry."""
    return [radius * torch.eye(n_basins)[i] for i in range(n_basins)]


def classify_nearest(vec: torch.Tensor, centers: List[torch.Tensor]) -> int:
    dists = [float(((vec - c) ** 2).sum()) for c in centers]
    return min(range(len(dists)), key=lambda i: dists[i])


def multi_well_loss(vec: torch.Tensor, centers: List[torch.Tensor], a: float) -> float:
    return a * min(float(((vec - c) ** 2).sum()) for c in centers)


# ---------------------------------------------------------------------------
# Config / Run
# ---------------------------------------------------------------------------


@dataclass
class MultiBasinConfig:
    name: str
    m: int = 8                 # clique size
    n_wells: int = 4           # total basins in the LANDSCAPE (occupied + empty traps)
    n_occupied: int = 3        # how many of those n_wells basins agents start in, M >= 2
    a: float = 1.0             # well curvature
    radius: float = 1.0        # basin separation scale
    seed: int = 0
    n_rounds: int = 10
    update_rule: str = "barycentre"   # "barycentre" (2.2b) | "plurality" (6.2)
    init_noise: float = 0.02


@dataclass
class MultiBasinRun:
    name: str
    m: int
    n_wells: int
    n_occupied: int
    a: float
    seed: int
    update_rule: str
    initial_occupancy: List[int]         # count of agents starting in each basin (len n_wells)
    # Per round (index 0 = after round 0): (consensus_reached, consensus_basin
    # or None, mean_loss, landed_in_unoccupied_basin)
    round_log: List[Tuple[bool, Optional[int], float, bool]] = field(default_factory=list)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultiBasinRun":
        with open(path, "rb") as f:
            return pickle.load(f)


def run_multi_basin_trial(config: MultiBasinConfig) -> MultiBasinRun:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.n_occupied < 2:
        raise ValueError(f"n_occupied must be >= 2, got {config.n_occupied}")
    if config.n_wells < config.n_occupied:
        raise ValueError(
            f"n_wells ({config.n_wells}) must be >= n_occupied ({config.n_occupied}) "
            "-- there must be at least as many basins in the landscape as are "
            "actually populated, or there is no unoccupied trap for the "
            "barycentre to land in and the destruction test is vacuous."
        )
    if config.update_rule not in ("barycentre", "plurality"):
        raise ValueError(f"Unknown update_rule {config.update_rule!r}")

    centers = basin_centers(config.n_wells, config.radius)

    # Balance m agents roughly evenly across the FIRST n_occupied basins,
    # leaving basins [n_occupied, n_wells) genuinely empty -- these are the
    # "unoccupied basin" traps Lemma 6.1's barycentre can land in.
    base_labels = [i % config.n_occupied for i in range(config.m)]
    initial_occupancy = [base_labels.count(i) for i in range(config.n_wells)]
    occupied_set = set(range(config.n_occupied))

    topo = NestedModularTopology(
        branching=1, depth=0, leaf_size=config.m, p=2.0, seed=config.seed,
    )
    loaders = [_dummy_loader_bistable() for _ in range(config.m)]
    dummy_loss_fn = lambda model, batch: torch.zeros(1, requires_grad=True)  # noqa: E731

    def _make_agent(i: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + i)
        init = centers[base_labels[i]].clone() + torch.randn(config.n_wells) * config.init_noise
        model = BistableParameterModule(d_param=config.n_wells, init_value=init)
        entry = ModelEntry(
            model=model, optimizer=torch.optim.SGD(model.parameters(), lr=1e-8),
            loss_fn=dummy_loss_fn, layers=frozenset({"social"}), train_locally=True,
            local_steps=1,
        )
        return Agent(i, ModelRegistry({"model": entry}), loaders[i])

    agents = [_make_agent(i) for i in range(config.m)]
    ml_topo = MultiLayerTopology({"social": topo})
    compositor = CoupledCompositor()

    protocol = (
        GossipAveraging() if config.update_rule == "barycentre"
        else LabelPluralityGossip(basin_centers=centers)
    )

    round_log: List[Tuple[bool, Optional[int], float, bool]] = []
    for round_idx in range(config.n_rounds):
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for comm_round in plan.rounds:
            protocol.execute(comm_round, agents)

        vecs = [list(a.registry["model"].model.parameters())[0].detach() for a in agents]
        labels = [classify_nearest(v, centers) for v in vecs]
        losses = [multi_well_loss(v, centers, config.a) for v in vecs]
        mean_loss = float(np.mean(losses))
        consensus_reached = len(set(labels)) == 1
        consensus_basin = labels[0] if consensus_reached else None
        landed_in_unoccupied = consensus_reached and consensus_basin not in occupied_set
        round_log.append((consensus_reached, consensus_basin, mean_loss, landed_in_unoccupied))

    return MultiBasinRun(
        name=config.name, m=config.m, n_wells=config.n_wells, n_occupied=config.n_occupied,
        a=config.a, seed=config.seed, update_rule=config.update_rule,
        initial_occupancy=initial_occupancy, round_log=round_log,
    )


# ---------------------------------------------------------------------------
# Experiment factory: E15MB
# ---------------------------------------------------------------------------


def experiment_E15MB(
    m_list: List[int] = (4, 8, 16),
    n_occupied_list: List[int] = (2, 3, 5),
    n_extra_wells: int = 1,
    update_rules: List[str] = ("barycentre", "plurality"),
    a: float = 1.0,
    radius: float = 1.0,
    seeds: List[int] = tuple(range(20)),
    n_rounds: int = 10,
) -> List[MultiBasinConfig]:
    """Annex B.2's E15 (multi-basin destruction), see this module's
    docstring for why it is prefixed E15MB rather than bare E15.
    n_occupied=2 is Remark 6.4's own coincidence check ("no diversity is
    destroyed that the chord geometry did not already order" -- barycentre
    and plurality should agree there); n_occupied in {3,5} is where Remark
    6.4's actual claim (barycentre becomes a LOCAL instance of basin
    destruction) is under test. n_extra_wells adds unoccupied basins beyond
    n_occupied (n_wells = n_occupied + n_extra_wells) so there is a genuine
    empty trap for the barycentre to land in -- without this the
    landed_in_unoccupied_basin observable is vacuously always False (every
    basin in the landscape is occupied, so nearest-centre classification
    can only ever pick an occupied one).
    """
    configs = []
    for update_rule in update_rules:
        for n_occupied in n_occupied_list:
            n_wells = n_occupied + n_extra_wells
            for m in m_list:
                if m < n_occupied:
                    continue  # can't occupy n_occupied distinct basins with fewer agents
                for seed in seeds:
                    configs.append(MultiBasinConfig(
                        name=f"E15MB/rule={update_rule}/M={n_occupied}/m={m}/seed={seed}",
                        m=m, n_wells=n_wells, n_occupied=n_occupied, a=a, radius=radius,
                        seed=seed, n_rounds=n_rounds, update_rule=update_rule,
                    ))
    return configs
