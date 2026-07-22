"""Generality-level scoping for tree and DAG-nested module hierarchies.

Implements the "in scope" test underlying Definition 4.3 (generality level,
tree case) and Section 11's H-ideal / boundary in-degree Delta_in (DAG case).
A single-innovation experiment (E6, E12(a), E13) needs only ONE bit per leaf
module: is it within the favourable ideal I(b) of an innovation originating
at `source_leaf` at hierarchy level `level`? Given that bit, the existing
per_leaf_loss_params mechanism (NaturalCascadeConfig, built for NMH-6) is
sufficient to realise the whole heterogeneity model -- no separate
tree-structured-loss subsystem is needed: in-scope leaves get a favourable
per-leaf bias, out-of-scope leaves get an unfavourable one, and containment/
attainment is read off the existing cascade_depth/fraction_reaching_level
observables exactly as in NMH-1..5.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


def module_at_level(leaf_idx: int, level: int, branching: int) -> int:
    """Index of the level-`level` module containing leaf `leaf_idx`."""
    return leaf_idx // (branching ** level)


@dataclass
class OverlapInfo:
    """Overlap metadata from topology.static.OverlappingModularTopology,
    needed to evaluate in_scope in the DAG case."""

    overlap_level: int
    extra_parents: Dict[int, List[int]]


def in_scope(
    source_leaf: int,
    target_leaf: int,
    level: int,
    branching: int,
    overlap: Optional[OverlapInfo] = None,
) -> bool:
    """Is target_leaf within the favourable ideal I(b) at hierarchy level
    `level`, for an innovation originating at source_leaf?

    Tree case (overlap=None, Def. 4.3): True iff source and target share the
    same level-`level` ancestor module.

    DAG case (Section 11, Remark 11.4 "multi-group pressure"): additionally
    True if target's level-overlap_level module was granted an extra parent
    equal to source's level-(overlap_level+1) ancestor -- target inherits
    scope through ANY parent, not just its natural one, at every level from
    overlap_level+1 upward (both ancestries merge into the same single root
    above the overlap point, so a match at overlap_level+1 holds at every
    level above it too).
    """
    if module_at_level(source_leaf, level, branching) == module_at_level(target_leaf, level, branching):
        return True
    if overlap is None or level < overlap.overlap_level + 1:
        return False
    tgt_ov_module = module_at_level(target_leaf, overlap.overlap_level, branching)
    src_anchor = module_at_level(source_leaf, overlap.overlap_level + 1, branching)
    return src_anchor in overlap.extra_parents.get(tgt_ov_module, [])


def per_leaf_loss_params_for_generality(
    n_leaf_types: int,
    source_leaf: int,
    generality_level: int,
    branching: int,
    a: float,
    b_in: float,
    b_out: float,
    overlap: Optional[OverlapInfo] = None,
) -> List[tuple]:
    """Build a per_leaf_loss_params list (NaturalCascadeConfig's existing
    heterogeneity mechanism) realising a single innovation of the given
    generality_level = G(b) (Def. 4.3) originating at source_leaf.

    In-scope leaves (in_scope(...)=True at level=generality_level) get bias
    +b_in (favours B); out-of-scope leaves get -b_out (favours A). This is
    the complete realisation of the tree-structured heterogeneity model
    (Def. 4.2) needed to test Theorem 4.5's containment/attainment for one
    controlled G(b) -- and, by the paper's own reduction (Prop. 4.9), of the
    overfitting-as-low-generality test (E12(a)) with generality_level = the
    injected overfitting scale l.
    """
    return [
        (a, b_in) if in_scope(source_leaf, tau, generality_level, branching, overlap) else (a, -b_out)
        for tau in range(n_leaf_types)
    ]
