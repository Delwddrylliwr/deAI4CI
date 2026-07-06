"""Closed-form theoretical predictions for the NMH gossip-SGD framework.

Implements formulas from the McKean-Vlasov / Fokker-Planck analysis:
  - NMH module-structure enumeration
  - Edge density and expected neighbourhood sizes by level
  - Level coupling strengths and well depths
  - Effective temperature and Kramers escape times
  - Uniqueness / ergodicity condition
"""

import math
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Module-structure enumeration
# ---------------------------------------------------------------------------


def nmh_modules_by_level(
    branching: int,
    depth: int,
    leaf_size: int,
) -> Dict[int, List[List[int]]]:
    """Return {level: [sorted node lists per module]} for all hierarchy levels.

    level 0 = leaf modules (leaf_size nodes each)
    level ℓ = super-modules containing branching^ℓ leaf modules each
    """
    n_leaf_modules = branching ** depth
    result: Dict[int, List[List[int]]] = {}
    for level in range(depth + 1):
        leaves_per_module = branching ** level
        n_modules = n_leaf_modules // leaves_per_module
        result[level] = [
            list(range(
                m * leaves_per_module * leaf_size,
                (m + 1) * leaves_per_module * leaf_size,
            ))
            for m in range(n_modules)
        ]
    return result


# ---------------------------------------------------------------------------
# Edge density and expected neighbourhood sizes
# ---------------------------------------------------------------------------


def edge_density(p: float, level: int) -> float:
    """Bernoulli edge probability between node pairs across modules at level ℓ.

    The NMH construction wires each cross-module pair with probability p / 4^ℓ.
    """
    return p / (4.0 ** level)


def expected_level_neighbours(
    p: float,
    branching: int,
    leaf_size: int,
    level: int,
) -> float:
    """Expected cross-module neighbours acquired at hierarchy level ℓ.

    Each node has (branching - 1) sibling child-modules in the same super-module
    at level ℓ, each of size leaf_size * branching^(ℓ-1). Each such partner node
    is wired with probability p / 4^ℓ.
    """
    partners = (branching - 1) * leaf_size * (branching ** (level - 1))
    return partners * edge_density(p, level)


def expected_degree(
    p: float,
    branching: int,
    leaf_size: int,
    depth: int,
) -> float:
    """Expected degree = (leaf_size - 1) intra-clique + sum of cross-level contributions."""
    intra = float(leaf_size - 1)
    cross = sum(
        expected_level_neighbours(p, branching, leaf_size, lv)
        for lv in range(1, depth + 1)
    )
    return intra + cross


def degree_limit(
    p: float,
    branching: int,
    leaf_size: int,
) -> Optional[float]:
    """Limiting expected degree as depth → ∞.

    The geometric series Σ_{ℓ=1}^∞ converges iff branching < 4, giving
    (leaf_size - 1) + p * leaf_size * (branching - 1) / (4 - branching).
    Returns None when branching ≥ 4 (series diverges).
    """
    if branching >= 4:
        return None
    intra = float(leaf_size - 1)
    cross_limit = p * leaf_size * (branching - 1) / (4.0 - branching)
    return intra + cross_limit


# ---------------------------------------------------------------------------
# McKean-Vlasov / Fokker-Planck quantities
# ---------------------------------------------------------------------------


def level_coupling_strength(
    gamma: float,
    leaf_size: int,
    p: float,
    level: int,
) -> float:
    """Consensus drift strength at hierarchy level ℓ: γ m p / 2^{ℓ+1}."""
    return gamma * leaf_size * p / (2.0 ** (level + 1))


def effective_temperature(eta: float, sigma2: float) -> float:
    """T_eff = η σ² / 2  (noise strength in the Langevin / Fokker-Planck limit)."""
    return eta * sigma2 / 2.0


def well_depth(
    gamma: float,
    leaf_size: int,
    p: float,
    level: int,
) -> float:
    """Free-energy well depth separating within- from cross-module basins at level ℓ.

    ΔV^(ℓ) ∝ γ m p / 4^ℓ — decays geometrically with level.
    """
    return gamma * leaf_size * p / (4.0 ** level)


def kramers_escape_time(T_eff: float, dV: float) -> float:
    """Mean exit time over a barrier of height dV at temperature T_eff.

    τ ~ exp(ΔV / T_eff).  Returns math.inf when T_eff ≤ 0.
    """
    if T_eff <= 0.0:
        return math.inf
    return math.exp(dV / T_eff)


def uniqueness_lhs(
    p: float,
    branching: int,
    leaf_size: int,
    depth: int,
) -> float:
    """Ratio of total cross-level coupling to intra-module isolation.

    Values > 1 indicate cross-level interactions dominate, supporting a unique
    global stationary measure rather than trapped local ones.
    """
    intra = float(max(leaf_size - 1, 1))
    cross = sum(
        expected_level_neighbours(p, branching, leaf_size, lv)
        for lv in range(1, depth + 1)
    )
    return cross / intra


def total_coupling_strength(
    gamma: float,
    leaf_size: int,
    p: float,
    depth: int,
) -> float:
    """γ_total = γ(m-1) + γmp·(1 - 2^{-L}) / 2 — sum of all fluctuation spring strengths."""
    intra = gamma * (leaf_size - 1)
    cross = gamma * leaf_size * p * (1.0 - 2.0 ** (-depth)) / 2.0
    return intra + cross


def gossip_timescale(gamma: float, leaf_size: int, p: float, level: int) -> float:
    """τ_gossip(ℓ) = ln(2) / γ_ℓ — gossip half-life at hierarchy level ℓ."""
    gamma_ell = level_coupling_strength(gamma, leaf_size, p, level)
    if gamma_ell <= 0.0:
        return math.inf
    return math.log(2.0) / gamma_ell


def catch_up_time(d: int, gamma: float, leaf_size: int, p: float) -> float:
    """T_catch(d) = ln(2)·2^{d+1} / (γ·m·p) — predicted t_50 at hierarchical distance d."""
    gamma_mp = gamma * leaf_size * p
    if gamma_mp <= 0.0:
        return math.inf
    return math.log(2.0) * (2 ** (d + 1)) / gamma_mp


# ---------------------------------------------------------------------------
# Active-escape (v2) quantities
# ---------------------------------------------------------------------------


def kappa_loc(a: float, delta_norm: float = 1.0) -> float:
    """κ_loc = a·||ΔΘ*||²/2 — curvature at the saddle of the quartic double-well."""
    return a * (delta_norm ** 2) / 2.0


def classify_regime(
    a: float,
    gamma: float,
    m: int,
    p: float,
    L: int,
    delta_norm: float = 1.0,
) -> str:
    """Return 'I', 'II', or 'III' for the active-escape propagation regime.

    Regime I   — all levels flip deterministically (γ_L > κ_loc/2)
    Regime II  — partial propagation; levels 1..ell_c det., rest thermal
    Regime III — no deterministic propagation (γ_1 < κ_loc/2)
    """
    kl = kappa_loc(a, delta_norm)
    gamma_1 = level_coupling_strength(gamma, m, p, 1)
    gamma_L = level_coupling_strength(gamma, m, p, L)
    if kl / 2.0 < gamma_L:
        return 'I'
    if kl / 2.0 >= gamma_1:
        return 'III'
    return 'II'


def critical_level(
    a: float,
    gamma: float,
    m: int,
    p: float,
    L: int,
    delta_norm: float = 1.0,
) -> int:
    """Return ell_c: deepest level ℓ where γ_ℓ > κ_loc/2 (deterministic flip).

    Returns 0 if no level satisfies the condition (Regime III).
    Returns L if all levels satisfy it (Regime I).
    """
    kl = kappa_loc(a, delta_norm)
    ell_c = 0
    for ell in range(1, L + 1):
        if level_coupling_strength(gamma, m, p, ell) > kl / 2.0:
            ell_c = ell
        else:
            break
    return ell_c


def deterministic_flip_time(d: int, gamma: float, m: int, p: float) -> float:
    """T_flip(d) = 2^(d+1) / (γ·m·p) — cascade propagation time at distance d."""
    gamma_mp = gamma * m * p
    if gamma_mp <= 0.0:
        return math.inf
    return (2 ** (d + 1)) / gamma_mp


def total_flip_time(
    d: int,
    gamma: float,
    m: int,
    p: float,
    ell_c: int,
    T_eff: float,
    dV_effective: float,
) -> float:
    """Mixed deterministic + thermal flip time for Regime II.

    For d ≤ ell_c: deterministic cascade → deterministic_flip_time(d).
    For d > ell_c: adds a Kramers thermal escape term at the stalled level.
    """
    t_det = deterministic_flip_time(d, gamma, m, p)
    if d <= ell_c or T_eff <= 0.0:
        return t_det
    return t_det + kramers_escape_time(T_eff, dV_effective)
