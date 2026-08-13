"""Closed-form theoretical predictions for the NMH gossip-SGD framework.

Implements formulas from the McKean-Vlasov / Fokker-Planck analysis:
  - NMH module-structure enumeration
  - Edge density and expected neighbourhood sizes by level
  - Level coupling strengths and well depths
  - Effective temperature and Kramers escape times
  - Uniqueness / ergodicity condition
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


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


# ---------------------------------------------------------------------------
# Bistable chord geometry (Lemma 2.1) and fixation (Lemma 3.1)
# ---------------------------------------------------------------------------
#
# active_escape.make_bistable_loss_fn implements
#   L(theta) = (a/2)*||theta-A||^2*||theta-B||^2/||Delta||^2 - b*(theta-M).dir_hat
# Restricting to the coordinate s along the A->B axis (theta = A + s*Delta,
# ||Delta|| = delta_norm) and writing mu = b / (a * delta_norm):
#   L(s) / (a * delta_norm^2) = (1/2)*s^2*(1-s)^2 - mu*(s - 1/2)
# Critical points solve g(s) := s*(1-s)*(1-2s) = mu, a cubic with three real
# roots (bistable) iff |mu| < max_{s in (0,1/2)} g(s) = 1/(6*sqrt(3)).  This
# is the repo-specific analogue of the paper's dimensionless tilt lambda and
# saddle-node constant kappa_phi (paper Lemma 2.1); it differs from the
# paper's kappa_phi = 1/(3*sqrt(3)) by a factor of 2 because this loss
# family's well term carries a 1/2 prefactor that the paper's canonical
# family (its eq. 2.6) does not -- the qualitative structure (chord fractions
# depend on (a,b) only through mu; bistability bound; saddle merges with a
# minimum as |mu| -> the bound) is identical.

KAPPA_PHI = 1.0 / (6.0 * math.sqrt(3.0))
"""Saddle-node bound on mu = b/(a*delta_norm) for this repo's bistable well."""


def dimensionless_tilt(a: float, b: float, delta_norm: float = 1.0) -> float:
    """lambda = mu / KAPPA_PHI, this repo's analogue of the paper's Lemma-2.1
    dimensionless tilt.  lambda in [0, 1); lambda -> 1 is the saddle-node
    point at which basin A merges with the saddle and ceases to exist.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    return (b / (a * delta_norm)) / KAPPA_PHI


def _saddle_node_roots(mu: float) -> Tuple[float, float, float]:
    """Solve g(s) = s*(1-s)*(1-2s) = mu, i.e. 2*s^3 - 3*s^2 + s - mu = 0.

    Returns the three real roots sorted ascending (s_A, s_dagger, s_B).
    Raises ValueError if |mu| >= KAPPA_PHI (outside the bistable range).
    """
    if abs(mu) >= KAPPA_PHI:
        raise ValueError(
            f"mu={mu} outside the bistable range (|mu| < {KAPPA_PHI:.6f}); "
            f"one basin has merged with the saddle at this (a,b)."
        )
    roots = np.roots([2.0, -3.0, 1.0, -mu])
    real_roots = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    if len(real_roots) != 3:
        raise ValueError(f"Expected 3 real roots at mu={mu}, got {len(real_roots)}")
    return real_roots[0], real_roots[1], real_roots[2]


def curvature_epsilon(r: float) -> float:
    """Amplitude-modulation parameter eps(r) = 2(r-1)/(r+1) for the asymmetric
    well active_escape.make_asymmetric_bistable_loss_fn implements (Lemma 10.1):

        L_r(theta) = (a/2)*Q(theta)*(1 + eps*T(theta)) - b*Bias(theta)

    where Q, Bias are as in the r=1 family and T(theta) = proj(theta) - 1/2.
    At mu=0 this gives curvature(s=1)/curvature(s=0) = (1+eps/2)/(1-eps/2)
    exactly, hence r = curvature_sharp/curvature_flat via this inversion
    (r >= 1 by convention; eps > 0 makes B the sharp side).  eps=0 (r=1)
    recovers the symmetric family of Lemma 2.1 exactly.
    """
    if r <= 0:
        raise ValueError("r (curvature ratio) must be positive")
    return 2.0 * (r - 1.0) / (r + 1.0)


def _asym_crit_eq(s: float, mu: float, eps: float) -> float:
    """f_r'(s) - mu, where f_r(s) = (1/2)*h(s)*(1 + eps*(s-1/2)), h(s)=s^2(1-s)^2.

    Derivative: f_r'(s) = g(s) + (eps/2)*(2*g(s)*(s-1/2) + h(s)), g(s)=s(1-s)(1-2s).
    """
    h = s ** 2 * (1.0 - s) ** 2
    g = s * (1.0 - s) * (1.0 - 2.0 * s)
    fprime = g + (eps / 2.0) * (2.0 * g * (s - 0.5) + h)
    return fprime - mu


def _saddle_node_roots_asym(
    mu: float, eps: float, s_lo: float = -0.5, s_hi: float = 1.5, n_scan: int = 4000,
) -> Tuple[float, float, float]:
    """Numeric root-finding for f_r'(s) = mu when eps != 0 (general r).

    f_r'(s) - mu is a quartic in s when eps != 0 (up to 4 real roots), so this
    scans for sign changes on a dense grid and refines each with bisection,
    rather than assuming a fixed polynomial degree. Exactly 3 roots are
    expected in the physical window for moderate |eps| and |mu| (the paper's
    own scope: Lemma 10.1's factorisation is stated as exact only in the
    "small-tilt, moderate-r regime") -- a 4th root, if present, indicates eps
    or mu has left that regime, and is treated as an error rather than
    silently disambiguated.
    """
    from scipy.optimize import brentq

    s_grid = np.linspace(s_lo, s_hi, n_scan)
    vals = np.array([_asym_crit_eq(s, mu, eps) for s in s_grid])
    roots: List[float] = []
    for i in range(len(s_grid) - 1):
        if vals[i] == 0.0:
            roots.append(float(s_grid[i]))
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(_asym_crit_eq, s_grid[i], s_grid[i + 1], args=(mu, eps)))
    if len(roots) != 3:
        raise ValueError(
            f"mu={mu}, eps={eps} (r): expected 3 critical points in the "
            f"small-tilt/moderate-r regime, found {len(roots)}. Reduce |b/a| "
            f"or bring r closer to 1."
        )
    roots.sort()
    return roots[0], roots[1], roots[2]


def chord_geometry(a: float, b: float, delta_norm: float = 1.0, r: float = 1.0) -> Dict[str, float]:
    """Chord-fraction single-kick success thresholds (eq. 2.5) computed from
    the loss geometry alone -- lambda, the three critical points s_A <
    s_dagger < s_B, and the normalised thresholds vartheta_{A->B},
    vartheta_{B->A} for crossing from one basin to the other.

    r is the curvature ratio of Lemma 10.1 (r=1: the equal-curvature family
    of Lemma 2.1, using the fast exact cubic solver; r!=1: the asymmetric
    well of curvature_epsilon, using numeric root-finding).
    """
    mu = b / (a * delta_norm)
    if r == 1.0:
        s_A, s_dagger, s_B = _saddle_node_roots(mu)
    else:
        eps = curvature_epsilon(r)
        s_A, s_dagger, s_B = _saddle_node_roots_asym(mu, eps)
    chord = s_B - s_A
    return {
        "lambda": mu / KAPPA_PHI,
        "r": r,
        "s_A": s_A,
        "s_dagger": s_dagger,
        "s_B": s_B,
        "vartheta_A_to_B": (s_dagger - s_A) / chord,
        "vartheta_B_to_A": (s_B - s_dagger) / chord,
    }


def fixation_bias(
    a: float, b: float, kick_weight: float, delta_norm: float = 1.0, r: float = 1.0,
) -> float:
    """Birth-death bias rho = p_-/p_+ of Lemma 3.1 (r=1.0, the default) or
    of Lemma 10.1's curvature-ratio generalisation (r!=1, as E12b sweeps),
    computed (not fitted) from the loss geometry and a single fixed kick
    weight. r=1 is the equal-curvature family of Lemma 2.1 that Lemma 3.1
    is stated for; r!=1 routes through chord_geometry's curvature_epsilon(r)
    (Lemma 10.1 Sec. 10), so the returned rho is rho_curv(r)-flavoured, not
    literally Lemma 3.1's rho_depth(lambda), whenever r!=1.

    This repo's kick-weight law mu_C is degenerate: AsynchronousGossip pulls
    with a single fixed mixing weight (`alpha`/`gossip_alpha`), not a
    continuous law, so p_+ = 1[kick_weight >= vartheta_{A->B}] and
    p_- = 1[kick_weight >= vartheta_{B->A}] are indicators rather than the
    genuinely continuous probabilities Lemma 3.1 anticipates from a richer
    kick-weight law. rho therefore reduces to a step function of lambda at
    fixed kick_weight (e.g. rho=0 for every lambda>0 at the default
    alpha=0.5, since vartheta_{A->B} < 0.5 < vartheta_{B->A} for any
    B-favouring tilt).

    Caveat (Experiment E7): whether this single-kick-then-fully-relax
    idealisation matches the fixation frequency actually observed under the
    repo's dynamics -- where local gradient steps and gossip kicks are
    interleaved `local_steps` times per measurement round rather than one
    kick fully relaxing before the next -- is exactly what E7 tests. This
    function is deliberately kept literal to Lemma 3.1 rather than curve-fit
    to simulation; disagreement with measured fixation frequency is
    informative, not a bug to silently patch over.

    See fixation_bias_distributed for the general (non-degenerate) reduction
    under a genuinely continuous kick-weight law -- this function stays as
    the literal degenerate-law prediction, kept deliberately unchanged so it
    remains available as the honest "what the fixed-alpha simulation
    actually implies" baseline the distributed version is compared against.
    """
    geom = chord_geometry(a, b, delta_norm, r=r)
    p_plus = 1.0 if kick_weight >= geom["vartheta_A_to_B"] else 0.0
    p_minus = 1.0 if kick_weight >= geom["vartheta_B_to_A"] else 0.0
    if p_plus == 0.0:
        return math.inf
    return p_minus / p_plus


def uniform_kick_cdf(x: float) -> float:
    """CDF of Uniform(0,1) kick weight: F(x) = x for x in [0,1], clipped
    outside. The natural, parameter-free, maximally-non-informative choice
    of a genuinely continuous kick-weight law mu_C -- the default for
    fixation_bias_distributed, and the law gossip.make_uniform_alpha_sampler
    draws from so the simulation and this prediction stay coupled to the
    same mu_C (see that function's docstring for why this matters)."""
    return float(np.clip(x, 0.0, 1.0))


def fixation_bias_distributed(
    a: float, b: float, delta_norm: float = 1.0, r: float = 1.0,
    kick_weight_cdf: Callable[[float], float] = uniform_kick_cdf,
) -> float:
    """Birth-death bias rho = p_-/p_+ (Lemma 3.1's general reduction, Sec.
    3.1: p_+ := mu_C{C >= vartheta_A_to_B}-average, i.e. Pr_{C~mu_C}[C >=
    vartheta] = 1 - CDF(vartheta)) under a genuinely continuous kick-weight
    law mu_C given by its CDF, rather than fixation_bias's single-fixed-
    kick-weight degenerate special case (which collapses p_+/p_- to hard
    indicators of whether one fixed alpha clears the threshold).

    Defaults to Uniform(0,1) (uniform_kick_cdf) -- the simplest non-
    degenerate choice, with no free parameters of its own. Pass a different
    kick_weight_cdf to test other kick-weight laws without touching this
    function's structure.

    For this to be a fair test against simulation (not a theory change
    tested against a simulation that still uses the OLD degenerate law),
    pair with a protocol whose kick weight is actually drawn from the SAME
    law each event -- see gossip.AsynchronousGossip/SynchronousPairwiseGossip's
    `alpha_sampler` parameter and gossip.make_uniform_alpha_sampler. Passing
    a mismatched cdf/sampler pair silently reintroduces the same kind of
    theory/simulation mismatch fixation_bias's docstring warns about for the
    degenerate case, just with a different, harder-to-notice cause.

    r=1.0 (default): Lemma 3.1's depth-only reduction. r!=1: Lemma 10.1's
    curvature-ratio generalisation (Sec. 10), matching fixation_bias's r
    handling exactly (same chord_geometry/curvature_epsilon(r) call).
    """
    geom = chord_geometry(a, b, delta_norm, r=r)
    p_plus = 1.0 - kick_weight_cdf(geom["vartheta_A_to_B"])
    p_minus = 1.0 - kick_weight_cdf(geom["vartheta_B_to_A"])
    if p_plus <= 0.0:
        return math.inf
    return p_minus / p_plus


def fixation_probability(j: int, m: int, rho: float) -> float:
    """q_fix(j; m, rho) of Lemma 3.1: probability that a module of size m
    started from j seeds fixes at all-B before reverting to all-A.

        q_fix = (1 - rho^j) / (1 - rho^m),   rho not in {0, 1, inf}
        q_fix = j / m,                       rho == 1 (symmetric random walk)
        q_fix = 1[j == m],                   rho == inf (up-moves impossible)
    """
    if m <= 0 or not (0 <= j <= m):
        raise ValueError(f"require 0 <= j <= m and m > 0, got j={j}, m={m}")
    if math.isinf(rho):
        return 1.0 if j == m else 0.0
    if math.isclose(rho, 1.0, abs_tol=1e-9):
        return j / m
    return (1.0 - rho ** j) / (1.0 - rho ** m)


# ---------------------------------------------------------------------------
# Hybrid (two-jump) protocol: Lemma 3.1' / Corollary 3.1'' (eq. 3.2a-3.2d)
# ---------------------------------------------------------------------------


def vartheta_dagger(a: float, b: float, delta_norm: float = 1.0, r: float = 1.0) -> float:
    """The favoured-direction threshold vartheta_dagger(lambda) :=
    vartheta_{A->B}(lambda) (Lemma 2.1'): the chord fraction an A-worker
    must receive to cross into B. Complementarity (2.7a) makes the reverse
    threshold 1 - vartheta_dagger, so this single value is all Lemma 2.4's
    clique-round threshold n_dagger needs.
    """
    geom = chord_geometry(a, b, delta_norm, r=r)
    return geom["vartheta_A_to_B"]


def n_dagger(m: int, vartheta_dagger_val: float) -> int:
    """Lemma 3.1' notation n^dagger := ceil(vartheta_dagger(lambda) * m):
    the round-threshold a clique's B-count must clear for Lemma 2.4's
    Type-N round map (eq. 2.9) to absorb it into all-B rather than all-A."""
    if not (0.0 <= vartheta_dagger_val <= 1.0):
        raise ValueError(f"vartheta_dagger_val must be in [0,1], got {vartheta_dagger_val}")
    return math.ceil(vartheta_dagger_val * m)


def fixation_probability_hybrid(j: int, m: int, rho: float, a: float, b: float,
                                 delta_norm: float = 1.0, r: float = 1.0,
                                 vartheta_dagger_val: Optional[float] = None) -> float:
    """q_hyb(j; m, rho, K)'s ruin factor q_ruin(j) (Lemma 3.1', eq. 3.2a-3.2b):
    ordinary gambler's-ruin fixation against the NEARER absorbing barrier
    n_dagger = ceil(vartheta_dagger(lambda) * m) rather than the full clique
    m -- i.e. fixation_probability(j, n_dagger, rho), reusing Lemma 3.1's
    formula with the round-threshold target Lemma 2.4 supplies.

    This is q_ruin(j) alone, NOT the full q_hyb = q_ruin(j) * Pr(T_hit <= K
    | hit): the conditional hit-time law is left open by the paper itself
    (Sec. 9 item 8, "the deadline factor... only to leading order"), so
    there is no closed form for the K-dependent factor here. q_hyb ~=
    q_ruin(j) exactly when K clears detection_floor's K_min (below); well
    below it, q_hyb is deadline-suppressed toward 0 regardless of q_ruin.
    j is clamped to n_dagger (j seeds beyond the round threshold already
    guarantee absorption at B, matching Lemma 2.4's projection).

    vartheta_dagger_val: if given, used directly instead of recomputing
    vartheta_dagger(a, b, ...) from chord_geometry -- for callers scoring
    against a MEASURED table (e.g. the H1 calibration gate's
    gateh1_vartheta_dagger_table) who want the comparison to depend only on
    that measured value, not on a second, independent chord_geometry call
    that happens to agree with it. a/b are still required (rho's own
    provenance may depend on them) even when this is set.
    """
    vd = vartheta_dagger_val if vartheta_dagger_val is not None else vartheta_dagger(a, b, delta_norm, r=r)
    nd = n_dagger(m, vd)
    return fixation_probability(min(j, nd), nd, rho)


def detection_floor(m: int, rho: float, p_plus: float, p_minus: float) -> float:
    """Corollary 3.1'' (eq. 3.2d): the round ratio K above which the hybrid
    protocol resolves an innovation of bias rho -- q_hyb(1) = Theta(q_ruin(1))
    -- rather than being deadline-suppressed by Type-N's per-round reset:

        K_min ~ m * log(m) / [(1 - rho) * (p_plus + p_minus)]

    A scaling-form bound, not an exact threshold -- matching the paper's own
    "~" status for (3.2d) (see fixation_probability_hybrid's docstring on
    the open exact-deadline-factor problem, Sec. 9 item 8). Callers compare
    their swept K against this: K >> K_min should sit near the q_ruin(1)
    plateau; K << K_min should be deadline-suppressed toward 0 (this is the
    E14/H2 calibration-ladder's two-sided test, per Annex B.1's "either
    bound alone is consistent with a monotone trend").

    rho=1 (unbiased, lambda=0) returns inf, matching the paper's statement
    that an unbiased basin only crosses when K = Theta~(m^2), not at any
    finite floor derived from (1 - rho).
    """
    if m <= 1:
        raise ValueError(f"m must be > 1, got {m}")
    if math.isclose(rho, 1.0, abs_tol=1e-12) or (p_plus + p_minus) <= 0.0:
        return math.inf
    return (m * math.log(m)) / ((1.0 - rho) * (p_plus + p_minus))


def cross_module_ceiling(p: float, vartheta_dagger_val: float) -> float:
    """Proposition 4.10 (eq. 4.9): l_theta, the highest hierarchical level
    the Type-N channel alone can ever cross (uniform neighbourhood weights,
    NMH wiring). Above l_theta no configuration of the sibling supermodule
    -- including full commitment to the favoured basin -- can carry a
    boundary worker across its threshold via Type-N alone, at any horizon:

        l_theta = log2( p / (vartheta_dagger * (1 + p/2)) ) - 1

    Not floored/ceiled here -- callers compare a measured integer "highest
    level ever crossed" (e.g. nmh_observables.cascade_depth under
    round_ratio=0.0, Experiment E16) against floor(l_theta) or just the
    sign/magnitude of this real-valued quantity, since (4.9)'s own
    derivation is a continuous inequality in l before the "no cross-module
    level transmits" conclusion is read off it.

    At p=2 (this repo's Safari default) with symmetric basins
    (vartheta_dagger=0.5), this evaluates to l_theta < 1 (in fact exactly
    log2(2) - 1 = 0): "no cross-module level transmits at all" -- the
    paper's own flagged degenerate case (Annex B.2's E16 design note).
    """
    if not (0.0 < vartheta_dagger_val <= 1.0):
        raise ValueError(f"vartheta_dagger_val must be in (0,1], got {vartheta_dagger_val}")
    if p <= 0.0:
        raise ValueError(f"p must be > 0, got {p}")
    return math.log2(p / (vartheta_dagger_val * (1.0 + p / 2.0))) - 1.0


def kick_success_probabilities_distributed(
    a: float, b: float, delta_norm: float = 1.0, r: float = 1.0,
    kick_weight_cdf: Callable[[float], float] = uniform_kick_cdf,
) -> Tuple[float, float]:
    """(p_plus, p_minus) under a continuous kick-weight law mu_C (eq. 3.1) --
    the same quantities fixation_bias_distributed collapses into their ratio
    rho, exposed separately here because detection_floor (3.2d) needs the
    SUM p_plus + p_minus, not just the ratio. Mirrors
    fixation_bias_distributed's internals exactly; kept as a separate
    function rather than changing that one's return type, since
    fixation_bias_distributed's callers already depend on it returning a
    bare rho.
    """
    geom = chord_geometry(a, b, delta_norm, r=r)
    p_plus = 1.0 - kick_weight_cdf(geom["vartheta_A_to_B"])
    p_minus = 1.0 - kick_weight_cdf(geom["vartheta_B_to_A"])
    return p_plus, p_minus
