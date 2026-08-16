"""Experiment factory functions for NCP Regime C experiments.

Each function returns a list of NCPSimConfig objects for experiments
NCP-1 through NCP-5 specified in the timesep specification.

NCP-1: Multi-layer periphery structure (graph-only; use ForestFireTopology directly)
NCP-2: Asymmetric nucleation between shells (clamped_shell configs)
NCP-3: Decoupling parameter profile χ_k
NCP-4: Multi-layer cascade suppression (propagation matrix)
NCP-5: Stationary distribution factorisation (conditional mutual information)

gossip_protocol parameter (NCP-2 through NCP-5):
  "async_poisson"        — AsynchronousGossip (default; Phase xA), no suffix
  "sync_pairwise"        — SynchronousPairwiseGossip: round-synchronous
                           maximal-matching pairwise kicks (Remark 4.3's
                           H-sched class), suffix "SP"
  "sync_neighbourhood"   — GossipAveraging: simultaneous m-way mean (Lemma
                           6.1's basin-destroying mechanism), suffix "SN"
  See dssgd.protocols.gossip.protocol_suffix / gossip_mechanisms.md: the bare
  "synchronous"/"S" label is retired (it was ambiguous between these two).
  NCP-1 is graph-only (no gossip) and has no gossip_protocol parameter.

Theory references use paper1_PDMP_wDAG_wData.md's current numbering
(Prop. 5.2/5.3/5.4 etc.), not the older timesep_NMH_NCP_experiment_spec.md
draft numbering ("Theorem 4", "§3.5/§3.9", "Eq (30)-(39)") some docstrings
below still carry over from that earlier draft.
"""

from typing import List, Optional, Tuple

from dssgd.protocols.gossip import protocol_suffix
from dssgd.topology.forest_fire import ForestFireTopology

from .ncp_runner import NCPSimConfig


# ---------------------------------------------------------------------------
# NCP-1: Multi-layer periphery structure (graph characterisation only)
# ---------------------------------------------------------------------------
# NCP-1 tests the Forest Fire graph structure itself — no simulation needed.
# Use ForestFireTopology directly and call shell_assignment().
# Config factories are provided here for consistency with the spec.


def experiment_NCP1_graph_configs(
    p_f_list: List[float] = (0.35, 0.37, 0.39),
    n_list: List[int] = (500, 1000, 2000, 4000),
    n_instances: int = 20,
) -> List[NCPSimConfig]:
    """Minimal configs for NCP-1 graph-structure validation.

    Tests the Sec. 5.1 shell/bridge structural facts (Annex A.4): k_max
    non-trivial and growing (logarithmically) with N, increasing with p_f,
    heavy-tailed shell sizes, and bridge counts B_{k,k+1} ~ |S_k|^gamma with
    gamma < 1 (the bridge exponent check_phase1.py's compute_ncp1_bridge_
    exponent fits).

    These configs are used only to instantiate ForestFireTopology and call
    shell_assignment(); the simulation loop is not run.  The n_meas_rounds=0
    flag signals that no measurement is needed.
    """
    configs = []
    seed = 0
    for p_f in p_f_list:
        for n in n_list:
            for _ in range(n_instances):
                configs.append(NCPSimConfig(
                    name=f"NCP1/p_f={p_f}/n={n}/seed={seed}",
                    n_nodes=n,
                    p_f=p_f,
                    seed=seed,
                    n_warmup=0,
                    n_meas_rounds=0,
                ))
                seed += 1
    return configs


# ---------------------------------------------------------------------------
# NCP-2: Asymmetric nucleation between shells
# ---------------------------------------------------------------------------


def experiment_NCP2(
    seeds: List[int] = tuple(range(50)),
    n_nodes: int = 1000,
    p_f: float = 0.37,
    r: float = 0.5,
    a: float = 0.5,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 800,
    clamped_shell: Optional[int] = None,  # None = auto-select innermost shell
    gossip_protocol: str = "async_poisson",
) -> List[NCPSimConfig]:
    """Asymmetric nucleation: outward (core→periphery) vs inward (periphery→core).

    Tests Proposition 5.2 (eq. 5.1): outward nucleation probability
    q^out_{k+1→k} >> q^in_{k→k+1}, with q^out/q^in ~ |S_{k+1}| at rho->1.

    Two scenario sets are generated:
    - Outward: clamp the innermost shell (highest k-core number) to B.
    - Inward: clamp the outermost shell (lowest k-core number) to B.

    The actual shell indices depend on the graph realisation; callers should
    determine shell indices from ForestFireTopology.shell_assignment() and
    pass them as clamped_shell.  This factory generates configs with a
    placeholder that callers replace after inspecting the graph.
    """
    prefix = f"NCP2{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        configs.append(NCPSimConfig(
            name=f"{prefix}/p_f={p_f}/seed={seed}/clamp={'auto' if clamped_shell is None else clamped_shell}",
            n_nodes=n_nodes,
            p_f=p_f,
            r=r,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a,
            b=b,
            clamped_shell=clamped_shell,
            gossip_protocol=gossip_protocol,
        ))
    return configs


# ---------------------------------------------------------------------------
# E8: Directional shell-crossing across EVERY adjacent boundary (Prop. 5.2)
# ---------------------------------------------------------------------------


def experiment_E8(
    seeds: List[int] = tuple(range(50)),
    n_nodes: int = 1000,
    p_f: float = 0.37,
    r: float = 0.5,
    a: float = 0.5,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 800,
    gossip_protocol: str = "async_poisson",
    hybrid_K_list: Tuple[float, ...] = (),
    epsilon_n_rounds: int = 5,
) -> List[NCPSimConfig]:
    """Directional shell-crossing measurement generalising NCP-2's clamping
    from just the innermost/outermost shells to EVERY shell in the graph.

    Clamping shell k to B lets both adjacent boundaries' entrainment rates
    be read off a single run post-hoc from flip_table (shell k-1's
    entrainment = outward from k; shell k+1's entrainment = inward from k),
    against the Lemma-3.1 prediction line computed from measured shell
    sizes (theory.fixation_bias, with the shell size as the effective
    clique size m -- q_fix(1; |S|, rho) -> 1/|S| as rho -> 1, eq. 5.1).

    paper1_computing_hybrid_gossip.md Annex B.3's E8 ("directional
    shell-crossing... at two K"), phase H4: hybrid_K_list=() (default)
    reproduces the ORIGINAL single-protocol E8 exactly, so phase 7a's
    existing call site is unaffected. Each K in hybrid_K_list adds a
    matched "E8H" arm at that round ratio over the SAME (p_f, seed, clamp)
    grid, gossip_protocol="hybrid" -- testing Remark 5.5's "protocol as
    repair lever" claim (a live Type-P channel at finite K should widen
    the polynomial inward route relative to K -> 0) against this
    K -> infinity (gossip_protocol="async_poisson") arm as the reference
    point, the same way E5H/E11H treat their non-hybrid arm.
    """
    prefix = f"E8{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        topo = ForestFireTopology(n=n_nodes, p_f=p_f, seed=seed)
        shell_ids = sorted(set(topo.shell_assignment().values()))
        for k in shell_ids:
            configs.append(NCPSimConfig(
                name=f"{prefix}/p_f={p_f}/seed={seed}/clamp={k}",
                n_nodes=n_nodes, p_f=p_f, r=r, seed=seed,
                n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                a=a, b=b, clamped_shell=k, gossip_protocol=gossip_protocol,
            ))
    for K in hybrid_K_list:
        for seed in seeds:
            topo = ForestFireTopology(n=n_nodes, p_f=p_f, seed=seed)
            shell_ids = sorted(set(topo.shell_assignment().values()))
            for k in shell_ids:
                configs.append(NCPSimConfig(
                    name=f"E8H/K={K}/p_f={p_f}/seed={seed}/clamp={k}",
                    n_nodes=n_nodes, p_f=p_f, r=r, seed=seed,
                    n_warmup=n_warmup, n_meas_rounds=n_meas, lr=lr, local_steps=local_steps,
                    a=a, b=b, clamped_shell=k, gossip_protocol="hybrid",
                    round_ratio=K, epsilon_n_rounds=epsilon_n_rounds,
                ))
    return configs


# ---------------------------------------------------------------------------
# NCP-3: Decoupling parameter profile
# ---------------------------------------------------------------------------


def experiment_NCP3(
    seeds: List[int] = tuple(range(30)),
    n_nodes: int = 1000,
    p_f: float = 0.37,
    r: float = 0.5,
    a: float = 0.5,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 800,
    gossip_protocol: str = "async_poisson",
) -> List[NCPSimConfig]:
    """Decoupling parameter χ_k profile across shells.

    Tests Proposition 5.3 (eq. 5.3): χ_k grows with shell outermost-ness.
    Free run (no clamped_shell); stationarity is reached by long n_meas.
    """
    prefix = f"NCP3{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        configs.append(NCPSimConfig(
            name=f"{prefix}/p_f={p_f}/seed={seed}",
            n_nodes=n_nodes,
            p_f=p_f,
            r=r,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a,
            b=b,
            gossip_protocol=gossip_protocol,
        ))
    return configs


# ---------------------------------------------------------------------------
# NCP-4: Multi-layer cascade suppression
# ---------------------------------------------------------------------------


def experiment_NCP4(
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
    gossip_protocol: str = "async_poisson",
) -> List[NCPSimConfig]:
    """Multi-layer cascade suppression: propagation matrix.

    Tests Proposition 5.4 (eq. 5.4): outermost-shell innovations reach
    innermost core with probability bounded by the inward product
    ∏_k q_fix(1; |S_{k+1}|, rho), super-exponentially small for heavy-tailed
    shell sizes. 200 seeds per source-shell (callers set clamped_shell after
    graph inspection).
    """
    prefix = f"NCP4{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        configs.append(NCPSimConfig(
            name=f"{prefix}/p_f={p_f}/seed={seed}",
            n_nodes=n_nodes,
            p_f=p_f,
            r=r,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a,
            b=b,
            gossip_protocol=gossip_protocol,
        ))
    return configs


# ---------------------------------------------------------------------------
# NCP-5: Stationary distribution factorisation
# ---------------------------------------------------------------------------


def experiment_NCP5(
    seeds: List[int] = tuple(range(10)),
    n_nodes: int = 1000,
    p_f: float = 0.37,
    r: float = 0.5,
    a: float = 0.5,
    b: float = 0.042,
    lr: float = 0.1,
    local_steps: int = 50,
    n_warmup: int = 400,
    n_meas: int = 3000,
    gossip_protocol: str = "async_poisson",
) -> List[NCPSimConfig]:
    """Factorisation test: I(B_k ; B_{k+2} | B_{k+1}) ≈ 0.

    Tests Proposition 5.3 (eq. 5.2): the conditional-Markov stationary
    factorisation. Long n_meas=3000 for CMI estimation accuracy. Few seeds
    (10) since each run is long.
    """
    prefix = f"NCP5{protocol_suffix(gossip_protocol)}"
    configs = []
    for seed in seeds:
        configs.append(NCPSimConfig(
            name=f"{prefix}/p_f={p_f}/seed={seed}",
            n_nodes=n_nodes,
            p_f=p_f,
            r=r,
            seed=seed,
            n_warmup=n_warmup,
            n_meas_rounds=n_meas,
            lr=lr,
            local_steps=local_steps,
            a=a,
            b=b,
            gossip_protocol=gossip_protocol,
        ))
    return configs
