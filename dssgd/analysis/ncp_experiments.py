"""Experiment factory functions for NCP Regime C experiments.

Each function returns a list of NCPSimConfig objects for experiments
NCP-1 through NCP-5 specified in the timesep specification.

NCP-1: Multi-layer periphery structure (graph-only; use ForestFireTopology directly)
NCP-2: Asymmetric nucleation between shells (clamped_shell configs)
NCP-3: Decoupling parameter profile χ_k
NCP-4: Multi-layer cascade suppression (propagation matrix)
NCP-5: Stationary distribution factorisation (conditional mutual information)

gossip_protocol parameter (NCP-2 through NCP-5):
  "async_poisson"  — AsynchronousGossip (default; Phase xA)
  "synchronous"    — GossipAveraging (Phase xS)
  When "synchronous", the experiment name prefix gains an "S" suffix.
  NCP-1 is graph-only (no gossip) and has no gossip_protocol parameter.
"""

from typing import List, Optional

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

    Tests §3.5, Equations (30)-(32): outward nucleation probability
    q^out_{k+1→k} >> q^in_{k→k+1}.

    Two scenario sets are generated:
    - Outward: clamp the innermost shell (highest k-core number) to B.
    - Inward: clamp the outermost shell (lowest k-core number) to B.

    The actual shell indices depend on the graph realisation; callers should
    determine shell indices from ForestFireTopology.shell_assignment() and
    pass them as clamped_shell.  This factory generates configs with a
    placeholder that callers replace after inspecting the graph.
    """
    prefix = "NCP2S" if gossip_protocol != "async_poisson" else "NCP2"
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

    Tests Theorem 4, Equation (35): χ_k grows with shell outermost-ness.
    Free run (no clamped_shell); stationarity is reached by long n_meas.
    """
    prefix = "NCP3S" if gossip_protocol != "async_poisson" else "NCP3"
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

    Tests §3.9, Equation (39): outermost-shell innovations reach innermost
    core with probability ∏_k 1/|S_k|, super-exponentially small.
    200 seeds per source-shell (callers set clamped_shell after graph inspection).
    """
    prefix = "NCP4S" if gossip_protocol != "async_poisson" else "NCP4"
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

    Tests Theorem 4, Equation (33).  Long n_meas=3000 for CMI estimation
    accuracy.  Few seeds (10) since each run is long.
    """
    prefix = "NCP5S" if gossip_protocol != "async_poisson" else "NCP5"
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
