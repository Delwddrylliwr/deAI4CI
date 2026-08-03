"""Runner for non-modular topologies: star (E9) and dumbbell (E10).

natural_cascade.py's runner hardcodes NestedModularTopology plus leaf/module
bookkeeping (leaf_assignments, hierarchical_distance) that a star or
dumbbell graph doesn't have. This module provides the minimal machinery
those two experiments actually need:

  - run_generic_cascade_simulation: a multi-round Regime-C cascade on an
    arbitrary Topology, recording per-node (not per-leaf) flip times --
    used by E10 to compare the dumbbell against the NMH at a matched
    spectral gap under an identical protocol.
  - run_collapse_test: a direct, one-shot realisation of Lemma 6.1's basin-
    destruction computation (uniform aggregation of a non-consensus
    configuration, then relaxation) plus a simplified permutation-alignment
    ablation (Remark 6.2) -- used by E9. This does not run gossip at all;
    the "star" framing is bookkeeping (E9 is about the aggregator's
    uniform-mean-then-broadcast rule, not about star-graph gossip dynamics,
    which dilute the exact-mean property GossipAveraging would otherwise
    need many rounds to reproduce on a star).

Checkpointing is implemented inside these runner functions directly (an
optional checkpoint_path/checkpoint_every pair), rather than via a second,
hand-inlined copy in hpc/worker.py -- unlike natural_cascade.py/ncp_runner.py,
which predate this convention.
"""

import itertools
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import (
    AsynchronousGossip,
    GossipAveraging,
    SynchronousPairwiseGossip,
)
from dssgd.topology.base import Topology
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import DumbbellTopology, StarTopology

from .active_escape import (
    BistableParameterModule,
    _dummy_loader_bistable,
    basin_label,
    find_t_flip,
    force_flip_module,
    make_bistable_loss_fn,
    verify_bistable_loss,
)
from .catchup import compute_all_centroids


@dataclass
class GenericTopologyConfig:
    """Degrees of freedom for a generic (non-modular) topology simulation."""

    name: str
    mode: str = "cascade"          # "cascade" (E10) or "collapse" (E9)
    topology: str = "dumbbell"     # "star" or "dumbbell"
    n_nodes: int = 128
    block_size: Optional[int] = None  # dumbbell: defaults to n_nodes // 2
    bridge_p: float = 0.05            # dumbbell bridge density
    seed: int = 0
    graph_seed: Optional[int] = None
    n_warmup: int = 400
    n_meas_rounds: int = 200
    lr: float = 0.1
    local_steps: int = 50
    d_param: int = 1
    theta_A: Optional[np.ndarray] = None
    theta_B: Optional[np.ndarray] = None
    a: float = 0.5
    b: float = 0.042
    epsilon: float = 0.2
    persistence: int = 3
    init_basin: str = "A"
    force_flip_source: bool = False
    flip_noise_scale: float = 0.0
    # "async_poisson", "sync_pairwise" (SynchronousPairwiseGossip), or
    # "sync_neighbourhood" (GossipAveraging, simultaneous m-way mean).
    gossip_protocol: str = "async_poisson"
    gossip_rate: Optional[float] = None
    gossip_alpha: float = 0.5
    # collapse-mode only (E9):
    collapse_fraction_B: float = 0.5
    collapse_broadcast_weight: float = 1.0
    collapse_align: bool = False
    collapse_align_groups: int = 2


@dataclass
class GenericTopologyRun:
    """Outputs from one generic-topology run (cascade or collapse mode)."""

    name: str
    mode: str
    topology: str
    seed: int
    n_nodes: int
    a: float
    b: float
    d_param: int
    theta_A: Optional[np.ndarray] = None
    theta_B: Optional[np.ndarray] = None
    # cascade-mode fields
    centroid_traj: Optional[np.ndarray] = None
    warmup_ok: Optional[bool] = None
    nucleation_node: Optional[int] = None
    t_nucleation: Optional[int] = None
    flip_table: Optional[List[dict]] = None
    graph_stats: Optional[dict] = None
    # collapse-mode fields
    pre_labels: Optional[List[str]] = None
    post_label: Optional[str] = None
    barrier_loss: Optional[float] = None
    aligned: Optional[bool] = None

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GenericTopologyRun":
        with open(path, "rb") as f:
            return pickle.load(f)


def _build_topology(config: GenericTopologyConfig) -> Topology:
    seed = config.graph_seed if config.graph_seed is not None else config.seed
    if config.topology == "star":
        return StarTopology(n=config.n_nodes)
    elif config.topology == "dumbbell":
        block_size = config.block_size or config.n_nodes // 2
        return DumbbellTopology(block_size=block_size, bridge_p=config.bridge_p, seed=seed)
    raise ValueError(f"Unknown topology {config.topology!r}; expected 'star' or 'dumbbell'.")


# ---------------------------------------------------------------------------
# Cascade mode (E10)
# ---------------------------------------------------------------------------


def run_generic_cascade_simulation(
    config: GenericTopologyConfig,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> GenericTopologyRun:
    """Multi-round Regime-C cascade on an arbitrary topology (E10: dumbbell).

    Structurally identical to run_natural_cascade_simulation, but with no
    leaf/module bookkeeping: nodes are their own singleton "leaf" (assigns =
    identity), so compute_all_centroids/force_flip_module are reused
    unchanged with n_leaf_types = n_nodes, and flip_table carries no
    distance_from_source (no meaningful hierarchy on these topologies).
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.init_basin not in ("A", "B"):
        raise ValueError(f"init_basin must be 'A' or 'B', got {config.init_basin!r}")

    theta_A_np = config.theta_A if config.theta_A is not None else np.zeros(config.d_param, dtype=np.float32)
    theta_B_np = (
        config.theta_B if config.theta_B is not None
        else np.array([1.0] + [0.0] * (config.d_param - 1), dtype=np.float32)
    )
    theta_A_t = torch.from_numpy(theta_A_np)
    theta_B_t = torch.from_numpy(theta_B_np)
    if config.b != 0.0:
        verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b, require_barrier=False)

    n_nodes = config.n_nodes
    assigns = np.arange(n_nodes)  # each node is its own singleton "leaf"
    topo = _build_topology(config)
    graph_stats = {"n_nodes": n_nodes, "topology": config.topology}

    loss_fn = make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b)
    loaders = [_dummy_loader_bistable() for _ in range(n_nodes)]
    init_target_t = theta_A_t if config.init_basin == "A" else theta_B_t

    def _make_agent(node_id: int) -> Agent:
        torch.manual_seed(config.seed * 10000 + node_id)
        init_val = init_target_t + torch.randn(config.d_param) * 0.05
        model = BistableParameterModule(d_param=config.d_param, init_value=init_val)
        entry = ModelEntry(
            model=model, optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
            loss_fn=loss_fn, layers=frozenset({"social"}), train_locally=True, local_steps=1,
        )
        return Agent(node_id, ModelRegistry({"model": entry}), loaders[node_id])

    agents = [_make_agent(i) for i in range(n_nodes)]
    ml_topo = MultiLayerTopology({"social": topo})
    compositor = CoupledCompositor()

    _async = config.gossip_protocol == "async_poisson"
    if _async:
        per_step_rate = (config.gossip_rate or float(n_nodes)) / float(config.local_steps)
        protocol = AsynchronousGossip(
            rate=per_step_rate, mode="poisson", alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )
    elif config.gossip_protocol == "sync_pairwise":
        protocol = SynchronousPairwiseGossip(
            alpha=config.gossip_alpha,
            rng=np.random.default_rng(config.seed + 42),
        )
    elif config.gossip_protocol == "sync_neighbourhood":
        protocol = GossipAveraging()
    else:
        raise ValueError(
            f"Unknown gossip_protocol {config.gossip_protocol!r}; expected "
            f"'async_poisson', 'sync_pairwise', or 'sync_neighbourhood'."
        )

    def _exec_round(round_idx: int) -> None:
        layer_graphs = ml_topo.step(round_idx)
        plan = compositor.compose(layer_graphs, agents[0].registry)
        for _ in range(config.local_steps):
            for agent in agents:
                agent.local_step()
            if _async:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
        if not _async:
            for comm_round in plan.rounds:
                protocol.execute(comm_round, agents)

    for round_idx in range(config.n_warmup):
        _exec_round(round_idx)

    post_warmup_centroids = compute_all_centroids(agents, assigns, n_nodes)
    warmup_ok = all(
        basin_label(post_warmup_centroids[i], theta_A_np, theta_B_np, config.epsilon) == config.init_basin
        for i in range(n_nodes)
    )

    resume_round = 0
    centroid_traj_list: List[np.ndarray] = []
    source_node: Optional[int] = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as fh:
            ckpt = pickle.load(fh)
        for agent, params in zip(agents, ckpt["agent_params"]):
            with torch.no_grad():
                list(agent.registry["model"].model.parameters())[0].copy_(torch.from_numpy(params))
        np.random.set_state(ckpt["np_rng_state"])
        torch.set_rng_state(pickle.loads(ckpt["torch_rng_state"]))
        if ckpt.get("gossip_rng_state") is not None and hasattr(protocol, "rng"):
            protocol.rng.bit_generator.state = ckpt["gossip_rng_state"]
        centroid_traj_list = list(ckpt["centroid_traj_so_far"])
        source_node = ckpt["source_node"]
        resume_round = ckpt["round_completed"]
        warmup_ok = ckpt["warmup_ok"]
    else:
        if config.force_flip_source:
            source_node = int(np.random.randint(n_nodes))
            force_flip_module(agents, assigns, source_node, theta_B_t, noise_scale=0.0)
        baseline = compute_all_centroids(agents, assigns, n_nodes)
        centroid_traj_list = [baseline.copy()]

    for round_idx in range(resume_round, config.n_meas_rounds):
        _exec_round(config.n_warmup + round_idx)
        if config.flip_noise_scale > 0:
            for agent in agents:
                model = agent.registry["model"].model
                with torch.no_grad():
                    for param in model.parameters():
                        noise = torch.from_numpy(
                            np.random.randn(*param.shape).astype(np.float32)
                        ) * config.flip_noise_scale
                        param.add_(noise)
        centroid_traj_list.append(compute_all_centroids(agents, assigns, n_nodes))
        rounds_done = round_idx + 1

        if checkpoint_path is not None and rounds_done % checkpoint_every == 0 and rounds_done < config.n_meas_rounds:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            state = {
                "agent_params": [
                    list(a.registry["model"].model.parameters())[0].detach().numpy().copy()
                    for a in agents
                ],
                "centroid_traj_so_far": np.stack(centroid_traj_list),
                "warmup_ok": warmup_ok,
                "source_node": source_node,
                "round_completed": rounds_done,
                "np_rng_state": np.random.get_state(),
                "torch_rng_state": pickle.dumps(torch.get_rng_state()),
                "gossip_rng_state": protocol.rng.bit_generator.state if hasattr(protocol, "rng") else None,
            }
            tmp = Path(str(checkpoint_path) + ".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(str(tmp), str(checkpoint_path))

    centroid_traj = np.stack(centroid_traj_list)

    t_flip_per_node: List[Optional[int]] = [
        find_t_flip(centroid_traj[:, i, :], theta_A_np, theta_B_np, config.epsilon, config.persistence)
        if warmup_ok else None
        for i in range(n_nodes)
    ]
    nucleation_node: Optional[int] = None
    t_nucleation: Optional[int] = None
    valid = [(i, t) for i, t in enumerate(t_flip_per_node) if t is not None]
    if valid:
        nucleation_node, t_nucleation = min(valid, key=lambda x: x[1])

    flip_table = [
        {
            "node": i,
            "t_flip_absolute": t_flip_per_node[i],
            "t_flip_relative": (
                t_flip_per_node[i] - t_nucleation
                if (t_flip_per_node[i] is not None and t_nucleation is not None) else None
            ),
            "warmup_ok": warmup_ok,
        }
        for i in range(n_nodes) if i != nucleation_node
    ]

    if checkpoint_path is not None:
        try:
            Path(checkpoint_path).unlink()
        except FileNotFoundError:
            pass

    return GenericTopologyRun(
        name=config.name, mode="cascade", topology=config.topology, seed=config.seed,
        n_nodes=n_nodes, a=config.a, b=config.b, d_param=config.d_param,
        theta_A=theta_A_np, theta_B=theta_B_np,
        centroid_traj=centroid_traj, warmup_ok=warmup_ok,
        nucleation_node=nucleation_node, t_nucleation=t_nucleation,
        flip_table=flip_table, graph_stats=graph_stats,
    )


# ---------------------------------------------------------------------------
# Collapse mode (E9: Lemma 6.1 basin destruction + Remark 6.2 alignment)
# ---------------------------------------------------------------------------


def _aligned_mean(params: np.ndarray, n_groups: int) -> np.ndarray:
    """Simplified permutation-alignment proxy for Remark 6.2's obstruction.

    Partitions the d-dim parameter vector into n_groups contiguous blocks;
    for every worker after the first, brute-force searches over block
    permutations for the one minimising L2 distance to a running reference
    (the first worker's vector), then averages the ALIGNED vectors. This is
    a toy stand-in for weight-space permutation symmetry in a real net's
    hidden units (Git Re-Basin-style alignment) -- exact alignment search is
    standard for small widths, brute force here since n_groups is small.
    n_groups<=1 (or d < n_groups) makes every permutation trivial, i.e. raw
    averaging: the ablation's baseline.
    """
    n, d = params.shape
    if n_groups <= 1 or d < n_groups:
        return params.mean(axis=0)
    block = d // n_groups
    ref = params[0].copy()
    aligned = [ref]
    for i in range(1, n):
        vec = params[i]
        blocks = [vec[g * block:(g + 1) * block] for g in range(n_groups)]
        remainder = vec[n_groups * block:]
        best, best_dist = None, np.inf
        for perm in itertools.permutations(range(n_groups)):
            cand = np.concatenate([blocks[g] for g in perm] + [remainder])
            dist = float(np.sum((ref - cand) ** 2))
            if dist < best_dist:
                best_dist, best = dist, cand
        aligned.append(best)
    return np.mean(aligned, axis=0)


def run_collapse_test(config: GenericTopologyConfig) -> GenericTopologyRun:
    """One-shot realisation of Lemma 6.1 (basin destruction) and, when
    collapse_align=True, the Remark 6.2 alignment ablation.

    No gossip/topology dynamics: this directly (i) samples a non-consensus
    configuration with collapse_fraction_B of workers at B, (ii) computes
    the uniform mean (or the aligned mean, if collapse_align), (iii)
    broadcasts it to every worker at weight collapse_broadcast_weight, and
    (iv) relaxes each worker by local_steps of gradient descent under the
    shared bistable loss, reading off the surviving basin label.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    theta_A_np = config.theta_A if config.theta_A is not None else np.zeros(config.d_param, dtype=np.float32)
    theta_B_np = (
        config.theta_B if config.theta_B is not None
        else np.array([1.0] + [0.0] * (config.d_param - 1), dtype=np.float32)
    )
    theta_A_t = torch.from_numpy(theta_A_np)
    theta_B_t = torch.from_numpy(theta_B_np)
    if config.b != 0.0:
        verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b, require_barrier=False)
    loss_fn = make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b)

    n = config.n_nodes
    n_B = int(round(config.collapse_fraction_B * n))
    labels = ["B"] * n_B + ["A"] * (n - n_B)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(labels)

    params = np.stack([
        (theta_B_t if lbl == "B" else theta_A_t).numpy() + rng.standard_normal(config.d_param).astype(np.float32) * 0.02
        for lbl in labels
    ])

    params_agg = (
        _aligned_mean(params, config.collapse_align_groups) if config.collapse_align
        else params.mean(axis=0)
    )
    c = config.collapse_broadcast_weight
    post_params = params + c * (params_agg[None, :] - params)

    final_params = np.zeros_like(post_params)
    for i in range(n):
        model = BistableParameterModule(d_param=config.d_param, init_value=torch.from_numpy(post_params[i]).float())
        opt = torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8))
        for _ in range(config.local_steps):
            opt.zero_grad()
            loss = loss_fn(model, None)
            loss.backward()
            opt.step()
        final_params[i] = model.theta.detach().numpy()

    final_labels = [basin_label(final_params[i], theta_A_np, theta_B_np, config.epsilon) for i in range(n)]

    t_vals = np.linspace(0.0, 1.0, 50)
    chord_losses = []
    for t in t_vals:
        pt = theta_A_np + t * (theta_B_np - theta_A_np)
        model = BistableParameterModule(d_param=config.d_param, init_value=torch.from_numpy(pt).float())
        with torch.no_grad():
            chord_losses.append(float(loss_fn(model, None).item()))
    barrier_loss = max(chord_losses)

    post_label = Counter(final_labels).most_common(1)[0][0]

    return GenericTopologyRun(
        name=config.name, mode="collapse", topology=config.topology, seed=config.seed,
        n_nodes=n, a=config.a, b=config.b, d_param=config.d_param,
        theta_A=theta_A_np, theta_B=theta_B_np,
        pre_labels=labels, post_label=post_label, barrier_loss=barrier_loss,
        aligned=config.collapse_align,
    )


def run_generic_topology_simulation(
    config: GenericTopologyConfig,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
) -> GenericTopologyRun:
    """Dispatch on config.mode: 'cascade' (E10) or 'collapse' (E9)."""
    if config.mode == "cascade":
        return run_generic_cascade_simulation(config, checkpoint_path, checkpoint_every)
    elif config.mode == "collapse":
        return run_collapse_test(config)  # single-shot; no checkpointing needed
    raise ValueError(f"Unknown mode {config.mode!r}; expected 'cascade' or 'collapse'.")
