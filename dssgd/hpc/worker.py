"""SLURM array-job worker for timesep HPC experiments.

Each worker process:
  1. Claims tasks atomically from a sharded pending queue (os.rename).
  2. Runs them via CheckpointableRunner (NMH / NCP) or a graph-only handler (NCP-1).
  3. Saves the full result pickle and appends a summary line to its JSONL file.
  4. Moves the task file to completed/ or failed/.

Checkpointing strategy
----------------------
`run_natural_cascade_simulation` and `run_ncp_simulation` contain simple
`for round_idx in range(n_meas_rounds)` measurement loops.  Rather than
modifying those runners, CheckpointableRunner inlines an equivalent loop and
saves a PartialRunState every `checkpoint_every` rounds.

On resume the warmup is re-executed from the same seed (deterministic, fast
relative to measurement) and then agent parameters and all RNG states are
overwritten from the checkpoint, giving bitwise-identical trajectories to an
uninterrupted run from the same seed.

Usage
-----
  python -m hpc.worker \\
      --queue-dir queue/phase1 \\
      --results-dir results/phase1 \\
      --checkpoint-dir results/phase1/checkpoints \\
      --worker-id 0 --shard-id 0
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.natural_cascade import NaturalCascadeRun
from analysis.ncp_runner import NCPRun, compute_shell_trajs
from dssgd.compositor.compositors import CoupledCompositor
from dssgd.nodes.agent import Agent
from dssgd.nodes.registry import ModelEntry, ModelRegistry
from dssgd.protocols.gossip import (
    AsynchronousGossip,
    BoundedStalenessGossip,
    GossipAveraging,
    SynchronousPairwiseGossip,
)
from dssgd.topology.base import Topology
from dssgd.topology.forest_fire import ForestFireTopology
from dssgd.topology.multilayer import MultiLayerTopology
from dssgd.topology.static import NestedModularTopology, OverlappingModularTopology
from hpc.serialization import dict_to_nc_config, dict_to_ncp_config

# Imported from runners for use in the inlined loops
from analysis.active_escape import (
    BistableParameterModule,
    _dummy_loader_bistable,
    basin_label,
    find_t_flip,
    force_flip_module,
    make_asymmetric_bistable_loss_fn,
    make_bistable_loss_fn,
    verify_bistable_loss,
)
from analysis.catchup import compute_all_centroids, hierarchical_distance, leaf_assignments
from analysis.provenance import GossipEvent, ProvenanceAsyncGossip, SeveredTopology
from analysis import theory


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------


@dataclass
class PartialRunState:
    """In-progress simulation state saved every checkpoint_every measurement rounds.

    centroid_traj_so_far has shape (round_completed + 1, n_entities, d_param)
    where the +1 is for the t=0 baseline snapshot.
    round_completed counts measurement rounds completed (not counting the baseline).
    """
    task_id: str
    round_completed: int
    agent_params: List[np.ndarray]          # one flat (d_param,) array per agent
    centroid_traj_so_far: np.ndarray
    warmup_ok: bool
    source_leaf: Optional[int]              # NMH force-flip source; None for NCP
    np_rng_state: Any                       # np.random.get_state() result
    torch_rng_state: bytes                  # pickle.dumps(torch.get_rng_state())
    gossip_rng_state: Optional[Any]         # protocol.rng.bit_generator.state (async only)
    clamp_rng_state: Optional[Any]          # NCP clamp rng state (async only)
    events_so_far: Optional[List[GossipEvent]] = None  # provenance log (track_provenance only)


# ---------------------------------------------------------------------------
# CheckpointableRunner
# ---------------------------------------------------------------------------


class CheckpointableRunner:
    """Run one NMH or NCP simulation task with periodic checkpointing.

    Inlines the measurement loops from run_natural_cascade_simulation and
    run_ncp_simulation to allow mid-run state saves.  On resume the warmup
    is re-executed (deterministic from seed) before restoring agent parameters
    and RNG states from the checkpoint.
    """

    def __init__(
        self,
        task: Dict[str, Any],
        checkpoint_dir: Path,
        checkpoint_every: int = 50,
    ) -> None:
        self.task = task
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = max(checkpoint_every, 1)
        self.checkpoint_path = self.checkpoint_dir / f"{task['task_id']}.ckpt.pkl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        sim_type = self.task["sim_type"]
        if sim_type == "clique_fixation":
            # No checkpointing: each trial is one small clique, seconds to run.
            from analysis.clique_fixation import CliqueFixationConfig, run_clique_fixation_trial

            config = CliqueFixationConfig(**self.task["config"])
            return run_clique_fixation_trial(config)

        if sim_type == "generic_topology":
            # Checkpointing lives inside generic_topology_runner itself (a
            # different convention from the two branches below, which
            # predate it -- see that module's docstring), so this branch
            # does not go through _load_checkpoint()/PartialRunState at all.
            from analysis.generic_topology_runner import run_generic_topology_simulation
            from hpc.serialization import dict_to_generic_config

            config = dict_to_generic_config(self.task["config"])
            return run_generic_topology_simulation(
                config, checkpoint_path=self.checkpoint_path,
                checkpoint_every=self.checkpoint_every,
            )

        ckpt = self._load_checkpoint()
        if sim_type == "natural_cascade":
            return self._run_nmh(ckpt)
        elif sim_type == "ncp_simulation":
            return self._run_ncp(ckpt)
        else:
            raise ValueError(f"Unexpected sim_type for CheckpointableRunner: {sim_type!r}")

    # ------------------------------------------------------------------
    # NMH (NaturalCascadeConfig) runner
    # ------------------------------------------------------------------

    def _run_nmh(self, ckpt: Optional[PartialRunState]) -> NaturalCascadeRun:
        config = dict_to_nc_config(self.task["config"])

        # -- Setup (mirrors run_natural_cascade_simulation lines 155-249) --
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if config.init_basin not in ("A", "B"):
            raise ValueError(f"init_basin must be 'A' or 'B', got {config.init_basin!r}")

        theta_A_np = (
            config.theta_A if config.theta_A is not None
            else np.zeros(config.d_param, dtype=np.float32)
        )
        theta_B_np = (
            config.theta_B if config.theta_B is not None
            else np.array([1.0] + [0.0] * (config.d_param - 1), dtype=np.float32)
        )
        theta_A_t = torch.from_numpy(theta_A_np)
        theta_B_t = torch.from_numpy(theta_B_np)

        if config.b != 0.0:
            verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b,
                                 require_barrier=False)

        regime = theory.classify_regime(config.a, 1.0, config.leaf_size, config.p, config.depth)
        ell_c = theory.critical_level(config.a, 1.0, config.leaf_size, config.p, config.depth)

        n_leaf_types = config.branching ** config.depth
        n_agents = n_leaf_types * config.leaf_size
        assigns = leaf_assignments(n_agents, config.leaf_size)

        graph_seed = config.graph_seed if config.graph_seed is not None else config.seed
        if config.overlap_level is not None:
            topo: Topology = OverlappingModularTopology(
                branching=config.branching,
                depth=config.depth,
                leaf_size=config.leaf_size,
                p=config.p,
                overlap_level=config.overlap_level,
                delta_in=config.delta_in,
                n_overlap=config.n_overlap,
                seed=graph_seed,
            )
        else:
            topo = NestedModularTopology(
                branching=config.branching,
                depth=config.depth,
                leaf_size=config.leaf_size,
                p=config.p,
                seed=graph_seed,
            )
        if config.sever_min_distance is not None:
            topo = SeveredTopology(topo, assigns, config.sever_min_distance)

        _make_loss = (
            (lambda a_i, b_i: make_bistable_loss_fn(theta_A_t, theta_B_t, a_i, b_i))
            if config.curvature_ratio == 1.0
            else (lambda a_i, b_i: make_asymmetric_bistable_loss_fn(
                theta_A_t, theta_B_t, a_i, b_i, r=config.curvature_ratio))
        )
        if config.per_leaf_loss_params is None:
            uniform_loss = _make_loss(config.a, config.b)
            loss_fn_for_leaf = {tau: uniform_loss for tau in range(n_leaf_types)}
        else:
            loss_fn_for_leaf = {
                tau: _make_loss(a_i, b_i)
                for tau, (a_i, b_i) in enumerate(config.per_leaf_loss_params)
            }

        loaders = [_dummy_loader_bistable() for _ in range(n_agents)]

        init_target_t = theta_A_t if config.init_basin == "A" else theta_B_t

        def _make_agent(agent_id: int) -> Agent:
            torch.manual_seed(config.seed * 10000 + agent_id)
            init_noise = torch.randn(config.d_param) * 0.05
            init_val = init_target_t + init_noise
            model = BistableParameterModule(d_param=config.d_param, init_value=init_val)
            leaf_idx = int(assigns[agent_id])
            entry = ModelEntry(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
                loss_fn=loss_fn_for_leaf[leaf_idx],
                layers=frozenset({config.layer_name}),
                train_locally=True,
                local_steps=1,
            )
            return Agent(agent_id, ModelRegistry({"model": entry}), loaders[agent_id])

        agents = [_make_agent(i) for i in range(n_agents)]
        ml_topo = MultiLayerTopology({config.layer_name: topo})
        compositor = CoupledCompositor()

        if config.track_provenance and config.gossip_protocol != "async_poisson":
            raise ValueError(
                "track_provenance requires gossip_protocol='async_poisson' "
                f"(event-level attribution is undefined for {config.gossip_protocol!r})"
            )

        _async_nmh = config.gossip_protocol in ("async_poisson", "bounded_staleness")
        if config.gossip_protocol == "async_poisson":
            per_step_rate = (
                config.gossip_rate if config.gossip_rate is not None else float(n_agents)
            ) / float(config.local_steps)
            if config.track_provenance:
                protocol = ProvenanceAsyncGossip(
                    rate=per_step_rate, mode="poisson", alpha=config.gossip_alpha,
                    rng=np.random.default_rng(config.seed + 42), leaf_assigns=assigns,
                )
            else:
                protocol = AsynchronousGossip(
                    rate=per_step_rate, mode="poisson", alpha=config.gossip_alpha,
                    rng=np.random.default_rng(config.seed + 42),
                )
        elif config.gossip_protocol == "bounded_staleness":
            per_step_rate = (
                config.gossip_rate if config.gossip_rate is not None else float(n_agents)
            ) / float(config.local_steps)
            protocol = BoundedStalenessGossip(
                rate=per_step_rate, mode="poisson", alpha=config.gossip_alpha,
                rng=np.random.default_rng(config.seed + 42), staleness_bound=config.staleness_bound,
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
                f"'async_poisson', 'sync_pairwise', 'sync_neighbourhood', or "
                f"'bounded_staleness'."
            )
        _has_round_idx = hasattr(protocol, "round_idx")

        # -- Warmup (always re-run; deterministic from seed) --
        for round_idx in range(config.n_warmup):
            if _has_round_idx:
                protocol.round_idx = round_idx
            layer_graphs = ml_topo.step(round_idx)
            plan = compositor.compose(layer_graphs, agents[0].registry)
            for _ in range(config.local_steps):
                for agent in agents:
                    agent.local_step()
                if _async_nmh:
                    for comm_round in plan.rounds:
                        protocol.execute(comm_round, agents)
            if not _async_nmh:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)

        post_warmup_centroids = compute_all_centroids(agents, assigns, n_leaf_types)
        warmup_ok = all(
            basin_label(post_warmup_centroids[tau], theta_A_np, theta_B_np, config.epsilon)
            == config.init_basin
            for tau in range(n_leaf_types)
        )

        # -- Restore from checkpoint or initialise fresh measurement state --
        if ckpt is not None:
            _restore_agent_params_nmh(agents, ckpt.agent_params, config.d_param)
            np.random.set_state(ckpt.np_rng_state)
            torch.set_rng_state(pickle.loads(ckpt.torch_rng_state))
            if ckpt.gossip_rng_state is not None and hasattr(protocol, "rng"):
                protocol.rng.bit_generator.state = ckpt.gossip_rng_state
            centroid_traj_list: List[np.ndarray] = list(ckpt.centroid_traj_so_far)
            source_leaf = ckpt.source_leaf
            warmup_ok = ckpt.warmup_ok
            resume_round = ckpt.round_completed
            if config.track_provenance and ckpt.events_so_far is not None:
                protocol.events = list(ckpt.events_so_far)
        else:
            source_leaf = None
            if config.force_flip_source:
                source_leaf = int(np.random.randint(n_leaf_types))
                flip_target_t = theta_B_t if config.init_basin == "A" else theta_A_t
                force_flip_module(agents, assigns, source_leaf, flip_target_t, noise_scale=0.0)
            baseline = compute_all_centroids(agents, assigns, n_leaf_types)
            centroid_traj_list = [baseline.copy()]
            resume_round = 0

        # -- Measurement loop --
        for round_idx in range(resume_round, config.n_meas_rounds):
            if _has_round_idx:
                protocol.round_idx = config.n_warmup + round_idx
            layer_graphs = ml_topo.step(config.n_warmup + round_idx)
            plan = compositor.compose(layer_graphs, agents[0].registry)
            for _ in range(config.local_steps):
                for agent in agents:
                    agent.local_step()
                if _async_nmh:
                    for comm_round in plan.rounds:
                        protocol.execute(comm_round, agents)
            if not _async_nmh:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
            if config.flip_noise_scale > 0:
                for agent in agents:
                    model = agent.registry["model"].model
                    with torch.no_grad():
                        for param in model.parameters():
                            noise = torch.from_numpy(
                                np.random.randn(*param.shape).astype(np.float32)
                            ) * config.flip_noise_scale
                            param.add_(noise)
            centroid_traj_list.append(compute_all_centroids(agents, assigns, n_leaf_types))
            rounds_done = round_idx + 1

            if rounds_done % self.checkpoint_every == 0 and rounds_done < config.n_meas_rounds:
                gossip_rng_state = (
                    protocol.rng.bit_generator.state
                    if hasattr(protocol, "rng") else None
                )
                self._save_checkpoint(PartialRunState(
                    task_id=self.task["task_id"],
                    round_completed=rounds_done,
                    agent_params=_collect_agent_params_nmh(agents, config.d_param),
                    centroid_traj_so_far=np.stack(centroid_traj_list),
                    warmup_ok=warmup_ok,
                    source_leaf=source_leaf,
                    np_rng_state=np.random.get_state(),
                    torch_rng_state=pickle.dumps(torch.get_rng_state()),
                    gossip_rng_state=gossip_rng_state,
                    clamp_rng_state=None,
                    events_so_far=list(protocol.events) if config.track_provenance else None,
                ))

        centroid_traj = np.stack(centroid_traj_list)

        # -- Post-hoc nucleation detection (mirrors runner lines 311-357) --
        t_flip_per_leaf: List[Optional[int]] = []
        for tau in range(n_leaf_types):
            t_flip_per_leaf.append(
                find_t_flip(
                    centroid_traj[:, tau, :], theta_A_np, theta_B_np,
                    config.epsilon, config.persistence,
                ) if warmup_ok else None
            )

        nucleation_leaf: Optional[int] = None
        t_nucleation: Optional[int] = None
        valid = [(tau, t) for tau, t in enumerate(t_flip_per_leaf) if t is not None]
        if valid:
            nucleation_leaf, t_nucleation = min(valid, key=lambda x: x[1])

        flip_table: List[dict] = []
        for tau in range(n_leaf_types):
            if tau == nucleation_leaf:
                continue
            t_abs = t_flip_per_leaf[tau]
            d = hierarchical_distance(nucleation_leaf, tau) if nucleation_leaf is not None else None
            t_rel = (
                (t_abs - t_nucleation) if (t_abs is not None and t_nucleation is not None)
                else None
            )
            flip_table.append({
                "target_leaf": tau,
                "distance_from_source": d,
                "t_flip_absolute": t_abs,
                "t_flip_relative": t_rel,
                "warmup_ok": warmup_ok,
                "a": config.a,
                "b": config.b,
                "lr": config.lr,
                "seed": config.seed,
                "local_steps": config.local_steps,
            })

        self._delete_checkpoint()
        return NaturalCascadeRun(
            name=config.name,
            lr=config.lr,
            seed=config.seed,
            branching=config.branching,
            depth=config.depth,
            leaf_size=config.leaf_size,
            p=config.p,
            a=config.a,
            b=config.b,
            d_param=config.d_param,
            local_steps=config.local_steps,
            n_warmup=config.n_warmup,
            n_meas_rounds=config.n_meas_rounds,
            theta_A=theta_A_np,
            theta_B=theta_B_np,
            centroid_traj=centroid_traj,
            warmup_ok=warmup_ok,
            nucleation_leaf=nucleation_leaf,
            t_nucleation=t_nucleation,
            flip_table=flip_table,
            regime=regime,
            ell_c=ell_c,
            events=protocol.events if config.track_provenance else None,
        )

    # ------------------------------------------------------------------
    # NCP (NCPSimConfig) runner
    # ------------------------------------------------------------------

    def _run_ncp(self, ckpt: Optional[PartialRunState]) -> NCPRun:
        from analysis.ncp_runner import _snapshot_params, _clamp_nodes

        config = dict_to_ncp_config(self.task["config"])

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        theta_A_np = (
            config.theta_A if config.theta_A is not None
            else np.zeros(config.d_param, dtype=np.float32)
        )
        theta_B_np = (
            config.theta_B if config.theta_B is not None
            else np.array([1.0] + [0.0] * (config.d_param - 1), dtype=np.float32)
        )
        theta_A_t = torch.from_numpy(theta_A_np)
        theta_B_t = torch.from_numpy(theta_B_np)

        require_barrier = (config.b == 0.0)
        verify_bistable_loss(theta_A_t, theta_B_t, config.a, config.b,
                             require_barrier=require_barrier)

        ff_topo = ForestFireTopology(
            n=config.n_nodes, p_f=config.p_f, r=config.r,
            seed=config.seed, ensure_connected=True,
        )
        G, _ = ff_topo.step(0)
        shell_assigns = ff_topo.shell_assignment()

        degrees = [d for _, d in G.degree()]
        graph_stats = {
            "n_nodes": config.n_nodes,
            "n_shells": len(set(shell_assigns.values())),
            "mean_degree": float(np.mean(degrees)),
            "max_shell": max(shell_assigns.values()),
        }

        loss_fn = make_bistable_loss_fn(theta_A_t, theta_B_t, config.a, config.b)
        loaders = [_dummy_loader_bistable() for _ in range(config.n_nodes)]

        def _make_agent(node_id: int) -> Agent:
            torch.manual_seed(config.seed * 100000 + node_id)
            init_noise = torch.randn(config.d_param) * 0.05
            init_val = theta_A_t + init_noise
            model = BistableParameterModule(d_param=config.d_param, init_value=init_val)
            entry = ModelEntry(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=max(config.lr, 1e-8)),
                loss_fn=loss_fn,
                layers=frozenset({config.layer_name}),
                train_locally=True,
                local_steps=1,
            )
            return Agent(node_id, ModelRegistry({"model": entry}), loaders[node_id])

        agents = [_make_agent(i) for i in range(config.n_nodes)]
        ml_topo = MultiLayerTopology({config.layer_name: ff_topo})
        compositor = CoupledCompositor()

        _async_ncp = config.gossip_protocol == "async_poisson"
        if _async_ncp:
            per_step_rate = (
                config.gossip_rate if config.gossip_rate is not None else float(config.n_nodes)
            ) / float(config.local_steps)
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

        # Warmup
        for round_idx in range(config.n_warmup):
            layer_graphs = ml_topo.step(round_idx)
            plan = compositor.compose(layer_graphs, agents[0].registry)
            for _ in range(config.local_steps):
                for agent in agents:
                    agent.local_step()
                if _async_ncp:
                    for comm_round in plan.rounds:
                        protocol.execute(comm_round, agents)
            if not _async_ncp:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)

        post_warmup_params = _snapshot_params(agents, config.d_param)
        warmup_ok = all(
            basin_label(post_warmup_params[i], theta_A_np, theta_B_np, config.epsilon) == 'A'
            for i in range(config.n_nodes)
        )

        clamped_nodes = (
            [node for node, sh in shell_assigns.items() if sh == config.clamped_shell]
            if config.clamped_shell is not None else []
        )

        # Restore from checkpoint or initialise fresh
        if ckpt is not None:
            _restore_agent_params_ncp(agents, ckpt.agent_params, config.d_param)
            np.random.set_state(ckpt.np_rng_state)
            torch.set_rng_state(pickle.loads(ckpt.torch_rng_state))
            if ckpt.gossip_rng_state is not None and hasattr(protocol, "rng"):
                protocol.rng.bit_generator.state = ckpt.gossip_rng_state
            clamp_rng = np.random.default_rng(config.seed + 1)
            if ckpt.clamp_rng_state is not None:
                clamp_rng.bit_generator.state = ckpt.clamp_rng_state
            traj_list: List[np.ndarray] = list(ckpt.centroid_traj_so_far)
            warmup_ok = ckpt.warmup_ok
            resume_round = ckpt.round_completed
        else:
            clamp_rng = np.random.default_rng(config.seed + 1)
            traj_list = [post_warmup_params.copy()]
            resume_round = 0

        # Measurement loop
        for round_idx in range(resume_round, config.n_meas_rounds):
            layer_graphs = ml_topo.step(config.n_warmup + round_idx)
            plan = compositor.compose(layer_graphs, agents[0].registry)
            for _ in range(config.local_steps):
                for agent in agents:
                    agent.local_step()
                if _async_ncp:
                    for comm_round in plan.rounds:
                        protocol.execute(comm_round, agents)
            if not _async_ncp:
                for comm_round in plan.rounds:
                    protocol.execute(comm_round, agents)
            if clamped_nodes:
                _clamp_nodes(agents, clamped_nodes, theta_B_t, clamp_rng)
            traj_list.append(_snapshot_params(agents, config.d_param))
            rounds_done = round_idx + 1

            if rounds_done % self.checkpoint_every == 0 and rounds_done < config.n_meas_rounds:
                gossip_rng_state = (
                    protocol.rng.bit_generator.state
                    if hasattr(protocol, "rng") else None
                )
                self._save_checkpoint(PartialRunState(
                    task_id=self.task["task_id"],
                    round_completed=rounds_done,
                    agent_params=_collect_agent_params_ncp(agents, config.d_param),
                    centroid_traj_so_far=np.stack(traj_list),
                    warmup_ok=warmup_ok,
                    source_leaf=None,
                    np_rng_state=np.random.get_state(),
                    torch_rng_state=pickle.dumps(torch.get_rng_state()),
                    gossip_rng_state=gossip_rng_state,
                    clamp_rng_state=clamp_rng.bit_generator.state if clamped_nodes else None,
                ))

        centroid_traj = np.stack(traj_list)
        shell_traj = compute_shell_trajs(centroid_traj, shell_assigns)

        t_flip_per_node: List[Optional[int]] = [
            find_t_flip(centroid_traj[:, i, :], theta_A_np, theta_B_np,
                        config.epsilon, config.persistence)
            if warmup_ok else None
            for i in range(config.n_nodes)
        ]

        nucleation_node: Optional[int] = None
        t_nucleation: Optional[int] = None
        valid = [(i, t) for i, t in enumerate(t_flip_per_node) if t is not None]
        if valid:
            nucleation_node, t_nucleation = min(valid, key=lambda x: x[1])

        flip_table: List[dict] = [
            {
                "node": i,
                "shell": shell_assigns[i],
                "t_flip_absolute": t_flip_per_node[i],
                "t_flip_relative": (
                    t_flip_per_node[i] - t_nucleation
                    if (t_flip_per_node[i] is not None and t_nucleation is not None)
                    else None
                ),
                "warmup_ok": warmup_ok,
            }
            for i in range(config.n_nodes)
            if i != nucleation_node
        ]

        self._delete_checkpoint()
        return NCPRun(
            name=config.name,
            seed=config.seed,
            n_nodes=config.n_nodes,
            p_f=config.p_f,
            r=config.r,
            a=config.a,
            b=config.b,
            d_param=config.d_param,
            local_steps=config.local_steps,
            n_warmup=config.n_warmup,
            n_meas_rounds=config.n_meas_rounds,
            theta_A=theta_A_np,
            theta_B=theta_B_np,
            centroid_traj=centroid_traj,
            shell_assigns=shell_assigns,
            shell_traj=shell_traj,
            warmup_ok=warmup_ok,
            nucleation_node=nucleation_node,
            t_nucleation=t_nucleation,
            flip_table=flip_table,
            graph_stats=graph_stats,
            clamped_shell=config.clamped_shell,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> Optional[PartialRunState]:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as fh:
                return pickle.load(fh)
        return None

    def _save_checkpoint(self, state: PartialRunState) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(tmp), str(self.checkpoint_path))

    def _delete_checkpoint(self) -> None:
        try:
            self.checkpoint_path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Agent parameter helpers
# ---------------------------------------------------------------------------


def _collect_agent_params_nmh(agents: List[Agent], d_param: int) -> List[np.ndarray]:
    result = []
    for agent in agents:
        params = list(agent.registry["model"].model.parameters())
        vec = torch.cat([p.detach().flatten() for p in params]).numpy()
        result.append(vec[:d_param].copy())
    return result


def _restore_agent_params_nmh(
    agents: List[Agent], saved: List[np.ndarray], d_param: int
) -> None:
    for agent, params in zip(agents, saved):
        t = torch.from_numpy(params)
        model_params = list(agent.registry["model"].model.parameters())
        with torch.no_grad():
            offset = 0
            for p in model_params:
                numel = p.numel()
                p.copy_(t[offset : offset + numel].view_as(p))
                offset += numel


def _collect_agent_params_ncp(agents: List[Agent], d_param: int) -> List[np.ndarray]:
    result = [None] * len(agents)
    for agent in agents:
        params = list(agent.registry["model"].model.parameters())
        vec = torch.cat([p.detach().flatten() for p in params]).numpy()
        result[agent.id] = vec[:d_param].copy()
    return result


def _restore_agent_params_ncp(
    agents: List[Agent], saved: List[np.ndarray], d_param: int
) -> None:
    for agent in agents:
        t = torch.from_numpy(saved[agent.id])
        model_params = list(agent.registry["model"].model.parameters())
        with torch.no_grad():
            offset = 0
            for p in model_params:
                numel = p.numel()
                p.copy_(t[offset : offset + numel].view_as(p))
                offset += numel


# ---------------------------------------------------------------------------
# NCP-1 graph-only handler
# ---------------------------------------------------------------------------


def run_ncp1_graph_only(task: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate ForestFireTopology and return shell structure stats.

    paper1_PDMP_wDAG_wData.md Sec. 5.1 / Annex A.4: shell decomposition and
    inter-shell bridge counts B_{k,k+1}, which check_phase1.py fits to
    B_{k,k+1} ~ |S_k|^gamma to recover the bridge exponent gamma feeding
    NCP-2's node-count choice (Sec. 5.1, Assumption P).
    """
    cfg = dict_to_ncp_config(task["config"])
    topo = ForestFireTopology(n=cfg.n_nodes, p_f=cfg.p_f, r=cfg.r, seed=cfg.seed,
                               ensure_connected=True)
    G, _ = topo.step(0)
    shells = topo.shell_assignment()
    shell_vals = list(shells.values())
    degrees = [d for _, d in G.degree()]
    shell_sizes = {}
    for sh in set(shell_vals):
        shell_sizes[str(sh)] = shell_vals.count(sh)
    # Bridge counts B_{k,k+1}: edges crossing exactly one shell boundary.
    bridge_counts: Dict[str, int] = {}
    for u, v in G.edges():
        su, sv = shells[u], shells[v]
        if abs(su - sv) == 1:
            k = min(su, sv)
            key = f"{k},{k + 1}"
            bridge_counts[key] = bridge_counts.get(key, 0) + 1
    return {
        "n_nodes": cfg.n_nodes,
        "p_f": cfg.p_f,
        "seed": cfg.seed,
        "n_shells": len(set(shell_vals)),
        "max_shell": max(shell_vals),
        "min_shell": min(shell_vals),
        "mean_degree": float(np.mean(degrees)),
        "shell_sizes": shell_sizes,
        "bridge_counts": bridge_counts,
    }


# ---------------------------------------------------------------------------
# Result summary extraction
# ---------------------------------------------------------------------------


def extract_result_summary(result, task: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key observables for the JSONL summary line (no centroid_traj)."""
    sim_type = task["sim_type"]
    if sim_type == "ncp_graph_only":
        return result  # already a plain dict

    if sim_type == "natural_cascade":
        flipped = sum(1 for row in result.flip_table if row.get("t_flip_absolute") is not None)
        return {
            "warmup_ok": result.warmup_ok,
            "nucleation_leaf": result.nucleation_leaf,
            "t_nucleation": result.t_nucleation,
            "n_flipped": flipped,
            "regime": result.regime,
            "ell_c": result.ell_c,
        }

    if sim_type == "ncp_simulation":
        flipped = sum(1 for row in result.flip_table if row.get("t_flip_absolute") is not None)
        return {
            "warmup_ok": result.warmup_ok,
            "nucleation_node": result.nucleation_node,
            "t_nucleation": result.t_nucleation,
            "n_flipped": flipped,
            "graph_stats": result.graph_stats,
        }

    if sim_type == "clique_fixation":
        return {
            "m": result.m, "j_seeds": result.j_seeds, "a": result.a, "b": result.b,
            "r": result.r, "fixed_at_B": result.fixed_at_B,
        }

    if sim_type == "generic_topology":
        if result.mode == "collapse":
            return {
                "mode": "collapse",
                "topology": result.topology,
                "post_label": result.post_label,
                "barrier_loss": result.barrier_loss,
                "aligned": result.aligned,
            }
        flipped = sum(1 for row in result.flip_table if row.get("t_flip_absolute") is not None)
        return {
            "mode": "cascade",
            "topology": result.topology,
            "warmup_ok": result.warmup_ok,
            "nucleation_node": result.nucleation_node,
            "t_nucleation": result.t_nucleation,
            "n_flipped": flipped,
            "graph_stats": result.graph_stats,
        }

    return {}


# ---------------------------------------------------------------------------
# Task claiming
# ---------------------------------------------------------------------------


def _shard_order(own_shard: int, n_shards: int = 100) -> List[int]:
    others = [s for s in range(n_shards) if s != own_shard]
    random.shuffle(others)
    return [own_shard] + others


def claim_task(
    queue_root: Path,
    own_shard: int,
    n_shards: int = 100,
    steal: bool = True,
) -> Optional[Tuple[Dict[str, Any], Path]]:
    """Atomically claim one task via os.rename from pending/ to claimed/.

    Returns (task_dict, claimed_path) or None if no tasks remain.
    """
    shards = _shard_order(own_shard, n_shards) if steal else [own_shard]
    for shard_idx in shards:
        pending_dir = queue_root / "pending" / f"shard_{shard_idx:03d}"
        claimed_dir = queue_root / "claimed" / f"shard_{shard_idx:03d}"
        claimed_dir.mkdir(parents=True, exist_ok=True)
        try:
            entries = list(pending_dir.iterdir())
        except FileNotFoundError:
            continue
        random.shuffle(entries)
        for entry in entries:
            if not entry.suffix == ".json":
                continue
            dest = claimed_dir / entry.name
            try:
                os.rename(str(entry), str(dest))
            except (FileNotFoundError, OSError):
                continue
            with open(dest) as fh:
                task = json.load(fh)
            return task, dest
    return None


def complete_task(claimed_path: Path, queue_root: Path) -> None:
    shard_name = claimed_path.parent.name
    dest_dir = queue_root / "completed" / shard_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(str(claimed_path), str(dest_dir / claimed_path.name))
    except FileNotFoundError:
        # File was moved by external recovery (e.g. manual claimed→pending sweep)
        # while this worker held it. Work is done; just log and continue.
        print(f"[warn] claimed file already moved: {claimed_path.name}", flush=True)


def fail_task(claimed_path: Path, queue_root: Path, error_msg: str) -> None:
    shard_name = claimed_path.parent.name
    dest_dir = queue_root / "failed" / shard_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / claimed_path.name
    try:
        os.rename(str(claimed_path), str(dest))
    except FileNotFoundError:
        print(f"[warn] claimed file already moved: {claimed_path.name}", flush=True)
        return
    with open(dest.with_suffix(".err"), "w") as fh:
        fh.write(error_msg)


# ---------------------------------------------------------------------------
# Per-task dispatch
# ---------------------------------------------------------------------------


def run_task(
    task: Dict[str, Any],
    results_dir: Path,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """Dispatch one task to the appropriate runner.  Returns result summary dict."""
    pkl_path = task.get("result_pkl_path")

    # Skip if already computed (supports re-submission of partially completed queues)
    if pkl_path and Path(pkl_path).exists():
        if task["sim_type"] == "natural_cascade":
            result = NaturalCascadeRun.load(pkl_path)
        elif task["sim_type"] == "generic_topology":
            from analysis.generic_topology_runner import GenericTopologyRun
            result = GenericTopologyRun.load(pkl_path)
        elif task["sim_type"] == "clique_fixation":
            from analysis.clique_fixation import CliqueFixationRun
            result = CliqueFixationRun.load(pkl_path)
        else:
            result = NCPRun.load(pkl_path)
        return extract_result_summary(result, task)

    sim_type = task["sim_type"]
    if sim_type == "ncp_graph_only":
        result = run_ncp1_graph_only(task)
    else:
        runner = CheckpointableRunner(task, checkpoint_dir,
                                      checkpoint_every=task.get("checkpoint_every", 50))
        result = runner.run()
        if pkl_path:
            result.save(pkl_path)

    return extract_result_summary(result, task)


# ---------------------------------------------------------------------------
# JSONL output helpers
# ---------------------------------------------------------------------------


def _jsonl_line(task: Dict[str, Any], summary: Dict[str, Any]) -> str:
    return json.dumps({"task": task, "result": summary}) + "\n"


def _atomic_append_lines(path: Path, lines: List[str]) -> None:
    """Append lines to a JSONL file via a tmp-and-rename pattern.

    Reads existing content, appends, writes to .tmp, then renames.
    Prevents corrupt partial lines if the worker is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(existing)
        fh.writelines(lines)
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------


def worker_main(
    queue_root: Path,
    results_dir: Path,
    checkpoint_dir: Path,
    worker_id: int,
    shard_id: int,
    n_shards: int = 100,
    checkpoint_every: int = 50,
    flush_every: int = 10,
    max_empty_retries: int = 3,
    retry_sleep: float = 30.0,
) -> None:
    jsonl_path = results_dir / f"worker_{worker_id:03d}.jsonl"
    buffer: List[str] = []
    empty_retries = 0

    while True:
        claimed = claim_task(queue_root, shard_id, n_shards=n_shards)
        if claimed is None:
            empty_retries += 1
            if empty_retries > max_empty_retries:
                break
            time.sleep(retry_sleep)
            continue
        empty_retries = 0

        task, claimed_path = claimed
        try:
            summary = run_task(task, results_dir, checkpoint_dir)
            buffer.append(_jsonl_line(task, summary))
            _atomic_append_lines(jsonl_path, buffer)
            buffer.clear()
            complete_task(claimed_path, queue_root)
        except Exception:
            error_msg = traceback.format_exc()
            fail_task(claimed_path, queue_root, error_msg)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="HPC task-queue worker process.")
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Defaults to worker-id %% n-shards.")
    parser.add_argument("--n-shards", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--max-empty-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    args = parser.parse_args()

    shard_id = args.shard_id if args.shard_id is not None else args.worker_id % args.n_shards
    checkpoint_dir = args.checkpoint_dir or args.results_dir / "checkpoints"

    worker_main(
        queue_root=args.queue_dir,
        results_dir=args.results_dir,
        checkpoint_dir=checkpoint_dir,
        worker_id=args.worker_id,
        shard_id=shard_id,
        n_shards=args.n_shards,
        checkpoint_every=args.checkpoint_every,
        flush_every=args.flush_every,
        max_empty_retries=args.max_empty_retries,
        retry_sleep=args.retry_sleep,
    )


if __name__ == "__main__":
    main()
