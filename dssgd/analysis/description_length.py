"""Operational description length (paper2_social Definition 4.1).

DL_delta(b): fix a receiver with loss L and relaxation dynamics, and a
coding scheme (prior) P over parameter space. DL_delta(b) is the fewest
bits k such that a k-bit codeword under P, decoded and then run through the
receiver's own relaxation, lands the receiver in basin b with probability
>= 1 - delta.

Shared by Experiment C1 (whose gossip_capacity_bits sweep already exercises
this quantize-then-relax mechanism live, inside a full cascade run via
dssgd.protocols.gossip's capacity-limited kicks) and Experiment C3 (which
needs a STANDALONE per-basin measurement -- not a full cascade -- to build a
family of (vartheta, operational DL) points to correlate against
generalisation gap, per Lemma 4.2 / Remark 4.4). Both reuse the exact same
`quantize_tensor` primitive and the same loss_fn/basin_label machinery from
gossip.py/active_escape.py, so "kick success" (C1) and "operational DL"
(C3) are provably the same test applied at different granularities (Claim
4.3), not independently-implemented approximations of one another.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from dssgd.protocols.gossip import quantize_tensor

from .active_escape import BistableParameterModule, basin_label


@dataclass
class OperationalDLResult:
    """Definition 4.1's DL_delta(b), plus the full sweep it was found from.

    `spec` carries whatever basin spec (a, b_in, b_out, generality_level,
    ...) produced this measurement -- check_phasec3.py correlates dl_bits
    against vartheta/generalization-gap keyed off these fields, so they
    travel with the result rather than needing to be re-derived from the
    task_id.
    """

    dl_bits: Optional[int]           # None if not achieved anywhere in k_search_range
    passage_probability: float       # passage probability at dl_bits (or at the largest k tried, if never achieved)
    delta: float
    passage_by_k: Dict[int, float] = field(default_factory=dict)
    name: str = ""
    spec: Dict = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OperationalDLResult":
        with open(path, "rb") as f:
            return pickle.load(f)


def lattice_from_thetas(
    theta_A: np.ndarray, theta_B: np.ndarray, ratio: float = 3.0,
) -> "tuple[float, float]":
    """(center, extent) for the SAME quantization lattice
    NaturalCascadeConfig.gossip_capacity_bits/capacity_lattice_ratio derives
    from theta_A/theta_B -- keeping this in one place so a DL measured here
    and a passage probability measured by a live C1 simulation at the same
    k are directly comparable (same codebook)."""
    center = float(((theta_A + theta_B) / 2.0).mean())
    extent = ratio * float(np.abs(theta_B - theta_A).max())
    return center, extent


def _relax(theta0: float, loss_fn: Callable, lr: float, relax_steps: int) -> float:
    """Run `relax_steps` gradient-descent steps on loss_fn from theta0 --
    the receiver's own relaxation dynamics (Definition 4.1: "decoded and
    relaxed by the receiver"). d_param=1 only (the campaign's scope)."""
    model = BistableParameterModule(d_param=1, init_value=torch.tensor([theta0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=max(lr, 1e-8))
    for _ in range(relax_steps):
        optimizer.zero_grad()
        loss = loss_fn(model, None)
        loss.backward()
        optimizer.step()
    return float(model.theta.detach().item())


def measure_passage_probability(
    loss_fn: Callable,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    sender_theta: float,
    receiver_init_theta: float,
    target_basin: str,
    capacity_bits: Optional[int],
    lattice_center: float,
    lattice_extent: float,
    alpha: float = 0.5,
    lr: float = 0.1,
    relax_steps: int = 50,
    n_trials: int = 200,
    init_noise_scale: float = 0.05,
    epsilon: float = 0.2,
    seed: int = 0,
) -> float:
    """Empirical Pr[receiver lands in target_basin] over n_trials
    independent receiver draws, each: quantize sender_theta to capacity_bits
    (or transmit exactly if None -- the uncompressed control), apply one
    kick x_i <- (1-alpha)*x_i + alpha*decoded_w_j from a freshly-initialised
    receiver x_i, then relax for relax_steps of the receiver's own local
    descent. This is exactly the mechanism Definition 4.1 describes and the
    one Experiment C1's live gossip_capacity_bits kicks implement inside a
    full cascade -- reusing quantize_tensor here (not re-deriving it) is
    what keeps the two consistent.
    """
    rng = np.random.default_rng(seed)
    n_success = 0
    for _ in range(n_trials):
        x_i = receiver_init_theta + float(rng.normal(0.0, init_noise_scale))
        w_j = torch.tensor([float(sender_theta)])
        if capacity_bits is not None:
            w_j, _, _ = quantize_tensor(w_j, capacity_bits, lattice_center, lattice_extent)
        kicked = (1.0 - alpha) * x_i + alpha * float(w_j.item())
        final_theta = _relax(kicked, loss_fn, lr, relax_steps)
        label = basin_label(np.array([final_theta]), theta_A, theta_B, epsilon)
        if label == target_basin:
            n_success += 1
    return n_success / n_trials


def operational_dl(
    loss_fn: Callable,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    sender_theta: float,
    receiver_init_theta: float,
    target_basin: str,
    delta: float = 0.1,
    k_search_range: Iterable[int] = range(1, 13),
    lattice_ratio: float = 3.0,
    **measure_kwargs,
) -> OperationalDLResult:
    """Search k_search_range (ascending) for the fewest bits DL_delta(b)
    achieving passage probability >= 1 - delta. The quantization lattice is
    derived from theta_A/theta_B via `lattice_from_thetas` -- the same rule
    NaturalCascadeConfig's gossip_capacity_bits uses -- so a DL value
    measured here is directly comparable to a live C1 sweep at the same k.
    """
    center, extent = lattice_from_thetas(theta_A, theta_B, lattice_ratio)
    passage_by_k: Dict[int, float] = {}
    dl_bits: Optional[int] = None
    for k in sorted(set(k_search_range)):
        p = measure_passage_probability(
            loss_fn, theta_A, theta_B, sender_theta, receiver_init_theta, target_basin,
            capacity_bits=k, lattice_center=center, lattice_extent=extent,
            **measure_kwargs,
        )
        passage_by_k[k] = p
        if dl_bits is None and p >= 1.0 - delta:
            dl_bits = k
            break

    if dl_bits is not None:
        passage_at_dl = passage_by_k[dl_bits]
    elif passage_by_k:
        last_k = max(passage_by_k)
        passage_at_dl = passage_by_k[last_k]
    else:
        passage_at_dl = 0.0

    return OperationalDLResult(
        dl_bits=dl_bits, passage_probability=passage_at_dl, delta=delta,
        passage_by_k=passage_by_k,
    )
