"""Post-hoc observable functions for NCP Regime C experiments.

All functions operate on shell_traj ({shell: (T, d_param)}) computed by
ncp_runner.compute_shell_trajs().  Pure functions — no simulation state.
"""

import math
from typing import Dict, List, Optional

import numpy as np

from .active_escape import basin_label


# ---------------------------------------------------------------------------
# Shell modal basin
# ---------------------------------------------------------------------------


def shell_modal_basin(
    shell_traj: Dict[int, np.ndarray],
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
) -> Dict[int, str]:
    """Modal basin ('A', 'B', or 'X') for each shell over the trajectory.

    For each shell, computes basin_label at each round and returns the
    majority vote.  Ties are broken in favour of 'X', then 'A'.
    """
    result: Dict[int, str] = {}
    for shell, traj in shell_traj.items():
        T = traj.shape[0]
        counts: Dict[str, int] = {'A': 0, 'B': 0, 'X': 0}
        for t in range(T):
            label = basin_label(traj[t], theta_A, theta_B, epsilon)
            counts[label] += 1
        result[shell] = max(counts, key=lambda k: counts[k])
    return result


# ---------------------------------------------------------------------------
# Decoupling parameter χ
# ---------------------------------------------------------------------------


def decoupling_chi(
    shell_traj: Dict[int, np.ndarray],
    inner_shell: int,
    outer_shell: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
) -> float:
    """Fraction of time-steps where inner and outer shells disagree on basin.

    χ = P(basin(inner) ≠ basin(outer)), estimated empirically over T rounds.
    Higher χ indicates more decoupling between the two shells.

    Returns NaN if either shell is missing from shell_traj or T=0.
    """
    if inner_shell not in shell_traj or outer_shell not in shell_traj:
        return float("nan")
    T = min(
        shell_traj[inner_shell].shape[0],
        shell_traj[outer_shell].shape[0],
    )
    if T == 0:
        return float("nan")

    disagreements = 0
    for t in range(T):
        l_inner = basin_label(shell_traj[inner_shell][t], theta_A, theta_B, epsilon)
        l_outer = basin_label(shell_traj[outer_shell][t], theta_A, theta_B, epsilon)
        if l_inner != l_outer:
            disagreements += 1
    return disagreements / T


# ---------------------------------------------------------------------------
# Conditional mutual information (NCP-5 factorisation test)
# ---------------------------------------------------------------------------


def conditional_mutual_information(
    shell_traj: Dict[int, np.ndarray],
    shell_k: int,
    shell_k1: int,
    shell_k2: int,
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    epsilon: float = 0.2,
) -> float:
    """I(B_{shell_k} ; B_{shell_k2} | B_{shell_k1}) estimated empirically.

    Tests Theorem 4 factorisation: conditional independence of non-adjacent
    shells given intermediate shell.  Should be ≈ 0 under the Markov property.

    Uses the identity I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    with empirical histograms over T time steps.

    Labels are discretised to {0=A, 1=B, 2=X}.  Returns NaN if any shell
    is missing or T is too small for reliable estimation.
    """
    required = [shell_k, shell_k1, shell_k2]
    if not all(s in shell_traj for s in required):
        return float("nan")

    T = min(shell_traj[s].shape[0] for s in required)
    if T < 10:
        return float("nan")

    # Discretise trajectories
    label_map = {'A': 0, 'B': 1, 'X': 2}

    def _labels(shell: int) -> np.ndarray:
        return np.array([
            label_map[basin_label(shell_traj[shell][t], theta_A, theta_B, epsilon)]
            for t in range(T)
        ], dtype=np.int32)

    X = _labels(shell_k)
    Z = _labels(shell_k1)
    Y = _labels(shell_k2)

    def _entropy(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _joint_counts_2(*arrays) -> np.ndarray:
        """Flat-index joint counts for up to 3 discrete variables (each 0..2)."""
        n = 3
        if len(arrays) == 2:
            idx = arrays[0] * n + arrays[1]
            return np.bincount(idx, minlength=n * n).reshape(n, n)
        # 3 variables
        idx = arrays[0] * n * n + arrays[1] * n + arrays[2]
        return np.bincount(idx, minlength=n * n * n).reshape(n, n, n)

    H_XZ = _entropy(_joint_counts_2(X, Z).flatten())
    H_YZ = _entropy(_joint_counts_2(Y, Z).flatten())
    H_XYZ = _entropy(_joint_counts_2(X, Y, Z).flatten())
    H_Z = _entropy(np.bincount(Z, minlength=3))

    cmi = H_XZ + H_YZ - H_XYZ - H_Z
    return max(0.0, cmi)  # CMI is non-negative; clamp numerical noise


# ---------------------------------------------------------------------------
# Propagation matrix (NCP-4)
# ---------------------------------------------------------------------------


def propagation_matrix(
    shell_trajs: List[Dict[int, np.ndarray]],
    shell_list: List[int],
    theta_A: np.ndarray,
    theta_B: np.ndarray,
    t_horizon: int,
    epsilon: float = 0.2,
) -> np.ndarray:
    """Empirical shell-to-shell propagation probability matrix.

    P[i, j] = fraction of runs (in shell_trajs list) where shell shell_list[j]
    is in basin B at t_horizon, conditioned on the i-th shell being the
    "source" (innermost shell in each run).

    In practice this is called across multiple runs that differ in which
    shell is initialised at B; the caller organises runs by source shell.

    Parameters
    ----------
    shell_trajs : list of shell_traj dicts, one per run.
    shell_list  : ordered list of shell indices to include.
    t_horizon   : measurement round for cascade-size check.

    Returns
    -------
    (n_shells, n_shells) float array.
    """
    n = len(shell_list)
    shell_to_idx = {s: i for i, s in enumerate(shell_list)}
    matrix = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    for run_idx, shell_traj in enumerate(shell_trajs):
        # Identify the "source" shell: shell in B at t=0
        source_shell = None
        for s in shell_list:
            if s in shell_traj and shell_traj[s].shape[0] > 0:
                if basin_label(shell_traj[s][0], theta_A, theta_B, epsilon) == 'B':
                    source_shell = s
                    break
        if source_shell is None or source_shell not in shell_to_idx:
            continue
        src_idx = shell_to_idx[source_shell]
        counts[src_idx] += 1

        t = min(t_horizon, min(shell_traj[s].shape[0] for s in shell_list if s in shell_traj) - 1)
        for s in shell_list:
            if s not in shell_traj:
                continue
            if basin_label(shell_traj[s][t], theta_A, theta_B, epsilon) == 'B':
                matrix[src_idx, shell_to_idx[s]] += 1.0

    # Normalise rows
    for i in range(n):
        if counts[i] > 0:
            matrix[i] /= counts[i]

    return matrix
