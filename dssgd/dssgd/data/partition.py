from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def iid_partition(dataset: Dataset, n_agents: int, seed: int = 0) -> List[Subset]:
    """Randomly split dataset into n equal parts."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, n_agents)
    return [Subset(dataset, split.tolist()) for split in splits]


def dirichlet_partition(
    dataset: Dataset,
    n_agents: int,
    alpha: float = 0.5,
    seed: int = 0,
) -> List[Subset]:
    """Non-IID split via Dirichlet allocation. Lower alpha → more heterogeneous."""
    rng = np.random.default_rng(seed)
    labels = _get_labels(dataset)
    classes = np.unique(labels)
    agent_indices: List[List[int]] = [[] for _ in range(n_agents)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_agents))
        counts = (proportions * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - counts[:-1].sum()
        splits = np.split(cls_idx, np.cumsum(counts)[:-1])
        for i, chunk in enumerate(splits):
            agent_indices[i].extend(chunk.tolist())

    return [Subset(dataset, idxs) for idxs in agent_indices]


def make_loaders(
    subsets: List[Subset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> List[DataLoader]:
    return [
        DataLoader(s, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        for s in subsets
    ]


def _get_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    return np.array([dataset[i][1] for i in range(len(dataset))])
