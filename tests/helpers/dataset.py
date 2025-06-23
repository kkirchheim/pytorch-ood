"""
Dataset Generators
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def sample_dataset(
    n_samples=100,
    n_dim=2,
    centers=None,
    std=1.0,
    loc=0.0,
    seed=None,
) -> TensorDataset:
    """
    Sample from some Gaussian's

    :param n_samples: number of samples from each cluster
    :param n_dim: number of dimensions
    :param centers: location, or number of centers
    :param std: std of clusters, same for all
    :param loc: mean of clusters, same for all
    :param seed: random seed for the generator
    """
    generator = np.random.default_rng(seed=seed)

    if centers is None:
        n_centers = 3
        centers = generator.normal(loc=loc, scale=5, size=(n_centers, n_dim))
    elif isinstance(centers, int):
        n_centers = centers
        centers = generator.normal(loc=loc, scale=5, size=(n_centers, n_dim))
    elif isinstance(centers, np.ndarray):
        n_centers = centers.shape[0]
        if centers.shape[1] != n_dim:
            raise ValueError()

    x = np.empty(shape=(n_samples * n_centers, n_dim), dtype=np.float32)
    y = np.empty(shape=(n_samples * n_centers,), dtype=int)

    for cluster in range(n_centers):
        x[cluster * n_samples : (cluster + 1) * n_samples] = generator.normal(
            loc=centers[cluster], scale=std, size=(n_samples, n_dim)
        )
        y[cluster * n_samples : (cluster + 1) * n_samples] = cluster

    return TensorDataset(torch.tensor(x), torch.tensor(y))
