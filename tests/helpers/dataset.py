"""
Dataset Generators
"""

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
    Sample from some Gaussians.

    :param n_samples: number of samples from each cluster
    :param n_dim: number of dimensions
    :param centers: number of centers (int) or center coordinates (Tensor of shape (K, n_dim))
    :param std: std of clusters, same for all
    :param loc: mean used when randomly placing centers
    :param seed: random seed for the generator
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    if centers is None or isinstance(centers, int):
        n_centers = 3 if centers is None else int(centers)
        centers = torch.randn(n_centers, n_dim, generator=g) * 5.0 + loc
    else:
        centers = torch.as_tensor(centers, dtype=torch.float32)
        n_centers = centers.shape[0]

    x_parts = []
    y_parts = []
    for c in range(n_centers):
        x_parts.append(torch.randn(n_samples, n_dim, generator=g) * std + centers[c])
        y_parts.append(torch.full((n_samples,), c, dtype=torch.long))

    return TensorDataset(torch.cat(x_parts), torch.cat(y_parts))
