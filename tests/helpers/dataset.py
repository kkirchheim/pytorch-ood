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


def points_with_covariance(centers: torch.Tensor, variances: torch.Tensor):
    """
    Points whose class means are exactly ``centers`` and whose shared within-class covariance
    (centered outer products divided by the number of points) is exactly ``diag(variances)``.

    Each class gets the :math:`2D` points :math:`c \\pm a_i e_i`; with :math:`a_i^2 = D v_i` the
    pooled covariance is :math:`\\mathrm{diag}(v)`.

    :param centers: class means of shape :math:`C \\times D`
    :param variances: diagonal of the covariance, shape :math:`D`
    :return: features of shape :math:`2CD \\times D` and labels of shape :math:`2CD`
    """
    n_classes, d = centers.shape
    offsets = torch.diag((variances * d).sqrt())
    offsets = torch.cat([offsets, -offsets])
    z = (centers[:, None, :] + offsets[None]).reshape(-1, d)
    y = torch.arange(n_classes).repeat_interleave(2 * d)
    return z, y
