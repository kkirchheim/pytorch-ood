import logging

import numpy as np
import torch
from torch import nn
from torch.nn import init

log = logging.getLogger(__name__)


def calc_openness(n_train, n_test, n_target):
    """
    In *Toward open set recognition* from Scheirer, Jain, Boult et al, the Openness was defined.

    .. math::
        \\mathcal{O} = 1 - \\sqrt{ \\frac{2 \\times  n_{train}}{n_{test} \\times n_{target}} }


    :return: Openness of the problem

    :see paper: https://ieeexplore.ieee.org/abstract/document/6365193
    """
    frac = 2 * n_train / (n_test + n_target)
    return 1 - np.sqrt(frac)


#######################################
# Helpers for labels
#######################################


def is_known(labels):
    """
    :returns: True, if label >= 0
    """
    return labels >= 0


# def is_known_unknown(labels):
#     return (labels < 0) & (labels > -1000)


# def is_unknown_unknown(labels) -> bool:
#     return labels <= -1000


def is_unknown(labels) -> bool:
    """
    :returns: True, if label < 0
    """
    return labels < 0


def contains_known_and_unknown(labels) -> bool:
    """
    :return: true if the labels contain known and unknown classes
    """
    return contains_known(labels) and contains_unknown(labels)


def contains_known(labels) -> bool:
    """
    :return: true if the labels contains any known labels
    """
    return is_known(labels).any()


def contains_unknown(labels) -> bool:
    """
    :return: true if the labels contains any unknown labels
    """
    return is_unknown(labels).any()


#######################################
# Distance functions etc.
#######################################


def estimate_class_centers(
    embedding: torch.Tensor, target: torch.Tensor, num_centers: int = None
) -> torch.Tensor:
    """
    Estimates class centers from the given embeddings and labels, using mean as estimator.

    TODO: the loop can prob. be replaced
    """
    batch_classes = torch.unique(target).long().to(embedding.device)

    if num_centers is None:
        num_centers = torch.max(target) + 1

    centers = torch.zeros((num_centers, embedding.shape[1]), device=embedding.device)

    for clazz in batch_classes:
        centers[clazz] = embedding[target == clazz].mean(dim=0)

    return centers


def torch_get_distances(centers, embeddings):
    """
    TODO: this can be done way more efficiently
    """

    n_instances = embeddings.shape[0]
    n_centers = centers.shape[0]
    distances = torch.empty((n_instances, n_centers)).to(embeddings.device)

    for clazz in torch.arange(n_centers):
        distances[:, clazz] = torch.norm(embeddings - centers[clazz], dim=1, p=2)

    return distances


def optimize_temperature(logits: torch.Tensor, y, init=1, steps=1000, device="cpu"):
    """
    Optimizing temperature for temperature scaling, by minimizing NLL on the given logits

    :see Paper: https://arxiv.org/pdf/1706.04599.pdf
    """
    log.info("Optimizing Temperature")

    if contains_unknown(y):
        raise ValueError("Do not optimize temperature on unknown labels")

    nll = torch.nn.NLLLoss().to(device)
    temperature = torch.nn.Parameter(torch.ones(size=(1,)), requires_grad=True).to(device)
    torch.fill_(temperature, init)
    logits = logits.clone().to(device)
    y = y.clone().to(device)
    optimizer = torch.optim.SGD([temperature], lr=0.1)

    with torch.enable_grad():
        for i in range(steps):
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            log_probs = torch.nn.functional.log_softmax(scaled_logits)
            loss = nll(log_probs, y)
            loss.backward()
            log.info(
                f"Step {i} Temperature {temperature.item()} NLL {loss.item()} Grad: {temperature.grad.item()}"
            )
            optimizer.step()

    best = temperature.detach().item()
    log.info("Finished Optimizing Temperature")
    return best


def pairwise_distances(x, y=None) -> torch.Tensor:
    """
    Calculate pairwise distance by quadratic expansion.

    :param x: is a Nxd matrix
    :param y:  Mxd matrix

    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

    :see Implementation: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)


class RunningCenters(nn.Module):
    """"""

    def __init__(self, n_centers, dim, **kwargs):
        """

        :param n_centers: number of centers
        :param dim: number of dimensions
        """
        self.n = n_centers
        self.dim = dim

        # create buffer for centers. those buffers will be updated during training, and are fixed during evaluation
        centers = torch.empty(size=(self.n_classes, self.dim), requires_grad=False).double()
        counter = torch.empty(size=(1,), requires_grad=False).double()

        self.register_buffer("centers", centers)
        self.register_buffer("counter", counter)
        self.reset_running_stats()

    def reset(self):
        log.info("Reset running stats")
        init.zeros_(self.running_centers)
        init.zeros_(self.num_batches_tracked)

    def forward(self, x, y):
        """
        Update centers

        :param x: points
        :param y: targets
        :return:
        """

        if self.training:
            batch_classes = torch.unique(y, sorted=False)
            mu = self._calculate_centers(x, y)

            # update running mean centers
            cma = mu[batch_classes] + self.centers[batch_classes] * self.counter
            self.centers[batch_classes] = cma / (self.counter + 1)
            self.counter += 1

    def _calculate_centers(self, x, y):
        mu = torch.full(size=(self.n, self.dim), fill_value=float("NaN"), device=x.device)
        for clazz in y.unique(sorted=False):
            mu[clazz] = x[y == clazz].mean(dim=0)
        return mu


class ToUnknown(object):
    """"""

    def __init__(self):
        pass

    def __call__(self, y):
        return -1
