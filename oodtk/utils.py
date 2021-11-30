import logging
import numpy as np
import torch

log = logging.getLogger(__name__)


def calc_openness(n_train, n_test, n_target):
    """
    In *Toward open set recognition* from Scheirer, Jain, Boult et al, the Openness was defined.

    .. math::
        \\mathcal{O} = 1 - \\sqrt{ \\frac{2 \\times  n_{train}}{n_{test} \\times n_{target}} }


    :return: Openness of the problem

    :paper: https://ieeexplore.ieee.org/abstract/document/6365193
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

#
# def torch_get_squared_distances(centers, embeddings):
#     return torch_get_distances(centers, embeddings).pow(2)


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
    log.info(f"Optimizing Temperature")

    if contains_unknown(y):
        raise ValueError(f"Do not optimize temperature on unknown labels")

    nll = torch.nn.NLLLoss().to(device)
    temperature = torch.nn.Parameter(torch.ones(size=(1,)), requires_grad=True).to(
        device
    )
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
    log.info(f"Finished Optimizing Temperature")
    return best
