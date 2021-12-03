"""

..  autoclass:: oodtk.mahalanobis.Mahalanobis
    :members:

"""
import torch


class Mahalanobis(torch.nn.Module):
    """
    Implements the Mahalanobis Distance Baseline Scoring.

    """

    def __init__(self):
        """"""
        super(Mahalanobis, self).__init__()
