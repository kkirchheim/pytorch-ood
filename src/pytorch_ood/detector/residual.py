"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.Residual
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional, TypeVar

import torch
from torch import Tensor

from .vim import ViM

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class Residual(ViM):
    """
    Implements the Residual detector from the paper
    *ViM: Out-Of-Distribution with Virtual-logit Matching* (Wang et al., CVPR 2022).

    Projects penultimate-layer features onto the **null subspace** — the subspace
    spanned by the eigenvectors of the empirical covariance matrix that correspond
    to the **smallest** eigenvalues (i.e. low-variance directions not explained by
    the training data). The L2 norm of those projected features serves as the OOD score.

    This class inherits from :class:`ViM` to reuse the PyTorch-native empirical
    covariance and principal subspace computation logic, removing duplicate code.

    :see Paper: `CVPR 2022 <https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-Of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf>`__
    :see Implementation: `GitHub <https://github.com/haoqiwang/vim>`__
    """

    requires_fit = True

    def __init__(
        self,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]],
        d: int,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        :param encoder: feature encoder. Can be
            ``None`` when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param d: dimensionality of the principal subspace
        :param w: weights :math:`W` of the last layer of the network
        :param b: biases :math:`b` of the last layer of the network
        """
        super().__init__(encoder=encoder, d=d, w=w, b=b)

    def fit_features(self: Self, features: Tensor, labels: Tensor) -> Self:
        """
        Extracts features, computes principle subspace using parent class.
        Ignores OOD samples.
        """
        super().fit_features(features, labels)
        return self

    @torch.no_grad()
    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features as given by the model
        """
        device = self.w.device
        x = x.detach().to(device).float()

        # Project centered features onto the null subspace and take L2 norm
        x_p_t = (x - self.u) @ self.principal_subspace  # (N, D-d)
        vlogit = x_p_t.norm(dim=-1)  # (N,)

        # Residual score is precisely the projection norm (vlogit).
        # Higher score = more likely OOD.
        return vlogit
