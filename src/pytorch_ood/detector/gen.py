"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge


..  autoclass:: pytorch_ood.detector.GEN
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional, TypeVar

from torch import Tensor
from torch.nn import Module

from ..api import LogitsDetector

Self = TypeVar("Self")


class GEN(LogitsDetector):
    """
    Implements *GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection*.

    GEN generalizes softmax-based OOD scoring by applying a power transform to the
    posterior probabilities. The score is defined as

    .. math::
        G_\\gamma(x) = \\sum_{j=1}^{C} p_j(x)^\\gamma \\, (1 - p_j(x))^\\gamma

    where :math:`p_j(x) = \\sigma_j(f(x))` is the :math:`j^{th}` softmax probability of the
    model output and :math:`\\gamma \\in (0, 1)` controls the sensitivity.

    A small :math:`\\gamma` (the paper recommends :math:`\\gamma = 0.1`) amplifies differences
    near :math:`p = 0` and :math:`p = 1`, making the score highly sensitive to the shape of
    the full softmax distribution rather than only its maximum. In-distribution samples produce
    confident (peaky) posteriors with low scores, while OOD samples yield higher scores.

    :see Paper:
        `CVPR 2023 <https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html>`__

    :see Implementation:
        `GitHub <https://github.com/XixiLiu95/GEN>`__
    """

    def __init__(self, model: Optional[Module], gamma: Optional[float] = 0.1):
        """
        :param model: the neural network :math:`f`. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        :param gamma: exponent :math:`\\gamma`. Default is 0.1 as recommended by the paper.
        """
        super(GEN, self).__init__()
        self.model = model
        self.gamma: float = gamma  #: Power-transform exponent

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by the model
        """
        return self.score(logits, gamma=self.gamma)

    @staticmethod
    def score(logits: Tensor, gamma: float = 0.1) -> Tensor:
        """
        :param logits: logits of input
        :param gamma: power-transform exponent :math:`\\gamma`
        """
        p = logits.softmax(dim=1).clamp(1e-7, 1 - 1e-7)
        return (p.pow(gamma) * (1 - p).pow(gamma)).sum(dim=1)
