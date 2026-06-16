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
        G_\\gamma(x) = \\sum_{j \\in \\text{top-}M} p_j(x)^\\gamma \\, (1 - p_j(x))^\\gamma

    where :math:`p_j(x) = \\sigma_j(f(x))` is the :math:`j^{th}` softmax probability of the
    model output, :math:`\\gamma \\in (0, 1)` controls the sensitivity, and the sum runs over
    the :math:`M` largest probabilities. Restricting to the top :math:`M` classes (the paper
    uses :math:`M = 100` for ImageNet) discards the noisy near-zero tail, which the power
    transform would otherwise amplify; for small label spaces it has little effect.

    A small :math:`\\gamma` (the paper recommends :math:`\\gamma = 0.1`) amplifies differences
    near :math:`p = 0` and :math:`p = 1`, making the score highly sensitive to the shape of
    the (truncated) softmax distribution rather than only its maximum. In-distribution samples
    produce confident (peaky) posteriors with low scores, while OOD samples yield higher scores.

    :see Paper:
        `CVPR 2023 <https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html>`__

    :see Implementation:
        `GitHub <https://github.com/XixiLiu95/GEN>`__
    """

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`, matching the
    #: ``gamma`` and ``M`` sweep used by OpenOOD.
    hyperparameter_space = {
        "gamma": [0.01, 0.1, 0.5, 1, 2, 5, 10],
        "M": [10, 50, 100, 200, 500, 1000],
    }

    def __init__(
        self, model: Optional[Module], gamma: Optional[float] = 0.1, M: Optional[int] = None
    ):
        """
        :param model: the neural network :math:`f`. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        :param gamma: exponent :math:`\\gamma`. Default is 0.1 as recommended by the paper.
        :param M: number of largest softmax probabilities to sum over. ``None`` (default)
            uses all classes; the paper uses ``M = 100`` for ImageNet.
        """
        super(GEN, self).__init__()
        self.model = model
        self.gamma: float = gamma  #: Power-transform exponent
        self.M: Optional[int] = M  #: Number of top classes to consider (``None`` = all)

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by the model
        """
        return self.score(logits, gamma=self.gamma, M=self.M)

    @staticmethod
    def score(logits: Tensor, gamma: float = 0.1, M: Optional[int] = None) -> Tensor:
        """
        :param logits: logits of input
        :param gamma: power-transform exponent :math:`\\gamma`
        :param M: number of largest softmax probabilities to sum over. ``None`` uses all.
        """
        p = logits.softmax(dim=1).clamp(1e-7, 1 - 1e-7)
        if M is not None:
            # keep the M largest probabilities per sample (top-M classes)
            p = p.sort(dim=1, descending=True).values[:, :M]
        return (p.pow(gamma) * (1 - p).pow(gamma)).sum(dim=1)
