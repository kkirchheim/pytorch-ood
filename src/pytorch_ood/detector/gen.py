"""

..  autoclass:: pytorch_ood.detector.GEN
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional

from torch import Tensor
from torch.nn import Module
from typing_extensions import Self

from ..api import DetectorInfo, LogitsDetector, Paper, Task


class GEN(LogitsDetector):
    """
    Implements *GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection*.

    GEN generalizes softmax-based OOD scoring by applying a power transform to the
    posterior probabilities. The score is defined as

    .. math::
        G_\\gamma(x) = \\frac{1}{M} \\sum_{j \\in \\text{top-}M} p_j(x)^\\gamma \\, (1 - p_j(x))^\\gamma

    where :math:`p_j(x) = \\sigma_j(f(x))` is the :math:`j^{th}` softmax probability of the
    model output, :math:`\\gamma > 0` controls the sensitivity, and the sum runs over
    the :math:`M` largest probabilities. Restricting to the top :math:`M` classes (the paper
    uses :math:`M = 100` for ImageNet) discards the noisy near-zero tail, which the power
    transform would otherwise amplify; for small label spaces it has little effect.

    .. note::
        The terms are averaged instead of summed, so the scores are bounded independent of the number of classes.

    A small :math:`\\gamma` (the paper recommends :math:`\\gamma = 0.1`) amplifies differences
    near :math:`p = 0` and :math:`p = 1`, making the score highly sensitive to the shape of
    the (truncated) softmax distribution rather than only its maximum. In-distribution samples
    produce confident (peaky) posteriors with low scores, while OOD samples yield higher scores.
    """

    info = DetectorInfo(
        paper=Paper(
            title="GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection",
            venue="CVPR",
            year=2023,
            url="https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html",
            code="https://github.com/XixiLiu95/GEN",
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
        ai_coded=True,
    )

    # matches the ``gamma`` and ``M`` sweep used by OpenOOD
    #: Default search space for :class:`pytorch_ood.utils.GridSearch`.
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
        :param logits: logits given by the model, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :return: outlier scores of shape :math:`B` (or :math:`B \\times H \\times W`)
        """
        return self.score(logits, gamma=self.gamma, M=self.M)

    @staticmethod
    def score(logits: Tensor, gamma: float = 0.1, M: Optional[int] = None) -> Tensor:
        """
        :param logits: logits of input, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :param gamma: power-transform exponent :math:`\\gamma`
        :param M: number of largest softmax probabilities to average over. ``None`` uses all.
        :return: outlier scores of shape :math:`B` (or :math:`B \\times H \\times W`).
            Probabilities are clamped to :math:`[10^{-7}, 1 - 10^{-7}]`.
        """
        p = logits.softmax(dim=1).clamp(1e-7, 1 - 1e-7)
        if M is not None:
            # keep the M largest probabilities per sample (top-M classes)
            p = p.sort(dim=1, descending=True).values[:, :M]
        # mean, not sum, to keep the score bounded regardless of class count; this rescales the score by a
        # constant factor (no effect on AUROC/AUPR) and avoids float32 saturation in torchmetrics' binary_auroc
        return (p.pow(gamma) * (1 - p).pow(gamma)).mean(dim=1)
