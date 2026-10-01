"""

..  autoclass:: pytorch_ood.detector.MCM
    :members:
    :inherited-members:
    :show-inheritance:
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)

log = logging.getLogger(__name__)


class MCM(FeaturesDetector):
    """
    Implements Maximum Concept Matching (MCM) from
    *Delving into Out-of-Distribution Detection with Vision-Language Representations*.

    MCM is a zero-shot OOD detection method designed for vision-language models such as CLIP.
    It exploits the alignment between image and text embeddings to score in-distribution vs
    out-of-distribution samples without requiring any training data or model fine-tuning.

    The method computes cosine similarities between an image's embedding and the embeddings
    of class name text prompts, then applies softmax to convert similarities to class
    probabilities. The negative maximum probability is used as the OOD score:

    .. math::
        -\\max_k \\left[ \\text{softmax}\\left(
            \\hat{z}(x) \\cdot \\hat{T}^T / \\tau
        \\right) \\right]_k

    where :math:`\\hat{z}(x)` is the L2-normalized image embedding, :math:`\\hat{T}` is
    the matrix of L2-normalized class text embeddings, :math:`\\tau` is the temperature
    the cosine similarities are divided by.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Delving into Out-of-Distribution Detection with Vision-Language Representations",
            venue="NeurIPS",
            year=2022,
            url="https://arxiv.org/abs/2211.13445",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    requires_fit = False

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        text_embeddings: Tensor,
        # default: the paper's tau = 1 (it reports similar results for tau in [0.5, 100])
        temperature: float = 1.0,
    ):
        """
        :param encoder: image feature encoder (e.g. CLIP's encode_image). Can be
            ``None`` when using ``predict_features(...)`` directly.
        :param text_embeddings: pre-computed class text embeddings, shape :math:`C \\times D`.
            Should be L2-normalized; if not, normalization is applied internally.
        :param temperature: temperature :math:`\\tau`; the cosine similarities are divided by
            it before the softmax
        """
        super().__init__()
        self.encoder = encoder
        self.text_embeddings = text_embeddings
        self.temperature = temperature

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute MCM scores directly from image features.

        :param z: image embeddings, shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        """
        if self.text_embeddings is None:
            raise RequiresFittingException

        z_norm = F.normalize(z.float(), dim=-1)
        t_norm = F.normalize(self.text_embeddings.to(z.device).float(), dim=-1)
        similarities = z_norm @ t_norm.T / self.temperature
        return -similarities.softmax(dim=-1).max(dim=-1).values

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through ``encoder``
        :return: outlier scores of shape :math:`B`
        """
        if self.encoder is None:
            raise ModelNotSetException
        features = self.encoder(x)
        return self.predict_features(features)
