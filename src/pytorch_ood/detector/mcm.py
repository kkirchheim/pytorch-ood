"""
.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.MCM
    :members:
    :inherited-members:
    :show-inheritance:
"""

import logging
from typing import Callable, Optional, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from ..api import FeaturesDetector, ModelNotSetException, RequiresFittingException

log = logging.getLogger(__name__)
Self = TypeVar("Self")


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
        s(x) = -\\max_k \\left[ \\text{softmax}\\left(
            \\hat{z}(x) \\cdot \\hat{T}^T \\cdot \\tau
        \\right) \\right]_k

    where :math:`\\hat{z}(x)` is the L2-normalized image embedding, :math:`\\hat{T}` is
    the matrix of L2-normalized class text embeddings, :math:`\\tau` is the temperature
    scaling factor, and higher scores indicate more likely OOD samples.

    :see Paper: `ArXiv <https://arxiv.org/abs/2211.13445>`__
    """

    requires_fit = False

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        text_embeddings: Tensor,
        temperature: float = 100.0,
    ):
        """
        :param encoder: image feature encoder (e.g. CLIP's encode_image). Can be
            ``None`` when using ``predict_features(...)`` directly.
        :param text_embeddings: pre-computed class text embeddings, shape (C, D).
            Should be L2-normalized; if not, normalization is applied internally.
        :param temperature: cosine similarity scale factor (CLIP's default is ~100)
        """
        super().__init__()
        self.encoder = encoder
        self.text_embeddings = text_embeddings
        self.temperature = temperature

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute MCM scores directly from image features.

        :param z: image embeddings, shape (B, D)
        """
        if self.text_embeddings is None:
            raise RequiresFittingException

        z_norm = F.normalize(z.float(), dim=-1)
        t_norm = F.normalize(self.text_embeddings.to(z.device).float(), dim=-1)
        similarities = z_norm @ t_norm.T * self.temperature
        return -similarities.softmax(dim=-1).max(dim=-1).values

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor
        """
        if self.encoder is None:
            raise ModelNotSetException
        features = self.encoder(x)
        return self.predict_features(features)
