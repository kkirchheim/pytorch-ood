Wide ResNet
===========

The model implements the representation interface used by detectors:
``features`` (pooled features of shape :math:`B \times D`, for the ``encoder`` argument of
:class:`~pytorch_ood.api.FeaturesDetector` subclasses) and ``feature_maps`` (spatial feature
maps, for the ``backbone`` argument of :class:`~pytorch_ood.api.FeatureMapsDetector`
subclasses). It is used by the pre-trained models of the :doc:`registry <registry>`.

.. autoclass:: pytorch_ood.model.WideResNet
    :members:
