"""
ResNet-18, ResNet-50

Pre-trained weights are available through the model registry, see
:func:`load_model <pytorch_ood.model.load_model>`.
"""

from typing import Optional

import torch
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNet18(ResNet):
    """
    ``torchvision.models.resnet18``, extended with the representation interface
    :class:`WideResNet <pytorch_ood.model.WideResNet>` exposes (``features``,
    ``feature_maps``, ``forward_feature_maps``), for detectors that need pooled
    features or spatial feature maps rather than just logits.

    Built by subclassing ``torchvision.models.resnet.ResNet`` directly with the same
    ``block``/``layers`` arguments ``torchvision.models.resnet18()`` uses internally,
    so the module tree -- and therefore the ``state_dict`` keys -- is identical to
    plain ``torchvision.models.resnet18``. Existing checkpoints load unchanged.

    :see Paper: `CVPR <https://arxiv.org/abs/1512.03385>`__
    """

    def __init__(self, num_classes: int = 1000, weights: Optional[object] = None, **kwargs):
        """
        :param num_classes: number of classes
        :param weights: unused; accepted for interface compatibility with
            ``torchvision.models.resnet18`` (weights are loaded separately by the
            model registry)
        """
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)

    def feature_maps(self, x: Tensor) -> Tensor:
        """
        Spatial feature maps before global average pooling.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_feature_maps(self, x: Tensor) -> Tensor:
        """
        Maps spatial feature maps (as returned by :meth:`feature_maps`) to logits.
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def features(self, x: Tensor) -> Tensor:
        """
        Extracts (flattened, pooled) features before the last fully connected layer.
        """
        x = self.avgpool(self.feature_maps(x))
        return torch.flatten(x, 1)


class ResNet50(ResNet):
    """
    ``torchvision.models.resnet50``, extended with the same representation interface
    as :class:`ResNet18`.

    :see Paper: `CVPR <https://arxiv.org/abs/1512.03385>`__
    """

    def __init__(self, num_classes: int = 1000, weights: Optional[object] = None, **kwargs):
        """
        :param num_classes: number of classes
        :param weights: unused; accepted for interface compatibility with
            ``torchvision.models.resnet50`` (weights are loaded separately by the
            model registry)
        """
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)

    def feature_maps(self, x: Tensor) -> Tensor:
        """
        Spatial feature maps before global average pooling.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_feature_maps(self, x: Tensor) -> Tensor:
        """
        Maps spatial feature maps (as returned by :meth:`feature_maps`) to logits.
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def features(self, x: Tensor) -> Tensor:
        """
        Extracts (flattened, pooled) features before the last fully connected layer.
        """
        x = self.avgpool(self.feature_maps(x))
        return torch.flatten(x, 1)
