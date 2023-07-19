"""
Wide Resnet

See https://github.com/wetliu/energy_ood/blob/master/CIFAR/models/wrn.py

Pretrained weights:

Pretrained on downscaled imagenet:
* https://github.com/hendrycks/pre-training/raw/master/downsampled_train/snapshots/40_2/imagenet_wrn_baseline_epoch_99.pt

"""
import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
from torch import Tensor
from torch.hub import load_state_dict_from_url

from pytorch_ood.utils import ToRGB


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)

        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Resnet Architecture with large number of channels and variable depth, which has been used in a number of
    publications.

    Provides a number of pre-trained weights for models trained with
    :class:`OutlierExposureLoss <pytorch_ood.loss.OutlierExposureLoss>`,
    :class:`Energy Regularization <pytorch_ood.loss.EnergyRegularizedLoss>` or
    :class:`PixMix <pytorch_ood.dataset.img.PixMixDataset>`.
    Also includes models pre-trained on the variants on the ImageNet, which is known to increase the robustness.

    :see Paper: `BMVC <https://arxiv.org/pdf/1605.07146v4.pdf>`__
    :see Implementation: `GitHub <https://github.com/wetliu/energy_ood/blob/master/CIFAR/models/wrn.py>`__
    """

    def __init__(
        self,
        num_classes,
        depth=40,
        widen_factor=2,
        drop_rate=0.3,
        in_channels=3,
        pretrained=None,
    ):
        """

        :param depth: depth of the network
        :param num_classes: number of classes
        :param widen_factor: factor used for channel increase per block
        :param drop_rate: dropout probability
        :param in_channels: number of input planes
        :param pretrained: identifier of pretrained weights to load

        Pretrained weights are taken from the corresponding publications.

        .. list-table:: Available Pre-Trained weights
           :widths: 25 75
           :header-rows: 1

           * - Key
             - Description
           * - imagenet32
             -  Pre-Trained on a downscaled version (:math:`32 \\times 32`) of the ImageNet.
           * - imagenet32-nocifar
             - Pre-Trained on a downscaled version (:math:`32 \\times 32`) of the ImageNet, excluding cifar10 classes.
           * - oe-cifar100-tune
             - Model trained with Outlier Exposure using the 80 milion TinyImages database on the CIFAR-100
           * - oe-cifar10-tune
             - Model trained with Outlier Exposure using the 80 milion TinyImages database on the CIFAR-10
           * - er-cifar10-tune
             - Model trained with Energy Regularization using the 80 milion TinyImages database on the CIFAR-10
           * - er-cifar100-tune
             - Model trained with Energy Regularization using the 80 milion TinyImages database on the CIFAR-100
           * - cifar100-pt
             - Pre-Trained model for CIFAR-100
           * - cifar10-pt
             - Pre-Trained model for CIFAR-10
           * - cifar10-pixmix
             - Model trained with PixMix on CIFAR-10. ``widen_factor=4``
           * - cifar100-pixmix
             - Model trained with PixMix on CIFAR-100. ``widen_factor=4``
        """
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            self._from_pretrained(pretrained)

    @staticmethod
    def norm_std_for(pretrained: str) -> List[float]:
        """
        Return normalization standard deviation values for pretrained model
        """
        if pretrained in ["cifar10-pt", "cifar100-pt"]:
            return [x / 255 for x in [63.0, 62.1, 66.7]]

        raise ValueError("Unknown Model")

    @staticmethod
    def transform_for(pretrained: str) -> tvt.Compose:
        """
        Return evaluation transform for pretrained model
        """
        if pretrained in ["cifar10-pt", "cifar100-pt"]:
            # Setup preprocessing
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            trans = tvt.Compose(
                [
                    tvt.Resize(size=(32, 32)),
                    ToRGB(),
                    tvt.ToTensor(),
                    tvt.Normalize(std=std, mean=mean),
                ]
            )
            return trans
        elif pretrained in ["imagenet32-nocifar"]:
            mean = [0.5] * 3
            std = [0.5] * 3
            trans = tvt.Compose(
                [
                    tvt.Resize(size=(32, 32)),
                    ToRGB(),
                    tvt.ToTensor(),
                    tvt.Normalize(std=std, mean=mean),
                ]
            )
            return trans

        raise ValueError("Unknown Model")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagate

        :param x: input images
        :return: class logits
        """
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def features_before_pool(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def forward_from_before_pool(self, x: Tensor) -> Tensor:
        out = F.avg_pool2d(x, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def features(self, x: Tensor) -> Tensor:
        """
        Extracts (flattened) features before the last fully connected layer.
        """
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out

    def feature_list(self, x: Tensor) -> List[Tensor]:
        """
        Extracts features after encoder, pooling, and fully connected layer
        """
        out_list = []
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out_list.append(out)
        out = out.view(-1, self.nChannels)
        out_list.append(self.fc(out))
        return out_list

    def _from_pretrained(self, name: str):
        """
        Load pre-trained weights
        """
        urls = {
            "imagenet32": "https://github.com/hendrycks/pre-training/raw/master/downsampled_train/snapshots/40_2/imagenet_wrn_baseline_epoch_99.pt",
            "imagenet32-nocifar": "https://github.com/hendrycks/pre-training/raw/master/uncertainty/CIFAR/snapshots/imagenet/cifar10_excluded/imagenet_wrn_baseline_epoch_99.pt",
            "oe-cifar100-tune": "https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots/oe_tune/cifar100_wrn_oe_tune_epoch_9.pt",
            "oe-cifar10-tune": "https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots/oe_tune/cifar10_wrn_oe_tune_epoch_9.pt",
            "er-cifar10-tune": "https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/energy_ft/cifar10_wrn_s1_energy_ft_epoch_9.pt",
            "er-cifar100-tune": "https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/energy_ft/cifar100_wrn_s1_energy_ft_epoch_9.pt",
            "cifar100-pt": "https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/pretrained/cifar100_wrn_pretrained_epoch_99.pt",
            "cifar10-pt": "https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
            "cifar10-pixmix": "https://cse.ovgu.de/files/cifar10-pixmix.pt",
            "cifar100-pixmix": "https://cse.ovgu.de/files/cifar100-pixmix.pt",
        }

        file_name = f"wrn-{name}.pt"

        if name in urls.keys():
            state_dict = load_state_dict_from_url(
                url=urls[name], map_location="cpu", file_name=file_name
            )

        else:
            raise ValueError(f"Unknown model identifier. Possible values are {list(urls)}")

        if "pixmix" in name:
            state_dict = state_dict["state_dict"]
            new_state_dict = copy.copy(state_dict)

            for key in state_dict.keys():
                if "conv_shortcut" in key:
                    new_state_dict[key.replace("conv_shortcut", "convShortcut")] = state_dict[key]
                    del new_state_dict[key]

            state_dict = new_state_dict

        # get last key in dict
        key = list(state_dict.keys())[-1]
        if key.startswith("module."):
            new_state_dict = {}
            for name, param in state_dict.items():
                new_state_dict[name.replace("module.", "")] = param

            state_dict = new_state_dict

        self.load_state_dict(state_dict)
