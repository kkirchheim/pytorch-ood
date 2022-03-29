"""
Vision Transformer

Much of this code is taken from:
https://github.com/asyml/vision-transformer-pytorch/blob/main/src/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.datasets.utils import download_file_from_google_drive


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


class MlpBlock(nn.Module):
    """Transformer Feed-Forward Block"""

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim**0.5
        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = self.out(out, dims=([2, 3], [0, 1]))
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_patches,
        emb_dim,
        mlp_dim,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
    ):
        super(Encoder, self).__init__()
        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        out = self.pos_embedding(x)
        for layer in self.encoder_layers:
            out = layer(out)
        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """
    Vision Transformer from *AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE*

    Transformer-Based architectures have also been used for OOD detection, for example in
    *OODDformer: Out-Of-Distribution Detection Transformer*.

    :see Implementation: https://github.com/asyml/vision-transformer-pytorch/blob/main/src/model.py
    :see Paper: https://arxiv.org/pdf/2010.11929.pdf
    """

    def __init__(
        self,
        image_size=(256, 256),
        patch_size=(16, 16),
        emb_dim=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        attn_dropout_rate=0.0,
        dropout_rate=0.1,
    ):
        super(VisionTransformer, self).__init__()
        h, w = image_size
        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        # classifier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x) -> torch.Tensor:
        """

        :param x: input
        :return: output tensor
        """
        emb = self.embedding(x)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)
        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)
        # transformer
        feat = self.transformer(emb)
        # classifier
        logits = self.classifier(feat[:, 0])
        return logits


class VisionTransformerPretrained(VisionTransformer):
    """
    Vision Transformer with different pre-trained weights.

    - **imagenet32**: Pre-Trained on a downscaled version (:math:`32 \\times 32`) of the ImageNet dataset.
    - **oe-cifar100-tune**: Model trained with Outlier Exposure using the 80 milion TinyImages database on the CIFAR-100 dataset
    - **oe-cifar10-tune**: Model trained with Outlier Exposure using the 80 milion TinyImages database on the CIFAR-10 dataset
    """

    urls = {
        "imagenet32": "https://github.com/hendrycks/pre-training/raw/master/downsampled_train/snapshots/40_2/imagenet_wrn_baseline_epoch_99.pt",
        "oe-cifar100-tune": "https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots/oe_tune/cifar100_wrn_oe_tune_epoch_9.pt",
        "oe-cifar10-tune": "https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots/oe_tune/cifar10_wrn_oe_tune_epoch_9.pt",
    }

    def __init__(self, pretrain, **kwargs):
        """

        :param pretrain: weights to load
        :param kwargs: arguments passed to WideResNet
        """
        super(VisionTransformerPretrained, self).__init__(**kwargs)
        url = VisionTransformerPretrained.urls[pretrain]
        state_dict = load_state_dict_from_url(url=url, map_location="cpu")
        self.load_state_dict(state_dict)
