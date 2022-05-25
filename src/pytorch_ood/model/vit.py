"""
Vision Transformer

Much of this code is taken from:
https://github.com/asyml/vision-transformer-pytorch/blob/main/src/model.py
"""

from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


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
    Vision Transformer from *An Image is worth 16x16 words: Transformers for Image Recognition at Scale*

    Transformer-Based architectures have also been used for OOD detection, for example in
    *OODDformer: Out-Of-Distribution Detection Transformer*.

    .. warning :: PyTorch adds vision transformers in v. 0.12.

    :see Implementation: https://github.com/asyml/vision-transformer-pytorch/
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
        pretrained=None,
    ):
        """
        :param image_size:
        :param patch_size:
        :param emb_dim:
        :param mlp_dim:
        :param num_heads:
        :param num_layers:
        :param num_classes:
        :param attn_dropout_rate:
        :param dropout_rate:
        :param pretrained: Idenfitier for pre-trained weights to load


        .. list-table:: Available models
           :widths: 25 75
           :header-rows: 1

           * - Key
             - Description
           * - b16-cifar10-tune
             - b16 trained on ImageNet 21k and fine tuned on the CIFAR10
           * - b16-cifar100-tune
             - b16 trained on ImageNet 21k and fine tuned on the CIFAR100

        """
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

    def _from_pretrained(self, name, **kwargs):
        """
        Vision Transformer with different pre-trained weights.



        .. note :: The original authors of the OODFormer did not provide weights for their final models. The
            weights for CIFAR-10 and CIFAR-100 here are provided by the maintainers of pytorch-ood.

        """
        urls = {
            "b16-cifar10-tune": "https://cse.ovgu.de/files/b16-cifar10-tune.pth",
            "b16-cifar100-tune": "https://cse.ovgu.de/files/b16-cifar100-tune.pth",
            "b16-im21k-224": "https://cse.ovgu.de/files/imagenet21k+imagenet2012_ViT-B_16-224.pth",
            "b16-im21k": "https://cse.ovgu.de/files/imagenet21k+imagenet2012_ViT-B_16.pth",
            "l16-im21k-224": "https://cse.ovgu.de/files/imagenet21k+imagenet2012_ViT-L_16-224.pth",
            "l16-im21k": "https://cse.ovgu.de/files/imagenet21k+imagenet2012_ViT-L_16.pth",
        }

        url = urls[name]

        state_dict = load_state_dict_from_url(url, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.load_state_dict(state_dict)
