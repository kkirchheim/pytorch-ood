"""
Backwards-compatibility shim. The ImageNet OpenOOD benchmarks now live in
:mod:`pytorch_ood.benchmark.img.openood` (imglist-driven, exact OpenOOD v1.5). This
module re-exports them so existing ``from ...img.imagenet import ...`` imports keep
working.
"""

from .openood import ImageNet1K_OpenOOD, ImageNet_OpenOOD, ImageNet200_OpenOOD

__all__ = ["ImageNet_OpenOOD", "ImageNet1K_OpenOOD", "ImageNet200_OpenOOD"]
