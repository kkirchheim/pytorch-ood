"""The exported, concrete components of each package, for tests that check all of them."""

import inspect

from torch import nn
from torch.utils.data import Dataset

from pytorch_ood import benchmark, detector, loss
from pytorch_ood.api import Detector
from pytorch_ood.dataset import audio, img, txt


def _exported(module, base, prefix):
    # a set of objects, so re-exported aliases count once
    return sorted(
        {
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, base)
            and not inspect.isabstract(obj)
            and not obj.__name__.startswith("_")
            and obj.__module__.startswith(prefix)
        },
        key=lambda cls: cls.__name__,
    )


DETECTORS = _exported(detector, Detector, "pytorch_ood.detector")
LOSSES = _exported(loss, nn.Module, "pytorch_ood.loss")
BENCHMARKS = _exported(benchmark, benchmark.Benchmark, "pytorch_ood.benchmark")
# generic loaders for the user's own data describe no dataset of their own
DATASETS = [
    cls
    for package in (img, txt, audio)
    for cls in _exported(package, Dataset, "pytorch_ood.dataset")
    if cls is not img.ImageListDataset
]
