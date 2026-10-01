"""
Benchmarks that recreate the OOD detection evaluations used in the literature.

Common Interface
----------------

Each benchmark implements a common interface.

.. note :: This is currently a draft and likely subject to change in the
    future.

.. code-block:: python

    from torch.utils.data import DataLoader

    from pytorch_ood.benchmark import CIFAR10_OpenOOD
    from pytorch_ood.detector import EnergyBased, Mahalanobis

    benchmark = CIFAR10_OpenOOD(root, transform)
    detector1 = EnergyBased(model)
    detector2 = Mahalanobis(model.features)  # requires fitting

    detector2.fit(DataLoader(benchmark.train_set(), batch_size=128))

    results1 = benchmark.evaluate(detector1, loader_kwargs={"batch_size": 128})
    results2 = benchmark.evaluate(detector2, loader_kwargs={"batch_size": 128})

``results1`` and ``results2`` are lists with one dictionary per OOD dataset (see
:meth:`Benchmark.evaluate <pytorch_ood.benchmark.Benchmark.evaluate>`).

Several detectors can also be evaluated together. Benchmark caching can reuse
intermediate logits or pooled features when evaluating multiple compatible detectors:

.. code-block:: python

    results = benchmark.evaluate(
        [detector1, detector2],
        loader_kwargs={"batch_size": 128},
        cache=True,
        cache_dir="cache/",
        cache_key="wrn-cifar10-v1",
    )

When possible, benchmarks reuse cached logits or pooled features for
``LogitsDetector`` and ``FeaturesDetector`` instances. With ``cache=True`` or
``cache_dir=...``, those cached representations are kept on the benchmark object and
can be reused across later ``evaluate(...)`` calls. With ``cache_dir=...`` and a
``cache_key``, they are also written to disk. Detectors of other kinds
(feature-map, structured, and gradient detectors) always run the full pipeline.

.. warning::

    File-backed cache reuse is keyed only by the user-supplied ``cache_key``
    and lightweight metadata. Users are responsible for changing the key when
    the model, weights, transforms, or benchmark configuration change.


..  autoclass:: pytorch_ood.benchmark.Benchmark
    :members:
"""

from .base import Benchmark
from .img import (
    CIFAR10_ODIN,
    CIFAR100_ODIN,
    CUB_SSB,
    Aircraft_SSB,
    CIFAR10_OpenOOD,
    CIFAR100_OpenOOD,
    ImageNet1K_OpenOOD,
    ImageNet200_OpenOOD,
    ImageNet_OpenOOD,
    MIDOG_OpenMIBOOD,
    OASIS3_OpenMIBOOD,
    PhaKIR_OpenMIBOOD,
    StanfordCars_SSB,
)

__all__ = [
    "Benchmark",
    "CIFAR10_ODIN",
    "CIFAR100_ODIN",
    "CIFAR10_OpenOOD",
    "CIFAR100_OpenOOD",
    "ImageNet_OpenOOD",
    "ImageNet1K_OpenOOD",
    "ImageNet200_OpenOOD",
    "MIDOG_OpenMIBOOD",
    "PhaKIR_OpenMIBOOD",
    "OASIS3_OpenMIBOOD",
    "CUB_SSB",
    "StanfordCars_SSB",
    "Aircraft_SSB",
]
