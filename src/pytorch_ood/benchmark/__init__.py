"""
Benchmarks that recreate the OOD detection evaluations used in the literature.

Common Interface
----------------

Each benchmark implements a common interface.

.. note :: This is currently a draft and likely subject to change in the
    future.

.. code:: python

    benchmark = Benchmark(root)
    detector = Detector(model)
    detector.fit(benchmark.train_set())

    results1 = benchmark.evaluate(detector1)
    results2 = benchmark.evaluate(detector2)

Several detectors can also be evaluated together. Benchmark caching can reuse
intermediate logits or pooled features when evaluating multiple compatible detectors:

.. code:: python

    results = benchmark.evaluate(
        [detector1, detector2],
        cache=True,
        cache_dir="cache/",
        cache_key="wrn-cifar10-v1",
    )

When possible, benchmarks reuse cached logits or pooled features for
``LogitsDetector`` and ``FeaturesDetector`` instances. With ``cache=True``,
those cached representations are kept on the benchmark object and can be
reused across later ``evaluate(...)`` calls. With ``cache_dir=...``, they
can also be written to disk.

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
