"""
Benchmarks
******************

Benchmark objects aim to provide a higher level interface to recreate the
OOD detection benchmarks used in the literature.


API
==================

Each benchmark implements a common interface.

.. note :: This is currently a draft and likely subject to change in the
    future.

.. code:: python

    benchmark = Benchmark(root)
    detector = Detector(model)
    detector.fit(benchmark.train_set())

    results1 = benchmark.evaluate(detector1)
    results2 = benchmark.evaluate(detector2)

Several detectors can also be evaluated together:

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


Image
==================

Examples can be found :doc:`here <auto_examples/benchmarks/index>`


CIFAR 10
^^^^^^^^

ODIN Benchmark
-----------------

.. autoclass:: pytorch_ood.benchmark.CIFAR10_ODIN
    :members:


OpenOOD Benchmark
-----------------

.. autoclass:: pytorch_ood.benchmark.CIFAR10_OpenOOD
    :members:


CIFAR 100
^^^^^^^^^^^

ODIN Benchmark
-----------------

.. autoclass:: pytorch_ood.benchmark.CIFAR100_ODIN
    :members:

OpenOOD Benchmark
-----------------

.. autoclass:: pytorch_ood.benchmark.CIFAR100_OpenOOD
    :members:


ImageNet
^^^^^^^^^^^

OpenOOD Benchmark
-----------------

.. autoclass:: pytorch_ood.benchmark.ImageNet_OpenOOD
    :members:


"""

from .base import Benchmark
from .img import (
    CIFAR10_ODIN,
    CIFAR100_ODIN,
    CIFAR10_OpenOOD,
    CIFAR100_OpenOOD,
    ImageNet_OpenOOD,
)

__all__ = [
    "Benchmark",
    "CIFAR10_ODIN",
    "CIFAR100_ODIN",
    "CIFAR10_OpenOOD",
    "CIFAR100_OpenOOD",
    "ImageNet_OpenOOD",
]
