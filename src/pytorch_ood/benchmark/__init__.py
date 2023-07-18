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
from .img import CIFAR10_ODIN, CIFAR100_ODIN, CIFAR10_OpenOOD, CIFAR100_OpenOOD, ImageNet_OpenOOD
