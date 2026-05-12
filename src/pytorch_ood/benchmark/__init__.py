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


Image
==================

Examples can be found :doc:`here <auto_examples/benchmarks/index>`


ODIN
^^^^^^

CIFAR-10
---------

.. autoclass:: pytorch_ood.benchmark.CIFAR10_ODIN
    :members:


CIFAR-100
---------

.. autoclass:: pytorch_ood.benchmark.CIFAR100_ODIN
    :members:


OpenOOD
^^^^^^^

CIFAR-10
---------

.. autoclass:: pytorch_ood.benchmark.CIFAR10_OpenOOD
    :members:


CIFAR-100
---------

.. autoclass:: pytorch_ood.benchmark.CIFAR100_OpenOOD
    :members:


ImageNet
---------

.. autoclass:: pytorch_ood.benchmark.ImageNet_OpenOOD
    :members:


SSB (Semantic Split Benchmark)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUB-200
---------

.. autoclass:: pytorch_ood.benchmark.CUB_SSB
    :members:


Stanford Cars
---------

.. autoclass:: pytorch_ood.benchmark.StanfordCars_SSB
    :members:


FGVC Aircraft
---------

.. autoclass:: pytorch_ood.benchmark.Aircraft_SSB
    :members:


OpenMIBOOD
^^^^^^^^^^

The benchmarks proposed in
*OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*
(`arXiv:2503.16247 <https://arxiv.org/abs/2503.16247>`_, CVPR 2025).
Each benchmark uses a 4-way split (ID, covariate-shifted ID, near-OOD, far-OOD).
Data must be prepared first following the
`OpenMIBOOD setup guide <https://github.com/remic-othr/OpenMIBOOD>`_.

.. image:: https://raw.githubusercontent.com/remic-othr/OpenMIBOOD/main/Datasets_Summary.jpg
   :alt: OpenMIBOOD datasets overview

MIDOG (microscopy / mitosis)
-----------------------------

.. autoclass:: pytorch_ood.benchmark.MIDOG_OpenMIBOOD
    :members:

PhaKIR (surgical video)
-----------------------------

.. autoclass:: pytorch_ood.benchmark.PhaKIR_OpenMIBOOD
    :members:

OASIS-3 (brain MRI)
-----------------------------

.. autoclass:: pytorch_ood.benchmark.OASIS3_OpenMIBOOD
    :members:


"""

from .base import Benchmark
from .img import (
    CIFAR10_ODIN,
    CIFAR100_ODIN,
    CIFAR10_OpenOOD,
    CIFAR100_OpenOOD,
    ImageNet_OpenOOD,
    MIDOG_OpenMIBOOD,
    OASIS3_OpenMIBOOD,
    PhaKIR_OpenMIBOOD,
    CUB_SSB,
    StanfordCars_SSB,
    Aircraft_SSB,
)

__all__ = [
    "Benchmark",
    "CIFAR10_ODIN",
    "CIFAR100_ODIN",
    "CIFAR10_OpenOOD",
    "CIFAR100_OpenOOD",
    "ImageNet_OpenOOD",
    "MIDOG_OpenMIBOOD",
    "PhaKIR_OpenMIBOOD",
    "OASIS3_OpenMIBOOD",
    "CUB_SSB",
    "StanfordCars_SSB",
    "Aircraft_SSB",
]
