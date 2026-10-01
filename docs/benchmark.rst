Benchmarks
==========

Benchmark objects aim to provide a higher level interface to recreate the
OOD detection benchmarks used in the literature. All of them implement the common
:doc:`Benchmark API <benchmarks/api>`. Examples can be found
:doc:`here <auto_examples/benchmarks/index>`.

.. py:module:: pytorch_ood.benchmark

.. toctree::
   :hidden:

   benchmarks/api


ODIN
----------

.. toctree::
   :maxdepth: 1

   CIFAR-10 <benchmarks/cifar10_odin>
   CIFAR-100 <benchmarks/cifar100_odin>


OpenOOD
----------

.. toctree::
   :maxdepth: 1

   CIFAR-10 <benchmarks/cifar10_openood>
   CIFAR-100 <benchmarks/cifar100_openood>
   ImageNet <benchmarks/imagenet_openood>
   ImageNet-200 <benchmarks/imagenet200_openood>


SSB (Semantic Split Benchmark)
------------------------------

SSB divides fine-grained visual datasets (birds, cars, aircraft) into known and unknown classes,
with unknown classes further split by semantic similarity: **Easy** OOD classes are visually dissimilar
from known classes, while **Hard** OOD classes are visually similar. This enables evaluation of OOD
detection methods on both straightforward and challenging cases.

.. toctree::
   :maxdepth: 1

   CUB-200 <benchmarks/cub_ssb>
   Stanford Cars <benchmarks/stanfordcars_ssb>
   FGVC Aircraft <benchmarks/aircraft_ssb>


OpenMIBOOD
----------

The benchmarks proposed in
*OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*
(`arXiv:2503.16247 <https://arxiv.org/abs/2503.16247>`_, CVPR 2025).
Each benchmark uses a 4-way split (ID, covariate-shifted ID, near-OOD, far-OOD).
Data must be prepared first following the
`OpenMIBOOD setup guide <https://github.com/remic-othr/OpenMIBOOD>`_.

.. image:: https://raw.githubusercontent.com/remic-othr/OpenMIBOOD/main/Datasets_Summary.jpg
   :alt: OpenMIBOOD datasets overview

.. toctree::
   :maxdepth: 1

   MIDOG (microscopy / mitosis) <benchmarks/midog_openmibood>
   PhaKIR (surgical video) <benchmarks/phakir_openmibood>
   OASIS-3 (brain MRI) <benchmarks/oasis3_openmibood>
