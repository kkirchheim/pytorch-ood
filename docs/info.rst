General Information
**************************


Terminology & Scope
-----------------------------------------

Out-of-Distribution Detection, Anomaly Detection, Novelty Detection,
Open-Set Recognition, and other related tasks share similarities in their
objectives and methodologies.
However, different researchers may use different terminologies, and there is,
to our knowledge, currently no clear consensus on the nomenclature.
Consequently, some of the terms may be used interchangeably.

The survey paper `Generalized Out-of-Distribution Detection: A Survey <https://arxiv.org/abs/2110.11334>`__
presents a possible nomenclature.

PyTorch-OOD aims to provide well tested implementation of methods for Out-of-Distribution Detection.
However, it may also cover approaches from closely related fields,
such as Anomaly Detection or Novelty Detection.

Experimental Workflow
=========================

OOD Detection Experiments usually involve the following steps:


1. Training a Deep Neural Network.
2. Creating an OOD detector, which is optionally fitted on some training data.
3. Evaluating the OOD detector on some benchmark dataset.


Design Choices
-----------------

Our goal is to provide a flexible and adaptable solution that can be easily
integrated into the entire workflow, enabling users
to test and compare various methods in a standardized and reproducible manner.

While PyTorch-OOD aims to be as general as possible, there are certain assumptions that we have to make.
These are as follows:


1) OOD Detection is Binary Classification
==========================================

PyTorch-OOD approaches Out-of-Distribution (OOD) detection as a binary
classification task with the objective of distinguishing between
in-distribution (ID) and out-of-distribution (OOD) data.
This binary classification is performed in addition to other tasks,
such as classification or segmentation.

2) Detectors predict Outlier Scores
===================================
PyTorch-OOD assumes that each OOD detector produces outlier scores,
which are numerical values that indicate the degree of outlierness of a
given sample, i.e., higher scores means higher certainty that it is OOD.

While this assumption may not be applicable to some detectors,
such as OpenMax, we believe that most methods can be modified
to produce outlier scores.


3) OOD Points have Negative Labels
===================================

PyTorch-OOD follows a labeling convention in which in-distribution data
samples are assigned target class labels greater
than or equal to zero (:math:`>= 0`). Out-of-distribution
data samples, whether known or unknown during training, are
assigned target values less than zero (:math:`< 0`).


Other design features
========================
We aim to make usage user friendly.
Sometimes, this comes at the price of performance.

In some cases, we might, for example, move tensors from one device to another so that computations
do not throw exceptions because of a device mismatch. While letting users manage tensor device placement on their own could lead to
better performance, it would place more burden on them.


Getting Started
****************


Setting up Environment
------------------------
It is recommended to set up an anaconda environment with

.. code-block:: shell

    conda install pytorch torchvision torchaudio torchtext==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia

Installing
----------------------

Installing from PyPI
======================


You can install the latest stable version directly via Python Packaging Index (PyPI)

.. code-block:: shell

   pip install pytorch-ood


Installing from Git
======================

To install the latest ``dev`` branch directly from git:

.. code-block:: shell

    pip install git+ssh://git@github.com/kkirchheim/pytorch-ood.git@dev



Editable Version
======================

You can install an editable version (developer version) with

.. code-block:: shell

   git clone https://github.com/kkirchheim/pytorch-ood
   cd pytorch-ood
   pip install -e .


Building Documentation
========================

To build the documentation, run

.. code-block:: shell

    pip install sphinx_gallery sphinx_rtd_theme sphinx
    cd docs
    make html


Quick Start
-----------------------------------------

You can find a lot of minimal examples :doc:`here <auto_examples/benchmarks/index>`.
