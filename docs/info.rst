General Information
**************************

Scope
-----------------------------------------

Out-of-Distribution Detection, Anomaly Detection, Novelty Detection,
Open-Set Recognition, and other related tasks share similarities in their
objectives and methodologies.
However, different researchers may use different terminologies, and there is,
to our knowledge, currently no clear consensus on the nomenclature.
Consequently, some of the terms may be used interchangeably.

This library aims to provide methods for Out-of-Distribution Detection.
However, it may also cover approaches from closely related fields,
such as Anomaly Detection or Novelty Detection.

Our goal is to provide a flexible and adaptable solution that can be easily
integrated into existing research workflows, enabling researchers
to test and compare various methods in a standardized and reproducible manner.
We intend to continue to update and expand the library to keep up with
the latest developments.


Assumptions
-------------

While PyTorch-OOD aims to be as general as possible, there are certain assumptions that we have to make.
These are as follows:


OOD as Binary Classification
==============================

PyTorch-OOD approaches Out-of-Distribution (OOD) detection as a binary
classification task with the objective of distinguishing between
in-distribution (IN) and out-of-distribution (OOD) data.
This binary classification is performed in addition to other tasks,
such as classification or segmentation.
PyTorch-OOD assumes that each OOD detector produces outlier scores,
which are numerical values that indicate the degree of outlierness of a
given sample.
While this assumption may not be applicable to some detectors,
such as OpenMax, we believe that most methods can be modified
to produce outlier scores.

Workflow
===============

We assume a workflow involving 3 Steps:

1. Training a Deep Neural Network
2. Creating an OOD detector, which is optionally fitted on some training data.
3. Evaluating the OOD detector on some benchmark dataset

Labeling
===============

PyTorch-OOD follows a labeling convention in which in-distribution data
samples are assigned target class labels greater
than or equal to zero (:math:`>= 0`). Out-of-distribution
data samples, whether known or unknown during training, are
assigned target values less than zero (:math:`< 0`).


Getting Started
****************

You can install PyTorch-OOD directly via Python Packaging Index (PyPI)

.. code-block:: shell

   pip install pytorch-ood

Editable Version
-----------------

You can install an editable version (developer version) by cloning the repository
to some directory ``<dir>`` and executing

.. code-block:: shell

   pip install -e <dir>
