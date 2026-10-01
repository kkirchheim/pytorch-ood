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

PyTorch-OOD aims to provide well-tested implementations of methods for Out-of-Distribution Detection.
However, it may also cover approaches from closely related fields,
such as Anomaly Detection or Novelty Detection.

Experimental Workflow
=========================

OOD Detection Experiments usually involve the following steps:


1. Training a Deep Neural Network.
2. Creating an OOD detector, which is optionally fitted on some training data.
3. Evaluating the OOD detector on some benchmark dataset.


.. _design-choices:

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
given sample, i.e., higher scores mean that the sample is more likely OOD.

While this assumption may not be applicable to some methods, we believe that most
methods can be modified to produce outlier scores. For example, the
:class:`OpenMax <pytorch_ood.detector.OpenMax>` detector is exposed through this interface
by using the probability of the unknown class as outlier score.


3) OOD Points have Negative Labels
===================================

PyTorch-OOD follows a labeling convention in which in-distribution data
samples (also called *known* data) are assigned target class labels greater
than or equal to zero (:math:`>= 0`). Out-of-distribution
data samples (also called *unknown* data), whether available during training or not, are
assigned target values less than zero (:math:`< 0`).


Other design features
========================
We aim to make usage user-friendly.
Sometimes, this comes at the price of performance.

In some cases, we might, for example, move tensors from one device to another so that computations
do not throw exceptions because of a device mismatch. While letting users manage tensor device placement on their own could lead to
better performance, it would place more burden on them.
