
Benchmarks
==================
The goal of this section is to describe how to quickly obtain some baseline results to compare against.

The following examples reproduces a common benchmark with 7 OOD detectors.
Each detector is tested against 5 OOD datasets. We then calculate the mean performance of each detector over all
datasets and sort the results by their AUROC in ascending order.

We test :class:`MaxSoftmax <pytorch_ood.detector.MaxSoftmax>`,
:class:`Energy-Based Out-of-Distribution Detection  <pytorch_ood.detector.EnergyBased>`,
:class:`MaxLogit <pytorch_ood.detector.MaxLogit>`,
:class:`ODIN <pytorch_ood.detector.ODIN>`,
:class:`KLMatching <pytorch_ood.detector.KLMatching>`,
:class:`ViM <pytorch_ood.detector.ViM>` and
:class:`Mahalanobis  <pytorch_ood.detector.Mahalanobis>`.

To run these examples, you have to install ``pandas`` as well as ``scikit-learn``
as additional dependencies:

.. code-block:: shell

    pip install pandas scikit-learn
