Image Benchmarks
====================

The objective of this section is to outline a quick method for obtaining
baseline results for comparison purposes.

We provide an example that replicates a
commonly used benchmark that includes 7 Out-of-Distribution detectors,
each tested against 5 OOD datasets.
We subsequently calculate the average performance of each detector
across all datasets and sort the outcomes based on their
Area Under Receiver Operating Characteristic (AUROC) score in ascending order.

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

    pip install pandas scikit-learn pandas
