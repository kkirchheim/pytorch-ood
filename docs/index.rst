:hide-toc:

.. The landing page. Layout comes from sphinx-design (grids, cards, tabs) plus the
   "landing" rules in _static/custom.css; the |n_*| counts are computed in conf.py.

==========================================
PyTorch Out-of-Distribution Detection
==========================================

.. container:: landing-hero

   .. container:: landing-lead

      Modular, tested, and well-documented implementations of Out-of-Distribution (OOD)
      detection methods, with a unified interface, pre-trained models, datasets, and
      benchmarks. Also covers closely related fields such as Open-Set Recognition,
      Novelty Detection, Confidence Estimation and Anomaly Detection.

   .. container:: landing-actions

      .. button-ref:: getting_started
         :ref-type: doc
         :color: primary
         :class: landing-button

         Get started

      .. button-ref:: detector
         :ref-type: doc
         :color: secondary
         :outline:
         :class: landing-button

         Browse detectors

      .. button-link:: https://github.com/kkirchheim/pytorch-ood
         :color: secondary
         :outline:
         :class: landing-button

         :octicon:`mark-github;1em;sd-mr-1` GitHub

.. grid:: 2 2 4 4
   :gutter: 2
   :class-container: landing-stats

   .. grid-item::

      .. div:: landing-stat

         .. div:: landing-stat-value

            |n_detectors|

         detectors

   .. grid-item::

      .. div:: landing-stat

         .. div:: landing-stat-value

            |n_losses|

         training objectives

   .. grid-item::

      .. div:: landing-stat

         .. div:: landing-stat-value

            |n_datasets|

         datasets

   .. grid-item::

      .. div:: landing-stat

         .. div:: landing-stat-value

            |n_models|

         pre-trained models


Quick Start
===========

Install the package from PyPI:

.. code-block:: shell

   pip install pytorch-ood

Every detector maps inputs to an **outlier score** that is larger for outliers, and
out-of-distribution samples carry **labels below zero**. With those two conventions,
evaluating a detector takes a few lines.

.. tab-set::

   .. tab-item:: Evaluate a detector

      Load a WideResNet-40 pre-trained on CIFAR-10 and score your data with the
      Energy-based detector:

      .. code-block:: python

         from pytorch_ood.detector import EnergyBased
         from pytorch_ood.model import load_model, load_transform
         from pytorch_ood.utils import OODMetrics

         data_loader = ...  # your data, OOD samples with label < 0

         model = load_model("wrn-40-2/cifar10/energy/s1").cuda()
         preprocess = load_transform("wrn-40-2/cifar10/energy/s1")

         detector = EnergyBased(model)
         metrics = OODMetrics()

         for x, y in data_loader:
             metrics.update(detector(preprocess(x).cuda()), y)

         print(metrics.compute())

   .. tab-item:: Run a benchmark

      Compare detectors on the OpenOOD v1.5 CIFAR-10 benchmark. All datasets are
      downloaded automatically, and model outputs are computed once and shared
      by all detectors:

      .. code-block:: python

         from pytorch_ood.benchmark import CIFAR10_OpenOOD
         from pytorch_ood.detector import EnergyBased, MaxSoftmax
         from pytorch_ood.model import load_model, load_transform

         model = load_model("wrn-40-2/cifar10/crossentropy").to("cuda:0")
         trans = load_transform("wrn-40-2/cifar10/crossentropy")

         benchmark = CIFAR10_OpenOOD(root="data", transform=trans)
         results = benchmark.evaluate(
             [MaxSoftmax(model), EnergyBased(model)], device="cuda:0", cache=True
         )

         for row in results:  # one row per detector and OOD dataset
             print(row)


Explore the Library
===================

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: landing-cards

   .. grid-item-card:: :octicon:`telescope;1.2em;sd-mr-2` Detectors
      :link: detector
      :link-type: doc

      Post-hoc OOD detectors: probability-, logit-, feature-, and gradient-based
      methods, plus activation pruning.

   .. grid-item-card:: :octicon:`flame;1.2em;sd-mr-2` Training Objectives
      :link: losses
      :link-type: doc

      Losses that make models better at separating known from unknown inputs,
      supervised and unsupervised.

   .. grid-item-card:: :octicon:`database;1.2em;sd-mr-2` Datasets
      :link: data
      :link-type: doc

      Image, text, and audio OOD datasets that download on first use,
      torchvision-style.

   .. grid-item-card:: :octicon:`cpu;1.2em;sd-mr-2` Models
      :link: models
      :link-type: doc

      Architectures from the literature and a registry of pre-trained,
      checksum-verified weights.

   .. grid-item-card:: :octicon:`graph;1.2em;sd-mr-2` Benchmarks
      :link: benchmark
      :link-type: doc

      Reproduce OpenOOD, OpenMIBOOD, SSB, and ODIN evaluations with one call,
      with cached representations.

   .. grid-item-card:: :octicon:`tools;1.2em;sd-mr-2` Utilities
      :link: utils
      :link-type: doc

      Metrics, feature extraction, hyperparameter search, and transforms that
      tie the pieces together.

.. container:: landing-more

   Looking for complete scripts? Browse the :doc:`examples <auto_examples/detectors/index>`,
   or use the :ref:`index <genindex>` and :ref:`module index <modindex>` to look up a name.


Citing PyTorch-OOD
==================

If you use PyTorch-OOD in your research, please cite
`the paper <https://openaccess.thecvf.com/content/CVPR2022W/HCIS/papers/Kirchheim_PyTorch-OOD_A_Library_for_Out-of-Distribution_Detection_Based_on_PyTorch_CVPRW_2022_paper.pdf>`__:

.. code-block:: bibtex

   @inproceedings{kirchheim2022pytorch,
     title={Pytorch-ood: A library for out-of-distribution detection based on pytorch},
     author={Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     pages={4351--4360},
     year={2022}
   }


.. The toctrees below are not shown on the page; they define the sidebar navigation.

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:

   info
   getting_started
   support

.. toctree::
   :maxdepth: 2
   :caption: Library
   :hidden:

   detector
   losses
   data
   augmentations
   models
   utils
   benchmark

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   auto_examples/benchmarks/index
   auto_examples/detectors/index
   auto_examples/loss/index
   auto_examples/segmentation/index
   auto_examples/text/index
   auto_examples/osr/index
   auto_examples/metrics/index
   auto_examples/hpo/index
