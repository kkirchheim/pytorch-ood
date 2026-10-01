Datasets
*************************

There are many datasets used in experiments with OOD methods. Finding, downloading and writing code for reading
these datasets can be tedious.

This package provides access to some of the most used datasets in the OOD literature. Most of these
implementations support auto-downloading.

.. py:module:: pytorch_ood.dataset.img
.. py:module:: pytorch_ood.dataset.txt
.. py:module:: pytorch_ood.dataset.audio


Image
====================================================

Classification
----------------------------------------------------

Contains datasets often used in anomaly detection, where the entire input is labeled as either ID or OOD.

.. toctree::
   :maxdepth: 1

   datasets/textures
   datasets/tinyimagenetcrop
   datasets/tinyimagenetresize
   datasets/lsuncrop
   datasets/lsunresize
   datasets/tinyimagenet
   datasets/places365
   datasets/tinyimages
   datasets/tinyimages300k
   datasets/imageneta
   datasets/imageneto
   datasets/imagenetr
   datasets/imagenet200
   datasets/imagenet800
   datasets/imagenetv2
   datasets/imagenetes
   datasets/mnistc
   datasets/cifar10c
   datasets/cifar100c
   datasets/cifar100gan
   datasets/imagenetc
   datasets/openimageso
   datasets/inaturalist
   datasets/ssbhard
   datasets/chars74k
   datasets/fractaldataset
   datasets/foolingimages
   datasets/ninco
   datasets/featurevisdataset
   datasets/gaussiannoise
   datasets/uniformnoise


Segmentation
----------------------------------------------------

.. toctree::
   :maxdepth: 1

   datasets/streethazards
   datasets/fishyscapes
   datasets/lostandfound
   datasets/roadanomaly
   datasets/segmentmeifyoucan
   datasets/mvtechad


Object Detection
----------------------------------------------------

.. toctree::
   :maxdepth: 1

   datasets/sumnist


Text
====================================================

.. toctree::
   :maxdepth: 1

   datasets/newsgroup20
   datasets/reuters8
   datasets/reuters52
   datasets/multi30k
   datasets/wmt16sentences
   datasets/wikitext2
   datasets/wikitext103


Audio
====================================================

.. toctree::
   :maxdepth: 1

   datasets/fsdd


Video
=====================================================

.. note :: There are, to our knowledge, no video datasets for OOD detection available.
            If you are aware of any, please, let us know.


Open Set Simulations
====================================================

.. toctree::
   :maxdepth: 1

   datasets/ossim


Custom Data
====================================================

Loaders for your own data in formats used by other benchmark suites.

.. toctree::
   :maxdepth: 1

   datasets/imagelistdataset
