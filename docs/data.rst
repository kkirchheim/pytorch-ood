Datasets
*************************

There are many datasets used in experiments with OOD methods. Finding, downloading and writing code for reading
these datasets can be tedious.

This package provides access to some of the most used datasets in the OOD literature. Most of these
implementations support auto-downloading with ``download=True``, which is off by default. Some data has to be
obtained manually: :class:`ImageNet200 <pytorch_ood.dataset.img.ImageNet200>` and
:class:`ImageNet800 <pytorch_ood.dataset.img.ImageNet800>` need an existing ImageNet directory,
:class:`TinyImages <pytorch_ood.dataset.img.TinyImages>` needs a local copy of the data file,
and :class:`FishyScapes <pytorch_ood.dataset.img.FishyScapes>` needs the CityScapes validation images.

Image datasets return :class:`PIL.Image.Image` objects (exceptions are noted on the respective pages), and text datasets
return :class:`str`.

.. _dataset-label-convention:

Labels
====================================================

Datasets follow the library's :ref:`labeling convention <design-choices>`: OOD samples have targets below zero.
Datasets that contain only OOD data, such as :class:`Textures <pytorch_ood.dataset.img.Textures>` or
:class:`WikiText2 <pytorch_ood.dataset.txt.WikiText2>`, return the target ``-1`` for every sample.
Some datasets, like :class:`ImageNetO <pytorch_ood.dataset.img.ImageNetO>` and
:class:`Chars74k <pytorch_ood.dataset.img.Chars74k>`, are used as OOD data but return their real class indices.
To use them as OOD data, map their targets with :class:`~pytorch_ood.utils.ToUnknown`:

.. code-block:: python

    from pytorch_ood.dataset.img import ImageNetO
    from pytorch_ood.utils import ToUnknown

    dataset = ImageNetO(root="data", target_transform=ToUnknown(), download=True)
    image, target = dataset[0]  # target is -1

In the segmentation datasets, the target is a mask in which ``-1`` marks anomalous pixels, ``0`` marks in-distribution
pixels and the ``VOID_LABEL`` (``1``) marks pixels that should be ignored. The ``transform`` of these datasets is
called with both the image and the mask, i.e. as ``transform(image, mask)``.

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
