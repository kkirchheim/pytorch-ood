Model Registry
==============

The registry provides pre-trained weights together with the input preprocessing they were trained
with, so experiments from the literature can be reproduced without re-training. Models are
identified by a string of the form ``arch/dataset/loss/seed`` (the seed is omitted when unknown)
and are loaded with a single call:

.. code-block:: python

    from pytorch_ood.model import load_model, load_transform

    model = load_model("wrn-40-2/cifar10/logitnorm/s0")
    transform = load_transform("wrn-40-2/cifar10/logitnorm/s0")

Weights are downloaded on first use, verified against a SHA-256 checksum, and cached in the
``torch.hub`` directory (``$TORCH_HOME``). Most of them are hosted in the
`pytorch-ood-models <https://huggingface.co/kkirchheim/pytorch-ood-models>`__ repository on
Hugging Face; weights published by third parties are downloaded from their original location,
which is shown next to the model.


Available Models
----------------

Models are grouped by the dataset they were trained on. To load a model, append one of the listed
seeds to its identifier, e.g. ``wrn-40-2/cifar10/oe/s0``; models without seeds are loaded by their
identifier as is. Accuracy is the top-1 test accuracy of the released checkpoint, given as a range
across seeds.

.. include:: ../generated/model_registry.rst


API
---

..  autofunction:: pytorch_ood.model.load_model

..  autofunction:: pytorch_ood.model.load_transform

..  autofunction:: pytorch_ood.model.get_model_info

..  autofunction:: pytorch_ood.model.list_models

..  autoclass:: pytorch_ood.model.ModelEntry
    :members:

..  autoclass:: pytorch_ood.model.Preprocessing
    :members:

..  autoclass:: pytorch_ood.model.ImagePreprocessing
    :members:
