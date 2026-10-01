"""

Open Set Simulations are frequently used to evaluate Open Set Recognition models.
The idea is to split a dataset with labels into subsets of in-distribution (ID) and out-of-distribution (OOD) classes.
These subsets are then used to train the model on the ID classes and to evaluate it on ID and OOD classes.

The classes are divided into

* known known classes (KKC): the ID classes, which are seen during training,
* known unknown classes (KUC): OOD classes that are available as outliers during training, and
* unknown unknown classes (UUC): OOD classes that are only seen during validation or testing.

A formal description can be found in this `paper <https://arxiv.org/abs/2203.00382>`__.


.. autoclass:: pytorch_ood.dataset.ossim.DynamicOSS
   :members:

.. autoclass:: pytorch_ood.dataset.ossim.OpenSetSimulation
   :members:


"""

from .ossim import DynamicOSS, OpenSetSimulation
