"""

Open Set Simulations are frequently used to evaluate Open Set Recognition models.
The idea is to split a dataset with labels into subsets of known (IN) and unknown (OOD) classes.
These subsets are then used to train the model on the known classes and evaluated on known and unknown classes.

A formal description can be found in this `paper <https://arxiv.org/abs/2203.00382>`__.


.. autoclass:: pytorch_ood.dataset.ossim.DynamicOSS
   :members:


"""
from .ossim import DynamicOSS, OpenSetSimulation
