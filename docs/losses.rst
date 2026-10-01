Training Objectives
*************************

Objective (or loss) functions for OOD detection usually aim to improve the
discriminability of ID and OOD points in the output or latent space of the model.

.. automodule:: pytorch_ood.loss
   :no-members:


Unsupervised
========================================

Unsupervised losses only use in-distribution data (or similarly, only on
examples from "known known" classes.)

Therefore, these loss functions do not require OOD samples. Samples with labels :math:`< 0`, if present, are
discarded with a warning (see :func:`~pytorch_ood.utils.drop_unknown`), so the loss equals the loss on the batch
without them. With ``reduction="none"``, the output has one entry per ID sample, except for
:class:`~pytorch_ood.loss.CrossEntropyLoss`, which keeps the shape of the targets (to support segmentation) and
returns zero for OOD entries.

.. toctree::
   :maxdepth: 1

   losses/deepsvddloss
   losses/cacloss
   losses/iiloss
   losses/centerloss
   losses/crossentropyloss
   losses/confidenceloss
   losses/logitnorm
   losses/virtualoutliersynthesizingregloss


Supervised
========================================

Supervised Losses make use of example Out-of-Distribution samples (or samples from known unknown classes).
Thus, these losses can handle samples with target values :math:`< 0`.

.. toctree::
   :maxdepth: 1

   losses/outlierexposureloss
   losses/entropicopensetloss
   losses/objectosphereloss
   losses/energyregularizedloss
   losses/vosregloss
   losses/mchadloss
   losses/deepsadloss
   losses/backgroundclassloss
   losses/energymarginloss
