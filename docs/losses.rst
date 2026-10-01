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

Therefore, all of these loss functions expect that the target labels are strictly :math:`\geq 0`.

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

Supervised Losses make use from example Out-of-Distribution samples (or samples from known unknown classes).
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
