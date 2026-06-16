Hyperparameter Optimization
***************************

This section demonstrates how to tune detector hyperparameters with
:class:`GridSearch <pytorch_ood.utils.GridSearch>`, following the
OpenOOD-style automatic parameter search protocol: candidate values are
evaluated on a held-out validation set containing both in-distribution and
out-of-distribution data, and the combination that maximizes the validation
metric (AUROC by default) is selected.

To run these examples, you have to install ``scikit-learn`` as an additional
dependency:

.. code-block:: shell

    pip install scikit-learn
