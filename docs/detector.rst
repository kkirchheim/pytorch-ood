Detectors
*************************

.. py:module:: pytorch_ood.detector

Out-of-Distribution detectors, grouped by method family. All of them implement the
common :doc:`detector interface </core_api/detectors>`: ``fit`` on in-distribution data (only where a detector
requires it), then call the detector to get outlier scores, which are larger for outliers.


Comparison
-------------------------------

.. include:: generated/detector_table.rst


Probability-based
-------------------------------

Probability-based methods are based on the observation that OOD inputs tend to be assigned lower posteriors with higher
entropy, i.e., the predicted distribution is often less concentrated on a single class.

.. toctree::
   :maxdepth: 1

   detectors/softmax
   detectors/mcd
   detectors/tscaling
   detectors/klmatching
   detectors/entropy
   detectors/gen


Logit-based
-------------------------------

Logit-based methods are based on the observation that OOD inputs tend to yield different logits compared to ID data.

.. toctree::
   :maxdepth: 1

   detectors/maxlogit
   detectors/openmax
   detectors/energy
   detectors/webo


Feature-based
-------------------------------

.. toctree::
   :maxdepth: 1

   detectors/mahalanobis
   detectors/mmahalanobis
   detectors/rmd
   detectors/vim
   detectors/knn
   detectors/nnguide
   detectors/she
   detectors/gram
   detectors/nci
   detectors/fdbd
   detectors/gmm
   detectors/pnml
   detectors/mcm
   detectors/lts


Gradient-based
-------------------------------

Gradient-based detectors are based on the observation that the gradients (w.r.t. the model parameters or
the inputs) for ID and OOD data behave differently. All gradient-based detectors inherit from
:class:`pytorch_ood.api.GradientDetector`.

.. toctree::
   :maxdepth: 1

   detectors/gradnorm
   detectors/gradnormkl
   detectors/odin
   detectors/mahalanobis_odin
   detectors/nac


Activation Pruning
-------------------------------

Activation pruning methods are based on the observation that OOD inputs cause unusual activations in the model,
and that, by rectifying these unusual activations, we can often improve discriminability of ID and OOD samples.

.. toctree::
   :maxdepth: 1

   detectors/ash
   detectors/react
   detectors/dice
   detectors/rankfeat
   detectors/vra
   detectors/scale
