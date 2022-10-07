PyTorch Out-of-Distribution Detection
######################################

PyTorch-OOD  provides implementation of methods and helper functions for Out-of-Distribution Detection based on PyTorch.
The library aims to provide modular, well-tested, and documented implementations of OOD detection methods with a unified
interface, as well as training and benchmark datasets, pre-trained models, and utility functions.

This documentation provides a user guide with general information on the
assumptions, nomenclature, as well as a structured API documentation to aid implementation.

.. warning:: The library is still work in progress. We do not claim that the provided implementations do not
    contain bugs. However, we count on the self correcting nature of open
    source software, which, as we hope, will ultimately lead to bug-free implementations.

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   info
   examples


.. toctree::
   :maxdepth: 3
   :caption: Library

   detector
   losses
   data
   models
   utils
