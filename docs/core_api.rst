Core API
*************************

The contracts the rest of the library is built on: the interfaces that detectors and
benchmarks implement, the metadata every component carries, and the exceptions they raise.
Most of it lives in :mod:`pytorch_ood.api`; implement these interfaces to add components of
your own.

.. py:module:: pytorch_ood.api

.. toctree::
   :maxdepth: 1

   core_api/detectors
   core_api/benchmarks
   core_api/metadata
   core_api/exceptions
