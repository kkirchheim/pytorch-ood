Component Metadata
==================

Detectors, training objectives, datasets and benchmarks carry a class attribute ``info`` that
describes them: the paper that introduced them, the tasks they support, the role a dataset usually
plays in OOD experiments, its license, and so on. The documentation renders badges, links and
comparison tables from it, and the tests check the claims it makes. Everything that can be read
from the class itself (e.g. the base class of a detector, or whether it needs fitting) is not
repeated here.

.. code-block:: python

    from pytorch_ood import detector
    from pytorch_ood.api import Task

    segmentation_detectors = [
        cls
        for cls in vars(detector).values()
        if getattr(cls, "info", None) and Task.SEGMENTATION in cls.info.tasks
    ]


Records
-------

.. autoclass:: pytorch_ood.api.DetectorInfo
    :members:
    :member-order: bysource

.. autoclass:: pytorch_ood.api.LossInfo
    :members:
    :member-order: bysource

.. autoclass:: pytorch_ood.api.DatasetInfo
    :members:
    :member-order: bysource

.. autoclass:: pytorch_ood.api.BenchmarkInfo
    :members:
    :member-order: bysource

.. autoclass:: pytorch_ood.api.Paper
    :members:
    :member-order: bysource


Enumerations
------------

.. autoclass:: pytorch_ood.api.Task
    :members:
    :member-order: bysource

.. autoclass:: pytorch_ood.api.Representation
    :members:
    :member-order: bysource

.. autoclass:: pytorch_ood.api.Role
    :members:
    :member-order: bysource
