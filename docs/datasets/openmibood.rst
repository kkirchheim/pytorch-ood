OpenMIBOOD (Dataset Collection)
===============================

The datasets used by the three medical imaging benchmarks from
*OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection* (CVPR 2025).
This is not a single dataset, but a collection of datasets from different sources, grouped below
by the benchmark they belong to.

Some of the datasets can be downloaded automatically with ``download=True``. The others require
registration or have to be preprocessed with the scripts from the
`OpenMIBOOD repository <https://github.com/remic-othr/OpenMIBOOD>`__. The documentation of each
dataset explains which applies and what to do.

..  autofunction:: pytorch_ood.dataset.img.download_all_openmibood


MIDOG
-----

Datasets of :class:`pytorch_ood.benchmark.MIDOG_OpenMIBOOD` (histopathology, mitosis classification).

..  autoclass:: pytorch_ood.dataset.img.MIDOG
    :members:

..  autoclass:: pytorch_ood.dataset.img.CCAgT
    :members:

..  autoclass:: pytorch_ood.dataset.img.FNAC2019
    :members:


PhaKIR
------

Datasets of :class:`pytorch_ood.benchmark.PhaKIR_OpenMIBOOD` (surgical video, instrument classification).

..  autoclass:: pytorch_ood.dataset.img.PhaKIR
    :members:

..  autoclass:: pytorch_ood.dataset.img.Cholec80
    :members:

..  autoclass:: pytorch_ood.dataset.img.EndoVis2018
    :members:

..  autoclass:: pytorch_ood.dataset.img.KvasirSEG
    :members:

..  autoclass:: pytorch_ood.dataset.img.CATARACTS
    :members:

EndoVis 2015, the second near-OOD dataset of this benchmark, is no longer available for download
and has no dataset class.


OASIS-3
-------

Datasets of :class:`pytorch_ood.benchmark.OASIS3_OpenMIBOOD` (brain MRI, Alzheimer's disease classification).

..  autoclass:: pytorch_ood.dataset.img.OASIS3
    :members:

..  autoclass:: pytorch_ood.dataset.img.ATLAS
    :members:

..  autoclass:: pytorch_ood.dataset.img.BraTS
    :members:

..  autoclass:: pytorch_ood.dataset.img.Task02Heart
    :members:

..  autoclass:: pytorch_ood.dataset.img.CHAOS
    :members:
