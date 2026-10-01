Getting Started
****************


Setting up Environment
------------------------
pytorch-ood runs on Python 3.8 or newer and any recent PyTorch. Install PyTorch first, with the
command for your platform and CUDA version from the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`__, for example in a fresh
virtual environment or conda environment.

Installing
----------------------

Installing from PyPI
======================


You can install the latest stable version directly via Python Packaging Index (PyPI)

.. code-block:: shell

   pip install pytorch-ood

Some components need optional dependencies (scikit-learn for e.g. KNN and ViM, gdown for some
dataset downloads). To install them as well:

.. code-block:: shell

   pip install "pytorch-ood[all]"


Installing from Git
======================

To install the latest ``dev`` branch directly from git:

.. code-block:: shell

    pip install git+https://github.com/kkirchheim/pytorch-ood.git@dev



Editable Version
======================

You can install an editable version (developer version) with

.. code-block:: shell

   git clone https://github.com/kkirchheim/pytorch-ood
   cd pytorch-ood
   pip install -e .


Building Documentation
========================

To build the documentation, run

.. code-block:: shell

    pip install -r docs/requirements.txt
    cd docs
    make html


Quick Start
-----------------------------------------

You can find a lot of minimal examples :doc:`here <auto_examples/benchmarks/index>`.
