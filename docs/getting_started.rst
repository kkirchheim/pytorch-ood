Getting Started
****************


Setting up Environment
------------------------
It is recommended to set up an anaconda environment with

.. code-block:: shell

    conda install pytorch torchvision torchaudio torchtext==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia

Installing
----------------------

Installing from PyPI
======================


You can install the latest stable version directly via Python Packaging Index (PyPI)

.. code-block:: shell

   pip install pytorch-ood


Installing from Git
======================

To install the latest ``dev`` branch directly from git:

.. code-block:: shell

    pip install git+ssh://git@github.com/kkirchheim/pytorch-ood.git@dev



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
