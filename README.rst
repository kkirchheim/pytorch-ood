PyTorch Out-of-Distribution Detection 
=====================================

.. image:: https://img.shields.io/pypi/v/pytorch-ood.svg?color=brightgreen
   :target: https://pypi.org/project/pytorch-ood/
   :alt: PyPI version


.. image:: https://img.shields.io/badge/-Python 3.8+-blue?logo=python&logoColor=white
   :target: https://www.python.org/
   :alt: Python


.. image:: https://img.shields.io/badge/code%20style-black-black.svg?labelColor=gray
   :target: https://black.readthedocs.io/en/stable/
   :alt: Code style: black


.. image:: https://static.pepy.tech/badge/pytorch-ood
   :target: 
   :alt: 


.. image:: https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/pipeline.svg
   :target: 
   :alt: 


.. image:: https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/coverage.svg
   :target: 
   :alt: 

.. image:: https://readthedocs.org/projects/pytorch-ood/badge/?version=latest
   :target: https://pytorch-ood.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

-----

Python library to accelerate research in Out-of-Distribution Detection, as well as related
fields such as Open-Set Recognition, Novelty Detection, Confidence Estimation and Anomaly Detection
based on Deep Neural Networks (with PyTorch).

This library provides

- Objective Functions
- OOD Detection Methods
- Datasets used in academic literature
- Neural Network Architectures used in academic literature, as well as pretrained weights
- Useful Utilities

and was created with the aim to speed up research and to facilitate reproducibility.
It is designed such that it should integrate seamlessly with frameworks that enable the scaling of model training,
like `pytorch-lightning <https://www.pytorchlightning.ai>`_.  


Installation
^^^^^^^^^^^^^^

.. code-block:: shell

   pip install pytorch-ood
   


Required Dependencies


* torch
* torchvision
* scipy
* torchmetrics


Optional Dependencies


* libmr for the OpenMax Detector, which is currently broken. You will have to install cython and libmr afterwards manually.
* pandas for the Cub200 Dataset


Quick Start
^^^^^^^^^^^
Load model pre-trained with energy regularization, and predict on some dataset `data_loader` using
Energy-based outlier scores.

.. code-block:: python

    from pytorch_ood.model import WideResNet
    from pytorch_ood import NegativeEnergy
    from pytorch_ood.utils import OODMetrics

    # create Neural Network
    model = WideResNet(pretrained="er-cifar10-tune").eval().cuda()

    # create detector
    detector = NegativeEnergy(model)

    # evaluate
    metrics = OODMetrics()

    for x, y in data_loader:
        metrics.update(detector(x.cuda()), y)

    print(metrics.compute())


Citing
^^^^^^^

pytorch-ood was presented on the CVPR Workshop on Human-centered Intelligent Services: Safe and Trustworthy.
If you use pytorch-ood in a scientific publication, please consider citing us::

    @article{kirchheim2022pytorch,
      author = {Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
      journal = {CVPR Workshop on Human-centered Intelligent Services: Safe and Trustworthy},
      number = {},
      pages = {},
      publisher = {IEEE},
      title = {PyTorch-OOD: A Library for Out-of-Distribution Detection based on PyTorch},
      year = {2022}
      }

or::

    Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank, 2022. PyTorch-OOD: A Library for Out-of-Distribution Detection based on PyTorch (IEEE)


Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

**Implemented Detectors** :

+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Detector                   | Description                                                                                    | Year | Ref                |
+============================+================================================================================================+======+====================+
| OpenMax                    | Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*. | 2016 | [#OpenMax]_        |
+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| ODIN                       | ODIN is a preprocessing method for inputs that aims to increase the discriminability of        | 2018 | [#ODIN]_           |
|                            | the softmax outputs for In- and Out-of-Distribution data.                                      |      |                    |
+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Mahalanobis                | This method calculates a class center :math:`\\mu_y` for each class, and a shared              | 2018 | [#Mahalanobis]_    |
|                            | covariance matrix :math:`\\Sigma` from the data.                                               |      |                    |
+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Monte Carlo Dropout        | Implements the Monte Carlo Dropout for OOD detection.                                          | 2022 | [#MonteCarloDrop]_ |
+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Softmax Thresholding       | Implements the Softmax Baseline for OOD detection.                                             | 2022 | [#Softmax]_        |
+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Energy-Based OOD Detection | Implements the Energy Score of  *Energy-based Out-of-distribution Detection*.                  | 2020 | [#EnergyBasedOOD]_ |
+----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+


**Implemented Objective Functions**:

+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| Objective Function         | Description                                                                                      | Year | Ref               |
+============================+==================================================================================================+======+===================+
| Objectosphere              | Implementation of the paper *Reducing Network Agnostophobia*.                                    | 2016 | [#Objectosphere]_ |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| Outlier Exposure           | Implementation of the paper *Deep Anomaly Detection With Outlier Exposure*.                      | 2018 | [#OE]_            |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| Deep SVDD                  | Implementation of the Deep Support Vector Data Description from the paper *Deep One-Class        | 2018 | [#SVDD]_          |
|                            | Classification*.                                                                                 |      |                   |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| II Loss                    | Implementation of II Loss function from *Learning a neural network-based representation for      | 2022 | [#IILoss]_        |
|                            | open set recognition*.                                                                           |      |                   |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| CAC Loss                   | Class Anchor Clustering Loss from *Class Anchor Clustering: a Distance-based Loss for Training   | 2022 | [#CACLoss]_       |
|                            | Open Set Classifiers*                                                                            |      |                   |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| Energy Regularization      | Adds a regularization term to the cross-entropy that aims to increase the energy gap between IN  | 2020 | [#EnergyReg]_     |
|                            | and OOD samples.                                                                                 |      |                   |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+
| Center Loss                | Generalized version of the *Center Loss* from the Paper *A Discriminative Feature Learning       | 2022 | [#CenterLoss]_    |
|                            | Approach for Deep Face Recognition*.                                                             |      |                   |
+----------------------------+--------------------------------------------------------------------------------------------------+------+-------------------+


Contributing
^^^^^^^^^^^^
We encourage everyone to contribute to this project by adding implementations of OOD Detection methods, datasets etc,
or check the existing implementations for bugs.

License
^^^^^^^
The code is licensed under Apache 2.0. We have taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc.
The legal implications of using pre-trained models in commercial services are, to our knowledge, not fully understood.

----

Reference
^^^^^^^^^
.. [#OpenMax]  OpenMax (2016) Towards open set deep networks, CVPR

.. [#ODIN] ODIN (2018)  Enhancing the reliability of out-of-distribution image detection in neural networks, ICLR

.. [#Mahalanobis] Mahalanobis (2018)  A simple unified framework for detecting out-of-distribution samples and adversarial attacks, NeurIPS

.. [#MonteCarloDrop] Monte Carlo Droput 

.. [#Softmax] Softmax Paper

.. [#EnergyBasedOOD] Energy-Based OOD (2020) Energy-based Out-of-distribution Detection, NeurIPS

.. [#Objectosphere] Object Sphere paper

.. [#OE] Outlier Exposure paper

.. [#SVDD] SVDD paper

.. [#IILoss] IILoss paper

.. [#CACLoss] CACLoss Paper

.. [#EnergyReg] Energy Regegularization Paper

.. [#CenterLoss] CenterLoss Paper

===============================
