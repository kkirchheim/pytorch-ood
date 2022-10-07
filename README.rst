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
   :target: https://pepy.tech/project/pytorch-ood
   :alt: Downloads

.. image:: https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/pipeline.svg
   :target: https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/pipeline.svg
   :alt: Pipeline

.. image:: https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/coverage.svg
   :target: https://gitlab.com/kkirchheim/pytorch-ood/badges/dev/coverage.svg
   :alt: Coverage

.. image:: https://readthedocs.org/projects/pytorch-ood/badge/?version=latest
   :target: https://pytorch-ood.readthedocs.io/en/latest/
   :alt: Documentation Status

-----

PyTorch-based library to accelerate research in Out-of-Distribution (OOD) Detection, as well as related
fields such as Open-Set Recognition, Novelty Detection, Confidence Estimation and Anomaly Detection
based on Deep Neural Networks.

This library provides

- Objective/Loss Functions
- Out-of-Distribution Detection Methods
- Datasets
- Neural Network Architectures as well as pretrained weights
- Useful Utilities

and is designed such that it should integrate seamlessly with frameworks that enable the scaling of model training,
like `pytorch-lightning <https://www.pytorchlightning.ai>`_.


Installation
^^^^^^^^^^^^^^
The package can be installed via PyPI:

.. code-block:: shell

   pip install pytorch-ood



**Dependencies**


* ``torch``
* ``torchvision``
* ``scipy``
* ``torchmetrics``


**Optional Dependencies**


* ``libmr``  for the OpenMax Detector [#OpenMax]_ . The library is currently broken and unlikely to be repaired. You will have to install ``cython`` and ``libmr`` afterwards manually.


Quick Start
^^^^^^^^^^^
Load model pre-trained on CIFAR-10 with the Energy-Bounded Learning Loss [#EnergyBasedOOD]_, and predict on some dataset ``data_loader`` using
Energy-based Out-of-Distribution Detection [#EnergyBasedOOD]_, calculating the common OOD detection metrics:

.. code-block:: python

    from pytorch_ood.model import WideResNet
    from pytorch_ood.detector import EnergyBased
    from pytorch_ood.utils import OODMetrics

    # Create Neural Network
    model = WideResNet(pretrained="er-cifar10-tune").eval().cuda()

    # Create detector
    detector = EnergyBased(model)

    # Evaluate
    metrics = OODMetrics()

    for x, y in data_loader:
        metrics.update(detector(x.cuda()), y)

    print(metrics.compute())


You can find more examples in the `documentation <https://pytorch-ood.readthedocs.io/en/latest/examples.html>`_.


Implemented
^^^^^^^^^^^^^^^^^^^^^^

**Detectors** :

+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Detector                    | Description                                                                                    | Year | Ref                |
+=============================+================================================================================================+======+====================+
| OpenMax                     | Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*. | 2016 | [#OpenMax]_        |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Monte Carlo Dropout         | Implements Monte Carlo Dropout.                                                                | 2016 | [#MonteCarloDrop]_ |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Maximum Softmax Probability | Implements the Softmax Baseline for OOD and Error detection.                                   | 2017 | [#Softmax]_        |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| ODIN                        | ODIN is a preprocessing method for inputs that aims to increase the discriminability of        | 2018 | [#ODIN]_           |
|                             | the softmax outputs for In- and Out-of-Distribution data.                                      |      |                    |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Mahalanobis                 | Implements the Mahalanobis Method.                                                             | 2018 | [#Mahalanobis]_    |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Energy-Based OOD Detection  | Implements the Energy Score of *Energy-based Out-of-distribution Detection*.                   | 2020 | [#EnergyBasedOOD]_ |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Maximum Logit               | Implements the MaxLogit method.                                                                | 2022 | [#StreeHaz]_       |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| KL-Matching                 | Implements the KL-Matching method for Multi-Class classification.                              | 2022 | [#StreeHaz]_       |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+

**Objective Functions**:

+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Objective Function         | Description                                                                                      | Year | Ref                |
+============================+==================================================================================================+======+====================+
| Objectosphere              | Implementation of the paper *Reducing Network Agnostophobia*.                                    | 2016 | [#Objectosphere]_  |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Center Loss                | Generalized version of the *Center Loss* from the Paper *A Discriminative Feature Learning       | 2016 | [#CenterLoss]_     |
|                            | Approach for Deep Face Recognition*.                                                             |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Outlier Exposure           | Implementation of the paper *Deep Anomaly Detection With Outlier Exposure*.                      | 2018 | [#OE]_             |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Deep SVDD                  | Implementation of the Deep Support Vector Data Description from the paper *Deep One-Class        | 2018 | [#SVDD]_           |
|                            | Classification*.                                                                                 |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Energy Regularization      | Adds a regularization term to the cross-entropy that aims to increase the energy gap between IN  | 2020 | [#EnergyBasedOOD]_ |
|                            | and OOD samples.                                                                                 |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| CAC Loss                   | Class Anchor Clustering Loss from *Class Anchor Clustering: a Distance-based Loss for Training   | 2021 | [#CACLoss]_        |
|                            | Open Set Classifiers*                                                                            |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| II Loss                    | Implementation of II Loss function from *Learning a neural network-based representation for      | 2022 | [#IILoss]_         |
|                            | open set recognition*.                                                                           |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| MCHAD Loss                 | Implementation of the MCHAD Loss friom the paper *Multi Class Hypersphere Anomaly Detection*.    | 2022 | [#MCHAD]_          |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+

**Image Datasets**:

+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Dataset               | Description                                                                                                     | Year | Ref           |
+=======================+=================================================================================================================+======+===============+
| TinyImages            | The TinyImages dataset is often used as auxiliary OOD training data. However, use is discouraged                | 2012 | [#TinyImgs]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Textures              | Textures dataset, also known as DTD, often used as OOD Examples                                                 | 2013 | [#Textures]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| FoolingImages         | OOD Images Generated to fool certain Deep Neural Networks                                                       | 2014 | [#FImages]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| TinyImages300k        | A cleaned version of the TinyImages Dataset with 300.000 images, often used as auxiliary OOD training data      | 2018 | [#OE]_        |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| MNIST-C               | Corrupted version of the MNIST                                                                                  | 2019 | [#MnistC]_    |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| CIFAR10-C             | Corrupted version of the CIFAR 10                                                                               | 2019 | [#Cifar10]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| CIFAR100-C            | Corrupted version of the CIFAR 100                                                                              | 2019 | [#Cifar10]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| ImageNet-C            | Corrupted version of the ImageNet                                                                               | 2019 | [#Cifar10]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| ImageNet - A, O, R    | Different Outlier Variants for the ImageNet                                                                     | 2019 | [#ImageNets]_ |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| MVTech-AD             | MVTech-AD                                                                                                       | 2021 | [#MVTech]_    |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| StreetHazards         | Anomaly Segmentation Dataset                                                                                    | 2022 | [#StreeHaz]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+

**Text Datasets**:

+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| Dataset     | Description                                                                                                               | Year | Ref             |
+=============+===========================================================================================================================+======+=================+
| Multi30k    | Multi-30k dataset, as used by Hendrycks et al. in the OOD baseline paper                                                  | 2016 | [#Multi30k]_    |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| WikiText2   | Texts from the wikipedia often used as auxiliary OOD training data                                                        | 2016 | [#WikiText2]_   |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| WikiText103 | Texts from the wikipedia often used as auxiliary OOD training data                                                        | 2016 | [#WikiText2]_   |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+


Citing
^^^^^^^

``pytorch-ood`` was presented on a CVPR Workshop in 2022.
If you use it in a scientific publication, please consider citing::

    @InProceedings{kirchheim2022pytorch,
        author    = {Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
        title     = {PyTorch-OOD: A Library for Out-of-Distribution Detection Based on PyTorch},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2022},
        pages     = {4351-4360}
    }


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
.. [#OpenMax]  Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. CVPR.

.. [#ODIN] Liang, S., Li, Y., & Srikant, R. (2017). Enhancing the reliability of out-of-distribution image detection in neural networks. ICLR.

.. [#Mahalanobis] Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. NeurIPS.

.. [#MonteCarloDrop] Miok, K., Nguyen-Doan, D., Zaharie, D., & Robnik-Šikonja, M. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. ICML.

.. [#Softmax] Hendrycks, D., & Gimpel, K. (2016). A baseline for detecting misclassified and out-of-distribution examples in neural networks. ICLR.

.. [#EnergyBasedOOD] Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. NeurIPS.

.. [#Objectosphere] Dhamija, A. R., Günther, M., & Boult, T. (2018). Reducing network agnostophobia. NeurIPS.

.. [#OE] Hendrycks, D., Mazeika, M., & Dietterich, T. (2018). Deep anomaly detection with outlier exposure. ICLR.

.. [#SVDD] Ruff, L.,  et al. (2018). Deep one-class classification. ICML.

.. [#IILoss] Hassen, M., & Chan, P. K. (2020). Learning a neural-network-based representation for open set recognition. SDM.

.. [#CACLoss] Miller, D., Sunderhauf, N., Milford, M., & Dayoub, F. (2021). Class anchor clustering: A loss for distance-based open set recognition. WACV.

.. [#CenterLoss] Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). A discriminative feature learning approach for deep face recognition. ECCV.

.. [#Cifar10] Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. ICLR.

.. [#FImages] Nguyen, A., Yosinski, J., & Clune, J. (2015). Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. CVPR.

.. [#ImageNets] Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., & Song, D. (2021). Natural adversarial examples. CVPR.

.. [#MnistC] Mu, N., & Gilmer, J. (2019). MNIST-C: A robustness benchmark for computer vision. ICLR Workshop.

.. [#StreeHaz] Hendrycks, D., Basart, S., Mazeika, M., Mostajabi, M., Steinhardt, J., & Song, D. (2022). Scaling out-of-distribution detection for real-world settings. ICML.

.. [#Textures] Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). Describing textures in the wild. CVPR.

.. [#TinyImgs] Torralba, A., Fergus, R., & Freeman, W. T. (2007). 80 million tiny images: a large dataset for non-parametric object and scene recognition. IEEE Transactions on Pattern Analysis and Machine Learning.

.. [#Multi30k] Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016). Multi30k: Multilingual english-german image descriptions. Proceedings of the 5th Workshop on Vision and Language.

.. [#WikiText2] Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models. `ArXiv <https://arxiv.org/abs/1609.07843>`_

.. [#MVTech] P. Bergmann, K. Batzner, et al. (2021) The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. IJCV.

.. [#MCHAD] K. Kirchheim, M. Filax, F. Ortmeier (2022) Multi Class Hypersphere Anomaly Detection. ICPR
