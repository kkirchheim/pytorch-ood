PyTorch Out-of-Distribution Detection
****************************************

|docs| |version| |license| |python-version| |downloads| |paper|


.. |docs| image:: https://img.shields.io/badge/docs-online-blue?style=for-the-badge
   :target: https://pytorch-ood.readthedocs.io/en/latest/
   :alt: Documentation
.. |version| image:: https://img.shields.io/pypi/v/pytorch-ood?color=light&style=for-the-badge
   :target: https://pypi.org/project/pytorch-ood/
   :alt: License
.. |license| image:: https://img.shields.io/pypi/l/pytorch-ood?style=for-the-badge
   :target: https://gitlab.com/kkirchheim/pytorch-ood/-/blob/master/LICENSE
   :alt: License
.. |python-version| image:: https://img.shields.io/badge/-Python 3.8+-blue?logo=python&logoColor=white&style=for-the-badge
   :target: https://www.python.org/
   :alt: Python
.. |downloads| image:: https://img.shields.io/pepy/dt/pytorch-ood?style=for-the-badge
   :target: https://pepy.tech/project/pytorch-ood
   :alt: Downloads
.. |paper| image:: https://img.shields.io/badge/paper-cvpr-blue?style=for-the-badge
   :target: https://openaccess.thecvf.com/content/CVPR2022W/HCIS/papers/Kirchheim_PyTorch-OOD_A_Library_for_Out-of-Distribution_Detection_Based_on_PyTorch_CVPRW_2022_paper.pdf
   :alt: Paper

.. image:: docs/_static/pytorch-ood-logo.jpg
   :align: center
   :width: 100%
   :alt: pytorch-ood-logo


A Python library for Out-of-Distribution (OOD) Detection with Deep Neural Networks based on PyTorch.

The library provides:

- Out-of-Distribution Detection Methods
- Loss Functions
- Datasets
- Neural Network Architectures, as well as pre-trained weights
- Data Augmentations
- Useful Utilities

and is designed to be compatible with frameworks
like `pytorch-lightning <https://www.pytorchlightning.ai>`_ and
`pytorch-segmentation-models <https://github.com/qubvel/segmentation_models.pytorch>`_.
The library also covers some methods from closely related fields, such as Open-Set Recognition, Novelty Detection,
Confidence Estimation and Anomaly Detection.


📚  Documentation
^^^^^^^^^^^^^^^^^^^
The documentation is available `here <https://pytorch-ood.readthedocs.io/en/latest/>`_.

**NOTE**: An important convention adopted in ``pytorch-ood`` is that **OOD detectors predict outlier scores**
that should be larger for outliers than for inliers.
If you notice that the scores predicted by a detector do not match the formulas in the corresponding publication, we may have adjusted the score calculation to comply with this convention.

⏳ Quick Start
^^^^^^^^^^^^^^^^^
Load a WideResNet-40 model (used in major publications), pre-trained on CIFAR-10 with the Energy-Bounded Learning Loss [#EnergyBasedOOD]_ (weights from to original paper), and predict on some dataset ``data_loader`` using
Energy-based OOD Detection (EBO) [#EnergyBasedOOD]_, calculating the common metrics.
OOD data must be marked with labels < 0.

.. code-block:: python


    from pytorch_ood.detector import EnergyBased
    from pytorch_ood.utils import OODMetrics
    from pytorch_ood.model import WideResNet

    data_loader = ... # your data, OOD with label < 0

    # Create Neural Network
    model = WideResNet(num_classes=10, pretrained="er-cifar10-tune").eval().cuda()
    preprocess = WideResNet.transform_for("er-cifar10-tune")

    # Create detector
    detector = EnergyBased(model)

    # Evaluate
    metrics = OODMetrics()

    for x, y in data_loader:
        x = preprocess(x).cuda()
        metrics.update(detector(x), y)

    print(metrics.compute())


You can find more examples in the `documentation <https://pytorch-ood.readthedocs.io/en/latest/auto_examples/benchmarks/>`_.

Benchmarks (Beta)
---------------------------

Evaluate detectors against common benchmarks, for example the OpenOOD ImageNet benchmark
(including ImageNet-O, OpenImages-O, Textures, SVHN, MNIST).  All datasets (except for ImageNet itself) will be downloaded automatically.

.. code-block:: python

   import pandas as pd
   from pytorch_ood.benchmark import ImageNet_OpenOOD
   from pytorch_ood.detector import MaxSoftmax
   from torchvision.models import resnet50
   from torchvision.models.resnet import ResNet50_Weights

   model = resnet50(ResNet50_Weights.IMAGENET1K_V1).eval().to("cuda:0")
   trans = ResNet50_Weights.IMAGENET1K_V1.transforms()

   benchmark = ImageNet_OpenOOD(root="data", image_net_root="data/imagenet-2012/", transform=trans)

   detector = MaxSoftmax(model)
   results = benchmark.evaluate(detector, loader_kwargs={"batch_size": 64}, device="cuda:0")
   df = pd.DataFrame(results)
   print(df)


This produces the following table:

+-------------+-------+---------+----------+----------+
| Dataset     | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=========+==========+==========+
| ImageNetO   | 28.64 | 2.52    | 94.85    | 91.20    |
+-------------+-------+---------+----------+----------+
| OpenImagesO | 84.98 | 62.61   | 94.67    | 49.95    |
+-------------+-------+---------+----------+----------+
| Textures    | 80.46 | 37.50   | 96.80    | 67.75    |
+-------------+-------+---------+----------+----------+
| SVHN        | 97.62 | 95.56   | 98.77    | 11.58    |
+-------------+-------+---------+----------+----------+
| MNIST       | 90.04 | 90.45   | 89.88    | 39.03    |
+-------------+-------+---------+----------+----------+


🛠 ️️Installation
^^^^^^^^^^^^^^^^^
The package can be installed via PyPI:

.. code-block:: shell

   pip install pytorch-ood



**Dependencies**


* ``torch``
* ``torchvision``
* ``scipy``
* ``torchmetrics``


**Optional Dependencies**

* ``scikit-learn`` for ViM and k-NN
* ``gdown`` to download some datasets and model weights
* ``pandas`` for the `examples <https://pytorch-ood.readthedocs.io/en/latest/auto_examples/benchmarks/>`_.
* ``segmentation-models-pytorch`` to run the examples for anomaly segmentation


📝 Citing
^^^^^^^^^^

If you use this project, please cite:

::

  @inproceedings{kirchheim2022pytorch,
    title={Pytorch-ood: A library for out-of-distribution detection based on pytorch},
    author={Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={4351--4360},
    year={2022}
  }



📦 Implemented
^^^^^^^^^^^^^^^

**Detectors**:

+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Detector                    | Description                                                                                    | Year | Ref                |
+=============================+================================================================================================+======+====================+
| OpenMax                     | Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*. | 2016 | [#OpenMax]_        |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Monte Carlo Dropout         | Implements Monte Carlo Dropout.                                                                | 2016 | [#MonteCarloDrop]_ |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Maximum Softmax Probability | Implements the Softmax Baseline for OOD and Error detection.                                   | 2017 | [#Softmax]_        |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Temperature Scaling         | Implements the Temperature Scaling for Softmax.                                                | 2017 | [#TempScaling]_    |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| ODIN                        | ODIN is a preprocessing method for inputs that aims to increase the discriminability of        | 2018 | [#ODIN]_           |
|                             | the softmax outputs for In- and Out-of-Distribution data.                                      |      |                    |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Mahalanobis                 | Implements the Mahalanobis Method.                                                             | 2018 | [#Mahalanobis]_    |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| GRAM                        | Detects OOD elements via deviations in the gram matrices                                       | 2019 | [#GramBased]_      |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Energy-Based OOD Detection  | Implements the energy score of *Energy-based Out-of-distribution Detection*.                   | 2020 | [#EnergyBasedOOD]_ |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Entropy                     | Uses entropy to detect OOD inputs.                                                             | 2021 | [#MaxEntropy]_     |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| ReAct                       | ReAct: Out-of-distribution detection with Rectified Activations.                               | 2021 | [#ReAct]_          |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Maximum Logit               | Implements the MaxLogit method.                                                                | 2022 | [#StreeHaz]_       |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| KL-Matching                 | Implements the KL-Matching method for Multi-Class classification.                              | 2022 | [#StreeHaz]_       |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| ViM                         | Implements Virtual Logit Matching.                                                             | 2022 | [#ViM]_            |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Weighted Energy-Based       | Implements Weighted Energy-Based for OOD Detection                                             | 2022 | [#WEBO]_           |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| Nearest Neighbor            | Implements Depp Nearest Neighbors for OOD Detection                                            | 2022 | [#kNN]_            |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| DICE                        | Implements Sparsification for OOD Detection                                                    | 2022 | [#DICE]_           |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| ASH                         | Implements Extremely Simple Activation Shaping                                                 | 2023 | [#Ash]_            |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| SHE                         | Implements Simplified Hopfield Networks                                                        | 2023 | [#She]_            |
+-----------------------------+------------------------------------------------------------------------------------------------+------+--------------------+
| NCI                         | Neural Collapse Inspired OOD Detection                                                         | 2025 | [#Nci]_            |
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
| Confidence Loss            | Model learn confidence additional to class membership prediction.                                | 2018 | [#ConfidenceLoss]_ |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Deep SVDD                  | Implementation of the Deep Support Vector Data Description from the paper *Deep One-Class        | 2018 | [#SVDD]_           |
|                            | Classification*.                                                                                 |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Energy-Bounded Loss        | Adds a regularization term to the cross-entropy that aims to increase the energy gap between IN  | 2020 | [#EnergyBasedOOD]_ |
|                            | and OOD samples.                                                                                 |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| CAC Loss                   | Class Anchor Clustering Loss from *Class Anchor Clustering: a Distance-based Loss for Training   | 2021 | [#CACLoss]_        |
|                            | Open Set Classifiers*                                                                            |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Entropic Open-Set Loss     | Entropy maximization and meta classification for OOD in semantic segmentation                    | 2021 | [#MaxEntropy]_     |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| II Loss                    | Implementation of II Loss function from *Learning a neural network-based representation for      | 2022 | [#IILoss]_         |
|                            | open set recognition*.                                                                           |      |                    |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| MCHAD Loss                 | Implementation of the MCHAD Loss from the paper *Multi Class Hypersphere Anomaly Detection*.     | 2022 | [#MCHAD]_          |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| VOS Energy-Based Loss      | Implementation of the paper *VOS: Learning what you don’t know by virtual outlier synthesis*.    | 2022 | [#WEBO]_           |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+
| Logit Normalization        | Implementation of the paper *Mitigating Neural Network Overconfidence with Logit Normalization*. | 2022 | [#LOGNORM]_        |
+----------------------------+--------------------------------------------------------------------------------------------------+------+--------------------+

**Image Datasets**:

+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Dataset               | Description                                                                                                     | Year | Ref           |
+=======================+=================================================================================================================+======+===============+
| Chars74k              | The Chars74K dataset contains 74,000 images across 64 classes, comprising English letters and Arabic numerals.  | 2012 | [#Chars74k]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| TinyImages            | The TinyImages dataset is often used as auxiliary OOD training data. However, use is discouraged.               | 2012 | [#TinyImgs]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Textures              | Textures dataset, also known as DTD, often used as OOD Examples.                                                | 2013 | [#Textures]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| FoolingImages         | OOD Images Generated to fool certain Deep Neural Networks.                                                      | 2015 | [#FImages]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Tiny ImageNet         | A derived version of ImageNet with 64x64-sized images.                                                          | 2015 | [#TinyIN]_    |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| TinyImages300k        | A cleaned version of the TinyImages Dataset with 300.000 images, often used as auxiliary OOD training data.     | 2018 | [#OE]_        |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| LSUN                  | A version of the Large-scale Scene UNderstanding Dataset with 10.000 images, often used as auxiliary            | 2018 | [#ODIN]_      |
|                       | OOD training data.                                                                                              |      |               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| MNIST-C               | Corrupted version of the MNIST.                                                                                 | 2019 | [#MnistC]_    |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| CIFAR10-C             | Corrupted version of the CIFAR 10.                                                                              | 2019 | [#Cifar10]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| CIFAR100-C            | Corrupted version of the CIFAR 100.                                                                             | 2019 | [#Cifar10]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| ImageNet-C            | Corrupted version of the ImageNet.                                                                              | 2019 | [#Cifar10]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| ImageNet - A, O, R    | Different Outlier Variants for the ImageNet.                                                                    | 2019 | [#ImageNets]_ |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| ImageNet - V2         | A new test set for the ImageNet.                                                                                | 2019 | [#ImageNV2]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| ImageNet - ES         | Event stream (ES) version of the ImageNet.                                                                      | 2021 | [#ImageNES]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| iNaturalist           | A Subset of iNaturalist, with 10.000 images.                                                                    | 2021 | [#INatural]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Fractals              | A dataset with Fractals from *PIXMIX: Dreamlike Pictures Comprehensively Improve Safety Measures*               | 2022 | [#PixMix]_    |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Feature               | A dataset with Feature visualizations from *PIXMIX: Dreamlike Pictures Comprehensively Improve Safety Measures* | 2022 | [#PixMix]_    |
| Visualizations        |                                                                                                                 |      |               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| FS Static             | The FishyScapes (FS) Static dataset contains real world OOD images from the CityScapes dataset.                 | 2021 | [#FS]_        |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| FS LostAndFound       | The FishyScapes dataset contains images from the CityScapes dataset blended with unknown objects scraped from   | 2021 | [#FS]_        |
|                       | the web.                                                                                                        |      |               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| MVTech-AD             | The MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.     | 2021 | [#MVTech]_    |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| StreetHazards         | Anomaly Segmentation Dataset                                                                                    | 2022 | [#StreeHaz]_  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| CIFAR100-GAN          | Images sampled from low likelihood regions of a BigGAN trained on CIFAR 100 from the paper *On Outlier Exposure | 2022 | [#CifarGAN]_  |
|                       | with Generative Models.*                                                                                        |      |               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| SSB - hard            | The hard split of the Semantic Shift Benchmark, which contains 49.00 images.                                    | 2022 | [#SSB]_       |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| NINCO                 | The NINCO (No ImageNet Class Objects) dataset which contains 5.879 images of 64 OOD classes.                    | 2023 | [#NINCO]_     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| SuMNIST               | The SuMNIST dataset is based on MNIST but each image display four numbers instead of one.                       | 2023 | [#SuMNIST]_   |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Gaussian Noise        | Dataset with samples drawn from a normal distribution.                                                          |      |               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+
| Uniform Noise         | Dataset with samples drawn from a uniform distribution.                                                         |      |               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+---------------+



**Text Datasets**:

+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| Dataset     | Description                                                                                                               | Year | Ref             |
+=============+===========================================================================================================================+======+=================+
| Multi30k    | Multi-30k dataset, as used by Hendrycks et al. in the OOD baseline paper.                                                 | 2016 | [#Multi30k]_    |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| WikiText2   | Texts from the wikipedia often used as auxiliary OOD training data.                                                       | 2016 | [#WikiText2]_   |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| WikiText103 | Texts from the wikipedia often used as auxiliary OOD training data.                                                       | 2016 | [#WikiText2]_   |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+
| NewsGroup20 | Texts from different newsgroups, as used by Hendrycks et al. in the OOD baseline paper.                                   |      |                 |
+-------------+---------------------------------------------------------------------------------------------------------------------------+------+-----------------+


**Augmentation Methods**:

+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+----------------+
| Augmentation          | Description                                                                                                     | Year | Ref            |
+=======================+=================================================================================================================+======+================+
| PixMix                | PixMix image augmentation method                                                                                | 2022 | [#PixMix]_     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+----------------+
| COCO Outlier Pasting  | From "Entropy maximization and meta classification for OOD in semantic segmentation"                            | 2021 | [#MaxEntropy]_ |
+-----------------------+-----------------------------------------------------------------------------------------------------------------+------+----------------+


🤝  Contributing
^^^^^^^^^^^^^^^^^
We encourage everyone to contribute to this project by adding implementations of OOD Detection methods, datasets etc,
or check the existing implementations for bugs.


🛡️ ️License
^^^^^^^^^^^

The code is licensed under Apache 2.0. We have taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc.
The legal implications of using pre-trained models in commercial services are, to our knowledge, not fully understood.

----

🔗 References
^^^^^^^^^^^^^^

.. [#OpenMax]  Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. CVPR.

.. [#ODIN] Liang, S., et al. (2017). Enhancing the reliability of out-of-distribution image detection in neural networks. ICLR.

.. [#Mahalanobis] Lee, K., et al. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. NeurIPS.

.. [#MonteCarloDrop] Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. ICML.

.. [#Softmax] Hendrycks, D., & Gimpel, K. (2016). A baseline for detecting misclassified and out-of-distribution examples in neural networks. ICLR.

.. [#TempScaling] Guo, C., et al. (2017). On calibration of modern neural networks. ICML.

.. [#ConfidenceLoss] DeVries, T., & Taylor, G. W. (2018). Learning confidence for out-of-distribution detection in neural networks. `ArXiv <https://arxiv.org/pdf/1802.04865>`__.

.. [#EnergyBasedOOD] Liu, W., et al. (2020). Energy-based out-of-distribution detection. NeurIPS.

.. [#Objectosphere] Dhamija, A. R., et al. (2018). Reducing network agnostophobia. NeurIPS.

.. [#OE] Hendrycks, D., Mazeika, M., & Dietterich, T. (2018). Deep anomaly detection with outlier exposure. ICLR.

.. [#SVDD] Ruff, L., et al. (2018). Deep one-class classification. ICML.

.. [#IILoss] Hassen, M., & Chan, P. K. (2020). Learning a neural-network-based representation for open set recognition. SDM.

.. [#CACLoss] Miller, D., et al. (2021). Class anchor clustering: A loss for distance-based open set recognition. WACV.

.. [#CenterLoss] Wen, Y., et al. (2016). A discriminative feature learning approach for deep face recognition. ECCV.

.. [#Cifar10] Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. ICLR.

.. [#FImages] Nguyen, A., et al. (2015). Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. CVPR.

.. [#TinyIN] Le, Y., et al. (2015). Tiny ImageNet Visual Recognition Challenge. `Stanford <https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf>`_.

.. [#ImageNets] Hendrycks, D., et al. (2021). Natural adversarial examples. CVPR.

.. [#ImageNV2] Recht, B., et al. (2019).  Do imagenet classifiers generalize to imagenet?. PMLR.

.. [#ImageNES] Lin, Y., et al. (2021).  ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks. `Front Neurosci <https://pubmed.ncbi.nlm.nih.gov/34899154/>`_.

.. [#MnistC] Mu, N., & Gilmer, J. (2019). MNIST-C: A robustness benchmark for computer vision. ICLR Workshop.

.. [#FS] Blum, H. et al (2021) The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation. IJCV.

.. [#MVTech] Bergmann, P. et al (2021) The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. IJCV

.. [#StreeHaz] Hendrycks, D., et al. (2022). Scaling out-of-distribution detection for real-world settings. ICML.

.. [#CifarGAN] Kirchheim, K., & Ortmeier, F. (2022) On Outlier Exposure with Generative Models. NeurIPS.

.. [#SSB] Vaze, S., et al. (2022)  Open-set recognition: A good closed-set classifier is all you need. ICLR.

.. [#NINCO] Bitterwolf, J., et al. (2023) In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation. ICML.

.. [#SuMNIST] Kirchheim, K. (2023) Towards Deep Anomaly Detection with Structured Knowledge Representations. SAFECOMP.

.. [#Textures] Cimpoi, M., et al. (2014). Describing textures in the wild. CVPR.

.. [#TinyImgs] Torralba, A., et al. (2007). 80 million tiny images: a large dataset for non-parametric object and scene recognition. IEEE Transactions on Pattern Analysis and Machine Learning.

.. [#Chars74k] de Campos, T. E., et al. (2009). Character recognition in natural images. In Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP).

.. [#Multi30k] Elliott, D., et al. (2016). Multi30k: Multilingual english-german image descriptions. Proceedings of the 5th Workshop on Vision and Language.

.. [#WikiText2] Merity, S., et al. (2016). Pointer sentinel mixture models. `ArXiv <https://arxiv.org/abs/1609.07843>`__

.. [#INatural] Huang, R., & Li, Y. (2021) MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space. CVPR.

.. [#MCHAD] Kirchheim, K., et al. (2022) Multi Class Hypersphere Anomaly Detection. ICPR.

.. [#ViM] Wang, H., et al. (2022) ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR.

.. [#WEBO] Du, X., et al. (2022) VOS: Learning What You Don't Know by Virtual Outlier Synthesis. ICLR.

.. [#kNN] Sun, Y., et al. (2022) Out-of-Distribution Detection with Deep Nearest Neighbors. ICML.

.. [#PixMix] Hendrycks, D, et al. (2022) PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures. CVPR.

.. [#MaxEntropy] Chan R,  et al. (2021) Entropy maximization and meta classification for out-of-distribution detection in semantic segmentation. CVPR.

.. [#DICE] Sun, et al. (2022) DICE: Leveraging Sparsification for Out-of-Distribution Detection. ECCV.

.. [#ASH] Djurisic,  et al. (2023) Extremely Simple Activation Shaping for Out-of-Distribution Detection, ICLR.

.. [#She] Zhang,  et al. (2023) Out-of-Distribution Detection Based on In-Distribution Data Patterns Memorization with Modern Hopfield Energy. ICLR.

.. [#ReAct] Sun,  et al. (2023) ReAct: Out-of-distribution Detection With Rectified Activations. NeurIPS.

.. [#GramBased] Shama,  et al. (2019) Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices. NeurIPS.

.. [#LOGNORM] Wei,  et al. (2022) Mitigating Neural Network Overconfidence with Logit Normalization. ICML.

.. [#Nci] Liu,  et al. (2025) Detecting Out-of-distribution through the Lens of Neural Collapse. CVPR.
