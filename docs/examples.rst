Examples
**************************

Baseline Results
==================
The goal of this section is to describe how to quickly obtain some baseline results to compare against.


CIFAR 10
--------------------------
The following code reproduces a common benchmark on the CIFAR10 with 7 OOD detectors.
Each detector is tested against 5 OOD datasets. We then calculate the mean performance of each detector over all
datasets and sort the results by their AUROC in ascending order. The table below is the output of the script.

We test :class:`MaxSoftmax <pytorch_ood.detector.MaxSoftmax>`,
:class:`Energy-Based Out-of-Distribution Detection  <pytorch_ood.detector.EnergyBased>`,
:class:`MaxLogit <pytorch_ood.detector.MaxLogit>`,
:class:`ODIN <pytorch_ood.detector.ODIN>`,
:class:`KLMatching <pytorch_ood.detector.KLMatching>`
:class:`OpenMax <pytorch_ood.detector.OpenMax>` and
:class:`Mahalanobis  <pytorch_ood.detector.Mahalanobis>`.



.. csv-table:: Mean Performance over 5 OOD Datasets
   :file: _static/baseline_cifar10.csv
   :header-rows: 1
   :class: longtable
   :widths: 1 1 1 1 1 1


.. literalinclude:: ../examples/cifar10_baseline.py


Objective Functions
=====================

Outlier Exposure
--------------------------
We train a model with  :class:`Outlier Exposure <pytorch_ood.loss.OutlierExposureLoss>` on the CIFAR10.

We can use a model pre-trained on the :math:`32 \times 32` resized version of the ImageNet as a foundation.
As outlier data, we use :class:`TinyImages300k <pytorch_ood.dataset.img.TinyImages300k>`, a cleaned version of the
TinyImages database, which contains random images scraped from the internet.

.. literalinclude:: ../examples/outlier_exposure.py

.. code :: python

    Epoch 0
    {'AUROC': 0.9181275963783264, 'AUPR-IN': 0.862043023109436, 'AUPR-OUT': 0.9391289949417114, 'ACC95TPR': 0.7166879773139954, 'FPR95TPR': 0.414900004863739}
    Epoch 1
    {'AUROC': 0.9658643007278442, 'AUPR-IN': 0.9305436611175537, 'AUPR-OUT': 0.9807648062705994, 'ACC95TPR': 0.8796675205230713, 'FPR95TPR': 0.1599999964237213}
    Epoch 2
    {'AUROC': 0.9716796875, 'AUPR-IN': 0.9417432546615601, 'AUPR-OUT': 0.9838283061981201, 'ACC95TPR': 0.9025575518608093, 'FPR95TPR': 0.1242000013589859}
    Epoch 3
    {'AUROC': 0.9516472816467285, 'AUPR-IN': 0.9256581664085388, 'AUPR-OUT': 0.9695837497711182, 'ACC95TPR': 0.8051789999008179, 'FPR95TPR': 0.27649998664855957}
    Epoch 4
    {'AUROC': 0.9773264527320862, 'AUPR-IN': 0.9591482281684875, 'AUPR-OUT': 0.9867616295814514, 'ACC95TPR': 0.9115728735923767, 'FPR95TPR': 0.11010000109672546}
    Epoch 5
    {'AUROC': 0.9767299890518188, 'AUPR-IN': 0.9531672596931458, 'AUPR-OUT': 0.9853834509849548, 'ACC95TPR': 0.9120204448699951, 'FPR95TPR': 0.10939999669790268}
    Epoch 6
    {'AUROC': 0.9764149785041809, 'AUPR-IN': 0.9478701949119568, 'AUPR-OUT': 0.9865636825561523, 'ACC95TPR': 0.9230179190635681, 'FPR95TPR': 0.09220000356435776}
    Epoch 7
    {'AUROC': 0.9865361452102661, 'AUPR-IN': 0.9664997458457947, 'AUPR-OUT': 0.9927579760551453, 'ACC95TPR': 0.9514066576957703, 'FPR95TPR': 0.04780000075697899}
    [...]


Deep One-Class Learning
---------------------------
We train a One-Class model (that is, a model that does not need class labels)
on MNIST, using :class:`Deep SVDD <pytorch_ood.loss.DeepSVDDLoss>`.
SVDD places a single center :math:`\mu` in the output space of a model :math:`f_{\theta}`.
During training, the parameters  :math:`\theta` are adjusted to minimize the (squared) sum of the distances of
representations :math:`f_{\theta}(x)` to this center.
Thus, the model is trained to map the training samples close to the center.
The hope is that the model learns to map **only** IN samples close to the center, and not OOD samples.
The distance to the center can be used as outlier score.

We test the model against FashionMNIST.

.. literalinclude:: ../examples/svdd.py

.. code :: python

    tensor([-0.0232,  0.0007])
    Epoch 0
    {'AUROC': 0.9716497659683228, 'AUPR-IN': 0.9781308174133301, 'AUPR-OUT': 0.9612259268760681, 'ACC95TPR': 0.8962500095367432, 'FPR95TPR': 0.1574999988079071}
    Epoch 1
    {'AUROC': 0.9785468578338623, 'AUPR-IN': 0.983160138130188, 'AUPR-OUT': 0.9711489677429199, 'ACC95TPR': 0.9200999736785889, 'FPR95TPR': 0.10980000346899033}
    [...]
    {'AUROC': 0.9923456907272339, 'AUPR-IN': 0.9935303926467896, 'AUPR-OUT': 0.9906884431838989, 'ACC95TPR': 0.9639000296592712, 'FPR95TPR': 0.022199999541044235}
    Epoch 18
    {'AUROC': 0.9926373362541199, 'AUPR-IN': 0.9937747716903687, 'AUPR-OUT': 0.9910157322883606, 'ACC95TPR': 0.9646499752998352, 'FPR95TPR': 0.02070000022649765}
    Epoch 19
    {'AUROC': 0.9926471710205078, 'AUPR-IN': 0.9937890768051147, 'AUPR-OUT': 0.9909848570823669, 'ACC95TPR': 0.9646999835968018, 'FPR95TPR': 0.020600000396370888}


Class Anchor Clustering
--------------------------------
:class:`Class Anchor Clustering <pytorch_ood.loss.CACLoss>` (CAC) can be seen as a multi-class generalization of Deep One-Class Learning, where there are
several centers :math:`\{\mu_1, \mu_2, ..., \mu_y\}` in the output space of the model, one for each class.
During training, the representation :math:`f_{\theta}(x)` from class :math:`y` is drawn
towards the corresponding center :math:`\mu_y`.

Here, we train the model for 10 epochs on the CIFAR10 dataset, using a backbone pre-trained on the
:math:`32 \times 32` resized version of the ImageNet as a foundation.

.. literalinclude:: ../examples/cac.py

.. code :: python

    Epoch 0
    {'AUROC': 0.7676149606704712, 'AUPR-IN': 0.5659406185150146, 'AUPR-OUT': 0.8522064685821533, 'ACC95TPR': 0.571675181388855, 'FPR95TPR': 0.641700029373169}
    Accuracy: 0.8070999979972839
    Epoch 1
    {'AUROC': 0.8150633573532104, 'AUPR-IN': 0.686570405960083, 'AUPR-OUT': 0.8723115921020508, 'ACC95TPR': 0.5786445140838623, 'FPR95TPR': 0.6308000087738037}
    Accuracy: 0.8166000247001648
    [...]
    Epoch 8
    {'AUROC': 0.8906528949737549, 'AUPR-IN': 0.8156697154045105, 'AUPR-OUT': 0.9292852282524109, 'ACC95TPR': 0.7187340259552002, 'FPR95TPR': 0.4117000102996826}
    Accuracy: 0.8980000019073486
    Epoch 9
    {'AUROC': 0.8763925433158875, 'AUPR-IN': 0.7935031056404114, 'AUPR-OUT': 0.9245198965072632, 'ACC95TPR': 0.700063943862915, 'FPR95TPR': 0.4408999979496002}
    Accuracy: 0.9020000100135803
