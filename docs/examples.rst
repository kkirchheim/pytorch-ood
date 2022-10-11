Examples
**************************

Baseline Results
==================
The goal of this section is to describe how to quickly obtain some baseline results to compare against.


CIFAR 10
--------------------------
The following code reproduces a common benchmark on the CIFAR10 with 8 OOD detectors.
Each detector is tested against 5 OOD datasets. We then calculate the mean performance of each detector over all
datasets and sort the results by their AUROC in ascending order. The table below is the output of the script.

We test :class:`MaxSoftmax <pytorch_ood.detector.MaxSoftmax>`,
:class:`Energy-Based Out-of-Distribution Detection  <pytorch_ood.detector.EnergyBased>`,
:class:`MaxLogit <pytorch_ood.detector.MaxLogit>`,
:class:`ODIN <pytorch_ood.detector.ODIN>`,
:class:`KLMatching <pytorch_ood.detector.KLMatching>`
:class:`ViM <pytorch_ood.detector.ViM>`,
:class:`OpenMax <pytorch_ood.detector.OpenMax>` and
:class:`Mahalanobis  <pytorch_ood.detector.Mahalanobis>`.



.. csv-table:: Mean Performance over 5 OOD Datasets
   :file: _static/baseline_cifar10.csv
   :header-rows: 1
   :class: longtable
   :widths: 1 1 1 1 1 1


.. literalinclude:: ../examples/cifar10_baseline.py


CIFAR 100
--------------------------
The evaluation is the same as for CIFAR 10.

.. csv-table:: Mean Performance over 5 OOD Datasets
   :file: _static/baseline_cifar100.csv
   :header-rows: 1
   :class: longtable
   :widths: 1 1 1 1 1 1


.. literalinclude:: ../examples/cifar100_baseline.py


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
    {'AUROC': 0.9431179165840149, 'AUPR-IN': 0.9154535531997681, 'AUPR-OUT': 0.9592585563659668, 'ACC95TPR': 0.7849743962287903, 'FPR95TPR': 0.30809998512268066}
    Epoch 1
    {'AUROC': 0.9741718769073486, 'AUPR-IN': 0.9422035813331604, 'AUPR-OUT': 0.9860184788703918, 'ACC95TPR': 0.9156649708747864, 'FPR95TPR': 0.10369999706745148}
    Epoch 2
    [...]
    Epoch 8
    {'AUROC': 0.9878846406936646, 'AUPR-IN': 0.9728320837020874, 'AUPR-OUT': 0.9936977028846741, 'ACC95TPR': 0.9515345096588135, 'FPR95TPR': 0.047600001096725464}
    Epoch 9
    {'AUROC': 0.9877899289131165, 'AUPR-IN': 0.9751396179199219, 'AUPR-OUT': 0.9933428168296814, 'ACC95TPR': 0.9497442245483398, 'FPR95TPR': 0.05040000006556511}


Deep One-Class Learning
---------------------------
We train a One-Class model (that is, a model that does not need class labels)
on MNIST, using :class:`Deep SVDD <pytorch_ood.loss.DeepSVDDLoss>`.
SVDD places a single center :math:`\mu` in the output space of a model :math:`f_{\theta}`.
During training, the parameters  :math:`\theta` are adjusted to minimize the (squared) sum of the distances of
representations :math:`f_{\theta}(x)` to this center.
Thus, the model is trained to map the training samples close to the center.
The idea is that the model learns to map **only** IN samples close to the center, and not OOD samples.
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
    {'AUROC': 0.8659521341323853, 'AUPR-IN': 0.7351062893867493, 'AUPR-OUT': 0.9259117245674133, 'ACC95TPR': 0.7191176414489746, 'FPR95TPR': 0.41110000014305115}
    Accuracy: 0.8587999939918518
    Epoch 1
    {'AUROC': 0.8711035847663879, 'AUPR-IN': 0.7894102334976196, 'AUPR-OUT': 0.9095292687416077, 'ACC95TPR': 0.6606777310371399, 'FPR95TPR': 0.5024999976158142}
    Accuracy: 0.857200026512146
    [...]
    Epoch 8
    {'AUROC': 0.9232172966003418, 'AUPR-IN': 0.8630521893501282, 'AUPR-OUT': 0.9534961581230164, 'ACC95TPR': 0.8020460605621338, 'FPR95TPR': 0.28139999508857727}
    Accuracy: 0.9060999751091003
    Epoch 9
    {'AUROC': 0.9008855223655701, 'AUPR-IN': 0.8215528130531311, 'AUPR-OUT': 0.9419905543327332, 'ACC95TPR': 0.7523657083511353, 'FPR95TPR': 0.35910001397132874}
    Accuracy: 0.9072999954223633
