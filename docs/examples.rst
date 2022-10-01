Examples
**************************

Detectors
==================

Softmax vs. Energy
---------------------

Compare the Maximum Softmax baseline method to Energy Based Out-of-Distribution Detection.

.. literalinclude:: ../examples/energy_softmax.py


Depending on the number of iterations, the output will be

.. code :: python

    {'AUROC': 0.601246178150177, 'AUPR-IN': 0.5482655763626099, 'AUPR-OUT': 0.6618213057518005, 'ACC95TPR': 0.3585677742958069, 'FPR95TPR': 0.9750000238418579}
    {'AUROC': 0.6039068698883057, 'AUPR-IN': 0.5614069104194641, 'AUPR-OUT': 0.6644740700721741, 'ACC95TPR': 0.3593989908695221, 'FPR95TPR': 0.9736999869346619}



Pretrained SoftMax
--------------------------
We can use the pre-trained model from the OOD baseline paper to obtain results without having to train a model.

.. literalinclude:: ../examples/softmax.py

.. code :: python

    {'AUROC': 0.8851455450057983, 'AUPR-IN': 0.7850116491317749, 'AUPR-OUT': 0.9299277663230896, 'ACC95TPR': 0.720716118812561, 'FPR95TPR': 0.40860000252723694}


Pretrained Monte Carlo Dropout
------------------------------------------
We can use Monte Carlo Dropout with a pre-trained model, since the model uses dropout in the convolutional layers.

.. literalinclude:: ../examples/mcd.py


.. code :: python

    {'AUROC': 0.8661140203475952, 'AUPR-IN': 0.744986891746521, 'AUPR-OUT': 0.9206404089927673, 'ACC95TPR': 0.7123401761054993, 'FPR95TPR': 0.42170000076293945}


Pretrained OpenMax
---------------------
Compared to other detectors, the OpenMax Layer has to be fitted to the training data.
While the method was proposed before the baseline, it outperforms it on this OOD dataset.

.. literalinclude:: ../examples/openmax.py


.. code :: python

    {'AUROC': 0.9078375101089478, 'AUPR-IN': 0.8213402628898621, 'AUPR-OUT': 0.9453056454658508, 'ACC95TPR': 0.7641304135322571, 'FPR95TPR': 0.3407000005245209}

Objective Functions
=====================

Outlier Exposure
--------------------------
We can use a model pre-trained on the :math:`32 \times 32` resized version of the ImageNet and train it with
:class:`Outlier Exposure <pytorch_ood.loss.OutlierExposureLoss>`.
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
