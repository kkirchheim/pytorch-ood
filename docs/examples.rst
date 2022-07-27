Examples
**************************

Softmax vs. Energy
---------------------

Compare the Maximum Softmax baseline method to Energy Based Out-of-Distribution Detection.

.. literalinclude:: ../examples/example.py


Depending on the number of iterations, the output will be

.. code :: python

    {'AUROC': 0.601246178150177, 'AUPR-IN': 0.5482655763626099, 'AUPR-OUT': 0.6618213057518005, 'ACC95TPR': 0.3585677742958069, 'FPR95TPR': 0.9750000238418579}
    {'AUROC': 0.6039068698883057, 'AUPR-IN': 0.5614069104194641, 'AUPR-OUT': 0.6644740700721741, 'ACC95TPR': 0.3593989908695221, 'FPR95TPR': 0.9736999869346619}



Pretrained SoftMax
--------------------------
We can use the pre-trained model from the OOD baseline paper to obtain results without having to train a model.

.. literalinclude:: ../examples/softmax.py

.. code :: python

    {'AUROC': 0.8360370993614197, 'AUPR-IN': 0.7002261281013489, 'AUPR-OUT': 0.8969095945358276, 'ACC95TPR': 0.6476342678070068, 'FPR95TPR': 0.5228999853134155}


Pretrained Monte Carlo Dropout
------------------------------------------
We can use Monte Carlo Dropout with a pre-trained model, since the model uses dropout in the convolutional layers.

.. literalinclude:: ../examples/mcd.py


.. code :: python

    {'AUROC': 0.8661140203475952, 'AUPR-IN': 0.744986891746521, 'AUPR-OUT': 0.9206404089927673, 'ACC95TPR': 0.7123401761054993, 'FPR95TPR': 0.42170000076293945}


Pretrained OpenMax
---------------------
Compared to other detectors, the OpenMax has to be fitted to the training data.
While the method was proposed before the baseline, it outperforms it by a large margin.

.. literalinclude:: ../examples/openmax.py


.. code :: python

    {'AUROC': 0.9078375101089478, 'AUPR-IN': 0.8213402628898621, 'AUPR-OUT': 0.9453056454658508, 'ACC95TPR': 0.7641304135322571, 'FPR95TPR': 0.3407000005245209}
