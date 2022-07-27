Examples
**************************

Softmax Thresholding
---------------------

Compare the Maximum Softmax baseline method to Energy Based Out-of-Distribution Detection.

.. literalinclude:: ../examples/example.py


Depending on the number of iterations, the output will be

.. code :: python

    {'AUROC': 0.601246178150177, 'AUPR-IN': 0.5482655763626099, 'AUPR-OUT': 0.6618213057518005, 'ACC95TPR': 0.3585677742958069, 'FPR95TPR': 0.9750000238418579}
    {'AUROC': 0.6039068698883057, 'AUPR-IN': 0.5614069104194641, 'AUPR-OUT': 0.6644740700721741, 'ACC95TPR': 0.3593989908695221, 'FPR95TPR': 0.9736999869346619}
