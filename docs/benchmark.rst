.. automodule:: pytorch_ood.benchmark

Benchmark caching can reuse intermediate logits or pooled features when
evaluating several compatible detectors on the same benchmark.

.. code:: python

    results = benchmark.evaluate(
        [detector1, detector2],
        cache=True,
        cache_dir="cache/",
        cache_key="wrn-cifar10-v1",
    )

Compatible cached detector families in the current implementation are
``LogitsDetector`` and pooled ``FeaturesDetector`` instances. Detectors whose
semantics depend on raw inputs, gradients, feature maps, or structured
multi-layer representations automatically fall back to their standard
``predict(...)`` path. ``Mahalanobis`` with ``eps > 0`` is treated as such a
fallback case explicitly.

.. warning::

    Disk-backed cache reuse is controlled by the user-provided ``cache_key``.
    Cache validity is therefore the caller's responsibility. Change the key
    whenever the model, weights, transforms, or benchmark setup change.
