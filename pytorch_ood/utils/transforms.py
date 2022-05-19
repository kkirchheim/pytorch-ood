"""

..  autoclass:: pytorch_ood.utils.ToUnknown
    :members:

..  autoclass:: pytorch_ood.utils.ToRGB
    :members:

..  autoclass:: pytorch_ood.utils.TargetMapping
    :members:
"""


class ToUnknown(object):
    """
    Callable that returns a negative number, used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1


class ToRGB(object):
    """
    Convert Image to RGB, if it is not already.
    """

    def __call__(self, x):
        try:
            return x.convert("RGB")
        except Exception as e:
            return x


class TargetMapping(object):
    """
    Maps known classes to index in :math:`[0,n]`, unknown classes to values in :math:`[-\\infty, -1]`.
    Required for open set simulations.

    **Example:**
    If we split up a dataset so that the classes 2,3,4,9 are considered *known* or *IN*, these class
    labels have to be remapped to 0,1,2,3 to be able to train
    using cross entropy with 1-of-K-vectors. All other classes have to be mapped to values :math:`<1`.

    Target mappings have to be known at evaluation time.
    """

    def __init__(self, train_in_classes, train_out_classes, test_out_classes):
        self.train_in_classes = train_in_classes
        self.train_out_classes = train_out_classes
        self.test_out_classes = test_out_classes
        self._map = dict()
        self._map.update({clazz: index for index, clazz in enumerate(train_in_classes)})
        # mapping test_out classes to < -1000
        self._map.update({clazz: (-clazz - 1000) for index, clazz in enumerate(test_out_classes)})
        # mapping train_out classes to < 0
        self._map.update({clazz: (-clazz) for index, clazz in enumerate(train_out_classes)})

    def __call__(self, target):
        # log.info(f"Target: {target} known: {target in self._map}")
        return self._map.get(target, -1)

    def __getitem__(self, item):
        return self._map[item]

    def items(self):
        return self._map.items()

    def __repr__(self):
        return str(self._map)
