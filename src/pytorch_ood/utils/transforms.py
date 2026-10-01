"""

..  autoclass:: pytorch_ood.utils.ToUnknown
    :members:

..  autoclass:: pytorch_ood.utils.ToRGB
    :members:

..  autoclass:: pytorch_ood.utils.TargetMapping
    :members:

"""

from typing import Set

import torch


class ToUnknown(object):
    """
    Target transform that maps every target to ``-1`` (unknown/OOD), regardless of its value.
    Used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1


class ToRGB(object):
    """
    Convert a PIL image to RGB, if it is not already. Inputs without a ``convert`` method are
    returned unchanged.
    """

    def __call__(self, x):
        try:
            return x.convert("RGB")
        except Exception as e:
            return x


class TargetMapping(object):
    """
    Maps ID (a.k.a. known) classes to labels :math:`\\{0, \\dots, n-1\\}` for :math:`n` known
    classes, in ascending order of the class ids, and OOD (a.k.a. unknown) classes :math:`c` to
    :math:`-(c+1)`.
    This is required for open set simulations.

    Classes that are in neither set are mapped to ``-1``.

    **Example:**
    If we split up a dataset so that the classes 2,3,4,9 are considered *known* or *ID*, these class
    labels have to be remapped to 0,1,2,3 to be able to train
    using cross entropy with 1-of-K-vectors. All other classes have to be mapped to values :math:`<0`
    to be marked as OOD.
    """

    def __init__(self, known: Set, unknown: Set):
        """
        :param known: set of integer class ids that are in-distribution; mapped to their rank, i.e.
            the smallest to ``0``
        :param unknown: set of non-negative integer class ids that are out-of-distribution; class
            :math:`c` is mapped to :math:`-(c+1)`
        :raises ValueError: if a class is in both sets
        """
        overlap = set(known) & set(unknown)
        if overlap:
            raise ValueError(f"Classes in both known and unknown: {sorted(overlap)}")

        self._map = dict()
        self._map.update({clazz: index for index, clazz in enumerate(sorted(set(known)))})
        # shifted by one, so that class 0 is mapped to a negative label as well
        self._map.update({clazz: -(clazz + 1) for clazz in set(unknown)})

    def __call__(self, target):
        if isinstance(target, torch.Tensor):
            return self._map.get(target.item(), -1)

        return self._map.get(target, -1)

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            return self._map[item.item()]

        return self._map[item]

    def items(self):
        return self._map.items()

    def __repr__(self):
        return str(self._map)
