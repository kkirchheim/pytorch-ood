from os.path import dirname, join

import pytest
import torch

example_dir = join(dirname(__file__), "..", "examples")


class TestTargetMapping:
    def test_known_classes_in_ascending_order(self):
        from pytorch_ood.utils import TargetMapping

        mapping = TargetMapping(known={9, 2, 3, 4}, unknown={0, 1})
        assert [mapping(c) for c in [2, 3, 4, 9]] == [0, 1, 2, 3]

    def test_unknown_classes_negative(self):
        from pytorch_ood.utils import TargetMapping

        mapping = TargetMapping(known={2, 3}, unknown={0, 1, 5})
        assert [mapping(c) for c in [0, 1, 5]] == [-1, -2, -6]
        assert mapping(torch.tensor(0)) == -1
        # classes in neither set
        assert mapping(7) == -1

    def test_overlap(self):
        from pytorch_ood.utils import TargetMapping

        with pytest.raises(ValueError):
            TargetMapping(known={1, 2}, unknown={2, 3})
