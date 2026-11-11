"""
Checks the metadata (``info``) of detectors, losses, datasets and benchmarks: that
every exported component has its own record of the right type, and that detectors
and losses really support the tasks their ``info`` claims.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import (
    BenchmarkInfo,
    DatasetInfo,
    DetectorInfo,
    LogitsDetector,
    LossInfo,
    Role,
    Task,
)
from tests.helpers.components import BENCHMARKS, DATASETS, DETECTORS, LOSSES
from tests.helpers.detectors import DETECTOR_CASES
from tests.helpers.losses import LOSS_INPUTS, SHAPES, C
from tests.helpers.segmentation import SEGMENTATION_DETECTORS


class TestInfoRecords:
    def assert_own_info(self, cls, info_type):
        # defined on the class itself, not inherited from a parent component
        assert "info" in vars(cls), f"{cls.__name__} has no info of its own"
        assert isinstance(cls.info, info_type)

    @pytest.mark.parametrize("cls", DETECTORS)
    def test_detectors(self, cls):
        self.assert_own_info(cls, DetectorInfo)
        assert cls.info.tasks

    @pytest.mark.parametrize("cls", LOSSES)
    def test_losses(self, cls):
        self.assert_own_info(cls, LossInfo)
        assert cls.info.tasks
        assert cls.info.inputs

    @pytest.mark.parametrize("cls", DATASETS)
    def test_datasets(self, cls):
        self.assert_own_info(cls, DatasetInfo)
        # text datasets have no established roles in OOD experiments
        if cls.__module__.startswith("pytorch_ood.dataset.img"):
            assert cls.info.roles

    @pytest.mark.parametrize("cls", BENCHMARKS)
    def test_benchmarks(self, cls):
        self.assert_own_info(cls, BenchmarkInfo)
        assert cls.info.tasks

    def test_sets_are_frozen(self):
        info = DatasetInfo(task=Task.CLASSIFICATION, roles=[Role.OOD_TEST], license=None)
        assert isinstance(info.roles, frozenset)
        with pytest.raises(AttributeError):
            info.license = "MIT"


def test_classification_claims_are_smoke_tested():
    # tests/detector/test_contract.py runs every case in DETECTOR_CASES
    tested = {type(case.build()) for case in DETECTOR_CASES}
    for cls in DETECTORS:
        if Task.CLASSIFICATION in cls.info.tasks:
            assert cls in tested, "missing from DETECTOR_CASES"


def test_segmentation():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    # fit on in-distribution images (one label per image), score per pixel
    loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=4)
    for cls in DETECTORS:
        if Task.SEGMENTATION not in cls.info.tasks:
            continue
        assert cls in SEGMENTATION_DETECTORS, "add a builder to SEGMENTATION_DETECTORS"
        built = SEGMENTATION_DETECTORS[cls]()
        if cls.requires_fit:
            built.fit(loader)
        scores = built(x)
        assert scores.shape == (4, 8, 8)
        assert torch.isfinite(scores).all()


def test_tasks():
    torch.manual_seed(0)
    for cls in LOSSES:
        for task in sorted(cls.info.tasks):
            assert cls in LOSS_INPUTS, "add an entry to LOSS_INPUTS"
            criterion, inputs = LOSS_INPUTS[cls](SHAPES[task])
            value = criterion(*inputs)
            assert value.dim() == 0
            assert torch.isfinite(value)


def _pixels_as_samples(tensor):
    """(B, C, H, W) -> (B * H * W, C): every pixel as a sample of its own."""
    return tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])


def test_logits_detectors_score_each_pixel_independently():
    torch.manual_seed(0)
    logits = torch.randn(2, C, 4, 4)
    for cls in DETECTORS:
        if Task.SEGMENTATION not in cls.info.tasks or not issubclass(cls, LogitsDetector):
            continue
        built = SEGMENTATION_DETECTORS[cls]()
        per_pixel = built.predict_logits(logits)
        as_samples = built.predict_logits(_pixels_as_samples(logits)).reshape(2, 4, 4)
        torch.testing.assert_close(per_pixel, as_samples)


def test_losses_treat_each_pixel_as_a_sample():
    torch.manual_seed(0)
    for cls in LOSSES:
        if Task.SEGMENTATION not in cls.info.tasks:
            continue
        criterion, (outputs, targets) = LOSS_INPUTS[cls](SHAPES[Task.SEGMENTATION])
        per_pixel = criterion(outputs, targets)
        as_samples = criterion(_pixels_as_samples(outputs), targets.reshape(-1))
        torch.testing.assert_close(per_pixel, as_samples)
