"""
Checks the metadata (``info``) of detectors, losses, datasets and benchmarks: that
every exported component has its own record of the right type, and that detectors
and losses really support the tasks their ``info`` claims.
"""

import inspect
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pytorch_ood import benchmark, detector, loss
from pytorch_ood.api import (
    BenchmarkInfo,
    DatasetInfo,
    Detector,
    DetectorInfo,
    LogitsDetector,
    LossInfo,
    Role,
    Task,
)
from pytorch_ood.dataset import audio, img, txt
from tests.detectors.test_all_detectors_smoke import TestAllDetectorsSmoke
from tests.helpers import SegmentationModel


def _exported(module, base, prefix):
    # a set of objects, so re-exported aliases count once
    return sorted(
        {
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, base)
            and not inspect.isabstract(obj)
            and not obj.__name__.startswith("_")
            and obj.__module__.startswith(prefix)
        },
        key=lambda cls: cls.__name__,
    )


DETECTORS = _exported(detector, Detector, "pytorch_ood.detector")
LOSSES = _exported(loss, nn.Module, "pytorch_ood.loss")
BENCHMARKS = _exported(benchmark, benchmark.Benchmark, "pytorch_ood.benchmark")
# generic loaders for the user's own data describe no dataset of their own
DATASETS = [
    cls
    for package in (img, txt, audio)
    for cls in _exported(package, Dataset, "pytorch_ood.dataset")
    if cls is not img.ImageListDataset
]

B, C, D = 8, 3, 5  # batch size, classes, feature dimension
SHAPES = {Task.CLASSIFICATION: (B,), Task.SEGMENTATION: (2, 4, 4)}


def _outputs(shape, channels):
    # channels second, as in (B, C) and (B, C, H, W)
    return torch.randn(shape[0], channels, *shape[1:])


def _targets(shape, outliers=True):
    targets = torch.arange(torch.Size(shape).numel()).reshape(shape) % (C + 1) - 1
    return targets if outliers else targets.clamp(min=0)


# Builds a loss and its arguments for outputs of the given shape (see SHAPES).
LOSS_INPUTS = {
    loss.BackgroundClassLoss: lambda s: (
        loss.BackgroundClassLoss(n_classes=C),
        (_outputs(s, C + 1), _targets(s)),
    ),
    loss.CACLoss: lambda s: (
        loss.CACLoss(n_classes=C),
        (_outputs(s, C).abs(), _targets(s, False)),
    ),
    loss.CenterLoss: lambda s: (
        loss.CenterLoss(n_classes=C, n_dim=D),
        (_outputs(s, C).abs(), _targets(s, False)),
    ),
    loss.ConfidenceLoss: lambda s: (
        loss.ConfidenceLoss(),
        (_outputs(s, C), torch.rand(s[0], 1), _targets(s, False)),
    ),
    loss.CrossEntropyLoss: lambda s: (loss.CrossEntropyLoss(), (_outputs(s, C), _targets(s))),
    loss.DeepSVDDLoss: lambda s: (
        loss.DeepSVDDLoss(n_dim=D),
        (_outputs(s, D), _targets(s, False)),
    ),
    loss.EnergyMarginLoss: lambda s: (
        loss.EnergyMarginLoss(full_train_loss=np.float32(1.0)),
        (_outputs(s, C), _targets(s), nn.Linear(1, 2)),
    ),
    loss.EnergyRegularizedLoss: lambda s: (
        loss.EnergyRegularizedLoss(),
        (_outputs(s, C), _targets(s)),
    ),
    loss.EntropicOpenSetLoss: lambda s: (
        loss.EntropicOpenSetLoss(),
        (_outputs(s, C), _targets(s)),
    ),
    loss.IILoss: lambda s: (
        loss.IILoss(n_classes=C, n_embedding=D),
        (_outputs(s, D), _targets(s, False)),
    ),
    loss.LogitNorm: lambda s: (loss.LogitNorm(), (_outputs(s, C), _targets(s, False))),
    loss.MCHADLoss: lambda s: (
        loss.MCHADLoss(n_classes=C, n_dim=D),
        (_outputs(s, C).abs(), _targets(s)),
    ),
    loss.ObjectosphereLoss: lambda s: (
        loss.ObjectosphereLoss(),
        (_outputs(s, C), _outputs(s, D), _targets(s)),
    ),
    loss.OutlierExposureLoss: lambda s: (
        loss.OutlierExposureLoss(),
        (_outputs(s, C), _targets(s)),
    ),
    loss.DeepSADLoss: lambda s: (
        loss.DeepSADLoss(n_features=D),
        (_outputs(s, D), _targets(s)),
    ),
    loss.VOSRegLoss: lambda s: (
        loss.VOSRegLoss(nn.Linear(1, 2), nn.Linear(C, 1)),
        (_outputs(s, C), _targets(s, False)),
    ),
    loss.VirtualOutlierSynthesizingRegLoss: lambda s: (
        loss.VirtualOutlierSynthesizingRegLoss(
            nn.Linear(1, 2),
            nn.Linear(C, 1),
            device="cpu",
            num_classes=C,
            num_input_last_layer=D,
            fc=nn.Linear(D, C),
            sample_number=4,
            sample_from=20,
        ),
        (_outputs(s, C), _outputs(s, D), _targets(s, False)),
    ),
}


class _ConvBackbone(nn.Module):
    """Feature maps (B, 8, H, W) and a per-pixel head (B, C, H, W), for segmentation."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU())
        self.head = nn.Conv2d(8, C, 1)


def _with_backbone(cls):
    def build():
        model = _ConvBackbone().eval()
        return cls(backbone=model.features, head=model.head)

    return build


# Builds each detector that claims segmentation, for (B, 3, H, W) inputs.
SEGMENTATION_DETECTORS = {
    detector.EnergyBased: lambda: detector.EnergyBased(SegmentationModel().eval()),
    detector.Entropy: lambda: detector.Entropy(SegmentationModel().eval()),
    detector.GEN: lambda: detector.GEN(SegmentationModel().eval()),
    detector.MaxLogit: lambda: detector.MaxLogit(SegmentationModel().eval()),
    detector.MaxSoftmax: lambda: detector.MaxSoftmax(SegmentationModel().eval()),
    detector.MCD: lambda: detector.MCD(SegmentationModel(), samples=4),
    detector.ReAct: _with_backbone(detector.ReAct),
    detector.WeightedEBO: lambda: detector.WeightedEBO(SegmentationModel().eval(), torch.ones(C)),
}


class TestInfoRecords(unittest.TestCase):
    def assert_own_info(self, cls, info_type):
        # defined on the class itself, not inherited from a parent component
        self.assertIn("info", vars(cls), f"{cls.__name__} has no info of its own")
        self.assertIsInstance(cls.info, info_type)

    def test_detectors(self):
        for cls in DETECTORS:
            with self.subTest(cls.__name__):
                self.assert_own_info(cls, DetectorInfo)
                self.assertTrue(cls.info.tasks)

    def test_losses(self):
        for cls in LOSSES:
            with self.subTest(cls.__name__):
                self.assert_own_info(cls, LossInfo)
                self.assertTrue(cls.info.tasks)
                self.assertTrue(cls.info.inputs)

    def test_datasets(self):
        for cls in DATASETS:
            with self.subTest(cls.__name__):
                self.assert_own_info(cls, DatasetInfo)
                # text datasets have no established roles in OOD experiments
                if cls.__module__.startswith("pytorch_ood.dataset.img"):
                    self.assertTrue(cls.info.roles)

    def test_benchmarks(self):
        for cls in BENCHMARKS:
            with self.subTest(cls.__name__):
                self.assert_own_info(cls, BenchmarkInfo)
                self.assertTrue(cls.info.tasks)

    def test_sets_are_frozen(self):
        info = DatasetInfo(task=Task.CLASSIFICATION, roles=[Role.OOD_TEST], license=None)
        self.assertIsInstance(info.roles, frozenset)
        with self.assertRaises(AttributeError):
            info.license = "MIT"


class TestDetectorTasks(unittest.TestCase):
    """Every task a detector's ``info`` lists must work."""

    def test_classification_claims_are_smoke_tested(self):
        # tests/detectors/test_all_detectors_smoke.py runs every detector it builds
        smoke = TestAllDetectorsSmoke("test_all_classification_detectors_smoke")
        registries = smoke._classification_detector_registry() + smoke._image_detector_registry()
        # by name: the smoke test imports the package as src.pytorch_ood
        tested = {type(build()).__qualname__ for _, build in registries}
        for cls in DETECTORS:
            if Task.CLASSIFICATION in cls.info.tasks:
                with self.subTest(cls.__name__):
                    self.assertIn(cls.__qualname__, tested, "missing from the smoke test")

    def test_segmentation(self):
        torch.manual_seed(0)
        x = torch.randn(4, 3, 8, 8)
        # fit on in-distribution images (one label per image), score per pixel
        loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=4)
        for cls in DETECTORS:
            if Task.SEGMENTATION not in cls.info.tasks:
                continue
            with self.subTest(cls.__name__):
                self.assertIn(
                    cls, SEGMENTATION_DETECTORS, "add a builder to SEGMENTATION_DETECTORS"
                )
                built = SEGMENTATION_DETECTORS[cls]()
                if cls.requires_fit:
                    built.fit(loader)
                scores = built(x)
                self.assertEqual(scores.shape, (4, 8, 8))
                self.assertTrue(torch.isfinite(scores).all())


class TestLossTasks(unittest.TestCase):
    """Every task a loss's ``info`` lists must work."""

    def test_tasks(self):
        torch.manual_seed(0)
        for cls in LOSSES:
            for task in sorted(cls.info.tasks):
                with self.subTest(f"{cls.__name__}, {task.value}"):
                    self.assertIn(cls, LOSS_INPUTS, "add an entry to LOSS_INPUTS")
                    criterion, inputs = LOSS_INPUTS[cls](SHAPES[task])
                    value = criterion(*inputs)
                    self.assertEqual(value.dim(), 0)
                    self.assertTrue(torch.isfinite(value))


def _pixels_as_samples(tensor):
    """(B, C, H, W) -> (B * H * W, C): every pixel as a sample of its own."""
    return tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])


class TestSegmentationSemantics(unittest.TestCase):
    """
    Accepting (B, C, H, W) inputs is not enough: a component that scores or trains
    per pixel must treat every pixel exactly like a classification input. Checked
    where this equivalence is the definition of the method's segmentation variant.
    """

    def test_logits_detectors_score_each_pixel_independently(self):
        torch.manual_seed(0)
        logits = torch.randn(2, C, 4, 4)
        for cls in DETECTORS:
            if Task.SEGMENTATION not in cls.info.tasks or not issubclass(cls, LogitsDetector):
                continue
            with self.subTest(cls.__name__):
                built = SEGMENTATION_DETECTORS[cls]()
                per_pixel = built.predict_logits(logits)
                as_samples = built.predict_logits(_pixels_as_samples(logits)).reshape(2, 4, 4)
                torch.testing.assert_close(per_pixel, as_samples)

    def test_losses_treat_each_pixel_as_a_sample(self):
        torch.manual_seed(0)
        for cls in LOSSES:
            if Task.SEGMENTATION not in cls.info.tasks:
                continue
            with self.subTest(cls.__name__):
                criterion, (outputs, targets) = LOSS_INPUTS[cls](SHAPES[Task.SEGMENTATION])
                per_pixel = criterion(outputs, targets)
                as_samples = criterion(_pixels_as_samples(outputs), targets.reshape(-1))
                torch.testing.assert_close(per_pixel, as_samples)
