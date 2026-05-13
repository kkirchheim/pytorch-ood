import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import (
    ASH,
    DICE,
    GEN,
    GMM,
    KNN,
    LTS,
    MCD,
    MCM,
    NACUE,
    NCI,
    ODIN,
    PNML,
    RMD,
    SHE,
    VRA,
    EnergyBased,
    Entropy,
    GradNorm,
    GradNormKL,
    Gram,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    MultiMahalanobis,
    NNGuide,
    OpenMax,
    RankFeat,
    ReAct,
    TemperatureScaling,
    ViM,
    WeightedEBO,
    fDBD,
)
from tests.helpers import ClassificationModel


class TinyConvDetectorModel(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=8, num_outputs=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(hidden_channels, num_outputs)

        # aliases used by several detectors
        self.block1 = self.relu
        self.block2 = torch.nn.Identity()
        self.block3 = torch.nn.Identity()
        self.bn1 = torch.nn.Identity()

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv1(x))

    def forward_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(self.pool(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(self.feature_maps(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class TestAllDetectorsSmoke(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.classification_loader = self._classification_loader()
        self.image_loader = self._image_loader()

    @staticmethod
    def _classification_loader(
        n_samples: int = 24, batch_size: int = 6, in_dim: int = 10, n_classes: int = 3
    ) -> DataLoader:
        x = torch.randn(n_samples, in_dim)
        y = torch.arange(n_samples) % n_classes
        return DataLoader(TensorDataset(x, y), batch_size=batch_size)

    @staticmethod
    def _image_loader(
        n_samples: int = 24,
        batch_size: int = 6,
        image_shape=(3, 8, 8),
        n_classes: int = 3,
    ) -> DataLoader:
        x = torch.randn(n_samples, *image_shape)
        y = torch.arange(n_samples) % n_classes
        return DataLoader(TensorDataset(x, y), batch_size=batch_size)

    @staticmethod
    def _classification_loader_matching_model(
        model: torch.nn.Module,
        n_samples: int = 24,
        batch_size: int = 6,
        in_dim: int = 10,
    ) -> DataLoader:
        x = torch.randn(n_samples, in_dim)
        with torch.no_grad():
            y = model(x).argmax(dim=1)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size)

    @staticmethod
    def _make_she_model() -> ClassificationModel:
        model = ClassificationModel(num_inputs=10, n_hidden=3, num_outputs=3)
        model.eval()
        with torch.no_grad():
            model.layer1.weight.zero_()
            model.layer1.bias.zero_()
            model.layer1.weight[:, :3] = torch.eye(3)
            model.classifier.weight.copy_(torch.eye(3))
            model.classifier.bias.zero_()
        return model

    @staticmethod
    def _she_fit_loader(batch_size: int = 6) -> DataLoader:
        x = torch.zeros(24, 10)
        y = torch.arange(24) % 3
        x[torch.arange(24), y] = 4.0
        return DataLoader(TensorDataset(x, y), batch_size=batch_size)

    @staticmethod
    def _assert_scores(scores: torch.Tensor, batch_size: int) -> None:
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (batch_size,)
        assert torch.isfinite(scores).all()

    def _classification_detector_registry(self):
        def eval_model():
            model = ClassificationModel()
            model.eval()
            return model

        def gradnorm_model():
            model = ClassificationModel()
            model.requires_grad_(False)
            model.classifier.requires_grad_(True)
            model.eval()
            return model

        return [
            ("MSP", lambda: MaxSoftmax(eval_model())),
            ("TemperatureScaling", lambda: TemperatureScaling(eval_model())),
            ("Entropy", lambda: Entropy(eval_model())),
            ("EnergyBased", lambda: EnergyBased(eval_model())),
            ("MaxLogit", lambda: MaxLogit(eval_model())),
            ("GEN", lambda: GEN(eval_model())),
            ("KLMatching", lambda: KLMatching(eval_model())),
            ("ODIN", lambda: ODIN(eval_model(), eps=0.001)),
            ("OpenMax", lambda: OpenMax(eval_model(), tailsize=3, alpha=2)),
            ("WeightedEBO", lambda: WeightedEBO(eval_model(), torch.randn(1, 3))),
            ("MCD", lambda: MCD(eval_model(), samples=4, mode="var")),
            (
                "KNN",
                lambda: (lambda model: KNN(model.features))(eval_model()),
            ),
            (
                "GMM",
                lambda: (lambda model: GMM(model.features))(eval_model()),
            ),
            (
                "MCM",
                lambda: (
                    lambda model: MCM(
                        encoder=model.features,
                        text_embeddings=F.normalize(torch.randn(3, 10), dim=-1),
                        temperature=1.0,
                    )
                )(eval_model()),
            ),
            (
                "PNML",
                lambda: (lambda model: PNML(model.features, model.classifier))(eval_model()),
            ),
            (
                "NNGuide",
                lambda: (
                    lambda model: NNGuide(
                        model.features,
                        model.classifier,
                        k=3,
                    )
                )(eval_model()),
            ),
            (
                "fDBD",
                lambda: (lambda model: fDBD(model.features, model.classifier))(eval_model()),
            ),
            (
                "Mahalanobis",
                lambda: (lambda model: Mahalanobis(model.features))(eval_model()),
            ),
            (
                "RMD",
                lambda: (lambda model: RMD(model.features))(eval_model()),
            ),
            (
                "ViM",
                lambda: (
                    lambda model: ViM(
                        model.features,
                        d=4,
                        w=model.classifier.weight,
                        b=model.classifier.bias,
                    )
                )(eval_model()),
            ),
            (
                "NCI",
                lambda: (
                    lambda model: NCI(
                        encoder=model.features,
                        head=model.classifier,
                        alpha=0.0,
                    )
                )(eval_model()),
            ),
            (
                "SHE",
                lambda: (lambda model: SHE(model.features, model.classifier))(
                    self._make_she_model()
                ),
            ),
            (
                "DICE",
                lambda: (
                    lambda model: DICE(
                        encoder=model.features,
                        w=model.classifier.weight,
                        b=model.classifier.bias,
                        p=65.0,
                    )
                )(eval_model()),
            ),
            (
                "LTS",
                lambda: (
                    lambda model: LTS(
                        encoder=model.features,
                        head=model.classifier,
                    )
                )(eval_model()),
            ),
            (
                "ReAct",
                lambda: (
                    lambda model: ReAct(
                        backbone=model.features,
                        head=model.classifier,
                        threshold=1.0,
                    )
                )(eval_model()),
            ),
            (
                "VRA",
                lambda: (
                    lambda model: VRA(
                        backbone=model.features,
                        head=model.classifier,
                    )
                )(eval_model()),
            ),
            (
                "GradNorm",
                lambda: GradNorm(
                    gradnorm_model(),
                    param_filter=lambda name: name.startswith("classifier"),
                ),
            ),
            (
                "GradNormKL",
                lambda: GradNormKL(
                    gradnorm_model(),
                    param_filter=lambda name: name.startswith("classifier"),
                ),
            ),
        ]

    @staticmethod
    def _image_detector_registry():
        return [
            (
                "ASH",
                lambda: (
                    lambda model: ASH(
                        backbone=model.feature_maps,
                        head=model.forward_feature_maps,
                    )
                )(TinyConvDetectorModel().eval()),
            ),
            (
                "RankFeat",
                lambda: (
                    lambda model: RankFeat(
                        backbone=model.feature_maps,
                        head=model.forward_feature_maps,
                    )
                )(TinyConvDetectorModel().eval()),
            ),
            (
                "MultiMahalanobis",
                lambda: (lambda model: MultiMahalanobis([model.conv1, model.relu]))(
                    TinyConvDetectorModel().eval()
                ),
            ),
            (
                "Gram",
                lambda: (
                    lambda model: Gram(
                        head=nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            model.fc,
                        ),
                        feature_layers=[model.conv1, model.relu],
                        num_classes=3,
                        num_poles_list=[1, 2],
                    )
                )(TinyConvDetectorModel().eval()),
            ),
            (
                "NACUE",
                lambda: (
                    lambda model: NACUE(
                        model=model,
                        layers=[model.conv1, model.relu],
                        m_bins=[10, 10],
                        alpha=[5.0, 5.0],
                        o_star=[2, 2],
                    )
                )(TinyConvDetectorModel().eval()),
            ),
        ]

    def test_all_classification_detectors_smoke(self):
        x_eval = torch.randn(8, 10)

        for detector_name, builder in self._classification_detector_registry():
            with self.subTest(detector=detector_name):
                detector = builder()
                if getattr(detector, "requires_fit", False):
                    fit_loader = self.classification_loader
                    if detector_name == "SHE":
                        fit_loader = self._she_fit_loader()
                    detector.fit(fit_loader)

                scores = detector(x_eval)
                self._assert_scores(scores, batch_size=8)

    def test_all_image_detectors_smoke(self):
        x_eval = torch.randn(8, 3, 8, 8)

        for detector_name, builder in self._image_detector_registry():
            with self.subTest(detector=detector_name):
                detector = builder()
                if getattr(detector, "requires_fit", False):
                    detector.fit(self.image_loader)

                scores = detector(x_eval)
                self._assert_scores(scores, batch_size=8)
