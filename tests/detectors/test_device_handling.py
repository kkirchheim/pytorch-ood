import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import (
    ASH,
    DICE,
    GEN,
    GMM,
    KNN,
    MCD,
    NACUE,
    NCI,
    NECO,
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


#: Seed used for every model built by the registries below. It is not arbitrary: SHE refuses to
#: fit unless the (untrained) model predicts at least one sample of every class correctly on
#: ``_classification_loader()``, so the seed and that loader have to agree. See
#: :class:`TestDeviceRegistryFixtures`, which fails loudly if this stops holding.
_MODEL_SEED = 0


def _seeded_classification_model(**kwargs) -> ClassificationModel:
    """
    Build a model from a fixed seed, so that every registry entry sees identical weights no
    matter where it sits in the registry.

    Several detectors have fit-time preconditions that depend on the randomly initialised
    weights. Without seeding, inserting a new entry shifts the RNG stream and breaks an
    unrelated detector further down the list.
    """
    torch.manual_seed(_MODEL_SEED)
    return ClassificationModel(**kwargs)


def _seeded_conv_model(**kwargs) -> "TinyConvDetectorModel":
    """
    Seeded counterpart of :func:`_seeded_classification_model` for the feature-map detectors.
    """
    torch.manual_seed(_MODEL_SEED)
    return TinyConvDetectorModel(**kwargs)


class TestDeviceRegistryFixtures(unittest.TestCase):
    """
    Guards the shared fixtures of the (CUDA-only) registries, so that a broken assumption is
    reported here instead of surfacing as a confusing failure inside an unrelated detector.
    """

    def test_seeded_model_satisfies_she_precondition(self):
        model = _seeded_classification_model().eval()
        loader = TestDetectorDeviceHandling._classification_loader()

        x = torch.cat([batch for batch, _ in loader])
        y = torch.cat([labels for _, labels in loader])
        with torch.no_grad():
            y_hat = model(x).argmax(dim=1)

        for clazz in y.unique():
            self.assertTrue(
                (y_hat[y == clazz] == clazz).any(),
                f"_MODEL_SEED={_MODEL_SEED} yields no correct prediction for class "
                f"{clazz.item()}; SHE cannot be fitted with it",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device handling tests")
class TestDetectorDeviceHandling(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.device = torch.device("cuda:0")
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
    def _assert_scores(scores: torch.Tensor, device: torch.device, batch_size: int) -> None:
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (batch_size,)
        assert scores.device == device
        assert torch.isfinite(scores).all()

    @staticmethod
    def _classification_fit_and_predict_features_registry():
        return [
            (
                "KNN",
                lambda: (lambda model: KNN(model.features))(_seeded_classification_model()),
            ),
            (
                "GMM",
                lambda: (lambda model: GMM(model.features))(_seeded_classification_model()),
            ),
            (
                "PNML",
                lambda: (lambda model: PNML(model.features, model.classifier))(
                    _seeded_classification_model()
                ),
            ),
            (
                "fDBD",
                lambda: (lambda model: fDBD(model.features, model.classifier))(
                    _seeded_classification_model()
                ),
            ),
            (
                "Mahalanobis",
                lambda: (lambda model: Mahalanobis(model.features))(
                    _seeded_classification_model()
                ),
            ),
            (
                "RMD",
                lambda: (lambda model: RMD(model.features))(_seeded_classification_model()),
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
                )(_seeded_classification_model()),
            ),
            (
                "NCI",
                lambda: (
                    lambda model: NCI(
                        encoder=model.features,
                        head=model.classifier,
                        alpha=0.0,
                    )
                )(_seeded_classification_model()),
            ),
            (
                "NECO",
                lambda: (
                    lambda model: NECO(
                        encoder=model.features,
                        head=model.classifier,
                        d=4,
                    )
                )(_seeded_classification_model()),
            ),
            (
                "SHE",
                lambda: (lambda model: SHE(model.features, model.classifier))(
                    _seeded_classification_model()
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
                )(_seeded_classification_model()),
            ),
            (
                "NNGuide",
                lambda: (
                    lambda model: NNGuide(
                        model.features,
                        model.classifier,
                        k=3,
                    )
                )(_seeded_classification_model()),
            ),
        ]

    @staticmethod
    def _classification_fit_and_predict_logits_registry():
        def make_eval_model():
            model = _seeded_classification_model()
            model.eval()
            return model

        return [
            ("MSP", lambda: MaxSoftmax(make_eval_model())),
            ("TemperatureScaling", lambda: TemperatureScaling(make_eval_model())),
            ("Entropy", lambda: Entropy(make_eval_model())),
            ("EnergyBased", lambda: EnergyBased(make_eval_model())),
            ("MaxLogit", lambda: MaxLogit(make_eval_model())),
            ("GEN", lambda: GEN(make_eval_model())),
            ("KLMatching", lambda: KLMatching(make_eval_model())),
            ("OpenMax", lambda: OpenMax(make_eval_model(), tailsize=3, alpha=2)),
            (
                "WeightedEBO",
                lambda: WeightedEBO(make_eval_model(), torch.randn(1, 3)),
            ),
        ]

    @staticmethod
    def _raw_predict_registry():
        def make_gradnorm_model():
            model = _seeded_classification_model()
            model.requires_grad_(False)
            model.classifier.requires_grad_(True)
            model.eval()
            return model

        return [
            ("ODIN", lambda: ODIN(_seeded_classification_model().eval(), eps=0.001)),
            ("MCD", lambda: MCD(_seeded_classification_model().eval(), samples=4, mode="var")),
            (
                "GradNorm",
                lambda: GradNorm(
                    make_gradnorm_model(),
                    param_filter=lambda name: name.startswith("classifier"),
                ),
            ),
            (
                "GradNormKL",
                lambda: GradNormKL(
                    make_gradnorm_model(),
                    param_filter=lambda name: name.startswith("classifier"),
                ),
            ),
        ]

    @staticmethod
    def _feature_map_registry():
        return [
            (
                "ASH",
                lambda: (
                    lambda model: ASH(
                        backbone=model.feature_maps,
                        head=model.forward_feature_maps,
                    )
                )(_seeded_conv_model()),
                False,
                "maps",
            ),
            (
                "RankFeat",
                lambda: (
                    lambda model: RankFeat(
                        backbone=model.feature_maps,
                        head=model.forward_feature_maps,
                    )
                )(_seeded_conv_model()),
                False,
                "maps",
            ),
            (
                "ReAct",
                lambda: (
                    lambda model: ReAct(
                        backbone=model.features,
                        head=model.classifier,
                    )
                )(_seeded_classification_model()),
                True,
                "features",
            ),
            (
                "VRA",
                lambda: (
                    lambda model: VRA(
                        backbone=model.features,
                        head=model.classifier,
                    )
                )(_seeded_classification_model()),
                True,
                "features",
            ),
        ]

    @staticmethod
    def _structured_registry():
        return [
            (
                "MultiMahalanobis",
                lambda: (lambda model: MultiMahalanobis([model.conv1, model.relu]))(
                    _seeded_conv_model()
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
                )(_seeded_conv_model()),
            ),
        ]

    def test_to_moves_detector_state(self):
        model = _seeded_classification_model()
        detector = MaxSoftmax(model)

        detector.to(self.device)

        self.assertEqual(detector.device, self.device)
        self.assertEqual(next(detector.model.parameters()).device, self.device)
        self.assertEqual(detector.t.device, self.device)

    def test_vim_infers_device_from_constructor_tensors_without_explicit_to(self):
        # ViM must not silently pin its state to CPU when constructed directly from
        # GPU-resident weights, without an explicit detector.to(device) call.
        model = _seeded_classification_model().to(self.device)
        detector = ViM(model.features, d=4, w=model.classifier.weight, b=model.classifier.bias)

        self.assertEqual(detector.w.device, self.device)
        self.assertEqual(detector.u.device, self.device)

        detector.fit(self.classification_loader)
        self.assertEqual(detector.principal_subspace.device, self.device)

        scores = detector(torch.randn(8, 10, device=self.device))
        self._assert_scores(scores, self.device, batch_size=8)

    def test_logits_fit_and_predict_logits_accept_cpu_tensors_for_cuda_detector(self):
        cached_logits = torch.randn(8, 3)

        for (
            detector_name,
            builder,
        ) in self._classification_fit_and_predict_logits_registry():
            with self.subTest(detector=detector_name):
                detector = builder().to(self.device)

                if getattr(detector, "requires_fit", False):
                    detector.fit(self.classification_loader)

                scores = detector.predict_logits(cached_logits)
                self._assert_scores(scores, self.device, batch_size=8)

    def test_feature_fit_and_predict_features_accept_cpu_tensors_for_cuda_detector(
        self,
    ):
        cached_features = torch.randn(8, 10)

        for (
            detector_name,
            builder,
        ) in self._classification_fit_and_predict_features_registry():
            with self.subTest(detector=detector_name):
                detector = builder().to(self.device)
                detector.fit(self.classification_loader)

                scores = detector.predict_features(cached_features)
                self._assert_scores(scores, self.device, batch_size=8)

    def test_feature_map_interfaces_accept_cpu_tensors_for_cuda_detector(self):
        cached_feature_maps = torch.randn(8, 8, 8, 8)
        cached_features = torch.randn(8, 10)

        for (
            detector_name,
            builder,
            requires_fit,
            input_kind,
        ) in self._feature_map_registry():
            with self.subTest(detector=detector_name):
                detector = builder().to(self.device)

                if requires_fit:
                    detector.fit(self.classification_loader)
                if input_kind == "features":
                    scores = detector.predict_feature_maps(cached_features)
                else:
                    scores = detector.predict_feature_maps(cached_feature_maps)

                self._assert_scores(scores, self.device, batch_size=8)

    def test_structured_interfaces_accept_cpu_tensors_for_cuda_detector(self):
        for detector_name, builder in self._structured_registry():
            with self.subTest(detector=detector_name):
                detector = builder().to(self.device)
                detector.fit(self.image_loader)

                if detector_name == "MultiMahalanobis":
                    zs = [torch.randn(8, 8), torch.randn(8, 8)]
                    scores = detector.predict_structured(zs)
                else:
                    logits = torch.randn(8, 3)
                    feature_list = [torch.randn(8, 8, 8, 8), torch.randn(8, 8, 8, 8)]
                    scores = detector.predict_structured(logits, feature_list)

                self._assert_scores(scores, self.device, batch_size=8)

    def test_raw_predict_accepts_cpu_inputs_for_cuda_detector(self):
        raw_inputs = torch.randn(8, 10)

        for detector_name, builder in self._raw_predict_registry():
            with self.subTest(detector=detector_name):
                detector = builder().to(self.device)
                scores = detector(raw_inputs)
                self._assert_scores(scores, self.device, batch_size=8)

    def test_nacue_accepts_cpu_inputs_for_cuda_detector(self):
        model = _seeded_conv_model().to(self.device)
        detector = NACUE(
            model=model,
            layers=[model.conv1, model.relu],
            m_bins=[10, 10],
            alpha=[5.0, 5.0],
            o_star=[2, 2],
            device=self.device,
        ).to(self.device)

        detector.fit(self.image_loader)
        scores = detector(torch.randn(8, 3, 8, 8))
        self._assert_scores(scores, self.device, batch_size=8)
