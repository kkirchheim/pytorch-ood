import unittest

from pytorch_ood.api import (
    Detector,
    FeatureMapsDetector,
    FeaturesDetector,
    LogitsDetector,
    StructuredDetector,
)
from pytorch_ood.detector import (
    ASH,
    DICE,
    EnergyBased,
    Entropy,
    fDBD,
    GEN,
    GMM,
    GradNorm,
    GradNormKL,
    Gram,
    KNN,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    MCD,
    MultiMahalanobis,
    NACUE,
    NCI,
    ODIN,
    OpenMax,
    RankFeat,
    ReAct,
    SHE,
    TemperatureScaling,
    ViM,
    VRA,
    WeightedEBO,
)


class TestDetectorTaxonomy(unittest.TestCase):
    def test_logits_detectors_inherit_logits_base(self):
        detectors = (
            MaxSoftmax,
            MaxLogit,
            EnergyBased,
            WeightedEBO,
            Entropy,
            GEN,
            TemperatureScaling,
            KLMatching,
            OpenMax,
        )

        for detector_cls in detectors:
            with self.subTest(detector=detector_cls.__name__):
                self.assertTrue(issubclass(detector_cls, LogitsDetector))

    def test_logits_detector_defaults(self):
        detectors = (
            MaxSoftmax,
            MaxLogit,
            EnergyBased,
            WeightedEBO,
            Entropy,
            GEN,
        )

        for detector_cls in detectors:
            with self.subTest(detector=detector_cls.__name__):
                self.assertFalse(detector_cls.requires_fit)
                self.assertTrue(hasattr(detector_cls, "predict_logits"))
                self.assertTrue(hasattr(detector_cls, "fit_logits"))
                self.assertFalse(hasattr(detector_cls, "predict_features"))
                self.assertFalse(hasattr(detector_cls, "predict_feature_maps"))
                self.assertFalse(hasattr(detector_cls, "predict_structured"))

    def test_fitted_logits_detector_overrides(self):
        self.assertTrue(TemperatureScaling.requires_fit)

        self.assertTrue(OpenMax.requires_fit)

        self.assertTrue(KLMatching.requires_fit)

    def test_explicit_method_families_are_exposed(self):
        self.assertTrue(hasattr(MaxSoftmax, "predict_logits"))
        self.assertTrue(hasattr(TemperatureScaling, "fit_logits"))
        self.assertTrue(hasattr(KLMatching, "predict_logits"))
        self.assertTrue(hasattr(ReAct, "predict_feature_maps"))
        self.assertTrue(hasattr(VRA, "fit_feature_maps"))
        self.assertTrue(hasattr(Gram, "predict_structured"))
        self.assertTrue(hasattr(MultiMahalanobis, "predict_structured"))
        self.assertFalse(hasattr(MaxSoftmax, "predict_features"))
        self.assertFalse(hasattr(TemperatureScaling, "fit_features"))
        self.assertFalse(hasattr(ReAct, "predict_features"))
        self.assertFalse(hasattr(VRA, "fit_features"))
        self.assertFalse(hasattr(Gram, "predict_features"))

    def test_feature_detectors_inherit_features_base(self):
        detectors = (
            KNN,
            Mahalanobis,
            GMM,
            ViM,
            NCI,
            fDBD,
            SHE,
            DICE,
        )

        for detector_cls in detectors:
            with self.subTest(detector=detector_cls.__name__):
                self.assertTrue(issubclass(detector_cls, FeaturesDetector))
                self.assertTrue(detector_cls.requires_fit)
                self.assertTrue(hasattr(detector_cls, "predict_features"))
                self.assertTrue(hasattr(detector_cls, "fit_features"))
                self.assertFalse(hasattr(detector_cls, "predict_logits"))
                self.assertFalse(hasattr(detector_cls, "predict_feature_maps"))
                self.assertFalse(hasattr(detector_cls, "predict_structured"))

    def test_feature_map_detectors_inherit_feature_maps_base(self):
        stateless = (ASH, ReAct, RankFeat)
        for detector_cls in stateless:
            with self.subTest(detector=detector_cls.__name__):
                self.assertTrue(issubclass(detector_cls, FeatureMapsDetector))
                self.assertFalse(detector_cls.requires_fit)
                self.assertTrue(hasattr(detector_cls, "predict_feature_maps"))
                self.assertTrue(hasattr(detector_cls, "fit_feature_maps"))
                self.assertFalse(hasattr(detector_cls, "predict_logits"))
                self.assertFalse(hasattr(detector_cls, "predict_features"))
                self.assertFalse(hasattr(detector_cls, "predict_structured"))

        self.assertTrue(issubclass(VRA, FeatureMapsDetector))
        self.assertTrue(VRA.requires_fit)

    def test_model_only_detectors_inherit_detector_base_only(self):
        detectors = (ODIN, MCD, GradNorm, GradNormKL, NACUE)

        for detector_cls in detectors:
            with self.subTest(detector=detector_cls.__name__):
                self.assertTrue(issubclass(detector_cls, Detector))
                self.assertFalse(issubclass(detector_cls, LogitsDetector))
                self.assertFalse(issubclass(detector_cls, FeaturesDetector))
                self.assertFalse(issubclass(detector_cls, FeatureMapsDetector))
                self.assertFalse(issubclass(detector_cls, StructuredDetector))
                self.assertFalse(hasattr(detector_cls, "predict_logits"))
                self.assertFalse(hasattr(detector_cls, "predict_features"))
                self.assertFalse(hasattr(detector_cls, "predict_feature_maps"))
                self.assertFalse(hasattr(detector_cls, "predict_structured"))

        self.assertTrue(NACUE.requires_fit)
        self.assertFalse(ODIN.requires_fit)
        self.assertFalse(MCD.requires_fit)
        self.assertFalse(GradNorm.requires_fit)
        self.assertFalse(GradNormKL.requires_fit)

    def test_structured_outliers_inherit_structured_base(self):
        self.assertTrue(issubclass(Gram, StructuredDetector))
        self.assertFalse(issubclass(Gram, LogitsDetector))
        self.assertTrue(Gram.requires_fit)
        self.assertTrue(hasattr(Gram, "predict_structured"))
        self.assertTrue(hasattr(Gram, "fit_structured"))

        self.assertTrue(issubclass(MultiMahalanobis, StructuredDetector))
        self.assertFalse(issubclass(MultiMahalanobis, FeaturesDetector))
        self.assertTrue(MultiMahalanobis.requires_fit)
        self.assertTrue(hasattr(MultiMahalanobis, "predict_structured"))
        self.assertTrue(hasattr(MultiMahalanobis, "fit_structured"))

    def test_detector_base_defaults_remain_conservative(self):
        self.assertFalse(Detector.requires_fit)
        self.assertFalse(hasattr(Detector, "predict_logits"))
        self.assertFalse(hasattr(Detector, "predict_features"))
        self.assertFalse(hasattr(Detector, "predict_feature_maps"))
        self.assertFalse(hasattr(Detector, "predict_structured"))
