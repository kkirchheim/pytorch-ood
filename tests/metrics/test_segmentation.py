"""
Segmentation metrics: pooled over all pixels, and per image.
"""

import math
import warnings

import pytest
import torch

from pytorch_ood.metrics import (
    AUPR,
    AUROC,
    AUTC,
    Accuracy,
    AveragePrecision,
    FPRAtTPR,
    MetricCollection,
    OODMetrics,
    OODPerImageSegmentationMetrics,
    OODSegmentationMetrics,
    PerImage,
)
from pytorch_ood.metrics import functional as F
from tests.helpers import requires


def _images(n=4, size=8, seed=0):
    """Score maps and label masks with ID (0, 1) and OOD (-1) pixels in every image."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n, size, size), generator=g)
    labels[:, : size // 2, : size // 2] = -1
    scores = torch.randn(n, size, size, generator=g) + (labels < 0).float()
    return scores, labels


def test_pools_all_pixels():
    scores, labels = _images()
    expected = OODMetrics().update(scores.flatten(), labels.flatten()).compute()
    result = OODSegmentationMetrics().update(scores[:2], labels[:2]).update(scores[2:], labels[2:])
    assert result.compute() == expected
    assert tuple(result.compute()) == ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR")


def test_images_without_ood_pixels_count():
    scores, labels = _images()
    labels[0] = labels[0].clamp(min=0)
    result = OODSegmentationMetrics().update(scores, labels).compute()
    assert result["AUROC"] == float(F.auroc(scores, labels))


def test_void_label_and_fpr_at():
    scores, labels = _images()
    labels[:, -1] = 255
    result = OODSegmentationMetrics(void_label=255, fpr_at=0.9).update(scores, labels)
    keep = labels != 255
    assert result.compute()["FPR90TPR"] == float(F.fpr_at_tpr(scores[keep], labels[keep], 0.9))


class TestPerImage:
    def _expected(self, scores, labels, fn=F.auroc):
        return sum(float(fn(s, y)) for s, y in zip(scores, labels)) / len(scores)

    def test_mean_over_images(self):
        scores, labels = _images()
        metric = PerImage(AUROC())
        metric.update(scores[:3], labels[:3]).update(scores[3:], labels[3:])
        assert metric.compute()["AUROC"] == pytest.approx(self._expected(scores, labels), abs=5e-8)

    @pytest.mark.parametrize("positive, key", (("ood", "AP-OUT"), ("id", "AP-IN")))
    def test_average_precision(self, positive, key):
        scores, labels = _images()
        # few-valued scores, where AP and AUPR differ
        scores = scores.round()
        metric = PerImage(AveragePrecision(positive))
        metric.update(scores[:3], labels[:3]).update(scores[3:], labels[3:])
        fn = lambda s, y: F.average_precision(s, y, positive)  # noqa: E731
        assert metric.compute()[key] == pytest.approx(
            self._expected(scores, labels, fn), abs=5e-13
        )
        pooled = AveragePrecision(positive).update(scores, labels).compute()[key]
        assert metric.compute()[key] != pytest.approx(pooled, abs=5e-8)

    def test_average_precision_skips_images_without_ood(self):
        scores, labels = _images()
        labels[1] = labels[1].clamp(min=0)  # only ID
        metric = PerImage(AveragePrecision()).update(scores, labels)
        keep = [0, 2, 3]
        with pytest.warns(UserWarning, match="Skipped 1 of 4 images"):
            result = metric.compute()
        expected = self._expected(scores[keep], labels[keep], F.average_precision)
        assert result["AP-OUT"] == pytest.approx(expected, abs=5e-13)

    def test_collection_matches_single_metrics(self):
        scores, labels = _images()
        result = OODPerImageSegmentationMetrics().update(scores, labels).compute()
        # AP is not part of the collection
        assert tuple(result) == ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR")
        for key, fn in (
            ("AUROC", F.auroc),
            ("AUTC", F.autc),
            ("AUPR-IN", lambda s, y: F.aupr(s, y, positive="id")),
            ("AUPR-OUT", lambda s, y: F.aupr(s, y, positive="ood")),
            ("FPR95TPR", lambda s, y: F.fpr_at_tpr(s, y, 0.95)),
        ):
            assert result[key] == pytest.approx(self._expected(scores, labels, fn), abs=5e-8), key

    def test_skips_images_without_id_or_ood(self):
        scores, labels = _images(n=5)
        labels[1] = labels[1].clamp(min=0)  # only ID
        labels[2] = -1  # only OOD
        labels[3] = 255  # only void
        metric = PerImage(AUROC(), void_label=255).update(scores, labels)
        keep = [0, 4]
        with pytest.warns(UserWarning, match="Skipped 3 of 5 images"):
            result = metric.compute()
        assert result["AUROC"] == pytest.approx(
            self._expected(scores[keep], labels[keep]), abs=5e-8
        )

    def test_no_warning_without_skipped_images(self):
        scores, labels = _images()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PerImage(AUROC()).update(scores, labels).compute()

    def test_all_skipped_raises(self):
        scores, labels = _images()
        metric = PerImage(AUROC()).update(scores, labels.clamp(min=0))
        with pytest.raises(ValueError, match="All 4 images were skipped"):
            metric.compute()
        with pytest.raises(ValueError):
            PerImage(AUROC()).compute()

    def test_void_pixels_are_removed_per_image(self):
        scores, labels = _images(n=2)
        labels[:, :, -2:] = 7
        result = PerImage(AUROC(), void_label=7).update(scores, labels).compute()
        keep = labels[0] != 7
        expected = (
            float(F.auroc(scores[0][keep], labels[0][keep]))
            + float(F.auroc(scores[1][keep], labels[1][keep]))
        ) / 2
        assert result["AUROC"] == pytest.approx(expected, abs=5e-8)

    def test_reset(self):
        scores, labels = _images()
        metric = PerImage(AUROC()).update(scores + 5 * (labels >= 0).float(), labels)
        metric.reset()
        metric.update(scores, labels)
        assert metric.compute() == PerImage(AUROC()).update(scores, labels).compute()

    def test_memory_does_not_grow(self):
        scores, labels = _images()
        metric = OODPerImageSegmentationMetrics()
        for _ in range(3):
            metric.update(scores, labels)
        assert len(metric._sums) == 5
        assert all(value.numel() == 1 for value in metric._sums.values())
        assert all(not values for values in metric.metric._buffers.values())

    def test_invalid_input(self):
        scores, labels = _images()
        metric = PerImage(AUROC())
        with pytest.raises(ValueError):
            metric.update(scores[0, 0, 0], labels[0, 0, 0])
        with pytest.raises(ValueError):
            metric.update(scores, labels[:2])
        with pytest.raises(ValueError):
            metric.update(scores)
        with pytest.raises(TypeError):
            metric.update(scores, labels.tolist())

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            PerImage(AUROC(device="cpu"))
        with pytest.raises(ValueError):
            PerImage(AUROC(void_label=1))
        with pytest.raises(ValueError):
            OODPerImageSegmentationMetrics(fpr_at=95)
        with pytest.raises(TypeError):
            OODPerImageSegmentationMetrics("cpu")

    def test_accuracy_only_skips_images_without_id_pixels(self):
        _, labels = _images(n=4)
        labels[1] = labels[1].clamp(min=0)  # no OOD pixels: accuracy is defined
        labels[2] = -1  # no ID pixels: accuracy is undefined
        predictions = labels.clamp(min=0)
        predictions[1] = 1 - predictions[1].clamp(max=1)  # image 1 is classified wrongly
        metric = PerImage(Accuracy()).update(predictions, labels)
        keep = [0, 1, 3]
        with pytest.warns(UserWarning, match="Skipped 1 of 4 images without ID pixels or"):
            result = metric.compute()
        assert result["ACC"] == pytest.approx(
            self._expected(predictions[keep], labels[keep], F.accuracy), abs=5e-8
        )
        assert result["ACC"] < 1

    def test_collection_skips_images_any_member_is_undefined_for(self):
        scores, labels = _images(n=4)
        labels[1] = labels[1].clamp(min=0)
        labels[2] = -1
        predictions = labels.clamp(min=0)
        metric = PerImage(MetricCollection([Accuracy(), AUROC()]))
        metric.update(scores=scores, predictions=predictions, labels=labels)
        with pytest.warns(
            UserWarning, match="Skipped 2 of 4 images without ID pixels, without OOD pixels or"
        ):
            result = metric.compute()
        keep = [0, 3]
        assert result["AUROC"] == pytest.approx(
            self._expected(scores[keep], labels[keep]), abs=5e-8
        )

    @pytest.mark.parametrize(
        "metric, needs_ood",
        [
            (AUROC(), True),
            (AUPR(), True),
            (AveragePrecision(), True),
            (FPRAtTPR(), True),
            (AUTC(), True),
            (OODMetrics(), True),
            (Accuracy(), False),
            (MetricCollection([Accuracy()]), False),
        ],
    )
    def test_needs(self, metric, needs_ood):
        assert (metric.needs_id, metric.needs_ood) == (True, needs_ood)

    def test_not_part_of_a_collection(self):
        with pytest.raises(ValueError, match="wrap the collection"):
            MetricCollection([AUROC(), PerImage(AUPR())])
        with pytest.raises(ValueError):
            MetricCollection([OODPerImageSegmentationMetrics()])
        with pytest.raises(ValueError):
            PerImage(PerImage(AUROC()))

    def test_wraps_streaming_metrics(self):
        _, labels = _images()
        predictions = labels.clamp(min=0)
        predictions[0] = 1 - predictions[0].clamp(max=1)
        metric = PerImage(MetricCollection([Accuracy()]))
        result = metric.update(predictions=predictions, labels=labels).compute()
        assert result["ACC"] == pytest.approx(
            self._expected(predictions, labels, F.accuracy), abs=5e-8
        )
        assert result["ACC"] < 1

    def test_nan_propagates(self):
        _, labels = _images(n=2)
        scores = torch.zeros(labels.shape)
        with pytest.warns(UserWarning):
            result = OODPerImageSegmentationMetrics().update(scores, labels).compute()
        assert math.isnan(result["AUTC"])
        assert result["AUROC"] == 0.5

    @requires(cuda=1)
    def test_cuda(self):
        scores, labels = _images()
        expected = OODPerImageSegmentationMetrics().update(scores, labels).compute()
        result = OODPerImageSegmentationMetrics().update(scores.cuda(), labels).compute()
        for key, value in expected.items():
            assert result[key] == pytest.approx(value, abs=5e-8), key
        metric = OODPerImageSegmentationMetrics(device="cpu").update(scores.cuda(), labels)
        assert all(v.device.type == "cpu" for v in metric._sums.values())
