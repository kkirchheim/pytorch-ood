"""
Segmentation metrics: pooled over all pixels, and per image.
"""

import math
import unittest
import warnings

import torch

from src.pytorch_ood.metrics import (
    AUPR,
    AUROC,
    AUTC,
    Accuracy,
    FPRAtTPR,
    MetricCollection,
    OODMetrics,
    OODPerImageSegmentationMetrics,
    OODSegmentationMetrics,
    PerImage,
)
from src.pytorch_ood.metrics import functional as F

CUDA = torch.cuda.is_available()


def _images(n=4, size=8, seed=0):
    """Score maps and label masks with ID (0, 1) and OOD (-1) pixels in every image."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n, size, size), generator=g)
    labels[:, : size // 2, : size // 2] = -1
    scores = torch.randn(n, size, size, generator=g) + (labels < 0).float()
    return scores, labels


class TestOODSegmentationMetrics(unittest.TestCase):
    def test_pools_all_pixels(self):
        scores, labels = _images()
        expected = OODMetrics().update(scores.flatten(), labels.flatten()).compute()
        result = (
            OODSegmentationMetrics().update(scores[:2], labels[:2]).update(scores[2:], labels[2:])
        )
        self.assertEqual(result.compute(), expected)
        self.assertEqual(
            tuple(result.compute()), ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR")
        )

    def test_images_without_ood_pixels_count(self):
        scores, labels = _images()
        labels[0] = labels[0].clamp(min=0)
        result = OODSegmentationMetrics().update(scores, labels).compute()
        self.assertEqual(result["AUROC"], float(F.auroc(scores, labels)))

    def test_void_label_and_fpr_at(self):
        scores, labels = _images()
        labels[:, -1] = 255
        result = OODSegmentationMetrics(void_label=255, fpr_at=0.9).update(scores, labels)
        keep = labels != 255
        self.assertEqual(
            result.compute()["FPR90TPR"], float(F.fpr_at_tpr(scores[keep], labels[keep], 0.9))
        )


class TestPerImage(unittest.TestCase):
    def _expected(self, scores, labels, fn=F.auroc):
        return sum(float(fn(s, y)) for s, y in zip(scores, labels)) / len(scores)

    def test_mean_over_images(self):
        scores, labels = _images()
        metric = PerImage(AUROC())
        metric.update(scores[:3], labels[:3]).update(scores[3:], labels[3:])
        self.assertAlmostEqual(metric.compute()["AUROC"], self._expected(scores, labels))

    def test_differs_from_pooled(self):
        # the second image is shifted, which changes the pooled AUROC but not the per-image one
        scores, labels = _images(n=2)
        scores[1] += 10
        per_image = PerImage(AUROC()).update(scores, labels).compute()["AUROC"]
        pooled = AUROC().update(scores, labels).compute()["AUROC"]
        self.assertAlmostEqual(per_image, self._expected(scores, labels))
        self.assertNotAlmostEqual(per_image, pooled)

    def test_collection_matches_single_metrics(self):
        scores, labels = _images()
        result = OODPerImageSegmentationMetrics().update(scores, labels).compute()
        for key, fn in (
            ("AUROC", F.auroc),
            ("AUTC", F.autc),
            ("AUPR-IN", lambda s, y: F.aupr(s, y, positive="id")),
            ("AUPR-OUT", lambda s, y: F.aupr(s, y, positive="ood")),
            ("FPR95TPR", lambda s, y: F.fpr_at_tpr(s, y, 0.95)),
        ):
            self.assertAlmostEqual(result[key], self._expected(scores, labels, fn), msg=key)

    def test_skips_images_without_id_or_ood(self):
        scores, labels = _images(n=5)
        labels[1] = labels[1].clamp(min=0)  # only ID
        labels[2] = -1  # only OOD
        labels[3] = 255  # only void
        metric = PerImage(AUROC(), void_label=255).update(scores, labels)
        keep = [0, 4]
        with self.assertWarnsRegex(UserWarning, "Skipped 3 of 5 images"):
            result = metric.compute()
        self.assertAlmostEqual(result["AUROC"], self._expected(scores[keep], labels[keep]))

    def test_no_warning_without_skipped_images(self):
        scores, labels = _images()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PerImage(AUROC()).update(scores, labels).compute()

    def test_all_skipped_raises(self):
        scores, labels = _images()
        metric = PerImage(AUROC()).update(scores, labels.clamp(min=0))
        with self.assertRaisesRegex(ValueError, "All 4 images were skipped"):
            metric.compute()
        with self.assertRaises(ValueError):
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
        self.assertAlmostEqual(result["AUROC"], expected)

    def test_reset(self):
        scores, labels = _images()
        metric = PerImage(AUROC()).update(scores + 5 * (labels >= 0).float(), labels)
        metric.reset()
        metric.update(scores, labels)
        self.assertEqual(metric.compute(), PerImage(AUROC()).update(scores, labels).compute())

    def test_memory_does_not_grow(self):
        scores, labels = _images()
        metric = OODPerImageSegmentationMetrics()
        for _ in range(3):
            metric.update(scores, labels)
        self.assertEqual(len(metric._sums), 5)
        self.assertTrue(all(value.numel() == 1 for value in metric._sums.values()))
        self.assertTrue(all(not values for values in metric.metric._buffers.values()))

    def test_invalid_input(self):
        scores, labels = _images()
        metric = PerImage(AUROC())
        with self.assertRaises(ValueError):
            metric.update(scores[0, 0, 0], labels[0, 0, 0])
        with self.assertRaises(ValueError):
            metric.update(scores, labels[:2])
        with self.assertRaises(ValueError):
            metric.update(scores)
        with self.assertRaises(TypeError):
            metric.update(scores, labels.tolist())

    def test_invalid_construction(self):
        with self.assertRaises(ValueError):
            PerImage(AUROC(device="cpu"))
        with self.assertRaises(ValueError):
            PerImage(AUROC(void_label=1))
        with self.assertRaises(ValueError):
            OODPerImageSegmentationMetrics(fpr_at=95)
        with self.assertRaises(TypeError):
            OODPerImageSegmentationMetrics("cpu")

    def test_accuracy_only_skips_images_without_id_pixels(self):
        _, labels = _images(n=4)
        labels[1] = labels[1].clamp(min=0)  # no OOD pixels: accuracy is defined
        labels[2] = -1  # no ID pixels: accuracy is undefined
        predictions = labels.clamp(min=0)
        predictions[1] = 1 - predictions[1].clamp(max=1)  # image 1 is classified wrongly
        metric = PerImage(Accuracy()).update(predictions, labels)
        keep = [0, 1, 3]
        with self.assertWarnsRegex(UserWarning, "Skipped 1 of 4 images without ID pixels or"):
            result = metric.compute()
        self.assertAlmostEqual(
            result["ACC"], self._expected(predictions[keep], labels[keep], F.accuracy)
        )
        self.assertLess(result["ACC"], 1)

    def test_collection_skips_images_any_member_is_undefined_for(self):
        scores, labels = _images(n=4)
        labels[1] = labels[1].clamp(min=0)
        labels[2] = -1
        predictions = labels.clamp(min=0)
        metric = PerImage(MetricCollection([Accuracy(), AUROC()]))
        metric.update(scores=scores, predictions=predictions, labels=labels)
        with self.assertWarnsRegex(
            UserWarning, "Skipped 2 of 4 images without ID pixels, without OOD pixels or"
        ):
            result = metric.compute()
        keep = [0, 3]
        self.assertAlmostEqual(result["AUROC"], self._expected(scores[keep], labels[keep]))

    def test_needs(self):
        for metric in (AUROC(), AUPR(), FPRAtTPR(), AUTC(), OODMetrics()):
            self.assertEqual((metric.needs_id, metric.needs_ood), (True, True))
        for metric in (Accuracy(), MetricCollection([Accuracy()])):
            self.assertEqual((metric.needs_id, metric.needs_ood), (True, False))

    def test_not_part_of_a_collection(self):
        with self.assertRaisesRegex(ValueError, "wrap the collection"):
            MetricCollection([AUROC(), PerImage(AUPR())])
        with self.assertRaises(ValueError):
            MetricCollection([OODPerImageSegmentationMetrics()])
        with self.assertRaises(ValueError):
            PerImage(PerImage(AUROC()))

    def test_wraps_streaming_metrics(self):
        _, labels = _images()
        predictions = labels.clamp(min=0)
        predictions[0] = 1 - predictions[0].clamp(max=1)
        metric = PerImage(MetricCollection([Accuracy()]))
        result = metric.update(predictions=predictions, labels=labels).compute()
        self.assertAlmostEqual(result["ACC"], self._expected(predictions, labels, F.accuracy))
        self.assertLess(result["ACC"], 1)

    def test_nan_propagates(self):
        _, labels = _images(n=2)
        scores = torch.zeros(labels.shape)
        with self.assertWarns(UserWarning):
            result = OODPerImageSegmentationMetrics().update(scores, labels).compute()
        self.assertTrue(math.isnan(result["AUTC"]))
        self.assertEqual(result["AUROC"], 0.5)

    @unittest.skipUnless(CUDA, "CUDA not available")
    def test_cuda(self):
        scores, labels = _images()
        expected = OODPerImageSegmentationMetrics().update(scores, labels).compute()
        result = OODPerImageSegmentationMetrics().update(scores.cuda(), labels).compute()
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value, msg=key)
        metric = OODPerImageSegmentationMetrics(device="cpu").update(scores.cuda(), labels)
        self.assertTrue(all(v.device.type == "cpu" for v in metric._sums.values()))


if __name__ == "__main__":
    unittest.main()
