import unittest
import warnings

import torch

from pytorch_ood.detector.klmatching import KLMatching


class TestKLMatching(unittest.TestCase):
    """
    Tests for KL Matching
    """

    @staticmethod
    def _paper_dists(logits):
        """d_k: mean posterior of the samples whose argmax is k (paper, Section 4)."""
        p = logits.softmax(dim=1)
        predictions = p.argmax(dim=1)
        return {int(k): p[predictions == k].mean(dim=0) for k in predictions.unique()}

    @staticmethod
    def _paper_scores(logits, dists):
        """min_k KL[p(y | x) || d_k] (paper, Section 4)."""
        p = logits.softmax(dim=1)
        kl = [(p * (p / d).log()).sum(dim=1) for d in dists.values()]
        return torch.stack(kl, dim=1).min(dim=1).values

    @staticmethod
    def _misclassified_logits(n=300, n_classes=3, seed=0):
        """Logits where 30% of the samples are predicted as another class than their label."""
        g = torch.Generator().manual_seed(seed)
        labels = torch.arange(n) % n_classes
        predicted = labels.clone()
        wrong = torch.rand(n, generator=g) < 0.3
        predicted[wrong] = (labels[wrong] + 1) % n_classes
        logits = torch.randn(n, n_classes, generator=g) + 3 * torch.eye(n_classes)[predicted]
        return logits, labels

    def test_fit_groups_by_predicted_class(self):
        logits, labels = self._misclassified_logits()
        detector = KLMatching(None).fit_logits(logits, labels)
        expected = self._paper_dists(logits)
        self.assertEqual(sorted(detector.dists.keys()), ["0", "1", "2"])
        for k, d_k in expected.items():
            torch.testing.assert_close(detector.dists[str(k)].data, d_k)

    def test_fit_does_not_need_labels(self):
        logits, labels = self._misclassified_logits()
        with_labels = KLMatching(None).fit_logits(logits, labels)
        without_labels = KLMatching(None).fit_logits(logits)
        shuffled = KLMatching(None).fit_logits(logits, labels[torch.randperm(len(labels))])
        test = torch.randn(20, 3)
        expected = with_labels.predict_logits(test)
        torch.testing.assert_close(without_labels.predict_logits(test), expected)
        torch.testing.assert_close(shuffled.predict_logits(test), expected)

    def test_score_is_min_kl_over_classes(self):
        logits, labels = self._misclassified_logits(n=3000, n_classes=10)
        detector = KLMatching(None).fit_logits(logits, labels)
        # flat posteriors, for which the closest typical posterior often is not the one of
        # the predicted class
        test = torch.randn(500, 10, generator=torch.Generator().manual_seed(1))
        p = test.softmax(dim=1)
        dists = self._paper_dists(logits)
        kl = torch.stack([(p * (p / d).log()).sum(dim=1) for d in dists.values()], dim=1)
        self.assertTrue((kl.argmin(dim=1) != p.argmax(dim=1)).any())
        torch.testing.assert_close(detector.predict_logits(test), self._paper_scores(test, dists))

    def test_classes_never_predicted_are_not_needed(self):
        """Classes without a typical posterior are left out of the minimum."""
        logits, labels = self._misclassified_logits()
        logits[:, 2] -= 20  # class 2 is never predicted
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector = KLMatching(None).fit_logits(logits, labels)
        self.assertEqual(sorted(detector.dists.keys()), ["0", "1"])
        test = torch.randn(20, 3)
        test[:, 2] += 5  # predicted as class 2
        scores = detector.predict_logits(test)
        torch.testing.assert_close(scores, self._paper_scores(test, self._paper_dists(logits)))
