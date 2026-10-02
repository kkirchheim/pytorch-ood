import unittest

import torch
from torch.utils.data import DataLoader

from pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import NNGuide
from tests.helpers import ClassificationModel, sample_dataset


class TestNNGuide(unittest.TestCase):
    """
    Tests for NNGuide Detector
    """

    def _make_detector(self):
        model = ClassificationModel(num_inputs=10, n_hidden=10, num_outputs=3)
        model.eval()
        return NNGuide(model.features, model.classifier, k=5)

    def test_requires_fitting(self):
        detector = self._make_detector()
        self.assertTrue(detector.requires_fit)
        z = torch.randn(16, 10)
        with self.assertRaises(RequiresFittingException):
            detector.predict_features(z)

    def test_fit_and_predict_features(self):
        detector = self._make_detector()
        ds = sample_dataset(n_dim=10)
        loader = DataLoader(ds, batch_size=32)

        detector.fit(loader)

        z = torch.randn(16, 10)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fit_features_direct(self):
        detector = self._make_detector()
        features = torch.randn(100, 10)
        labels = torch.zeros(100, dtype=torch.long)

        detector.fit_features(features, labels)

        z = torch.randn(8, 10)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (8,))

    def test_ignores_ood_in_fit(self):
        detector = self._make_detector()
        features = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.full((50,), -1)])

        detector.fit_features(features, labels)

        # only 50 ID samples should be in the bank
        self.assertEqual(detector._scaled_features.shape[0], 50)


def _reference_scores(head, z_train, z, k):
    # the paper, written out: normalized training features scaled by their energy, mean of the
    # k largest inner products with the normalized input features, times the energy of the input
    def energy(features):
        return torch.logsumexp(head(features), dim=1)

    def normalize(features):
        return features / features.norm(dim=1, keepdim=True)

    bank = normalize(z_train) * energy(z_train)[:, None]
    scores = []
    for zi in z:
        similarities = sorted((normalize(zi[None]) @ bank.T)[0].tolist(), reverse=True)
        guidance = sum(similarities[:k]) / k
        scores.append(-guidance * energy(zi[None])[0])
    return torch.stack(scores)


class TestNNGuideReference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = torch.nn.Linear(6, 4)
        self.z_train = torch.randn(200, 6)
        self.z = torch.randn(16, 6)

    def test_matches_reference(self):
        for k in (1, 5, 10):
            with self.subTest(k=k):
                detector = NNGuide(encoder=None, head=self.head, k=k)
                detector.fit_features(self.z_train, torch.zeros(200, dtype=torch.long))
                with torch.no_grad():
                    expected = _reference_scores(self.head, self.z_train, self.z, k)
                torch.testing.assert_close(detector.predict_features(self.z), expected)

    def test_bank_chunks_do_not_change_scores(self):
        detector = NNGuide(encoder=None, head=self.head, k=10)
        detector.fit_features(self.z_train, torch.zeros(200, dtype=torch.long), batch_size=7)
        torch.testing.assert_close(
            detector.predict_features(self.z, batch_size=3), detector.predict_features(self.z)
        )

    def test_neighbors_weighted_by_energy(self):
        # two training samples with the direction of the input, one with a larger energy: with
        # k = 1, the guidance is the inner product with the high-energy sample, while the cosine
        # similarity of both is 1
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        z_train = torch.tensor([[1.0, 0.0], [4.0, 0.0], [0.0, 1.0]])
        detector = NNGuide(encoder=None, head=head, k=1)
        detector.fit_features(z_train, torch.zeros(3, dtype=torch.long))
        z = torch.tensor([[2.0, 0.0]])
        energy = lambda f: torch.logsumexp(head(f), dim=1)  # noqa: E731
        with torch.no_grad():
            expected = -energy(z_train[1:2]) * energy(z)
        torch.testing.assert_close(detector.predict_features(z), expected)

    def test_fewer_samples_than_k_raises(self):
        detector = NNGuide(encoder=None, head=self.head, k=10)
        labels = torch.cat([torch.zeros(5, dtype=torch.long), torch.full((20,), -1)])
        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(25, 6), labels)
