import unittest

import torch

from pytorch_ood.detector import NNGuide
from pytorch_ood.utils import TorchNearestNeighbors


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

    def test_chunks_do_not_change_scores(self):
        labels = torch.zeros(200, dtype=torch.long)
        expected = NNGuide(encoder=None, head=self.head, k=10).fit_features(self.z_train, labels)
        search = TorchNearestNeighbors(metric="inner_product", batch_size=3)
        detector = NNGuide(encoder=None, head=self.head, k=10, search=search)
        detector.fit_features(self.z_train, labels, batch_size=7)
        torch.testing.assert_close(
            detector.predict_features(self.z), expected.predict_features(self.z)
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
