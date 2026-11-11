import pytest
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


class TestNNGuideReference:
    def setup_method(self):
        torch.manual_seed(0)
        self.head = torch.nn.Linear(6, 4)
        self.z_train = torch.randn(200, 6)
        self.z = torch.randn(16, 6)

    @pytest.mark.parametrize("k", (1, 5, 10))
    def test_matches_reference(self, k):
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
