import unittest

import torch

from pytorch_ood.detector import fDBD


def _reference_fdbd(z, train_mean, w, b, distance_as_normalizer):
    """
    Reference implementation following the code of the authors and of OpenMIBOOD
    (without the constant factor 1 / (C - 1) and with negated sign).
    """
    n_classes = w.shape[0]
    denominator_matrix = torch.zeros((n_classes, n_classes))
    for p in range(n_classes):
        denominator = torch.norm(w - w[p, :], dim=1)
        denominator[p] = 1
        denominator_matrix[p, :] = denominator

    logits = z @ w.T + b
    values, nn_idx = logits.max(1)
    logits_sub = torch.abs(logits - values.repeat(n_classes, 1).T)
    if distance_as_normalizer:
        normalizer = torch.norm(z - train_mean, dim=1)
    else:
        normalizer = torch.norm(z, dim=1)
    return -torch.sum(logits_sub / denominator_matrix[nn_idx], axis=1) / normalizer


class fDBDNormalizerTest(unittest.TestCase):
    """
    Test the distance_as_normalizer parameter of fDBD
    """

    def setUp(self) -> None:
        torch.manual_seed(123)
        self.head = torch.nn.Linear(10, 4)
        self.z_train = torch.randn(size=(100, 10)) + 1
        self.z = torch.randn(size=(32, 10))

    def _check(self, distance_as_normalizer):
        detector = fDBD(None, self.head, distance_as_normalizer=distance_as_normalizer)
        detector.fit_features(self.z_train)
        scores = detector.predict_features(self.z)

        expected = _reference_fdbd(
            self.z,
            self.z_train.mean(dim=0),
            self.head.weight.data,
            self.head.bias.data,
            distance_as_normalizer,
        )
        n_classes = self.head.out_features
        torch.testing.assert_close(scores, expected / (n_classes - 1))

    def test_distance_as_normalizer(self):
        self._check(distance_as_normalizer=True)

    def test_feature_norm_as_normalizer(self):
        self._check(distance_as_normalizer=False)
