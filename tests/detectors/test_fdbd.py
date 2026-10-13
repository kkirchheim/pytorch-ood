import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import fDBD
from tests.helpers import ClassificationModel


class fDBDTest(unittest.TestCase):
    """
    Test fDBD detector
    """

    def setUp(self) -> None:
        torch.manual_seed(123)
        self.nn = ClassificationModel()
        self.head = self.nn.classifier

    def test_fit_predict(self):
        # Wrap the features method as a module
        encoder = torch.nn.Sequential(self.nn.layer1, torch.nn.Tanh())
        detector = fDBD(encoder, self.head)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader)
        scores = detector(x)

        self.assertEqual(scores.shape, (20,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fit_predict_features(self):
        detector = fDBD(encoder=None, head=self.head)

        z = torch.randn(size=(20, 10))
        y = torch.zeros(size=(20,))

        detector.fit_features(z)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (20,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_nofit(self):
        detector = fDBD(self.nn.features, self.head)
        z = torch.randn(size=(5, 10))

        with self.assertRaises(RequiresFittingException):
            detector.predict_features(z)

    def test_ood_scores_higher(self):
        """ID points near class centers should score lower (less OOD) than far-away points."""
        detector = fDBD(self.nn.features, self.head)

        # create well-separated training data
        z_train = torch.cat(
            [
                torch.randn(50, 10) + 5,  # class 0 cluster
                torch.randn(50, 10) - 5,  # class 1 cluster
            ]
        )
        y_train = torch.cat([torch.zeros(50), torch.ones(50)])
        detector.fit_features(z_train)

        # ID-like features (near training clusters)
        z_id = torch.randn(20, 10) + 5
        # OOD-like features (far from everything)
        z_ood = torch.randn(20, 10) * 100

        scores_id = detector.predict_features(z_id)
        scores_ood = detector.predict_features(z_ood)

        # on average, OOD scores should be higher
        self.assertGreater(scores_ood.mean().item(), scores_id.mean().item())


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

    def test_default_is_distance(self):
        self.assertTrue(fDBD(None, self.head).distance_as_normalizer)

    def test_distance_as_normalizer(self):
        self._check(distance_as_normalizer=True)

    def test_feature_norm_as_normalizer(self):
        self._check(distance_as_normalizer=False)
