import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import ModelNotSetException, RequiresFittingException
from pytorch_ood.detector import PNML
from tests.helpers import ClassificationModel


class PNMLTest(unittest.TestCase):
    """
    Test PNML detector
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_fit_predict(self):
        nn = ClassificationModel()
        detector = PNML(nn.features, nn.classifier)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader)
        scores = detector(x)

        self.assertEqual(scores.shape, (20,))
        self.assertIsNotNone(scores)

    def test_fit_predict_features(self):
        head = torch.nn.Linear(10, 3)
        detector = PNML(encoder=None, head=head)

        z = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])

        detector.fit_features(z, y)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (20,))

    def test_nofit(self):
        nn = ClassificationModel()
        detector = PNML(nn.features, nn.classifier)
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            detector(x)

    def test_no_backbone(self):
        detector = PNML(encoder=None, head=torch.nn.Linear(10, 3))

        z = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        detector.fit_features(z, y)

        with self.assertRaises(ModelNotSetException):
            detector(torch.randn(size=(5, 10)))

    def test_no_head(self):
        detector = PNML(encoder=None, head=None)

        z = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        detector.fit_features(z, y)

        with self.assertRaises(ModelNotSetException):
            detector.predict_features(torch.randn(size=(5, 10)))

    def test_matches_paper_formula(self):
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0]]))

        detector = PNML(encoder=None, head=head)

        z_train = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = torch.tensor([0, 1, 0, 1])
        detector.fit_features(z_train, y_train)

        z_test = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        scores = detector.predict_features(z_test)

        z_train_norm = torch.nn.functional.normalize(z_train.float(), p=2, dim=1)
        z_test_norm = torch.nn.functional.normalize(z_test.float(), p=2, dim=1)
        projector = torch.linalg.pinv(z_train_norm) @ torch.linalg.pinv(z_train_norm).T
        probs = torch.softmax(head(z_test_norm), dim=1).clamp(min=1e-12, max=1.0 - 1e-12)
        x_proj = (z_test_norm @ projector * z_test_norm).sum(dim=1)
        x_t_g = x_proj / (1.0 + x_proj)
        regret_terms = probs / (probs + (1.0 - probs) * probs.pow(x_t_g.unsqueeze(1)))
        expected = torch.log(regret_terms.sum(dim=1)) / torch.log(torch.tensor(2.0))

        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))
        self.assertGreater(scores[1].item(), scores[0].item())


class PNMLNumericsTest(unittest.TestCase):
    """Ill-conditioned training features: float32 inputs must give the scores of a float64 computation."""

    def test_ill_conditioned_features_match_float64_reference(self):
        torch.manual_seed(0)
        n, d, c = 500, 64, 3
        scales = torch.logspace(0, -5, d)  # Gram eigenvalues span 10 orders of magnitude
        z_train = torch.randn(n, d) * scales
        z_test = torch.randn(16, d) * scales
        head = torch.nn.Linear(d, c)

        detector = PNML(encoder=None, head=head).fit_features(z_train, torch.zeros(n).long())
        scores = detector.predict_features(z_test)

        zn = torch.nn.functional.normalize(z_train.double(), dim=1)
        xn = torch.nn.functional.normalize(z_test.double(), dim=1)
        projector = torch.linalg.pinv(zn.T @ zn, hermitian=True)
        x_proj = (xn @ projector * xn).sum(dim=1)
        x_t_g = x_proj / (1.0 + x_proj)
        with torch.no_grad():
            probs = torch.softmax(head(xn.float()), dim=1).double().clamp(1e-12, 1 - 1e-12)
        terms = probs / (probs + (1.0 - probs) * probs.pow(x_t_g.unsqueeze(1)))
        expected = torch.log(terms.sum(dim=1)) / torch.log(
            torch.tensor(float(c), dtype=torch.float64)
        )

        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores.double(), expected, rtol=1e-4, atol=1e-5)
