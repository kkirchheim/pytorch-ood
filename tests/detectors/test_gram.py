import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import Gram
from tests.helpers.model import ConvClassifier


class InitGram:
    def __init__(self):
        nn = ConvClassifier(in_channels=3, num_outputs=2)

        class MyHead(torch.nn.Module):
            def __init__(self, classifier, pool):
                super(MyHead, self).__init__()
                self.classifier = classifier
                self.pool = pool
                self.flatten = torch.nn.Flatten()

            def forward(self, x):
                x = self.pool(x)
                x = self.flatten(x)
                x = self.classifier(x)
                return x

        self.model = Gram(MyHead(nn.classifier, nn.pool), [nn.layer1], 2, [1])


class GramTest(unittest.TestCase):
    """
    Test gram matrix based method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_something(self):
        model = InitGram().model
        y = torch.cat(
            [
                torch.zeros(size=(10,), dtype=torch.int),
                torch.ones(size=(10,), dtype=torch.int),
            ]
        )
        x = torch.randn(size=(20, 3, 16, 16))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(f"Scores: {scores}")

        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape[0], 20)

    def test_nofit(self):
        model = InitGram().model
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            model(x)

    def test_scores_non_negative(self):
        """
        Deviations are sums of relu terms, so scores must be non-negative.
        """
        model = InitGram().model
        y = torch.cat([torch.zeros(50, dtype=torch.int), torch.ones(50, dtype=torch.int)])
        x = torch.randn(size=(100, 3, 16, 16))
        model.fit(DataLoader(TensorDataset(x, y), batch_size=16))

        scores = model(x)
        self.assertTrue((scores >= 0).all())

    def test_score_direction(self):
        """
        Inputs whose gram statistics fall far outside the training bounds must
        receive higher outlier scores than the fitting data (higher = more OOD).
        The previous implementation returned inverted scores.
        """
        from src.pytorch_ood.utils import OODMetrics

        model = InitGram().model
        y = torch.cat([torch.zeros(50, dtype=torch.int), torch.ones(50, dtype=torch.int)])
        x_in = torch.randn(size=(100, 3, 16, 16))
        model.fit(DataLoader(TensorDataset(x_in, y), batch_size=16))

        x_out = 10.0 * torch.randn(size=(100, 3, 16, 16))

        scores_in = model(x_in)
        scores_out = model(x_out)

        self.assertGreater(scores_out.mean().item(), scores_in.mean().item())

        metrics = OODMetrics()
        metrics.update(
            torch.cat([scores_in, scores_out]),
            torch.cat([torch.zeros(100), -torch.ones(100)]),
        )
        self.assertGreater(metrics.compute()["AUROC"], 0.95)
