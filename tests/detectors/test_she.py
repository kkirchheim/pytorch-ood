import unittest

import torch

from src.pytorch_ood.detector import SHE
from src.pytorch_ood.model import WideResNet


class TestASH(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    @torch.no_grad()
    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = SHE(
            encoder=model.features,
            head=model.fc,
        )

        detector.fit_features(
            z=torch.randn(1000, 128),
            y=torch.arange(
                1000,
            )
            % 10,
            batch_size=128,
        )

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)


class SHEScoreTest(unittest.TestCase):
    def test_negated_inner_product_with_pattern_of_predicted_class(self):
        # the head is the identity, so the predicted class is the argmax of the features
        detector = SHE(encoder=None, head=torch.nn.Identity())
        detector.patterns = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
        z = torch.tensor([[2.0, 1.0], [0.0, 2.0]])
        # predicted classes 0 and 1; inner products 2 and 6
        torch.testing.assert_close(detector.predict_features(z), torch.tensor([-2.0, -6.0]))

    def test_fit_without_encoder(self):
        from torch.utils.data import DataLoader, TensorDataset

        from src.pytorch_ood.api import ModelNotSetException

        loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.arange(8) % 2), batch_size=4)
        with self.assertRaises(ModelNotSetException):
            SHE(encoder=None, head=torch.nn.Linear(4, 2)).fit(loader)
