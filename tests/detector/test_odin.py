import unittest

import torch
from torch import nn

from pytorch_ood.detector import ODIN, odin_preprocessing
from tests.helpers.model import ConvClassifier

NORM_STD = [0.25, 0.5, 2.0]


def _reference_odin(model, inputs, eps, temperature, std):
    # Eq. 2 of the paper, written out per sample: one signed gradient step on -log S_y(x; T),
    # with the step of each channel divided by its standard deviation, then the maximum softmax
    # probability at temperature T of the perturbed input
    scores = []
    for x in inputs:
        x = x.unsqueeze(0).clone().requires_grad_()
        logits = model(x) / temperature
        log_prob = logits.log_softmax(dim=1)[0, logits.argmax()]
        (grad,) = torch.autograd.grad(-log_prob, x)
        step = grad.sign() / torch.tensor(std).view(1, -1, 1, 1)
        x_hat = x.detach() - eps * step
        with torch.no_grad():
            scores.append((model(x_hat) / temperature).softmax(dim=1).max())
    return torch.stack(scores)


class TestODIN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ConvClassifier(in_channels=3, out_channels=16, num_outputs=5).eval()
        self.x = torch.randn(8, 3, 8, 8)

    def test_matches_reference(self):
        for temperature, eps in ((1000.0, 0.0014), (10.0, 0.0028), (1.0, 0.05)):
            with self.subTest(temperature=temperature, eps=eps):
                detector = ODIN(self.model, eps=eps, temperature=temperature, norm_std=NORM_STD)
                expected = _reference_odin(self.model, self.x, eps, temperature, NORM_STD)
                torch.testing.assert_close(detector.predict(self.x), -expected)

    def test_perturbation_follows_cross_entropy(self):
        x = self.x.clone().requires_grad_()
        logits = self.model(x) / 1000.0
        nn.functional.cross_entropy(logits, logits.argmax(dim=1)).backward()
        expected = self.x - 0.01 * x.grad.sign()
        x_hat = odin_preprocessing(self.model, self.x, eps=0.01, temperature=1000.0)
        torch.testing.assert_close(x_hat, expected)

    def test_inference_mode(self):
        detector = ODIN(self.model, norm_std=NORM_STD)
        expected = detector.predict(self.x)
        with torch.inference_mode():
            with self.assertWarns(UserWarning):
                scores = detector.predict(self.x)
        torch.testing.assert_close(scores.clone(), expected)


if __name__ == "__main__":
    unittest.main()
