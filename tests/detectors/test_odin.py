import unittest

import torch
from torch import nn

from src.pytorch_ood.detector import ODIN, odin_preprocessing
from tests.helpers import ClassificationModel
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

    def test_odin(self):
        model = ClassificationModel()
        odin = ODIN(model)

        x = torch.zeros(size=(1, 10))
        y = odin.predict(x)

        self.assertIsNotNone(y)

    def test_matches_reference(self):
        for temperature, eps in ((1000.0, 0.0014), (10.0, 0.0028), (1.0, 0.05)):
            with self.subTest(temperature=temperature, eps=eps):
                detector = ODIN(self.model, eps=eps, temperature=temperature, norm_std=NORM_STD)
                expected = _reference_odin(self.model, self.x, eps, temperature, NORM_STD)
                torch.testing.assert_close(detector.predict(self.x), -expected)

    def test_score_uses_temperature(self):
        # without perturbation, the score is the negative MSP at temperature T
        detector = ODIN(self.model, eps=0.0, temperature=100.0)
        with torch.no_grad():
            expected = -(self.model(self.x) / 100.0).softmax(dim=1).max(dim=1).values
        torch.testing.assert_close(detector.predict(self.x), expected)

    def test_perturbation_follows_cross_entropy(self):
        x = self.x.clone().requires_grad_()
        logits = self.model(x) / 1000.0
        nn.functional.cross_entropy(logits, logits.argmax(dim=1)).backward()
        expected = self.x - 0.01 * x.grad.sign()
        x_hat = odin_preprocessing(self.model, self.x, eps=0.01, temperature=1000.0)
        torch.testing.assert_close(x_hat, expected)

    def test_perturbation_increases_confidence(self):
        # the step decreases -log S_y(x; T), so the softmax probability of the prediction grows
        x_hat = odin_preprocessing(self.model, self.x, eps=0.01, temperature=1000.0)
        with torch.no_grad():
            before = (self.model(self.x) / 1000.0).softmax(dim=1).max(dim=1).values
            after = (self.model(x_hat) / 1000.0).softmax(dim=1).max(dim=1).values
        self.assertTrue((after > before).all())

    def test_norm_std_scales_channels(self):
        plain = odin_preprocessing(self.model, self.x, eps=0.01) - self.x
        scaled = odin_preprocessing(self.model, self.x, eps=0.01, norm_std=NORM_STD) - self.x
        for c, std in enumerate(NORM_STD):
            torch.testing.assert_close(scaled[:, c], plain[:, c] / std)

    def test_model_parameters_collect_no_gradients(self):
        ODIN(self.model).predict(self.x)
        for name, param in self.model.named_parameters():
            self.assertIsNone(param.grad, name)

    def test_defaults(self):
        detector = ODIN(self.model)
        self.assertEqual(detector.eps, 0.0014)
        self.assertEqual(detector.temperature, 1000.0)
        self.assertIs(detector.criterion, nn.functional.cross_entropy)

    def test_inference_mode(self):
        detector = ODIN(self.model, norm_std=NORM_STD)
        expected = detector.predict(self.x)
        with torch.inference_mode():
            with self.assertWarns(UserWarning):
                scores = detector.predict(self.x)
        torch.testing.assert_close(scores.clone(), expected)


if __name__ == "__main__":
    unittest.main()
