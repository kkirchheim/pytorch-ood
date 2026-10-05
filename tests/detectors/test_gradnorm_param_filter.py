"""
GradUncertainty and GradNorm compute gradients only for the parameters that ``param_filter``
selects, independent of ``requires_grad``, and leave no gradients on the model.
"""

import unittest
from unittest import mock

import torch

from src.pytorch_ood.detector import GradNorm, GradUncertainty
from src.pytorch_ood.detector import gradnorm as gradnorm_module
from src.pytorch_ood.detector import graduncertainty as graduncertainty_module
from src.pytorch_ood.model import WideResNet
from tests.helpers import ClassificationModel

DETECTORS = ((GradUncertainty, graduncertainty_module), (GradNorm, gradnorm_module))


def _only_classifier_weight(name):
    return name == "classifier.weight"


class TestParamFilter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ClassificationModel(num_inputs=10, n_hidden=16, num_outputs=5).eval()
        self.x = torch.randn(6, 10)

    def test_gradients_only_for_selected_parameters(self):
        for cls, module in DETECTORS:
            with self.subTest(cls.__name__):
                original = module._func_grad
                seen = []

                def spy(fn):
                    def wrapped(params, x):
                        seen.append(sorted(params))
                        return original(fn)(params, x)

                    return wrapped

                with mock.patch.object(module, "_func_grad", spy):
                    cls(self.model, param_filter=_only_classifier_weight)(self.x)
                self.assertTrue(seen)
                self.assertTrue(all(names == ["classifier.weight"] for names in seen))

    def test_batched_and_sequential_paths_agree(self):
        for cls, module in DETECTORS:
            with self.subTest(cls.__name__):
                detector = cls(self.model, param_filter=_only_classifier_weight)
                batched = detector(self.x)
                with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", False):
                    sequential = detector(self.x)
                torch.testing.assert_close(batched, sequential)

    def test_paths_agree_on_conv_model(self):
        # batch normalization buffers, several selected layers, and chunking in GradNorm
        model = WideResNet(num_classes=10, depth=10, widen_factor=1).eval()
        x = torch.randn(5, 3, 32, 32)
        filters = {
            "fc.weight": lambda name: name == "fc.weight",
            "block1 and fc": lambda name: name.startswith(("block1", "fc")),
            "all": None,
        }
        for cls, module in DETECTORS:
            for label, param_filter in filters.items():
                with self.subTest(cls.__name__, param_filter=label):
                    kwargs = {"micro_batch_size": 2} if cls is GradNorm else {}
                    detector = cls(model, param_filter=param_filter, **kwargs)
                    batched = detector(x)
                    with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", False):
                        sequential = detector(x)
                    torch.testing.assert_close(batched, sequential)

    def test_independent_of_requires_grad(self):
        for cls, module in DETECTORS:
            for batched in (True, False):
                with self.subTest(cls.__name__, batched=batched):
                    detector = cls(self.model, param_filter=_only_classifier_weight)
                    with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", batched):
                        expected = detector(self.x)
                        self.model.requires_grad_(False)
                        try:
                            scores = detector(self.x)
                        finally:
                            self.model.requires_grad_(True)
                    torch.testing.assert_close(scores, expected)

    def test_no_gradients_on_model(self):
        for cls, module in DETECTORS:
            for batched in (True, False):
                with self.subTest(cls.__name__, batched=batched):
                    with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", batched):
                        cls(self.model, param_filter=_only_classifier_weight)(self.x)
                    for name, param in self.model.named_parameters():
                        self.assertIsNone(param.grad, name)

    def test_scores_do_not_require_grad(self):
        for cls, _ in DETECTORS:
            with self.subTest(cls.__name__):
                scores = cls(self.model, param_filter=_only_classifier_weight)(self.x)
                self.assertFalse(scores.requires_grad)

    def test_empty_selection_raises(self):
        for cls, module in DETECTORS:
            for batched in (True, False):
                with self.subTest(cls.__name__, batched=batched):
                    detector = cls(self.model, param_filter=lambda name: False)
                    with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", batched):
                        with self.assertRaises(ValueError):
                            detector(self.x)


if __name__ == "__main__":
    unittest.main()
