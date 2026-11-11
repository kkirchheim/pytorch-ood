"""
GradUncertainty and GradNorm compute gradients only for the parameters that ``param_filter``
selects, independent of ``requires_grad``, and leave no gradients on the model. Both the batched and
the sequential path refuse models in training mode.
"""

from unittest import mock

import pytest
import torch

from pytorch_ood.api import ModelModeException
from pytorch_ood.detector import GradNorm, GradUncertainty
from pytorch_ood.detector import gradnorm as gradnorm_module
from pytorch_ood.detector import graduncertainty as graduncertainty_module
from pytorch_ood.model import WideResNet
from tests.helpers import ClassificationModel
from tests.helpers.snapshot import snapshot

DETECTORS = ((GradUncertainty, graduncertainty_module), (GradNorm, gradnorm_module))


def _only_classifier_weight(name):
    return name == "classifier.weight"


class TestParamFilter:
    def setup_method(self):
        torch.manual_seed(0)
        self.model = ClassificationModel(num_inputs=10, n_hidden=16, num_outputs=5).eval()
        self.x = torch.randn(6, 10)

    @pytest.mark.parametrize("cls, module", DETECTORS)
    def test_gradients_only_for_selected_parameters(self, cls, module):
        original = module._func_grad
        seen = []

        def spy(fn):
            def wrapped(params, x):
                seen.append(sorted(params))
                return original(fn)(params, x)

            return wrapped

        with mock.patch.object(module, "_func_grad", spy):
            cls(self.model, param_filter=_only_classifier_weight)(self.x)
        assert seen
        assert all(names == ["classifier.weight"] for names in seen)

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
                kwargs = {"micro_batch_size": 2} if cls is GradNorm else {}
                detector = cls(model, param_filter=param_filter, **kwargs)
                batched = detector(x)
                with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", False):
                    sequential = detector(x)
                torch.testing.assert_close(batched, sequential)

    @pytest.mark.parametrize("cls, module", DETECTORS)
    @pytest.mark.parametrize("batched", (True, False))
    def test_training_mode_raises(self, cls, module, batched):
        # without the check, the sequential path would update the batch norm statistics silently
        model = WideResNet(num_classes=10, depth=10, widen_factor=1).train()
        before = snapshot(model=model)
        with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", batched):
            with pytest.raises(ModelModeException, match=r"model\.eval\(\)"):
                cls(model)(torch.randn(2, 3, 32, 32))
        assert snapshot(model=model) == before  # raised before the first forward pass

    @pytest.mark.parametrize("cls, module", DETECTORS)
    @pytest.mark.parametrize("batched", (True, False))
    def test_independent_of_requires_grad(self, cls, module, batched):
        detector = cls(self.model, param_filter=_only_classifier_weight)
        with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", batched):
            expected = detector(self.x)
            self.model.requires_grad_(False)
            try:
                scores = detector(self.x)
                assert not any(p.requires_grad for p in self.model.parameters())
            finally:
                self.model.requires_grad_(True)
        torch.testing.assert_close(scores, expected)

    @pytest.mark.parametrize("cls, module", DETECTORS)
    @pytest.mark.parametrize("batched", (True, False))
    def test_empty_selection_raises(self, cls, module, batched):
        detector = cls(self.model, param_filter=lambda name: False)
        with mock.patch.object(module, "_TORCH_FUNC_AVAILABLE", batched):
            with pytest.raises(ValueError):
                detector(self.x)
