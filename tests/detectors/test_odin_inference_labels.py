import unittest
from contextlib import nullcontext

import torch

from src.pytorch_ood.detector import odin_preprocessing


class TestODINInferenceLabels(unittest.TestCase):
    def test_supplied_labels(self):
        for dtype in (torch.float32, torch.float64):
            model = torch.nn.Linear(2, 3, bias=False).to(dtype).eval()
            with torch.no_grad():
                model.weight.copy_(
                    torch.tensor([[1.0, -2.0], [2.0, 1.0], [-1.0, 3.0]], dtype=dtype)
                )
            inputs = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]], dtype=dtype)
            labels = torch.tensor([0, 1, 2])
            temperature, eps = 10.0, 0.02
            with torch.no_grad():
                residual = (model(inputs) / temperature).softmax(
                    dim=1
                ) - torch.nn.functional.one_hot(labels, 3)
                gradient = residual @ model.weight / (len(inputs) * temperature)
                expected = inputs - eps * gradient.sign()
            for inference_labels in (False, True):
                for inference_inputs in (False, True):
                    for ambient_inference in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            labels=inference_labels,
                            inputs=inference_inputs,
                            ambient=ambient_inference,
                        ):
                            with torch.inference_mode(inference_labels):
                                target = labels.clone()
                            with torch.inference_mode(inference_inputs):
                                x = inputs.clone()
                            with torch.inference_mode() if ambient_inference else nullcontext():
                                result = odin_preprocessing(
                                    model, x, target, eps=eps, temperature=temperature
                                )
                            torch.testing.assert_close(result, expected)
                            torch.testing.assert_close(x, inputs)
                            torch.testing.assert_close(target, labels)
                            self.assertEqual(torch.is_inference(target), inference_labels)
                            self.assertFalse(result.requires_grad)
                            self.assertIsNone(model.weight.grad)
