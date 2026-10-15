import unittest

import torch

from src.pytorch_ood.detector import KNN


class TestKNNLowPrecision(unittest.TestCase):
    def test_relu_features(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for normalize in (True, False):
                for include_zero in (True, False):
                    with self.subTest(dtype=dtype, normalize=normalize, zero=include_zero):
                        model = torch.nn.Sequential(
                            torch.nn.Linear(2, 2, bias=False), torch.nn.ReLU()
                        ).to(dtype)
                        with torch.no_grad():
                            model[0].weight.copy_(torch.eye(2, dtype=dtype))
                            train = model(
                                torch.tensor(
                                    [[-1.0, -2.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0]], dtype=dtype
                                )
                            )
                            query = model(torch.tensor([[-2.0, -1.0], [2.0, 3.0]], dtype=dtype))
                        if not include_zero:
                            train, query = train[1:], query[1:]
                        original_train, original_query = train.clone(), query.clone()
                        reference_train = (
                            train.float() if dtype in (torch.float16, torch.bfloat16) else train
                        )
                        reference_query = (
                            query.float() if dtype in (torch.float16, torch.bfloat16) else query
                        )
                        if normalize:
                            reference_train = torch.nn.functional.normalize(reference_train, dim=1)
                            reference_query = torch.nn.functional.normalize(reference_query, dim=1)
                        expected = (
                            torch.cdist(reference_query.double(), reference_train.double())
                            .sort(dim=1)
                            .values[:, 1]
                        )
                        detector = KNN(model, k=2, normalize=normalize)
                        detector.fit_features(train, torch.zeros(len(train), dtype=torch.long))
                        scores = detector.predict_features(query)
                        self.assertTrue(torch.isfinite(scores).all())
                        torch.testing.assert_close(detector.predict(query), scores)
                        torch.testing.assert_close(scores.double(), expected, rtol=1e-6, atol=1e-7)
                        torch.testing.assert_close(train, original_train)
                        torch.testing.assert_close(query, original_query)
