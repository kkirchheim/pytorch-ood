"""
The wrappers that move tensor arguments to the detector's device keep the signatures of the
wrapped predict_* methods, which the docs show.
"""

import inspect
import unittest

from src.pytorch_ood import detector as detector_module
from src.pytorch_ood.api import Detector

METHODS = ("predict_logits", "predict_features", "predict_feature_maps")


class TestSignatures(unittest.TestCase):
    def test_predict_methods_keep_their_signature(self):
        checked = 0
        for name in detector_module.__all__:
            cls = getattr(detector_module, name)
            if not (inspect.isclass(cls) and issubclass(cls, Detector)):
                continue
            for method in METHODS:
                if method not in cls.__dict__:
                    continue
                with self.subTest(detector=name, method=method):
                    parameters = inspect.signature(getattr(cls, method)).parameters.values()
                    kinds = {p.kind for p in parameters}
                    self.assertNotIn(inspect.Parameter.VAR_POSITIONAL, kinds)
                    self.assertEqual(
                        list(inspect.signature(getattr(cls, method)).parameters)[0], "self"
                    )
                    checked += 1
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
