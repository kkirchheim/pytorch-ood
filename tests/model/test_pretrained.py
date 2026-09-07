import os
import unittest

import torch

from src.pytorch_ood.model import list_models, load_model
from src.pytorch_ood.model.registry import get_model_info


@unittest.skipUnless(
    os.environ.get("PYTORCH_OOD_DOWNLOAD_TESTS"),
    "Downloads all pre-trained models; set PYTORCH_OOD_DOWNLOAD_TESTS=1 to run",
)
class TestPreTrainedModels(unittest.TestCase):
    """
    Downloads every registered model and checks that it loads and produces
    correctly shaped output. Gated behind an environment variable so CI does
    not download checkpoints.
    """

    def test_load_all_registered_models(self):
        for key in list_models():
            with self.subTest(model=key):
                entry = get_model_info(key)
                model = load_model(key)
                self.assertFalse(model.training)

                x = torch.ones(size=(1, 3, 32, 32))
                y = model(x)
                self.assertEqual(y.shape, (1, entry.arch_kwargs["num_classes"]))


if __name__ == "__main__":
    unittest.main()
