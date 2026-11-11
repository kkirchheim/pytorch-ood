import torch

from pytorch_ood.detector import SHE


class TestSHEScore:
    def test_negated_inner_product_with_pattern_of_predicted_class(self):
        # the head is the identity, so the predicted class is the argmax of the features
        detector = SHE(encoder=None, head=torch.nn.Identity())
        detector.patterns = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
        z = torch.tensor([[2.0, 1.0], [0.0, 2.0]])
        # predicted classes 0 and 1; inner products 2 and 6
        torch.testing.assert_close(detector.predict_features(z), torch.tensor([-2.0, -6.0]))
