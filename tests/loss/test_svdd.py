import pytest
import torch

from pytorch_ood.loss import DeepSADLoss, DeepSVDDLoss


class TestDeepSAD:
    def test_deprecated_name(self):
        import pytorch_ood.loss as losses

        with pytest.warns(DeprecationWarning):
            assert losses.SSDeepSVDDLoss is DeepSADLoss


class TestDeepSADValues:
    def _loss(self, **kwargs):
        criterion = DeepSADLoss(n_dim=2, reduction="none", **kwargs)
        criterion.center.params.data.zero_()
        return criterion

    def test_squared_distance(self):
        x = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        loss = self._loss()(x, torch.tensor([0, -1]))
        torch.testing.assert_close(loss, torch.tensor([4.0, 1 / (4.0 + 1e-6)]))

    def test_eta_weights_outliers(self):
        x = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        loss = self._loss(eta=3.0)(x, torch.tensor([0, -1]))
        torch.testing.assert_close(loss, torch.tensor([4.0, 3 / (4.0 + 1e-6)]))

    def test_outlier_at_center_is_finite(self):
        loss = self._loss()(torch.zeros(1, 2), torch.tensor([-1]))
        assert torch.isfinite(loss).all()


class TestSVDDLossDefaultRadius:
    def test_default_radius(self):
        criterion = DeepSVDDLoss(n_dim=2)
        loss = DeepSVDDLoss.svdd_loss(torch.randn(3, 2), criterion.center)
        assert loss.shape == (3,)


class TestDeepSADCenterAndDistance:
    def test_center_argument(self):
        center = torch.tensor([1.0, 2.0])
        criterion = DeepSADLoss(n_dim=2, center=center)
        torch.testing.assert_close(criterion.center.params, center.reshape(1, 2))

    def test_distance_is_squared_distance_to_center(self):
        criterion = DeepSADLoss(n_dim=2, center=torch.tensor([1.0, 0.0]))
        x = torch.tensor([[1.0, 0.0], [3.0, 0.0], [1.0, -1.0]])
        torch.testing.assert_close(criterion.distance(x), torch.tensor([0.0, 4.0, 1.0]))
