import torch

from pytorch_ood.loss import EnergyRegularizedLoss

torch.manual_seed(123)


class TestEnergyRegularization:
    """
    Test code for energy bounded learning
    """

    def test_matches_paper_formula(self):
        """CE over ID samples plus separately averaged ID and OOD hinges (Liu et al., 2020)."""
        criterion = EnergyRegularizedLoss(alpha=0.5, margin_in=-3.0, margin_out=1.0)
        logits = torch.randn(size=(12, 4))
        target = torch.arange(12) % 5 - 1  # some -1
        known = target >= 0
        energy = -torch.logsumexp(logits, dim=1)
        expected = torch.nn.functional.cross_entropy(logits[known], target[known]) + 0.5 * (
            (energy[known] + 3.0).relu().pow(2).mean()
            + (1.0 - energy[~known]).relu().pow(2).mean()
        )
        torch.testing.assert_close(criterion(logits, target), expected)

    def test_segmentation_equals_pixels_as_samples(self):
        criterion = EnergyRegularizedLoss(margin_in=-3.0, margin_out=1.0)
        logits = torch.randn(size=(2, 4, 5, 5))
        target = torch.arange(50).reshape(2, 5, 5) % 5 - 1
        flat = criterion(logits.permute(0, 2, 3, 1).reshape(-1, 4), target.reshape(-1))
        torch.testing.assert_close(criterion(logits, target), flat)
