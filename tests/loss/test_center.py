import torch

from pytorch_ood.loss import CenterLoss


def test_squared_distances_to_the_centers():
    criterion = CenterLoss(n_classes=3, n_dim=2)
    criterion.centers.params.data = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    z = torch.tensor([[1.0, 1.0]])
    torch.testing.assert_close(criterion.distance(z), torch.tensor([[2.0, 1.0, 2.0]]))
