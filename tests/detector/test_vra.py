import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import VRA


def test_fit_and_predict_with_spatial_feature_maps():
    """
    Regression test: fit() must preserve the (N, C, H, W) shape of feature maps
    from a real conv backbone (H, W > 1), since predict_feature_maps() clips
    un-flattened feature maps of that same shape.
    """
    backbone = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
    head = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3)
    )
    detector = VRA(backbone=backbone, head=head)

    x = torch.randn(20, 3, 8, 8)
    y = torch.zeros(20, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    detector.fit(loader)

    scores = detector(torch.randn(5, 3, 8, 8))
    assert scores.shape == (5,)
