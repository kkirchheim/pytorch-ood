import torch


class ClassificationModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.p = torch.nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)


class SegmentationModel(torch.nn.Module):
    """
    Mock model for semantic segmentation
    """

    def __init__(self, in_channels=3, our_channels=3):
        super(SegmentationModel, self).__init__()
        self.p = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=our_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)
