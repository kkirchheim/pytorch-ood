import torch


class ClassificationModel(torch.nn.Module):
    def __init__(self, num_inputs=10, n_hidden=10, num_outputs=3):
        super(ClassificationModel, self).__init__()
        self.layer1 = torch.nn.Linear(num_inputs, n_hidden)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.classifier = torch.nn.Linear(n_hidden, num_outputs)
        self.encoder = torch.nn.Sequential(self.layer1, self.dropout)
    def features(self, x):
        return self.layer1(x).tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        return self.classifier(x)


class SegmentationModel(torch.nn.Module):
    """
    Mock model for semantic segmentation
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(SegmentationModel, self).__init__()
        self.p = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)


class ConvClassifier(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=16, n_hidden=10, num_outputs=3):
        super(ConvClassifier, self).__init__()
        self.layer1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.classifier = torch.nn.Linear(n_hidden, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.pool(x)
        print(x.shape)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
