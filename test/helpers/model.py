import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.p = torch.nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)
