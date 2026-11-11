import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import ViM
from tests.helpers import ClassificationModel, requires


def test_ill_conditioned_features_match_float64_reference():
    torch.manual_seed(0)
    n, dim, c, d = 2000, 64, 4, 16
    scales = torch.logspace(0, -6, dim)
    rotation = torch.linalg.qr(torch.randn(dim, dim))[0]
    # ReLU features with a large mean, as penultimate CNN features: ViM centers at u, not at the mean
    mean = 5.0 * torch.rand(dim)
    features = ((torch.randn(n, dim) * scales) @ rotation.T + mean).relu()
    x = ((torch.randn(32, dim) * scales) @ rotation.T + 1.2 * mean).relu()
    w, b = torch.randn(c, dim), torch.randn(c)

    detector = ViM(None, d=d, w=w, b=b).fit_features(features, torch.zeros(n).long())
    residual = ((x - detector.u) @ detector.residual_subspace).norm(dim=1)

    u = -(torch.linalg.pinv(w.double()) @ b.double())
    centered = features.double() - u
    _, vecs = torch.linalg.eigh(centered.T @ centered / n)
    expected = ((x.double() - u) @ vecs[:, : dim - d]).norm(dim=1)

    torch.testing.assert_close(residual.double(), expected, rtol=1e-3, atol=1e-6)


@requires(cuda=1)
def test_device_follows_the_tensors_given_to_the_constructor(accelerator: torch.device):
    # state must not be pinned to the CPU when the weights are on the GPU, without calling to()
    model = ClassificationModel().to(accelerator)
    detector = ViM(model.features, d=4, w=model.classifier.weight, b=model.classifier.bias)
    assert detector.w.device == detector.u.device == accelerator

    x = torch.randn(24, 10)
    detector.fit(DataLoader(TensorDataset(x, torch.arange(24) % 3), batch_size=6))

    assert detector.residual_subspace.device == accelerator
    assert detector(torch.randn(8, 10, device=accelerator)).device == accelerator
