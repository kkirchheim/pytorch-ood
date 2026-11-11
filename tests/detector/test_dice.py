import torch

from pytorch_ood.detector import DICE


def test_scores_are_the_energy_of_the_sparsified_head():
    g = torch.Generator().manual_seed(0)
    w, b = torch.randn(5, 16, generator=g), torch.randn(5, generator=g)
    z, x = torch.rand(100, 16, generator=g), torch.rand(8, 16, generator=g)
    p = 0.7

    detector = DICE(encoder=None, w=w, b=b, p=p).fit_features(z, torch.zeros(100).long())

    # keep the weights whose contribution (mean ID activation times weight) is above the p-quantile
    contribution = z.double().mean(dim=0) * w.double()
    keep = contribution > torch.quantile(contribution.flatten(), p)
    logits = (x.double()[:, None, :] * (w.double() * keep)).sum(dim=2) + b.double()
    expected = -torch.logsumexp(logits, dim=1)

    torch.testing.assert_close(detector.predict_features(x).double(), expected)
