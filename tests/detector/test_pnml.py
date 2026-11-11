import pytest
import torch

from pytorch_ood.api import ModelNotSetException
from pytorch_ood.detector import PNML


def test_no_head():
    torch.manual_seed(123)
    detector = PNML(encoder=None, head=None)

    z = torch.randn(size=(20, 10))
    y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
    detector.fit_features(z, y)

    with pytest.raises(ModelNotSetException):
        detector.predict_features(torch.randn(size=(5, 10)))


def test_ill_conditioned_features_match_float64_reference():
    torch.manual_seed(0)
    n, d, c = 500, 64, 3
    scales = torch.logspace(0, -5, d)  # Gram eigenvalues span 10 orders of magnitude
    z_train = torch.randn(n, d) * scales
    z_test = torch.randn(16, d) * scales
    head = torch.nn.Linear(d, c)

    detector = PNML(encoder=None, head=head).fit_features(z_train, torch.zeros(n).long())
    scores = detector.predict_features(z_test)

    zn = torch.nn.functional.normalize(z_train.double(), dim=1)
    xn = torch.nn.functional.normalize(z_test.double(), dim=1)
    projector = torch.linalg.pinv(zn.T @ zn, hermitian=True)
    x_proj = (xn @ projector * xn).sum(dim=1)
    x_t_g = x_proj / (1.0 + x_proj)
    with torch.no_grad():
        probs = torch.softmax(head(xn.float()), dim=1).double().clamp(1e-12, 1 - 1e-12)
    terms = probs / (probs + (1.0 - probs) * probs.pow(x_t_g.unsqueeze(1)))
    expected = torch.log(terms.sum(dim=1)) / torch.log(torch.tensor(float(c), dtype=torch.float64))

    torch.testing.assert_close(scores.double(), expected, rtol=1e-4, atol=1e-5)
