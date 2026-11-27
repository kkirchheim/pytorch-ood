import torch

from pytorch_ood.detector import EnergyBased, JointEnergy


def _official(logits):
    # score of the reference implementation (lib.py, --ood energy --method sum), larger is ID
    return torch.sum(torch.log(1 + torch.exp(logits)), dim=1)


def test_matches_reference_implementation():
    """Scores are the negated JointEnergy of the official code."""
    logits = torch.randn(16, 20, dtype=torch.float64) * 3
    assert torch.allclose(JointEnergy.score(logits), -_official(logits))


def test_hand_computed_value():
    """log(1 + e^0) + log(1 + e^{log 3}) = log 2 + log 4 = log 8."""
    logits = torch.tensor([[0.0, torch.log(torch.tensor(3.0))]])
    assert torch.allclose(JointEnergy.score(logits), -torch.log(torch.tensor([8.0])))


def test_stable_for_large_logits():
    """The naive log(1 + exp(x)) overflows to inf for large logits; the detector does not."""
    logits = torch.tensor([[100.0, 200.0, -200.0]])
    scores = JointEnergy.score(logits)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, torch.tensor([-300.0]))


def test_more_confident_labels_are_less_ood():
    """Raising any label's logit lowers the outlier score."""
    logits = torch.randn(8, 5)
    higher = logits.clone()
    higher[:, 2] += 1.0
    assert (JointEnergy.score(higher) < JointEnergy.score(logits)).all()


def test_differs_from_softmax_energy():
    """Unlike the softmax energy, JointEnergy is not invariant to shifting all logits."""
    logits = torch.randn(8, 5)
    shifted = logits + 2.0
    assert torch.allclose(
        EnergyBased.score(shifted) - EnergyBased.score(logits), torch.full((8,), -2.0)
    )
    assert not torch.allclose(
        JointEnergy.score(shifted) - JointEnergy.score(logits), torch.full((8,), -2.0)
    )


def test_predict_logits_and_model():
    """predict_logits and predict on a model both return the score of shape B."""
    model = torch.nn.Linear(4, 6)
    x = torch.randn(10, 4)
    detector = JointEnergy(model)
    expected = JointEnergy.score(model(x))
    assert detector.predict(x).shape == (10,)
    assert torch.allclose(detector.predict(x), expected)
    assert torch.allclose(JointEnergy(None).predict_logits(model(x)), expected)
