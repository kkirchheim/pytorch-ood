import torch

from pytorch_ood.detector import OpenMax


def test_score_does_not_depend_on_which_classes_the_other_samples_are_predicted_as():
    # Four classes whose training logits lie around the unit vectors, two per class
    centers = 5 * torch.eye(4)
    offsets = torch.tensor([[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]])
    logits = (centers[:, None, :] + offsets).reshape(8, 4)
    y = torch.arange(4).repeat_interleave(2)
    detector = OpenMax(None, tailsize=2, alpha=2).fit_logits(logits, y)

    # Predicted as class 0, second-ranked class 1 (alpha=2 revises both), and far from the
    # center of class 1.
    sample = torch.tensor([[4.0, 3.0, 0.0, 0.0]])
    # Predicted as class 1, so class 1 is the top-1 class of a sample in this batch.
    other = torch.tensor([[0.0, 5.0, 0.0, 0.0]])

    alone = detector.predict_logits(sample)
    in_batch = detector.predict_logits(torch.cat([sample, other]))[:1]

    torch.testing.assert_close(alone, in_batch)
