import torch


def apply_reduction(tensor: torch.Tensor, reduction: str):
    """
    Apply specific reduction to a tensor
    """
    if reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    elif reduction is None:
        return tensor
    else:
        raise ValueError
