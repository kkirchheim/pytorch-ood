"""Per-iteration cosine LR schedule, matching OpenOOD's ``BaseTrainer`` (used
for every dataset, including CIFAR): a ``LambdaLR`` cosine decay stepped once
per training iteration (not per epoch) over ``total_steps`` optimizer steps,
with no warmup.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_annealing(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(step / total_steps * math.pi))


def per_iteration_cosine(optimizer: Optimizer, total_steps: int, lr_min: float = 1e-6) -> LambdaLR:
    lr_max = optimizer.defaults["lr"]
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step, total_steps, 1, lr_min / lr_max),
    )
