from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def linear_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    epochs_decay: int,
) -> LambdaLR:
    """Keep the learning rate constant, then linearly decay it to zero."""

    def lambda_rule(epoch_index: int) -> float:
        if epochs_decay == 0:
            return 1.0

        if epoch_index < epochs:
            return 1.0

        decay_step = epoch_index - epochs + 1
        return max(0.0, 1.0 - decay_step / float(epochs_decay + 1))

    return LambdaLR(optimizer, lr_lambda=lambda_rule)
