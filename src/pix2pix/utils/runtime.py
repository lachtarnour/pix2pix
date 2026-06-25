from __future__ import annotations

import random

import torch
from torch import nn


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Enable or disable gradients for every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad
