from __future__ import annotations

import torch


def make_labels_like(predictions: torch.Tensor, value: float) -> torch.Tensor:
    """Create labels with the same shape as a PatchGAN logits map."""
    return torch.full_like(predictions, fill_value=value)


extend_labels = make_labels_like
