from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.utils import make_grid, save_image


def denormalize(
    image_batch: torch.Tensor,
    *,
    detach: bool = False,
    cpu: bool = False,
) -> torch.Tensor:
    """Convert image tensors from [-1, 1] to [0, 1]."""
    images = image_batch

    if detach:
        images = images.detach()

    if cpu:
        images = images.cpu()

    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def save_sample_images(
    generator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: Path,
    max_images: int = 4,
) -> None:
    """Save source, generated and target images as one comparison grid."""
    generator_was_training = generator.training
    generator.eval()

    batch = next(iter(dataloader))
    source = batch["source"][:max_images].to(device)
    target = batch["target"][:max_images].to(device)
    generated = generator(source)

    images = torch.cat(
        [
            denormalize(source, detach=True, cpu=True),
            denormalize(generated, detach=True, cpu=True),
            denormalize(target, detach=True, cpu=True),
        ],
        dim=0,
    )
    grid = make_grid(images, nrow=max_images)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)

    if generator_was_training:
        generator.train()
