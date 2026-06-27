from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
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
    image_count = source.size(0)

    images = torch.cat(
        [
            denormalize(source, detach=True, cpu=True),
            denormalize(generated, detach=True, cpu=True),
            denormalize(target, detach=True, cpu=True),
        ],
        dim=0,
    )
    grid = make_grid(images, nrow=image_count)
    grid = add_grid_titles(
        grid,
        image_size=(source.size(-1), source.size(-2)),
        column_count=image_count,
        row_titles=("Source", "Generated", "Target"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)

    if generator_was_training:
        generator.train()


def add_grid_titles(
    grid: torch.Tensor,
    *,
    image_size: tuple[int, int],
    column_count: int,
    row_titles: tuple[str, ...],
    padding: int = 2,
) -> torch.Tensor:
    """Draw a short title on each image in a tensor grid."""
    if column_count == 0:
        return grid

    grid_uint8 = (grid.clamp(0.0, 1.0) * 255).byte()
    image = grid_uint8.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, mode="RGBA")
    font = ImageFont.load_default()
    tile_width, tile_height = image_size

    for row_index, title in enumerate(row_titles):
        for column_index in range(column_count):
            x = padding + column_index * (tile_width + padding) + 4
            y = padding + row_index * (tile_height + padding) + 4
            text_box = draw.textbbox((x, y), title, font=font)
            background_box = (
                text_box[0] - 3,
                text_box[1] - 2,
                text_box[2] + 3,
                text_box[3] + 2,
            )

            draw.rectangle(background_box, fill=(0, 0, 0, 180))
            draw.text((x, y), title, fill=(255, 255, 255, 255), font=font)

    return pil_to_tensor(pil_image).float() / 255.0
