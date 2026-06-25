from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def save_checkpoint(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: LambdaLR,
    scheduler_d: LambdaLR,
    epoch: int,
    global_step: int,
    best_val_mae: float,
    best_epoch: int,
    output_path: Path,
) -> None:
    """Save the full training state needed to resume training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
        },
        output_path,
    )


def load_checkpoint(
    *,
    checkpoint_path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: LambdaLR,
    scheduler_d: LambdaLR,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint and restore all training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    scheduler_d.load_state_dict(checkpoint["scheduler_d"])

    return checkpoint


def get_epoch_from_checkpoint_path(path: Path) -> int | None:
    """Extract an epoch number from a checkpoint path like epoch_0020.pth."""
    match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)

    if match is None:
        return None

    return int(match.group(1))


def find_last_periodic_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent periodic checkpoint named epoch_XXXX.pth."""
    if not checkpoint_dir.exists():
        return None

    checkpoint_paths: list[tuple[int, Path]] = []

    for path in checkpoint_dir.glob("epoch_*.pth"):
        epoch = get_epoch_from_checkpoint_path(path)

        if epoch is not None:
            checkpoint_paths.append((epoch, path))

    if not checkpoint_paths:
        return None

    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths[-1][1]
