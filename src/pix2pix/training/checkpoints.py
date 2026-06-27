from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


LATEST_CHECKPOINT_NAME = "latest.pth"


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

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    checkpoint = {
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
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(output_path)


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


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the checkpoint used to resume training."""
    latest_checkpoint = checkpoint_dir / LATEST_CHECKPOINT_NAME

    if latest_checkpoint.is_file():
        return latest_checkpoint

    return None
