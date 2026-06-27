from __future__ import annotations

import os
from typing import Any

from pix2pix.config import TrainConfig


try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def init_wandb(config: TrainConfig) -> None:
    """Initialize Weights & Biases if tracking is enabled."""
    if not config.use_wandb:
        return

    if wandb is None:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")

    run_id = os.getenv("WANDB_RUN_ID") or config.wandb_run_id
    entity = os.getenv("WANDB_ENTITY") or config.wandb_entity
    resume = os.getenv("WANDB_RESUME") or config.wandb_resume

    init_kwargs: dict[str, Any] = {
        "entity": entity,
        "project": config.wandb_project,
        "name": config.wandb_run_name,
        "config": config.to_wandb_config(),
        "save_code": False,
        "settings": wandb.Settings(
            disable_code=True,
            x_disable_machine_info=True,
            x_disable_stats=True,
        ),
    }

    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = resume

    wandb.init(**init_kwargs)


def log_to_wandb(metrics: dict[str, float], step: int) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()
