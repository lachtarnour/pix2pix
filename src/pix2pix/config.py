from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Direction = Literal["AtoB", "BtoA"]
InitType = Literal["normal", "xavier", "kaiming", "orthogonal"]
NormType = Literal["batch", "instance", "none"]
Phase = Literal["train", "val", "test"]
WandbResume = Literal["allow", "must", "never", "auto"]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42

    # Dataset
    data_root: Path = Path("datasets/maps")
    direction: Direction = "AtoB"
    eval_phase: Phase = "val"
    train_load_size: int = 286
    image_size: int = 256

    # Training schedule
    epochs: int = 100
    epochs_decay: int = 100
    batch_size: int = 1
    num_workers: int = 0

    # Optimizer
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0

    # Model
    input_channels: int = 3
    output_channels: int = 3
    generator_filters: int = 64
    discriminator_filters: int = 64
    norm_type: NormType = "batch"
    init_type: InitType = "normal"
    init_gain: float = 0.02
    use_dropout: bool = True

    # Outputs
    output_dir: Path = Path("outputs/maps_pix2pix")
    checkpoint_dir: Path = Path("checkpoints/maps_pix2pix")
    print_every: int = 100
    sample_every: int = 1
    eval_every: int = 1
    checkpoint_archive_every: int = 5

    # Weights & Biases
    use_wandb: bool = True
    wandb_entity: str | None = None
    wandb_project: str = "pix2pix-maps"
    wandb_run_name: str = "pix2pix-maps"
    wandb_run_id: str | None = None
    wandb_resume: WandbResume = "allow"

    def to_dict(self) -> dict:
        config = asdict(self)

        for key, value in config.items():
            if isinstance(value, Path):
                config[key] = str(value)

        return config

    def to_wandb_config(self) -> dict:
        return self.to_dict()


CONFIG = TrainConfig()
