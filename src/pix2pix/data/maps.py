from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from pix2pix.config import Direction, Phase


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_SIZE = 256
DEFAULT_TRAIN_LOAD_SIZE = 286


class MapsDataset(Dataset):
    """Load aligned image pairs from the Pix2Pix Maps dataset."""

    def __init__(
        self,
        root: str | Path = "datasets/maps",
        phase: Phase = "train",
        direction: Direction = "AtoB",
        load_size: int = DEFAULT_TRAIN_LOAD_SIZE,
        crop_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if direction not in {"AtoB", "BtoA"}:
            raise ValueError("direction must be either 'AtoB' or 'BtoA'.")

        if load_size < crop_size:
            raise ValueError("load_size must be greater than or equal to crop_size.")

        self.phase = phase
        self.direction = direction
        self.load_size = load_size
        self.crop_size = crop_size
        self.image_directory = Path(root) / phase

        if not self.image_directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.image_directory}\n"
                "Run the dataset download script first."
            )

        self.image_paths = sorted(
            path
            for path in self.image_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(f"No images were found in {self.image_directory}")

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def split_pair(paired_image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Split a horizontally concatenated image into image A and image B."""
        width, height = paired_image.size
        middle = width // 2

        image_a = paired_image.crop((0, 0, middle, height))
        image_b = paired_image.crop((middle, 0, width, height))

        return image_a, image_b

    def transform_pair(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply identical spatial transforms to both paired images."""
        resize_shape = [self.load_size, self.load_size]

        image_a = TF.resize(
            image_a,
            resize_shape,
            interpolation=InterpolationMode.BICUBIC,
        )
        image_b = TF.resize(
            image_b,
            resize_shape,
            interpolation=InterpolationMode.BICUBIC,
        )

        if self.phase == "train":
            maximum_offset = self.load_size - self.crop_size
            crop_top = random.randint(0, maximum_offset)
            crop_left = random.randint(0, maximum_offset)
            apply_horizontal_flip = random.random() > 0.5
        else:
            crop_top = (self.load_size - self.crop_size) // 2
            crop_left = (self.load_size - self.crop_size) // 2
            apply_horizontal_flip = False

        image_a = TF.crop(
            image_a,
            top=crop_top,
            left=crop_left,
            height=self.crop_size,
            width=self.crop_size,
        )
        image_b = TF.crop(
            image_b,
            top=crop_top,
            left=crop_left,
            height=self.crop_size,
            width=self.crop_size,
        )

        if apply_horizontal_flip:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)

        normalization_mean = [0.5, 0.5, 0.5]
        normalization_std = [0.5, 0.5, 0.5]

        tensor_a = TF.normalize(
            TF.to_tensor(image_a),
            mean=normalization_mean,
            std=normalization_std,
        )
        tensor_b = TF.normalize(
            TF.to_tensor(image_b),
            mean=normalization_mean,
            std=normalization_std,
        )

        return tensor_a, tensor_b

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            paired_image = image.convert("RGB")

        image_a, image_b = self.split_pair(paired_image)
        image_a, image_b = self.transform_pair(image_a, image_b)

        if self.direction == "AtoB":
            source_image = image_a
            target_image = image_b
        else:
            source_image = image_b
            target_image = image_a

        return {
            "source": source_image,
            "target": target_image,
            "path": str(image_path),
        }


def create_dataloader(
    root: str | Path = "datasets/maps",
    phase: Phase = "train",
    direction: Direction = "AtoB",
    batch_size: int = 1,
    num_workers: int = 0,
    load_size: int | None = None,
    crop_size: int = DEFAULT_IMAGE_SIZE,
) -> DataLoader:
    """Create a DataLoader for training, validation, or testing."""
    if load_size is None:
        load_size = DEFAULT_TRAIN_LOAD_SIZE if phase == "train" else crop_size

    dataset = MapsDataset(
        root=root,
        phase=phase,
        direction=direction,
        load_size=load_size,
        crop_size=crop_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
