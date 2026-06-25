from __future__ import annotations

from pathlib import Path

import torch

from pix2pix.models import (
    Pix2PixArchitectureConfig,
    build_discriminator,
    build_generator,
)

def test_pix2pix_models_forward_pass() -> None:
    config = Pix2PixArchitectureConfig(
        generator_filters=8,
        discriminator_filters=8,
    )
    generator = build_generator(config).eval()
    discriminator = build_discriminator(config).eval()
    source = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        generated = generator(source)

    assert generated.shape == source.shape
