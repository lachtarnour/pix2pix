from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from pix2pix.config import InitType, NormType


NormLayer = Callable[[int], nn.Module]


@dataclass(frozen=True)
class Pix2PixArchitectureConfig:
    input_channels: int = 3
    output_channels: int = 3
    generator_filters: int = 64
    discriminator_filters: int = 64
    norm_type: NormType = "batch"
    init_type: InitType = "normal"
    init_gain: float = 0.02
    use_dropout: bool = True


def get_norm_layer(norm_type: NormType = "batch") -> NormLayer:
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)

    if norm_type == "instance":
        return functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )

    if norm_type == "none":
        return lambda _channels: nn.Identity()

    raise ValueError(
        f"Unsupported norm_type={norm_type!r}; expected 'batch', 'instance' or 'none'."
    )


def _uses_bias(norm_type: NormType) -> bool:
    return norm_type in {"instance", "none"}


def initialize_weights(
    network: nn.Module,
    init_type: InitType = "normal",
    init_gain: float = 0.02,
) -> None:
    """Initialize network weights following the original Pix2Pix conventions."""

    def init_module(module: nn.Module) -> None:
        class_name = module.__class__.__name__
        has_weight = hasattr(module, "weight") and module.weight is not None
        is_conv_or_linear = "Conv" in class_name or "Linear" in class_name

        if has_weight and is_conv_or_linear:
            if init_type == "normal":
                nn.init.normal_(module.weight.data, mean=0.0, std=init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise ValueError(f"Unsupported init_type={init_type!r}")

            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)

        elif "BatchNorm2d" in class_name:
            nn.init.normal_(module.weight.data, mean=1.0, std=init_gain)
            nn.init.constant_(module.bias.data, 0.0)

    network.apply(init_module)


class UNetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_channels: int,
        inner_channels: int,
        *,
        norm_type: NormType,
        input_channels: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()

        if outermost and submodule is None:
            raise ValueError("The outermost U-Net block requires a submodule.")

        if innermost and submodule is not None:
            raise ValueError("The innermost U-Net block cannot have a submodule.")

        self.outermost = outermost

        if input_channels is None:
            input_channels = outer_channels

        norm_layer = get_norm_layer(norm_type)
        use_bias = _uses_bias(norm_type)

        down_conv = nn.Conv2d(
            input_channels,
            inner_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias,
        )
        down_relu = nn.LeakyReLU(0.2, inplace=True)
        down_norm = norm_layer(inner_channels)

        up_relu = nn.ReLU(inplace=True)
        up_norm = norm_layer(outer_channels)

        if outermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            layers = [down_conv, submodule, up_relu, up_conv, nn.Tanh()]

        elif innermost:
            up_conv = nn.ConvTranspose2d(
                inner_channels,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            layers = [down_relu, down_conv, up_relu, up_conv, up_norm]

        else:
            up_conv = nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            layers = [
                down_relu,
                down_conv,
                down_norm,
                submodule,
                up_relu,
                up_conv,
                up_norm,
            ]

            if use_dropout:
                layers.append(nn.Dropout(p=0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(image)

        return torch.cat([image, self.model(image)], dim=1)


class UNetGenerator(nn.Module):
    """U-Net generator used by Pix2Pix for 256x256 images."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_filters: int = 64,
        num_downs: int = 8,
        norm_type: NormType = "batch",
        use_dropout: bool = True,
        init_type: InitType = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()

        if num_downs < 5:
            raise ValueError("num_downs must be at least 5.")

        unet_block: nn.Module = UNetSkipConnectionBlock(
            outer_channels=base_filters * 8,
            inner_channels=base_filters * 8,
            innermost=True,
            norm_type=norm_type,
        )

        for _ in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(
                outer_channels=base_filters * 8,
                inner_channels=base_filters * 8,
                submodule=unet_block,
                norm_type=norm_type,
                use_dropout=use_dropout,
            )

        unet_block = UNetSkipConnectionBlock(
            outer_channels=base_filters * 4,
            inner_channels=base_filters * 8,
            submodule=unet_block,
            norm_type=norm_type,
        )
        unet_block = UNetSkipConnectionBlock(
            outer_channels=base_filters * 2,
            inner_channels=base_filters * 4,
            submodule=unet_block,
            norm_type=norm_type,
        )
        unet_block = UNetSkipConnectionBlock(
            outer_channels=base_filters,
            inner_channels=base_filters * 2,
            submodule=unet_block,
            norm_type=norm_type,
        )

        self.model = UNetSkipConnectionBlock(
            outer_channels=output_channels,
            inner_channels=base_filters,
            input_channels=input_channels,
            submodule=unet_block,
            outermost=True,
            norm_type=norm_type,
        )

        initialize_weights(self, init_type=init_type, init_gain=init_gain)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.model(source)


class PatchGANDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN discriminator."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_filters: int = 64,
        n_layers: int = 3,
        norm_type: NormType = "batch",
        init_type: InitType = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        norm_layer = get_norm_layer(norm_type)
        use_bias = _uses_bias(norm_type)
        discriminator_channels = input_channels + output_channels

        layers: list[nn.Module] = [
            nn.Conv2d(
                discriminator_channels,
                base_filters,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        previous_multiplier = 1
        current_multiplier = 1

        for layer_index in range(1, n_layers):
            previous_multiplier = current_multiplier
            current_multiplier = min(2**layer_index, 8)

            layers += [
                nn.Conv2d(
                    base_filters * previous_multiplier,
                    base_filters * current_multiplier,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(base_filters * current_multiplier),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        previous_multiplier = current_multiplier
        current_multiplier = min(2**n_layers, 8)

        layers += [
            nn.Conv2d(
                base_filters * previous_multiplier,
                base_filters * current_multiplier,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(base_filters * current_multiplier),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        layers.append(
            nn.Conv2d(
                base_filters * current_multiplier,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

        self.model = nn.Sequential(*layers)
        initialize_weights(self, init_type=init_type, init_gain=init_gain)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[0] != target.shape[0]:
            raise ValueError("source and target must have the same batch size.")

        if source.shape[2:] != target.shape[2:]:
            raise ValueError("source and target must have the same spatial size.")

        pair = torch.cat([source, target], dim=1)
        return self.model(pair)


def build_generator(
    config: Pix2PixArchitectureConfig | None = None,
) -> UNetGenerator:
    """Create a Pix2Pix U-Net generator."""
    config = config or Pix2PixArchitectureConfig()

    return UNetGenerator(
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        base_filters=config.generator_filters,
        norm_type=config.norm_type,
        use_dropout=config.use_dropout,
        init_type=config.init_type,
        init_gain=config.init_gain,
    )


def build_discriminator(
    config: Pix2PixArchitectureConfig | None = None,
) -> PatchGANDiscriminator:
    """Create a Pix2Pix PatchGAN discriminator."""
    config = config or Pix2PixArchitectureConfig()

    return PatchGANDiscriminator(
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        base_filters=config.discriminator_filters,
        norm_type=config.norm_type,
        init_type=config.init_type,
        init_gain=config.init_gain,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
