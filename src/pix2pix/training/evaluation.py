from __future__ import annotations

import math

import torch
from torch import nn

from pix2pix.training.losses import make_labels_like
from pix2pix.utils.images import denormalize


def psnr_from_mse(mse: float) -> float:
    """Compute PSNR assuming image values are in [0, 1]."""
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    adversarial_loss: nn.Module,
    reconstruction_loss: nn.Module,
    lambda_l1: float,
) -> dict[str, float]:
    """Evaluate Pix2Pix models on a validation split."""
    generator_was_training = generator.training
    discriminator_was_training = discriminator.training

    generator.eval()
    discriminator.eval()

    total_loss_d = 0.0
    total_loss_g = 0.0
    total_loss_g_gan = 0.0
    total_loss_g_l1 = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_d_real_score = 0.0
    total_d_fake_score = 0.0
    total_samples = 0

    for batch in dataloader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        batch_size = source.size(0)

        fake_target = generator(source)
        pred_real = discriminator(source, target)
        pred_fake = discriminator(source, fake_target)

        loss_d_real = adversarial_loss(pred_real, make_labels_like(pred_real, 1.0))
        loss_d_fake = adversarial_loss(pred_fake, make_labels_like(pred_fake, 0.0))
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        loss_g_gan = adversarial_loss(pred_fake, make_labels_like(pred_fake, 1.0))
        loss_g_l1 = reconstruction_loss(fake_target, target) * lambda_l1
        loss_g = loss_g_gan + loss_g_l1

        fake_01 = denormalize(fake_target)
        target_01 = denormalize(target)
        mae = torch.mean(torch.abs(fake_01 - target_01))
        mse = torch.mean((fake_01 - target_01) ** 2)

        d_real_score = torch.sigmoid(pred_real).mean()
        d_fake_score = torch.sigmoid(pred_fake).mean()

        total_loss_d += loss_d.item() * batch_size
        total_loss_g += loss_g.item() * batch_size
        total_loss_g_gan += loss_g_gan.item() * batch_size
        total_loss_g_l1 += loss_g_l1.item() * batch_size
        total_mae += mae.item() * batch_size
        total_mse += mse.item() * batch_size
        total_d_real_score += d_real_score.item() * batch_size
        total_d_fake_score += d_fake_score.item() * batch_size
        total_samples += batch_size

    if generator_was_training:
        generator.train()

    if discriminator_was_training:
        discriminator.train()

    if total_samples == 0:
        raise RuntimeError("Evaluation dataloader is empty.")

    mean_mse = total_mse / total_samples

    return {
        "val/loss_d": total_loss_d / total_samples,
        "val/loss_g": total_loss_g / total_samples,
        "val/loss_g_gan": total_loss_g_gan / total_samples,
        "val/loss_g_l1": total_loss_g_l1 / total_samples,
        "val/mae": total_mae / total_samples,
        "val/mse": mean_mse,
        "val/psnr": psnr_from_mse(mean_mse),
        "val/d_real_score": total_d_real_score / total_samples,
        "val/d_fake_score": total_d_fake_score / total_samples,
    }
