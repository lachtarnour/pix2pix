from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam

from pix2pix.config import CONFIG, TrainConfig
from pix2pix.data import create_dataloader
from pix2pix.models import (
    Pix2PixArchitectureConfig,
    build_discriminator,
    build_generator,
)
from pix2pix.training.checkpoints import (
    find_last_periodic_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from pix2pix.training.evaluation import evaluate
from pix2pix.training.losses import make_labels_like
from pix2pix.training.schedulers import linear_decay_scheduler
from pix2pix.utils.images import save_sample_images
from pix2pix.utils.logging import configure_logging, log_metrics, logger
from pix2pix.utils.runtime import get_device, set_requires_grad, set_seed
from pix2pix.utils.system_metrics import collect_system_metrics


try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def init_wandb(config: TrainConfig) -> None:
    """Initialize Weights & Biases only when enabled."""
    if not config.use_wandb:
        return

    if wandb is None:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.to_wandb_config(),
        save_code=False,
        settings=wandb.Settings(
            x_disable_stats=True,
            x_disable_machine_info=True,
            disable_code=True,
        ),
    )


def log_to_wandb(metrics: dict[str, float], step: int) -> None:
    """Log metrics to Weights & Biases if a run is active."""
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def train(config: TrainConfig = CONFIG) -> None:
    configure_logging()
    set_seed(config.seed)
    device = get_device()
    init_wandb(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", device)
    logger.info("Data root: %s", config.data_root)
    logger.info("Direction: %s", config.direction)

    train_loader = create_dataloader(
        root=config.data_root,
        phase="train",
        direction=config.direction,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        load_size=config.train_load_size,
        crop_size=config.image_size,
    )
    val_loader = create_dataloader(
        root=config.data_root,
        phase=config.eval_phase,
        direction=config.direction,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        load_size=config.image_size,
        crop_size=config.image_size,
    )

    logger.info(
        "Training images: %d | Evaluation images: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    architecture_config = Pix2PixArchitectureConfig(
        input_channels=config.input_channels,
        output_channels=config.output_channels,
        generator_filters=config.generator_filters,
        discriminator_filters=config.discriminator_filters,
        norm_type=config.norm_type,
        init_type=config.init_type,
        init_gain=config.init_gain,
        use_dropout=config.use_dropout,
    )

    generator = build_generator(architecture_config).to(device)
    discriminator = build_discriminator(architecture_config).to(device)

    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()

    optimizer_g = Adam(
        generator.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = Adam(
        discriminator.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )

    scheduler_g = linear_decay_scheduler(
        optimizer_g,
        epochs=config.epochs,
        epochs_decay=config.epochs_decay,
    )
    scheduler_d = linear_decay_scheduler(
        optimizer_d,
        epochs=config.epochs,
        epochs_decay=config.epochs_decay,
    )

    total_epochs = config.epochs + config.epochs_decay
    global_step = 0
    best_val_mae = float("inf")
    best_epoch = 0
    start_epoch = 1

    last_checkpoint_path = find_last_periodic_checkpoint(config.checkpoint_dir)

    if last_checkpoint_path is not None:
        logger.info("Resuming from periodic checkpoint: %s", last_checkpoint_path)
        checkpoint = load_checkpoint(
            checkpoint_path=last_checkpoint_path,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            device=device,
        )

        checkpoint_epoch = int(checkpoint["epoch"])
        start_epoch = checkpoint_epoch + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_mae = float(checkpoint.get("best_val_mae", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))

        logger.info(
            "Resumed from epoch %d | next epoch=%d | best_epoch=%d | best_val_mae=%.6f",
            checkpoint_epoch,
            start_epoch,
            best_epoch,
            best_val_mae,
        )
    else:
        logger.info("No periodic checkpoint found. Starting from scratch.")

    if start_epoch > total_epochs:
        logger.info(
            "Training already completed. start_epoch=%d > total_epochs=%d",
            start_epoch,
            total_epochs,
        )
        return

    for epoch in range(start_epoch, total_epochs + 1):
        generator.train()
        discriminator.train()

        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_loss_g_gan = 0.0
        epoch_loss_g_l1 = 0.0

        for batch_index, batch in enumerate(train_loader, start=1):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            fake_target = generator(source)

            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)

            pred_real = discriminator(source, target)
            loss_d_real = adversarial_loss(pred_real, make_labels_like(pred_real, 1.0))

            pred_fake = discriminator(source, fake_target.detach())
            loss_d_fake = adversarial_loss(pred_fake, make_labels_like(pred_fake, 0.0))

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            optimizer_d.step()

            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)

            pred_fake_for_g = discriminator(source, fake_target)
            loss_g_gan = adversarial_loss(
                pred_fake_for_g,
                make_labels_like(pred_fake_for_g, 1.0),
            )
            loss_g_l1 = reconstruction_loss(fake_target, target) * config.lambda_l1
            loss_g = loss_g_gan + loss_g_l1

            loss_g.backward()
            optimizer_g.step()

            global_step += 1

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_loss_g_gan += loss_g_gan.item()
            epoch_loss_g_l1 += loss_g_l1.item()

            if config.print_every > 0 and batch_index % config.print_every == 0:
                step_metrics = {
                    "step/loss_d": loss_d.item(),
                    "step/loss_g": loss_g.item(),
                    "step/loss_g_gan": loss_g_gan.item(),
                    "step/loss_g_l1": loss_g_l1.item(),
                    "step/lr": optimizer_g.param_groups[0]["lr"],
                }
                step_metrics.update(collect_system_metrics(device))

                logger.info(
                    "Epoch [%d/%d] Batch [%d/%d] "
                    "D: %.4f G: %.4f G_GAN: %.4f G_L1: %.4f",
                    epoch,
                    total_epochs,
                    batch_index,
                    len(train_loader),
                    loss_d.item(),
                    loss_g.item(),
                    loss_g_gan.item(),
                    loss_g_l1.item(),
                )
                log_to_wandb(step_metrics, step=global_step)

        scheduler_g.step()
        scheduler_d.step()

        num_batches = len(train_loader)
        epoch_metrics = {
            "epoch/loss_d": epoch_loss_d / num_batches,
            "epoch/loss_g": epoch_loss_g / num_batches,
            "epoch/loss_g_gan": epoch_loss_g_gan / num_batches,
            "epoch/loss_g_l1": epoch_loss_g_l1 / num_batches,
            "epoch/lr": optimizer_g.param_groups[0]["lr"],
        }

        log_metrics(f"End epoch [{epoch}/{total_epochs}]", epoch_metrics)
        log_to_wandb(epoch_metrics, step=global_step)

        if config.eval_every > 0 and epoch % config.eval_every == 0:
            val_metrics = evaluate(
                generator=generator,
                discriminator=discriminator,
                dataloader=val_loader,
                device=device,
                adversarial_loss=adversarial_loss,
                reconstruction_loss=reconstruction_loss,
                lambda_l1=config.lambda_l1,
            )

            log_metrics(f"Validation epoch [{epoch}/{total_epochs}]", val_metrics)
            log_to_wandb(val_metrics, step=global_step)

            current_val_mae = val_metrics["val/mae"]

            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                best_epoch = epoch
                save_checkpoint(
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scheduler_g=scheduler_g,
                    scheduler_d=scheduler_d,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_mae=best_val_mae,
                    best_epoch=best_epoch,
                    output_path=config.checkpoint_dir / "best.pth",
                )

                logger.info(
                    "New best checkpoint saved | epoch=%d | val/mae=%.6f",
                    best_epoch,
                    best_val_mae,
                )
                log_to_wandb(
                    {"best/epoch": best_epoch, "best/val_mae": best_val_mae},
                    step=global_step,
                )

        if config.sample_every > 0 and epoch % config.sample_every == 0:
            sample_path = config.output_dir / "samples" / f"epoch_{epoch:04d}.png"
            save_sample_images(generator, val_loader, device, sample_path)

        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            checkpoint_path = config.checkpoint_dir / f"epoch_{epoch:04d}.pth"
            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                epoch=epoch,
                global_step=global_step,
                best_val_mae=best_val_mae,
                best_epoch=best_epoch,
                output_path=checkpoint_path,
            )
            logger.info("Periodic checkpoint saved: %s", checkpoint_path)

    if wandb is not None and wandb.run is not None:
        wandb.finish()

    logger.info("Training finished.")


if __name__ == "__main__":
    train()
