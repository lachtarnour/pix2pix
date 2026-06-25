from __future__ import annotations

import logging


logger = logging.getLogger("pix2pix")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    formatted = " ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
    logger.info("%s %s", prefix, formatted)
