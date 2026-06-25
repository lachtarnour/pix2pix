from __future__ import annotations

import os

import psutil
import torch


def collect_system_metrics(device: torch.device) -> dict[str, float]:
    process = psutil.Process(os.getpid())
    memory = psutil.virtual_memory()

    metrics = {
        "system/process_ram_gb": process.memory_info().rss / (1024**3),
        "system/ram_used_gb": memory.used / (1024**3),
        "system/ram_available_gb": memory.available / (1024**3),
    }

    if device.type == "mps":
        metrics.update(
            {
                "system/mps_memory_allocated_gb": (
                    torch.mps.current_allocated_memory() / (1024**3)
                ),
                "system/mps_memory_driver_gb": (
                    torch.mps.driver_allocated_memory() / (1024**3)
                ),
            }
        )

    return metrics
