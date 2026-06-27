from __future__ import annotations

import os
import subprocess

import psutil
import torch


def collect_system_metrics(device: torch.device) -> dict[str, float]:
    metrics = _collect_ram_metrics()

    if device.type == "cuda" and torch.cuda.is_available():
        metrics.update(_collect_cuda_metrics(device))
    elif device.type == "mps":
        metrics.update(_collect_mps_metrics())

    return metrics


def _collect_ram_metrics() -> dict[str, float]:
    process = psutil.Process(os.getpid())
    memory = psutil.virtual_memory()

    return {
        "system/process_ram_gb": process.memory_info().rss / (1024**3),
        "system/ram_used_gb": memory.used / (1024**3),
    }


def _collect_cuda_metrics(device: torch.device) -> dict[str, float]:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu="
                "power.draw,clocks.sm,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 5:
        return {}

    try:
        power_w, frequency_mhz, utilization_percent, memory_used_mb, memory_total_mb = (
            float(value)
            for value in values
        )
    except ValueError:
        return {}

    return {
        "system/gpu_power_w": power_w,
        "system/gpu_frequency_mhz": frequency_mhz,
        "system/gpu_utilization_percent": utilization_percent,
        "system/gpu_memory_used_gb": memory_used_mb / 1024,
        "system/gpu_memory_total_gb": memory_total_mb / 1024,
    }


def _collect_mps_metrics() -> dict[str, float]:
    return {
        "system/mps_memory_allocated_gb": (
            torch.mps.current_allocated_memory() / (1024**3)
        ),
        "system/mps_memory_driver_gb": torch.mps.driver_allocated_memory() / (1024**3),
    }
