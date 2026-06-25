from __future__ import annotations

import tarfile
from pathlib import Path
from urllib.request import urlretrieve

from pix2pix.utils.logging import configure_logging, logger


URL = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
DATA_DIR = Path("datasets")
ARCHIVE_PATH = DATA_DIR / "maps.tar.gz"
DATASET_DIR = DATA_DIR / "maps"


def show_progress(block_number: int, block_size: int, total_size: int) -> None:
    downloaded = block_number * block_size

    if total_size > 0:
        percentage = min(downloaded * 100 / total_size, 100)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)

        print(
            f"\rDownloading: {percentage:6.2f}% "
            f"({downloaded_mb:.1f} MB / {total_mb:.1f} MB)",
            end="",
            flush=True,
        )
        return

    downloaded_mb = downloaded / (1024 * 1024)
    print(f"\rDownloading: {downloaded_mb:.1f} MB", end="", flush=True)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()

    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()

        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")

    archive.extractall(destination)


def main() -> None:
    configure_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_DIR.exists():
        logger.info("Dataset already available: %s", DATASET_DIR)
        return

    if not ARCHIVE_PATH.exists():
        logger.info("Downloading Maps dataset...")
        urlretrieve(URL, str(ARCHIVE_PATH), reporthook=show_progress)
        print()
        logger.info("Download finished.")

    logger.info("Extracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        _safe_extract(archive, DATA_DIR)

    ARCHIVE_PATH.unlink()
    logger.info("Dataset ready: %s", DATASET_DIR)


if __name__ == "__main__":
    main()
