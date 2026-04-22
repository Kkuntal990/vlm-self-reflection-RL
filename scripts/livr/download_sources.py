#!/usr/bin/env python3
"""Download all source datasets for LIVR perception MCQ construction.

Idempotent: skips datasets that are already downloaded.
Run on the pod (vlm-jupyter-eval2) where /outputs/ is mounted.

Usage:
    python scripts/livr/download_sources.py
    python scripts/livr/download_sources.py --only coco_annotations pixmo_count
"""

import argparse
import logging
import os
import subprocess
import sys
import tarfile
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = "/outputs/livr_sources"


def _set_base_dir(new_dir: str) -> None:
    """Update the module-level BASE_DIR.

    Args:
        new_dir: New base directory path.
    """
    global BASE_DIR
    BASE_DIR = new_dir
    os.makedirs(BASE_DIR, exist_ok=True)


def _run(cmd: str, desc: str) -> bool:
    """Run a shell command, logging output.

    Args:
        cmd: Shell command string.
        desc: Description for logging.

    Returns:
        True if successful.
    """
    logger.info(f"[{desc}] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[{desc}] Failed: {result.stderr}")
        return False
    return True


def _marker(name: str) -> str:
    """Path to a download-complete marker file.

    Args:
        name: Dataset name.

    Returns:
        Path to marker file.
    """
    return os.path.join(BASE_DIR, f".{name}_done")


def _is_done(name: str) -> bool:
    """Check if a dataset has already been downloaded.

    Args:
        name: Dataset name.

    Returns:
        True if marker file exists.
    """
    return os.path.exists(_marker(name))


def _mark_done(name: str) -> None:
    """Create a download-complete marker.

    Args:
        name: Dataset name.
    """
    with open(_marker(name), "w") as f:
        f.write("done\n")


def download_coco_annotations() -> None:
    """Download COCO 2017 train/val annotations (~240MB zip)."""
    name = "coco_annotations"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "annotations_trainval2017.zip")

    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if not _run(f"wget -q -O {zip_path} '{url}'", name):
        return

    logger.info(f"[{name}] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Only extract instances_train2017.json
        for member in zf.namelist():
            if "instances_train2017" in member:
                zf.extract(member, out_dir)
                logger.info(f"[{name}] Extracted {member}")

    os.remove(zip_path)
    _mark_done(name)
    logger.info(f"[{name}] Done.")


def download_pixmo_count() -> None:
    """Download PixMo-Count from HuggingFace."""
    name = "pixmo_count"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[{name}] Loading from HuggingFace allenai/pixmo-count...")
    from datasets import load_dataset

    ds = load_dataset("allenai/pixmo-count", split="train")
    save_path = os.path.join(out_dir, "dataset")
    ds.save_to_disk(save_path)
    logger.info(f"[{name}] Saved {len(ds)} samples to {save_path}")

    _mark_done(name)
    logger.info(f"[{name}] Done.")


def download_hpatches() -> None:
    """Download HPatches sequences (~1.6GB)."""
    name = "hpatches"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    tar_path = os.path.join(out_dir, "hpatches-sequences-release.tar.gz")

    url = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"
    if not _run(f"wget -q -O {tar_path} '{url}'", name):
        return

    logger.info(f"[{name}] Extracting...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)

    os.remove(tar_path)
    _mark_done(name)
    logger.info(f"[{name}] Done.")


def download_artbench() -> None:
    """Download ArtBench-10 from HuggingFace."""
    name = "artbench"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[{name}] Loading from HuggingFace Doub7e/ArtBench-10...")
    from datasets import load_dataset

    ds = load_dataset("Doub7e/ArtBench-10", split="train")
    save_path = os.path.join(out_dir, "dataset")
    ds.save_to_disk(save_path)
    logger.info(f"[{name}] Saved {len(ds)} samples to {save_path}")

    _mark_done(name)
    logger.info(f"[{name}] Done.")


def download_spair71k() -> None:
    """Download SPair-71k dataset."""
    name = "spair71k"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    tar_path = os.path.join(out_dir, "SPair-71k.tar.gz")

    url = "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
    if not _run(f"wget -q -O {tar_path} '{url}'", name):
        # Fallback: try HTTPS
        url2 = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
        if not _run(f"wget -q --no-check-certificate -O {tar_path} '{url2}'", name):
            logger.error(f"[{name}] Download failed from both HTTP and HTTPS.")
            return

    logger.info(f"[{name}] Extracting...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)

    os.remove(tar_path)
    _mark_done(name)
    logger.info(f"[{name}] Done.")


def download_funkpoint() -> None:
    """Download FunKPoint dataset from GitHub."""
    name = "funkpoint"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # Clone the repo (contains annotations + image download scripts)
    repo_url = "https://github.com/ZJULearning/FunKPoint.git"
    repo_dir = os.path.join(out_dir, "FunKPoint")
    if not os.path.exists(repo_dir):
        if not _run(f"git clone --depth 1 {repo_url} {repo_dir}", name):
            logger.warning(f"[{name}] Git clone failed. Will use fallback at build time.")
            return

    _mark_done(name)
    logger.info(f"[{name}] Done. Note: images may need separate download via repo scripts.")


def download_mid() -> None:
    """Download Multi-Illumination Dataset (albedo subset).

    Downloads a subset of scenes for relative reflectance task.
    """
    name = "mid"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # MID is hosted at MIT CSAIL. Download albedo images for ~50 scenes.
    # The dataset provides per-scene directories with albedo/diffuse/specular maps.
    base_url = "https://projects.csail.mit.edu/illumination/data"

    # First try to get the scene list
    scene_list_path = os.path.join(out_dir, "scene_list.txt")
    if not _run(
        f"wget -q -O {scene_list_path} '{base_url}/scene_list.txt'",
        f"{name}/scene_list",
    ):
        # Generate a manual list of known scenes
        logger.warning(f"[{name}] Could not fetch scene list. Creating placeholder.")
        with open(scene_list_path, "w") as f:
            f.write("# Scene list could not be fetched. Download manually.\n")
            f.write("# See: https://projects.csail.mit.edu/illumination/\n")

    _mark_done(name)
    logger.info(f"[{name}] Done (may need manual scene downloads).")


def download_nyuv2() -> None:
    """Download NYUv2 depth dataset from HuggingFace."""
    name = "nyuv2"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[{name}] Loading from HuggingFace sayakpaul/nyu_depth_v2...")
    from datasets import load_dataset

    ds = load_dataset("sayakpaul/nyu_depth_v2", split="train", trust_remote_code=True)
    save_path = os.path.join(out_dir, "dataset")
    ds.save_to_disk(save_path)
    logger.info(f"[{name}] Saved {len(ds)} samples to {save_path}")

    _mark_done(name)
    logger.info(f"[{name}] Done.")


def download_nights() -> None:
    """Download DreamSim NIGHTS triplet dataset."""
    name = "nights"
    if _is_done(name):
        logger.info(f"[{name}] Already downloaded, skipping.")
        return

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[{name}] Loading NIGHTS from HuggingFace...")
    from datasets import load_dataset

    try:
        ds = load_dataset("night-bench/nights", split="train")
        save_path = os.path.join(out_dir, "dataset")
        ds.save_to_disk(save_path)
        logger.info(f"[{name}] Saved {len(ds)} samples to {save_path}")
    except Exception as e:
        logger.warning(f"[{name}] HF download failed: {e}")
        logger.info(f"[{name}] Trying DreamSim GitHub release...")
        # Fallback: download from DreamSim repo releases
        url = "https://github.com/ssundaram21/dreamsim/releases/download/v0.1.0/nights.zip"
        zip_path = os.path.join(out_dir, "nights.zip")
        if _run(f"wget -q -O {zip_path} '{url}'", name):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            os.remove(zip_path)
        else:
            logger.error(f"[{name}] All download methods failed.")
            return

    _mark_done(name)
    logger.info(f"[{name}] Done.")


ALL_DATASETS = {
    "coco_annotations": download_coco_annotations,
    "pixmo_count": download_pixmo_count,
    "hpatches": download_hpatches,
    "artbench": download_artbench,
    "spair71k": download_spair71k,
    "funkpoint": download_funkpoint,
    "mid": download_mid,
    "nyuv2": download_nyuv2,
    "nights": download_nights,
}


def main() -> None:
    """Download all (or selected) LIVR source datasets."""
    parser = argparse.ArgumentParser(
        description="Download source datasets for LIVR MCQ construction."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(ALL_DATASETS.keys()),
        help="Download only these datasets.",
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_DIR,
        help="Base directory for downloads.",
    )
    args = parser.parse_args()

    _set_base_dir(args.base_dir)

    targets = args.only if args.only else list(ALL_DATASETS.keys())
    logger.info(f"Downloading {len(targets)} datasets to {BASE_DIR}")

    for name in targets:
        logger.info(f"--- {name} ---")
        try:
            ALL_DATASETS[name]()
        except Exception as e:
            logger.error(f"[{name}] Unexpected error: {e}", exc_info=True)

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
