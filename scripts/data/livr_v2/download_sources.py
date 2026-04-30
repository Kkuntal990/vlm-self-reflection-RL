#!/usr/bin/env python3
"""Download all source datasets for the LIVR-v2 build.

Mirrors the construction in Appendix A of "Latent Implicit Visual
Reasoning" (arXiv:2512.21218). Replaces the original LIVR build, which
substituted synthetic homographies on COCO and a luminance proxy for
reflectance — both are removed in v2.

Sources downloaded:
  * HPatches (Viewpoint + Illumination sequences)        — visual_correspondence
  * SPair-71k (training split)                           — semantic_correspondence
  * Multi-Illumination Dataset (MID, training split)     — relative_reflectance
  * ArtBench-10 (256x256 ImageFolder variant)            — art_style
  * DreamSim NIGHTS                                      — visual_similarity
  * BLINK val sets (for cross-dataset deduplication)

Already-on-PVC (symlinked, not re-downloaded):
  * PixMo-Count (allenai/pixmo-count) — counting
  * COCO 2017 (train2017 + val2017)   — jigsaw, object_localization

Usage (one-shot from the helper pod or a dedicated download Job):
    python scripts/data/livr_v2/download_sources.py \\
        --root /outputs/livr_v2_sources \\
        --pixmo-existing /outputs/livr_sources/pixmo_count/dataset \\
        --coco-existing /outputs/image_base/coco

Each download step is idempotent: if the destination dir already
contains a marker file (`.download_complete`), the step is skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Marker file written when a download/extract step completes successfully.
_DONE_MARKER = ".download_complete"


def _is_done(path: Path) -> bool:
    return (path / _DONE_MARKER).exists()


def _mark_done(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / _DONE_MARKER).touch()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _wget(url: str, out: Path) -> None:
    """Download a single URL to `out` using wget with resume support."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        logger.info("Already present, skipping: %s (%.1f MB)", out, out.stat().st_size / 1e6)
        return
    _run(["wget", "-c", "-O", str(out), url])


def _extract(archive: Path, dest: Path) -> None:
    """Extract a .tar.gz / .tar / .zip archive into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    logger.info("Extracting %s -> %s", archive, dest)
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
    elif name.endswith(".tar"):
        with tarfile.open(archive, "r:") as tf:
            tf.extractall(dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive format: {archive}")


# =============================================================================
# Per-source downloaders
# =============================================================================


def download_hpatches(root: Path) -> None:
    """HPatches release tarball (Viewpoint + Illumination sequences).

    Source: https://github.com/hpatches/hpatches-dataset

    The release tarball `hpatches-sequences-release.tar.gz` (~580 MB)
    contains 116 sequences (59 viewpoint + 57 illumination). Each
    sequence has 6 images and 5 ground-truth homographies (1->2..1->6).
    """
    dest = root / "hpatches"
    if _is_done(dest):
        logger.info("HPatches already complete, skipping")
        return
    archive_url = (
        "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/"
        "hpatches-sequences-release.tar.gz"
    )
    archive = dest / "hpatches-sequences-release.tar.gz"
    _wget(archive_url, archive)
    _extract(archive, dest)
    archive.unlink(missing_ok=True)
    _mark_done(dest)
    logger.info("HPatches done")


def download_spair71k(root: Path) -> None:
    """SPair-71k training split.

    Source: https://cvlab.postech.ac.kr/research/SPair-71k/

    Provides 70,958 image pairs with annotated semantic keypoints
    across 18 object categories.
    """
    dest = root / "spair_71k"
    if _is_done(dest):
        logger.info("SPair-71k already complete, skipping")
        return
    archive_url = "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"
    archive = dest / "SPair-71k.tar.gz"
    _wget(archive_url, archive)
    _extract(archive, dest)
    archive.unlink(missing_ok=True)
    _mark_done(dest)
    logger.info("SPair-71k done")


def download_mid(root: Path) -> None:
    """Multi-Illumination Dataset (MID) training split — albedo maps.

    Source: https://projects.csail.mit.edu/illumination/

    LIVR uses albedo maps (decomposition output) from this dataset, NOT
    the raw RGB. The training split is large (~25 GB raw); we want only
    albedo maps. The official release packages albedo separately.
    """
    dest = root / "mid"
    if _is_done(dest):
        logger.info("MID already complete, skipping")
        return
    # MID release URLs (from the project's data page).
    # The "diffuse_25" subset includes the albedo + luminance maps.
    archive_url = (
        "https://data.csail.mit.edu/multilum/"
        "multi_illumination_train_mip2_jpg.zip"
    )
    archive = dest / "multi_illumination_train_mip2_jpg.zip"
    _wget(archive_url, archive)
    _extract(archive, dest)
    archive.unlink(missing_ok=True)
    _mark_done(dest)
    logger.info("MID done")


def download_artbench10(root: Path) -> None:
    """ArtBench-10 (256x256 ImageFolder variant).

    Source: https://github.com/liaopeiyuan/artbench

    10 art styles x ~6000 images each, 256x256, ImageFolder layout.
    """
    dest = root / "artbench10"
    if _is_done(dest):
        logger.info("ArtBench-10 already complete, skipping")
        return
    archive_url = (
        "https://artbench.eecs.berkeley.edu/files/"
        "artbench-10-imagefolder-split.tar"
    )
    archive = dest / "artbench-10-imagefolder-split.tar"
    _wget(archive_url, archive)
    _extract(archive, dest)
    archive.unlink(missing_ok=True)
    _mark_done(dest)
    logger.info("ArtBench-10 done")


def download_nights(root: Path) -> None:
    """DreamSim NIGHTS triplet dataset.

    Source: https://dreamsim-nights.github.io/

    20K human-annotated triplets (reference + 2 distortions + vote)
    for perceptual similarity.
    """
    dest = root / "nights"
    if _is_done(dest):
        logger.info("NIGHTS already complete, skipping")
        return
    # NIGHTS is hosted on HuggingFace via the DreamSim project.
    archive_url = "https://data.csail.mit.edu/nights/nights.zip"
    archive = dest / "nights.zip"
    _wget(archive_url, archive)
    _extract(archive, dest)
    archive.unlink(missing_ok=True)
    _mark_done(dest)
    logger.info("NIGHTS done")


def download_blink_val(root: Path) -> None:
    """BLINK validation set TSVs (for cross-dataset deduplication).

    Source: https://github.com/zeyofu/BLINK_Benchmark

    We need the val TSVs for: Counting, Object_Localization, Art_Style,
    Visual_Similarity, Semantic_Correspondence (the tasks for which our
    paper does explicit cross-dataset dedup against BLINK).
    """
    dest = root / "blink_val"
    if _is_done(dest):
        logger.info("BLINK val already complete, skipping")
        return
    # Use the HuggingFace dataset mirror (BLINK/BLINK).
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "huggingface-cli",
            "download",
            "BLINK-Benchmark/BLINK",
            "--repo-type",
            "dataset",
            "--local-dir",
            str(dest),
        ]
    )
    _mark_done(dest)
    logger.info("BLINK val done")


def link_pixmo(root: Path, existing: Path) -> None:
    """Symlink the existing PixMo-Count snapshot into livr_v2_sources/."""
    dest = root / "pixmo_count"
    if dest.exists() or dest.is_symlink():
        logger.info("PixMo-Count link already present at %s", dest)
        return
    if not existing.exists():
        raise FileNotFoundError(
            f"PixMo-Count snapshot not found at {existing}. "
            f"Either download it via `datasets.load_dataset('allenai/pixmo-count')` "
            f"and `save_to_disk(...)`, or pass the actual path via --pixmo-existing."
        )
    os.symlink(existing, dest)
    _mark_done(dest)
    logger.info("PixMo-Count linked: %s -> %s", dest, existing)


def link_coco(root: Path, existing: Path) -> None:
    """Symlink the existing COCO 2017 directory into livr_v2_sources/."""
    dest = root / "coco"
    if dest.exists() or dest.is_symlink():
        logger.info("COCO link already present at %s", dest)
        return
    if not existing.exists():
        raise FileNotFoundError(
            f"COCO 2017 directory not found at {existing}. Pass the correct "
            f"path via --coco-existing (must contain train2017/, val2017/, "
            f"and annotations/)."
        )
    os.symlink(existing, dest)
    _mark_done(dest)
    logger.info("COCO linked: %s -> %s", dest, existing)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/outputs/livr_v2_sources",
        help="Root directory for all downloads.",
    )
    parser.add_argument(
        "--pixmo-existing",
        default="/outputs/livr_sources/pixmo_count/dataset",
        help="Path to existing PixMo-Count HF dataset on PVC (will be symlinked).",
    )
    parser.add_argument(
        "--coco-existing",
        default="/outputs/image_base/coco",
        help="Path to existing COCO 2017 directory on PVC (will be symlinked).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=[
            "hpatches",
            "spair_71k",
            "mid",
            "artbench10",
            "nights",
            "blink_val",
            "pixmo",
            "coco",
        ],
        help="Sources to skip.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    logger.info("LIVR-v2 source download root: %s", root)

    # Public downloads (idempotent)
    if "hpatches" not in args.skip:
        download_hpatches(root)
    if "spair_71k" not in args.skip:
        download_spair71k(root)
    if "mid" not in args.skip:
        download_mid(root)
    if "artbench10" not in args.skip:
        download_artbench10(root)
    if "nights" not in args.skip:
        download_nights(root)
    if "blink_val" not in args.skip:
        download_blink_val(root)

    # Already-on-PVC symlinks
    if "pixmo" not in args.skip:
        link_pixmo(root, Path(args.pixmo_existing))
    if "coco" not in args.skip:
        link_coco(root, Path(args.coco_existing))

    logger.info("=== All sources ready under %s ===", root)
    # Final inventory
    for entry in sorted(root.iterdir()):
        if entry.is_dir() or entry.is_symlink():
            try:
                size_mb = sum(
                    f.stat().st_size for f in entry.rglob("*") if f.is_file()
                ) / 1e6
                logger.info("  %s: %.0f MB", entry.name, size_mb)
            except OSError:
                logger.info("  %s: (symlinked)", entry.name)


if __name__ == "__main__":
    main()
