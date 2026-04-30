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
    """Download a single URL to `out` using Python requests with resume support.

    Avoids the wget dependency so we can run on minimal images
    (e.g. huggingface/trl:0.29.0) without needing apt-get.
    """
    import requests

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        # Try a HEAD to compare size; if equal, we're done.
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            content_len = int(head.headers.get("content-length", 0))
            if content_len > 0 and content_len == out.stat().st_size:
                logger.info("Already complete: %s (%.1f MB)", out, out.stat().st_size / 1e6)
                return
        except Exception:
            pass
        logger.info("Resuming partial: %s (%.1f MB)", out, out.stat().st_size / 1e6)

    headers = {}
    mode = "wb"
    if out.exists() and out.stat().st_size > 0:
        headers["Range"] = f"bytes={out.stat().st_size}-"
        mode = "ab"

    logger.info("Downloading: %s -> %s", url, out)
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + (
            out.stat().st_size if mode == "ab" else 0
        )
        downloaded = out.stat().st_size if mode == "ab" else 0
        last_log_mb = 0
        with open(out, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    mb = downloaded // (1 << 20)
                    if mb >= last_log_mb + 100:
                        logger.info("  ... %.0f / %.0f MB", mb, total / (1 << 20) if total else mb)
                        last_log_mb = mb
    logger.info("Downloaded: %s (%.1f MB)", out, out.stat().st_size / 1e6)


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
    """HPatches release zip (Viewpoint + Illumination sequences).

    Primary source (Imperial College) at icvl.ee.ic.ac.uk fails DNS
    on many cluster nodes; we use the official HuggingFace mirror by
    V. Balntas (the dataset author) instead. Same 116 sequences (59
    viewpoint + 57 illumination), each with 6 images + 5 ground-truth
    homographies (1->2..1->6).

    HF mirror: https://huggingface.co/datasets/vbalnt/hpatches
    """
    from huggingface_hub import hf_hub_download

    dest = root / "hpatches"
    if _is_done(dest):
        logger.info("HPatches already complete, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HPatches via HF Hub (vbalnt/hpatches)")
    archive_path = hf_hub_download(
        repo_id="vbalnt/hpatches",
        filename="hpatches-sequences-release.zip",
        repo_type="dataset",
        local_dir=str(dest),
    )
    _extract(Path(archive_path), dest)
    Path(archive_path).unlink(missing_ok=True)
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
    archive_url = "https://data.csail.mit.edu/multilum/multi_illumination_train_mip2_jpg.zip"
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
    archive_url = "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar"
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

    BLINK is distributed as parquet files on HuggingFace
    (BLINK-Benchmark/BLINK). After download, we extract embedded image
    bytes into per-task `images/` subdirectories so dedup helpers can
    point at them as files.
    """
    dest = root / "blink_val"
    if _is_done(dest):
        logger.info("BLINK val already complete, skipping")
        return
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
    _extract_blink_images(dest)
    _mark_done(dest)
    logger.info("BLINK val done")


def _extract_blink_images(blink_root: Path) -> None:
    """Walk BLINK task subdirs, read each `*-of-*.parquet`, and write the
    embedded image columns out as PNG files under `<task>/images/`.

    BLINK rows typically carry images as either `image_1`, `image_2`, ...
    columns or as a single `image` column with PIL-encoded bytes.
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image as PILImage

    for task_dir in sorted(blink_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name in {"assets"}:
            continue
        out_dir = task_dir / "images"
        out_dir.mkdir(exist_ok=True)
        parquets = list(task_dir.glob("*.parquet"))
        if not parquets:
            continue
        n_extracted = 0
        for pq_path in parquets:
            try:
                table = pq.read_table(pq_path)
            except Exception as e:
                logger.warning("Failed to read %s: %s", pq_path, e)
                continue
            cols = table.column_names
            img_cols = [c for c in cols if c == "image" or c.startswith("image_")]
            if not img_cols:
                continue
            for col in img_cols:
                arr = table.column(col).to_pylist()
                for i, cell in enumerate(arr):
                    if cell is None:
                        continue
                    raw = cell.get("bytes") if isinstance(cell, dict) else cell
                    if raw is None:
                        continue
                    try:
                        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        continue
                    out_path = out_dir / f"{pq_path.stem}_{col}_{i:05d}.png"
                    img.save(out_path)
                    n_extracted += 1
        if n_extracted:
            logger.info("BLINK %s: extracted %d images", task_dir.name, n_extracted)


def download_coco_annotations(root: Path) -> None:
    """COCO 2017 detection annotations (instances_train2017.json + val).

    The PVC's COCO directory typically only has the image splits
    (train2017/, val2017/) without the annotation JSONs. Object
    localization needs the annotations for bbox + segmentation masks,
    so we download them separately into the symlinked coco/ directory.

    Annotation tar: ~241 MB.
    """
    coco_dir = root / "coco"
    if not coco_dir.exists():
        logger.warning("COCO symlink missing at %s; skipping annotations", coco_dir)
        return
    target = coco_dir / "annotations" / "instances_train2017.json"
    if target.exists():
        logger.info("COCO annotations already present, skipping")
        return
    archive_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    archive = root / "coco_annotations" / "annotations_trainval2017.zip"
    _wget(archive_url, archive)
    # The zip extracts to `annotations/instances_*.json`.
    # If coco_dir is a symlink, we extract into the symlink target.
    extract_into = coco_dir.resolve()
    _extract(archive, extract_into)
    archive.unlink(missing_ok=True)
    archive.parent.rmdir()
    logger.info("COCO annotations extracted into %s", extract_into)


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
        # COCO annotations (instances_train/val 2017) are NOT in the PVC's
        # image-only symlink target — fetch them separately and drop into
        # the linked coco/ directory.
        download_coco_annotations(root)

    logger.info("=== All sources ready under %s ===", root)
    # Final inventory
    for entry in sorted(root.iterdir()):
        if entry.is_dir() or entry.is_symlink():
            try:
                size_mb = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file()) / 1e6
                logger.info("  %s: %.0f MB", entry.name, size_mb)
            except OSError:
                logger.info("  %s: (symlinked)", entry.name)


if __name__ == "__main__":
    main()
