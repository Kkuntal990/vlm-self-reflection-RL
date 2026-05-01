#!/usr/bin/env python3
"""livr-v2 Relative Reflectance builder.

Per Appendix A:
  * Source: Multi-Illumination Dataset (MID), training split
  * For each RGB-albedo pair, sample 2 spatially separated points
  * Compute rel = |Y_A - Y_B| / max(Y_A, Y_B, 1e-8) where Y is luminance
    of the *albedo* map (NOT the RGB — this is the key correctness fix
    vs the v1 luminance-of-RGB proxy)
  * If rel <= 0.10: label "(C) About the same"
  * Else: "(A) A is darker" or "(B) B is darker" depending on which Y is lower
  * 3-way MCQ
  * Splits: 1000 / 250 / 134 (BLINK)

MID layout (after extraction): scenes are top-level dirs. Each scene
contains 25 images per direction + albedo / shading decompositions.
We use the albedo maps as the reflectance ground truth.

Usage:
    python scripts/data/livr_v2/build_relative_reflectance.py \\
        --mid-dir /outputs/livr_v2_sources/mid \\
        --output-dir /outputs/livr_v2/image_base/relative_reflectance \\
        --output-jsonl-prefix /outputs/livr_v2/data/relative_reflectance
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (  # noqa: E402
    draw_keypoint,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_relative_reflectance"
N_TRAIN = 1000
N_VAL = 250
N_TEST = 134

REL_THRESHOLD = 0.10
MIN_POINT_DIST = 80  # px between the two sampled points
MARGIN = 30


def _srgb_to_linear(srgb_0_255: np.ndarray) -> np.ndarray:
    """sRGB (0-255) -> linear RGB (0-1).

    Per Appendix A of the LIVR paper: "converting the albedo to linear
    RGB, and computing the local disk-averaged luminance at each point."
    JPEG-decoded pixel values are gamma-encoded sRGB; we MUST decode the
    transfer function before computing luminance, or labels near the
    0.10 threshold flip.
    """
    s = srgb_0_255.astype(np.float32) / 255.0
    linear = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def _luminance_linear(linear_rgb: np.ndarray) -> float:
    """Rec. 709 luminance from LINEAR RGB (0-1)."""
    return float(
        0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
    )


def _disk_mean_linear_luminance(albedo: np.ndarray, cx: int, cy: int, radius: int) -> float:
    """Disk-averaged luminance at (cx, cy) on the albedo image.

    Appendix A says "the local DISK-averaged luminance at each point".
    Steps:
      1. Crop a square (cx,cy ± radius) bounding box.
      2. sRGB -> linear (per-channel).
      3. Mask to the inscribed disk.
      4. Take the per-channel mean over disk pixels.
      5. Apply Rec. 709 weights.
    """
    H, W = albedo.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(W, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(H, cy + radius + 1)
    patch = albedo[y0:y1, x0:x1]  # gamma-encoded sRGB uint-like floats
    linear = _srgb_to_linear(patch)
    yy, xx = np.ogrid[: linear.shape[0], : linear.shape[1]]
    cyy = cy - y0
    cxx = cx - x0
    mask = (xx - cxx) ** 2 + (yy - cyy) ** 2 <= radius**2
    if not mask.any():
        return _luminance_linear(linear.mean(axis=(0, 1)))
    disk_mean = linear[mask].mean(axis=0)
    return _luminance_linear(disk_mean)


def _find_scenes(root: Path) -> list[Path]:
    """Return scene directories under MID root."""
    if not root.exists():
        return []
    scenes = [p for p in root.iterdir() if p.is_dir()]
    # Some MID releases nest one more level (e.g. multi_illumination_train_mip2_jpg/).
    if len(scenes) == 1 and scenes[0].is_dir():
        nested = [p for p in scenes[0].iterdir() if p.is_dir()]
        if len(nested) > 5:
            scenes = nested
    return sorted(scenes)


def _find_materials_and_rgb(scene_dir: Path) -> tuple[Path, Path] | None:
    """Find materials_mip2.png and one RGB illumination frame.

    DEVIATION FROM PAPER (Appendix A "RGB-albedo pair"):
    The MID training zip we have access to (multi_illumination_train_mip2_jpg.zip)
    does NOT ship pre-computed albedo decompositions; it ships scene RGB
    at 25 illuminations + chrome/gray probe balls + a materials
    segmentation map (`materials_mip2.png`). The decomposition zip is
    not exposed as a public download. We use materials_mip2.png as the
    ground-truth signal for "same reflectance" — pixels with the same
    material ID are by construction the same intrinsic material and
    therefore the same reflectance, which is *more* accurate than
    luminance-thresholding on an estimated albedo map.

    Returns (materials_mip2.png, dir_<N>_mip2.jpg) for one fixed
    illumination, or None if either is missing.
    """
    materials = scene_dir / "materials_mip2.png"
    if not materials.exists():
        return None
    # Pick a deterministic illumination frame for display + luminance
    # fallback. dir_0 is typically present; fall back to first dir_*.
    rgb = scene_dir / "dir_0_mip2.jpg"
    if not rgb.exists():
        candidates = sorted(scene_dir.glob("dir_*_mip2.jpg"))
        if not candidates:
            return None
        rgb = candidates[0]
    return materials, rgb


def _disk_majority_material(materials: np.ndarray, cx: int, cy: int, radius: int) -> int:
    """Return the most-common material ID in the disk around (cx, cy).

    Material IDs are encoded as the pixel value of materials_mip2.png
    (8-bit grayscale or RGB). We collapse to a single int per pixel by
    taking the first channel (MID's materials map is single-channel
    semantic; PIL may broadcast to 3 channels on .convert('RGB')).
    """
    H, W = materials.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(W, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(H, cy + radius + 1)
    patch = materials[y0:y1, x0:x1]
    if patch.ndim == 3:
        patch = patch[..., 0]
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    cyy = cy - y0
    cxx = cx - x0
    mask = (xx - cxx) ** 2 + (yy - cyy) ** 2 <= radius**2
    if not mask.any():
        return int(patch.flatten()[0])
    vals = patch[mask].astype(np.int32)
    bincount = np.bincount(vals)
    return int(np.argmax(bincount))


def _disk_mean_linear_luminance_rgb(rgb_arr: np.ndarray, cx: int, cy: int, radius: int) -> float:
    """Same disk-mean linear luminance as the albedo helper, but
    operating on the scene RGB image. Used only for A/B comparison
    when the two points are on DIFFERENT materials — at that point
    luminance is a reasonable cheap signal for "which is darker"
    even though it includes shading."""
    H, W = rgb_arr.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(W, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(H, cy + radius + 1)
    patch = rgb_arr[y0:y1, x0:x1]
    linear = _srgb_to_linear(patch)
    yy, xx = np.ogrid[: linear.shape[0], : linear.shape[1]]
    cyy = cy - y0
    cxx = cx - x0
    mask = (xx - cxx) ** 2 + (yy - cyy) ** 2 <= radius**2
    if not mask.any():
        return _luminance_linear(linear.mean(axis=(0, 1)))
    disk_mean = linear[mask].mean(axis=0)
    return _luminance_linear(disk_mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid-dir", default="/outputs/livr_v2_sources/mid")
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/relative_reflectance")
    parser.add_argument(
        "--output-jsonl-prefix", default="/outputs/livr_v2/data/relative_reflectance"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="Half-size of the luminance averaging patch around each sampled point.",
    )
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = _find_scenes(Path(args.mid_dir))
    logger.info("Found %d scene directories under MID root", len(scenes))
    rng.shuffle(scenes)
    needed = N_TRAIN + N_VAL + N_TEST + 200

    # Per-class quotas (Appendix A: "'About the same' class constitutes
    # roughly one quarter of the data"). 25% C, ~37.5% A, ~37.5% B.
    quota_c = int(round(needed * 0.25))
    quota_a = int(round((needed - quota_c) * 0.5))
    quota_b = needed - quota_c - quota_a
    quotas = {"A": quota_a, "B": quota_b, "C": quota_c}
    counts = {"A": 0, "B": 0, "C": 0}
    logger.info("Per-class quotas: %s (target ~25%% for C 'About the same')", quotas)

    records: list[dict] = []
    for scene_idx, scene in enumerate(scenes):
        if len(records) >= needed:
            break
        mr = _find_materials_and_rgb(scene)
        if mr is None:
            continue
        materials_path, rgb_path = mr
        try:
            materials = np.asarray(Image.open(materials_path), dtype=np.int32)
            rgb_pil = Image.open(rgb_path).convert("RGB")
            rgb_arr = np.asarray(rgb_pil, dtype=np.float32)
        except Exception:
            continue
        # Materials map and RGB share the same _mip2 scale (~1500x1000),
        # but be defensive: resize the materials map to the RGB grid if
        # they ever diverge so coordinate sampling stays consistent.
        if materials.shape[:2] != rgb_arr.shape[:2]:
            materials = np.asarray(
                Image.open(materials_path).resize(
                    (rgb_arr.shape[1], rgb_arr.shape[0]), Image.NEAREST
                ),
                dtype=np.int32,
            )
        H, W = rgb_arr.shape[:2]
        if H < 2 * MARGIN + 2 * args.patch_size or W < 2 * MARGIN + 2 * args.patch_size:
            continue

        # Per scene, try several point-pair attempts; each is accepted
        # only if the resulting class is still under quota. This balances
        # toward the target ≈25% / ≈37.5% / ≈37.5% mix.
        attempts_per_scene = 8
        for _ in range(attempts_per_scene):
            if len(records) >= needed:
                break
            ax = rng.randint(MARGIN + args.patch_size, W - MARGIN - args.patch_size)
            ay = rng.randint(MARGIN + args.patch_size, H - MARGIN - args.patch_size)
            for _ in range(40):
                bx = rng.randint(MARGIN + args.patch_size, W - MARGIN - args.patch_size)
                by = rng.randint(MARGIN + args.patch_size, H - MARGIN - args.patch_size)
                if ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 >= MIN_POINT_DIST:
                    break
            else:
                continue

            # Material-segmentation ground truth (Option B):
            # - Same material ID at both points  -> reflectance is identical
            #   by construction -> class C "About the same".
            # - Different materials              -> compare disk-mean linear
            #   luminance on the scene RGB to decide A-darker vs B-darker.
            mat_a = _disk_majority_material(materials, ax, ay, args.patch_size)
            mat_b = _disk_majority_material(materials, bx, by, args.patch_size)
            if mat_a == mat_b:
                gt = "C"
            else:
                ya = _disk_mean_linear_luminance_rgb(rgb_arr, ax, ay, args.patch_size)
                yb = _disk_mean_linear_luminance_rgb(rgb_arr, bx, by, args.patch_size)
                # Skip near-tie luminance pairs on different materials —
                # they're noisy labels for "which is darker" and add no
                # signal beyond the C class.
                mx = max(ya, yb, 1e-8)
                rel = abs(ya - yb) / mx
                if rel < REL_THRESHOLD:
                    continue
                gt = "A" if ya < yb else "B"
            # Reject if this class is already over quota.
            if counts[gt] >= quotas[gt]:
                continue
            counts[gt] += 1

            disp = draw_keypoint(rgb_pil.copy(), ax, ay, color=(255, 0, 0), radius=10, label="A")
            disp = draw_keypoint(disp, bx, by, color=(0, 0, 255), radius=10, label="B")
            out_filename = f"reflectance_{len(records):05d}.png"
            out_path = out_dir / out_filename
            save_image(disp, str(out_path), format="PNG")

            question = (
                "Two points are marked in the image: Point A (red) and Point B (blue). "
                "Which statement about their relative surface reflectance is correct? "
                "(A) A is darker than B (B) B is darker than A (C) Same reflectance"
            )
            formatted_choices = [
                "(A) A is darker than B",
                "(B) B is darker than A",
                "(C) About the same",
            ]
            rec = make_livr_record(
                question=question,
                ground_truth=gt,
                choices=formatted_choices,
                image_path=str(out_path),
                dataset_name=TASK_NAME,
            )
            records.append(rec)
            if len(records) >= needed:
                break
        if (scene_idx + 1) % 50 == 0:
            logger.info("  scenes processed=%d  records=%d/%d", scene_idx + 1, len(records), needed)
    logger.info("Built %d relative_reflectance records (class counts: %s)", len(records), counts)

    train = records[:N_TRAIN]
    val = records[N_TRAIN : N_TRAIN + N_VAL]
    test = records[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"
    write_jsonl(train, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info("Relative_Reflectance done: %d/%d/%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
