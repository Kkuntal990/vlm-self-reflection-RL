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


def _luminance(rgb: np.ndarray) -> float:
    """Rec. 709 luminance from a single RGB pixel (or mean of patch)."""
    return float(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


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


def _find_albedo_and_rgb(scene_dir: Path) -> tuple[Path, Path] | None:
    """Find the albedo map and an RGB image to display in this scene.

    MID provides per-scene 'probes' and material decomposition outputs.
    The naming convention varies; we look for files containing 'albedo'
    (or 'reflectance') and pair with the first available RGB frame.
    """
    files = list(scene_dir.iterdir())
    albedo = None
    rgb = None
    for f in files:
        n = f.name.lower()
        if (
            albedo is None
            and ("albedo" in n or "reflectance" in n)
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".exr"}
        ):
            albedo = f
        elif (
            rgb is None
            and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and "albedo" not in n
            and "shading" not in n
            and "normal" not in n
            and "depth" not in n
        ):
            rgb = f
    if albedo and rgb:
        return albedo, rgb
    return None


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
        ar = _find_albedo_and_rgb(scene)
        if ar is None:
            continue
        albedo_path, rgb_path = ar
        try:
            albedo = np.asarray(Image.open(albedo_path).convert("RGB"), dtype=np.float32)
            rgb = Image.open(rgb_path).convert("RGB")
        except Exception:
            continue
        H, W = albedo.shape[:2]
        if H < 2 * MARGIN + 2 * args.patch_size or W < 2 * MARGIN + 2 * args.patch_size:
            continue

        # Try a generous number of point-pair attempts per scene; for
        # each, accept only if the resulting class is still under quota.
        # This balances toward the target ≈25% / ≈37.5% / ≈37.5% mix.
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
            patch_a = albedo[
                ay - args.patch_size : ay + args.patch_size,
                ax - args.patch_size : ax + args.patch_size,
            ].mean(axis=(0, 1))
            patch_b = albedo[
                by - args.patch_size : by + args.patch_size,
                bx - args.patch_size : bx + args.patch_size,
            ].mean(axis=(0, 1))
            ya = _luminance(patch_a)
            yb = _luminance(patch_b)
            mx = max(ya, yb, 1e-8)
            rel = abs(ya - yb) / mx
            if rel <= REL_THRESHOLD:
                gt = "C"  # About the same
            elif ya < yb:
                gt = "A"  # A is darker
            else:
                gt = "B"  # B is darker
            # Reject if this class is already over quota.
            if counts[gt] >= quotas[gt]:
                continue
            counts[gt] += 1

            # Annotate the displayed RGB (resize albedo points to RGB coords if dims differ).
            disp = rgb.copy()
            if disp.size != (W, H):
                rax = int(ax * disp.size[0] / W)
                ray = int(ay * disp.size[1] / H)
                rbx = int(bx * disp.size[0] / W)
                rby = int(by * disp.size[1] / H)
            else:
                rax, ray, rbx, rby = ax, ay, bx, by
            disp = draw_keypoint(disp, rax, ray, color=(255, 0, 0), radius=10, label="A")
            disp = draw_keypoint(disp, rbx, rby, color=(0, 0, 255), radius=10, label="B")

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
