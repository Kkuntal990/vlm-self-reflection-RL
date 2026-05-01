#!/usr/bin/env python3
"""livr-v2 Visual Correspondence builder.

Per Appendix A:
  * Source: HPatches (Viewpoint + Illumination sequences)
  * Forward image pairs (i, j) with 1 <= i < j <= 6 and j-i >= 2
  * Use the provided ground-truth homographies (H_1_2 ... H_1_6) to
    map points from image i to image j
  * 4-way MCQ + REF marker on source
  * 3 distractor keypoints sampled to be far from the true location
  * Splits: 1000 train / 500 val / 700 test (HPatches-based test, not BLINK)

This replaces the v1 synthetic-warp-on-COCO build entirely.

Usage:
    python scripts/data/livr_v2/build_visual_correspondence.py \\
        --hpatches-dir /outputs/livr_v2_sources/hpatches \\
        --output-dir /outputs/livr_v2/image_base/visual_correspondence \\
        --output-jsonl-prefix /outputs/livr_v2/data/visual_correspondence
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
    OPTION_COLORS,
    OPTION_LETTERS,
    create_side_by_side,
    draw_keypoint,
    draw_ref_marker,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices_with_index,
    write_jsonl,
)

TASK_NAME = "livr_visual_correspondence"
N_TRAIN = 1000
N_VAL = 500
N_TEST = 700

# Distance threshold (in target-image pixels) for distractor sampling:
# distractors must be at least MIN_DISTRACTOR_DIST from the true location
# but within MAX_DISTRACTOR_DIST so they remain visible/plausible.
MIN_DISTRACTOR_DIST = 60
MAX_DISTRACTOR_DIST = 250

MARGIN = 30  # keep keypoints away from image edges


def _list_hpatches_sequences(root: Path) -> list[Path]:
    """Find HPatches sequence directories.

    The release tarball extracts into hpatches-sequences-release/
    containing both `v_*` (viewpoint) and `i_*` (illumination) subdirs.
    Each subdir has 1.ppm..6.ppm and H_1_2 .. H_1_6 homography files.
    """
    candidates = list(root.rglob("hpatches-sequences-release"))
    if candidates:
        seq_root = candidates[0]
    else:
        seq_root = root
    seqs = sorted(
        [
            p
            for p in seq_root.iterdir()
            if p.is_dir() and (p.name.startswith("v_") or p.name.startswith("i_"))
        ]
    )
    return seqs


def _load_seq(seq_dir: Path):
    """Load images 1..6 and homographies H_1_2..H_1_6.

    Returns (images, homographies) where images[i] is the i+1-th frame
    (1-indexed in HPatches convention) and homographies[k] (k=0..4)
    is H_1_(k+2) — the 3x3 matrix mapping points from image 1 to
    image k+2.
    """
    imgs = []
    for k in range(1, 7):
        for ext in (".ppm", ".png", ".jpg", ".jpeg"):
            p = seq_dir / f"{k}{ext}"
            if p.exists():
                imgs.append(Image.open(p).convert("RGB"))
                break
        else:
            return None, None
    Hs = []
    for k in range(2, 7):
        hp = seq_dir / f"H_1_{k}"
        if not hp.exists():
            return None, None
        H = np.loadtxt(hp)
        if H.shape != (3, 3):
            return None, None
        Hs.append(H)
    return imgs, Hs


def _warp_pt(pt, H):
    """Apply 3x3 homography to a 2D point."""
    x, y = pt
    v = np.array([x, y, 1.0])
    u = H @ v
    return float(u[0] / u[2]), float(u[1] / u[2])


def _Hi_j_from_H1(H_1_i, H_1_j):
    """Compose homography from image i to image j: H_i_j = H_1_j @ inv(H_1_i)."""
    return H_1_j @ np.linalg.inv(H_1_i)


def _sample_pair_indices(rng):
    """Yield (i, j) with 1<=i<j<=6 and j-i >= 2 in random order."""
    pairs = [(i, j) for i in range(1, 7) for j in range(i + 1, 7) if j - i >= 2]
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpatches-dir", default="/outputs/livr_v2_sources/hpatches")
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/visual_correspondence")
    parser.add_argument(
        "--output-jsonl-prefix", default="/outputs/livr_v2/data/visual_correspondence"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = _list_hpatches_sequences(Path(args.hpatches_dir))
    logger.info("Found %d HPatches sequences", len(seqs))
    if not seqs:
        raise SystemExit("No HPatches sequences found.")

    rng.shuffle(seqs)
    needed = N_TRAIN + N_VAL + N_TEST + 100
    records: list[dict] = []
    seq_round = 0

    while len(records) < needed and seq_round < 5:
        for seq in seqs:
            if len(records) >= needed:
                break
            imgs, Hs = _load_seq(seq)
            if imgs is None:
                continue
            for i, j in _sample_pair_indices(rng):
                if len(records) >= needed:
                    break
                src_img = imgs[i - 1]
                tgt_img = imgs[j - 1]
                # H from src(i) to tgt(j)
                H_1_i = np.eye(3) if i == 1 else Hs[i - 2]
                H_1_j = Hs[j - 2]
                H_ij = _Hi_j_from_H1(H_1_i, H_1_j)

                sw, sh = src_img.size
                tw, th = tgt_img.size

                # Try a few times to find an in-bounds correspondence.
                gt_pt_src = None
                gt_pt_tgt = None
                for _ in range(50):
                    sx = rng.randint(MARGIN, sw - MARGIN)
                    sy = rng.randint(MARGIN, sh - MARGIN)
                    tx, ty = _warp_pt((sx, sy), H_ij)
                    if MARGIN <= tx <= tw - MARGIN and MARGIN <= ty <= th - MARGIN:
                        gt_pt_src = (sx, sy)
                        gt_pt_tgt = (tx, ty)
                        break
                if gt_pt_src is None:
                    continue

                # 3 distractors in target image, "far from the true location
                # and from each other" (Appendix A). Each distractor must be
                # at least MIN_DISTRACTOR_DIST from the GT AND from every
                # already-placed distractor.
                distractors: list[tuple[float, float]] = []
                for _ in range(3):
                    for _ in range(50):
                        a = rng.uniform(0, 2 * np.pi)
                        r = rng.uniform(MIN_DISTRACTOR_DIST, MAX_DISTRACTOR_DIST)
                        dx = gt_pt_tgt[0] + r * np.cos(a)
                        dy = gt_pt_tgt[1] + r * np.sin(a)
                        if not (MARGIN <= dx <= tw - MARGIN and MARGIN <= dy <= th - MARGIN):
                            continue
                        # Must be far from every previously-placed distractor.
                        ok = True
                        for px, py in distractors:
                            if ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5 < MIN_DISTRACTOR_DIST:
                                ok = False
                                break
                        if not ok:
                            continue
                        distractors.append((dx, dy))
                        break
                if len(distractors) < 3:
                    continue

                points = [gt_pt_tgt] + distractors  # 0 is GT
                correct_pos, ordering = shuffle_choices_with_index(0, 4, rng)
                ordered = [points[k] for k in ordering]

                # Annotate.
                src_an = draw_ref_marker(src_img, int(gt_pt_src[0]), int(gt_pt_src[1]))
                tgt_an = tgt_img.copy()
                for k, pt in enumerate(ordered):
                    tgt_an = draw_keypoint(
                        tgt_an,
                        int(pt[0]),
                        int(pt[1]),
                        color=OPTION_COLORS[k],
                        radius=10,
                        label=f"({OPTION_LETTERS[k]})",
                    )

                composite = create_side_by_side(
                    src_an,
                    tgt_an,
                    left_label="Source (REF = query point)",
                    right_label="Target (find corresponding point)",
                )
                out_filename = f"viscorr_{len(records):05d}.png"
                out_path = out_dir / out_filename
                save_image(composite, str(out_path), format="PNG")

                # Paper Appendix A.4.2 prompt verbatim — explicit framing
                # of the camera/lighting variation and "(A) Point A" choice
                # format. Brick-wall rate on terse prompts was 45.4%; the
                # paper's prompt primes the task properly.
                question = (
                    "A point is circled on the first image, labeled with REF. "
                    "We change the camera position or lighting and shoot the second image. "
                    "You are given multiple red-circled points on the second image, "
                    'choices of "A, B, C, D" are drawn beside each circle. '
                    "Which point on the second image corresponds to the point in the first image? "
                    "Select from the following options.\n"
                    "(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D"
                )
                formatted_choices = [
                    f"({OPTION_LETTERS[k]}) Point {OPTION_LETTERS[k]}" for k in range(4)
                ]
                rec = make_livr_record(
                    question=question,
                    ground_truth=OPTION_LETTERS[correct_pos],
                    choices=formatted_choices,
                    image_path=str(out_path),
                    dataset_name=TASK_NAME,
                )
                records.append(rec)
                if len(records) % 200 == 0:
                    logger.info("  built %d/%d", len(records), needed)
        seq_round += 1
    logger.info("Built %d viscorr composites", len(records))

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
    logger.info("Visual_Correspondence done: %d/%d/%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
