#!/usr/bin/env python3
"""livr-v2 Functional Correspondence builder — strict Appendix A replication.

Per Appendix A of the LIVR paper (arXiv:2512.21218):

  Source     : FunKPoint (Lai, Purushwalkam, Gupta — ICCV 2021).
               20 object categories x 10 actions, 5 functional keypoints
               per (image, action) row, normalized [0..1] coordinates.
  Pairing    : "pair images that share the same action using a one-use-
               per-image policy" with an "action-aware balancing scheme"
               (cap pairs per action so no single action dominates).
  Construct  : On the LEFT (source) image, pick a reference keypoint
               index k in {1..5}, mark it with a red "REF" circle.
               On the RIGHT (target) image, place 4 candidate keypoint
               markers (A-D): the keypoint with the SAME index k (true
               correspondence) + 3 distractors sampled from the
               remaining indices {1..5} \\ {k}.
  Format     : 4-way MCQ.
  Splits     : disjoint 80/10/10 split of FunKPoint IMAGES (the entire
               image is held out, not just one row) -> 1000 train /
               144 val / 146 test.
  Dedup      : not required (no shared source with BLINK).

FunKPoint layout (after extraction):
  funkpoint/FunKPoint/
    labels.csv                  (4776 rows: p1..p5 (x,y) normalized,
                                 difficulty, image_path, action, wnid,
                                 object_category)
    images/<wnid>/<file>.jpg

Usage:
    python scripts/data/livr_v2/build_functional_correspondence.py \\
        --funkpoint-dir /outputs/livr_v2_sources/funkpoint/FunKPoint \\
        --output-dir /outputs/livr_v2/image_base/functional_correspondence \\
        --output-jsonl-prefix /outputs/livr_v2/data/functional_correspondence
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

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

TASK_NAME = "livr_functional_correspondence"

# Appendix A: 80/10/10 split of FunKPoint -> 1000/144/146
N_TRAIN = 1000
N_VAL = 144
N_TEST = 146

# Margin from image edges (in pixels of the destination image — applied
# after we resolve normalized [0..1] coords to pixel coords).
MARGIN = 20
# Min pixel distance between any two of the 4 candidate keypoint markers
# in the target image (so the disks don't overlap visually).
MIN_DISTRACTOR_DIST = 30


def _read_labels(funkpoint_root: Path) -> list[dict]:
    """Parse labels.csv into a list of row dicts.

    Returns rows with normalized (0..1) keypoint coordinates kept as
    floats, plus image_path/action/wnid/object_category strings.
    """
    csv_path = funkpoint_root / "labels.csv"
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                kps = [(float(r[f"p{i}_x"]), float(r[f"p{i}_y"])) for i in range(1, 6)]
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "kps": kps,  # list of 5 (x_norm, y_norm) tuples
                    "difficulty": r.get("difficulty", "").strip(),
                    "image_path": r["image_path"].strip(),
                    "action": r["action"].strip(),
                    "wnid": r.get("wnid", "").strip(),
                    "object_category": r.get("object_category", "").strip(),
                }
            )
    return rows


def _split_images(image_paths: list[str], rng: random.Random) -> tuple[set, set, set]:
    """80/10/10 disjoint partition of FunKPoint images.

    The paper's "disjoint 80/10/10 split of FunKPoint images" requires
    that an entire image (and all of its action-rows) ends up in
    exactly one of train/val/test. We partition by image_path identity.
    """
    unique = sorted(set(image_paths))
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_set = set(unique[:n_train])
    val_set = set(unique[n_train : n_train + n_val])
    test_set = set(unique[n_train + n_val :])
    return train_set, val_set, test_set


def _make_pairs(
    rows: list[dict], allowed_images: set, rng: random.Random
) -> list[tuple[dict, dict]]:
    """Pair rows that share the same action using a one-use-per-image policy.

    "Action-aware balancing scheme" interpretation: within each action,
    shuffle the rows whose image is in `allowed_images`, then pair
    adjacent rows greedily, ensuring each IMAGE is used at most once
    within this action.
    """
    by_action: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["image_path"] in allowed_images:
            by_action[r["action"]].append(r)

    pairs: list[tuple[dict, dict]] = []
    for action, action_rows in by_action.items():
        rng.shuffle(action_rows)
        used_images: set = set()
        unused: list[dict] = []
        for r in action_rows:
            if r["image_path"] in used_images:
                continue
            unused.append(r)
        # Pair adjacent unused entries (1<->2, 3<->4, ...).
        for i in range(0, len(unused) - 1, 2):
            a, b = unused[i], unused[i + 1]
            if a["image_path"] == b["image_path"]:
                # Edge case: two distinct rows but same image — skip.
                continue
            pairs.append((a, b))
        logger.info(
            "  action=%s pairs=%d", action, sum(1 for p in pairs if p[0]["action"] == action)
        )
    return pairs


def _norm_to_pixel(xn: float, yn: float, w: int, h: int) -> tuple[int, int]:
    """Normalized (0..1) FunKPoint coords -> pixel (x, y) in image of size w x h."""
    return int(round(xn * w)), int(round(yn * h))


def _build_one(
    src_row: dict,
    tgt_row: dict,
    funkpoint_root: Path,
    out_dir: Path,
    out_idx: int,
    rng: random.Random,
) -> dict | None:
    """Build a single 4-way MCQ record from a (src_row, tgt_row) pair.

    Returns None if the example must be skipped (image load failure,
    keypoints too close to edges, etc.).
    """
    src_p = funkpoint_root / src_row["image_path"]
    tgt_p = funkpoint_root / tgt_row["image_path"]
    try:
        src_img = Image.open(src_p).convert("RGB")
        tgt_img = Image.open(tgt_p).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load %s or %s: %s", src_p, tgt_p, e)
        return None

    sw, sh = src_img.size
    tw, th = tgt_img.size

    # Pick reference keypoint index k in {1..5} (1-indexed in paper, 0-indexed here).
    k = rng.randint(0, 4)
    src_kp_n = src_row["kps"][k]
    tgt_kp_n = tgt_row["kps"][k]
    src_xy = _norm_to_pixel(src_kp_n[0], src_kp_n[1], sw, sh)
    tgt_xy = _norm_to_pixel(tgt_kp_n[0], tgt_kp_n[1], tw, th)

    if not (MARGIN <= src_xy[0] <= sw - MARGIN and MARGIN <= src_xy[1] <= sh - MARGIN):
        return None
    if not (MARGIN <= tgt_xy[0] <= tw - MARGIN and MARGIN <= tgt_xy[1] <= th - MARGIN):
        return None

    # 3 distractors: the OTHER 4 indices from the SAME target row.
    other_idxs = [i for i in range(5) if i != k]
    rng.shuffle(other_idxs)
    distractors_xy: list[tuple[int, int]] = []
    for di in other_idxs:
        if len(distractors_xy) >= 3:
            break
        dx, dy = _norm_to_pixel(*tgt_row["kps"][di], tw, th)
        if not (MARGIN <= dx <= tw - MARGIN and MARGIN <= dy <= th - MARGIN):
            continue
        # Min visual separation from GT and from prior distractors.
        if ((dx - tgt_xy[0]) ** 2 + (dy - tgt_xy[1]) ** 2) ** 0.5 < MIN_DISTRACTOR_DIST:
            continue
        too_close = False
        for px, py in distractors_xy:
            if ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5 < MIN_DISTRACTOR_DIST:
                too_close = True
                break
        if too_close:
            continue
        distractors_xy.append((dx, dy))
    if len(distractors_xy) < 3:
        return None

    # Shuffle GT among 4 letters (A-D).
    points = [tgt_xy] + distractors_xy  # index 0 is GT
    correct_pos, ordering = shuffle_choices_with_index(0, 4, rng)
    ordered = [points[i] for i in ordering]

    # Annotate.
    src_an = draw_ref_marker(src_img, src_xy[0], src_xy[1])
    tgt_an = tgt_img.copy()
    for j, (px, py) in enumerate(ordered):
        tgt_an = draw_keypoint(
            tgt_an,
            px,
            py,
            color=OPTION_COLORS[j],
            radius=10,
            label=f"({OPTION_LETTERS[j]})",
        )

    composite = create_side_by_side(
        src_an,
        tgt_an,
        left_label=f"Source — REF ({src_row.get('object_category', 'object')})",
        right_label="Target (find the same functional point)",
    )
    out_filename = f"funccorr_{out_idx:05d}.png"
    out_path = out_dir / out_filename
    save_image(composite, str(out_path), format="PNG")

    action = src_row["action"]
    src_cat = src_row.get("object_category", "object")
    tgt_cat = tgt_row.get("object_category", "object")
    question = (
        f"A functional keypoint relevant to action '{action}' is marked with REF on the "
        f"{src_cat} in the source image (left). Which point in the target image (right) "
        f"on the {tgt_cat} corresponds to the same functional role? (A) (B) (C) (D)"
    )
    formatted_choices = [f"({OPTION_LETTERS[i]})" for i in range(4)]
    return make_livr_record(
        question=question,
        ground_truth=OPTION_LETTERS[correct_pos],
        choices=formatted_choices,
        image_path=str(out_path),
        dataset_name=TASK_NAME,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--funkpoint-dir",
        default="/outputs/livr_v2_sources/funkpoint/FunKPoint",
        help="Path to extracted FunKPoint dataset (must contain labels.csv + images/)",
    )
    parser.add_argument(
        "--output-dir", default="/outputs/livr_v2/image_base/functional_correspondence"
    )
    parser.add_argument(
        "--output-jsonl-prefix", default="/outputs/livr_v2/data/functional_correspondence"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    funkpoint_root = Path(args.funkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_labels(funkpoint_root)
    logger.info("Loaded %d FunKPoint label rows", len(rows))
    if not rows:
        raise SystemExit("No labels.csv rows parsed — check funkpoint-dir.")

    train_imgs, val_imgs, test_imgs = _split_images([r["image_path"] for r in rows], rng)
    logger.info(
        "Image-disjoint split: train=%d val=%d test=%d unique images",
        len(train_imgs),
        len(val_imgs),
        len(test_imgs),
    )

    targets = [
        ("train", train_imgs, N_TRAIN),
        ("val", val_imgs, N_VAL),
        ("test", test_imgs, N_TEST),
    ]
    all_records: dict[str, list[dict]] = {}
    for split_name, allowed, target in targets:
        logger.info("=== building split=%s (target=%d) ===", split_name, target)
        pairs = _make_pairs(rows, allowed, rng)
        rng.shuffle(pairs)
        recs: list[dict] = []
        idx_offset = 0
        # Try pairs until we have enough records.
        for src_row, tgt_row in pairs:
            if len(recs) >= target:
                break
            global_idx = {"train": 0, "val": 100000, "test": 200000}[split_name] + idx_offset
            rec = _build_one(src_row, tgt_row, funkpoint_root, out_dir, global_idx, rng)
            if rec is not None:
                rec["split"] = split_name
                recs.append(rec)
            idx_offset += 1
        if len(recs) < target:
            logger.warning(
                "split=%s: built only %d/%d records (pool exhausted)",
                split_name,
                len(recs),
                target,
            )
        all_records[split_name] = recs[:target]

    write_jsonl(all_records["train"], f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(all_records["val"], f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(all_records["test"], f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info(
        "Functional_Correspondence done: train=%d val=%d test=%d",
        len(all_records["train"]),
        len(all_records["val"]),
        len(all_records["test"]),
    )


if __name__ == "__main__":
    main()
