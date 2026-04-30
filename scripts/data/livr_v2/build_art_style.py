#!/usr/bin/env python3
"""livr-v2 Art Style builder.

Per Appendix A:
  * Source: ArtBench-10 (256x256 ImageFolder split variant)
  * Construction: reference painting + same-style candidate + different-style
    candidate. Model must pick the same-style candidate.
  * 2-way MCQ (A or B)
  * Cross-dataset CLIP dedup vs BLINK Art_Style val
  * Splits: 1000 / 250 / 117 (BLINK)

ArtBench-10 layout: artbench-10-imagefolder-split/{train,test}/<style>/<image>.jpg
where <style> is one of 10 art styles.

Usage:
    python scripts/data/livr_v2/build_art_style.py \\
        --artbench-dir /outputs/livr_v2_sources/artbench10 \\
        --blink-val-dir /outputs/livr_v2_sources/blink_val \\
        --output-dir /outputs/livr_v2/image_base/art_style \\
        --output-jsonl-prefix /outputs/livr_v2/data/art_style
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from dedup import full_dedup  # noqa: E402
from livr_common import (  # noqa: E402
    OPTION_LETTERS,
    get_font,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices_with_index,
    write_jsonl,
)

TASK_NAME = "livr_art_style"
N_TRAIN = 1000
N_VAL = 250
N_TEST = 117


def _find_artbench_root(root: Path) -> Path:
    candidates = list(root.rglob("artbench-10-imagefolder-split"))
    if candidates:
        return candidates[0]
    return root


def _load_styles(root: Path) -> dict[str, list[Path]]:
    """Return style -> list of image paths (use train+test combined)."""
    by_style: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "test"):
        sd = root / split
        if not sd.exists():
            continue
        for style_dir in sd.iterdir():
            if not style_dir.is_dir():
                continue
            for img in style_dir.glob("*.jpg"):
                by_style[style_dir.name].append(img)
            for img in style_dir.glob("*.png"):
                by_style[style_dir.name].append(img)
    return by_style


def _stack3(reference: Image.Image, a: Image.Image, b: Image.Image) -> Image.Image:
    """Reference on top, two candidates side-by-side below labeled (A) (B)."""
    cell = 256
    pad = 12
    label_h = 36
    width = 2 * cell + 3 * pad
    height = cell + 2 * label_h + 3 * pad + cell
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    font = get_font(28)
    # Reference (centered).
    rref = reference.copy()
    rref.thumbnail((cell, cell), Image.LANCZOS)
    rx = (width - rref.width) // 2
    ry = pad + label_h
    canvas.paste(rref, (rx, ry))
    ImageDraw.Draw(canvas).text((rx, pad), "Reference", fill=(0, 0, 0), font=font)
    # Candidates row.
    cy_label = cell + 2 * pad + label_h
    cy = cy_label + label_h
    for k, img in enumerate([a, b]):
        c = img.copy()
        c.thumbnail((cell, cell), Image.LANCZOS)
        cx = pad + k * (cell + pad)
        canvas.paste(c, (cx + (cell - c.width) // 2, cy))
        ImageDraw.Draw(canvas).text(
            (cx + cell // 2 - 12, cy_label),
            f"({OPTION_LETTERS[k]})",
            fill=(0, 0, 0),
            font=font,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artbench-dir", default="/outputs/livr_v2_sources/artbench10")
    parser.add_argument("--blink-val-dir", default="/outputs/livr_v2_sources/blink_val")
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/art_style")
    parser.add_argument("--output-jsonl-prefix", default="/outputs/livr_v2/data/art_style")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--clip-device", default="cuda")
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = _find_artbench_root(Path(args.artbench_dir))
    by_style = _load_styles(root)
    styles = sorted(by_style.keys())
    logger.info(
        "Loaded ArtBench: %d styles, total %d images",
        len(styles),
        sum(len(v) for v in by_style.values()),
    )
    if not styles:
        raise SystemExit("ArtBench-10 not found.")

    needed = N_TRAIN + N_VAL + N_TEST + 200
    saved: list[tuple[Path, dict]] = []
    while len(saved) < needed:
        ref_style = rng.choice(styles)
        if len(by_style[ref_style]) < 2:
            continue
        ref, pos = rng.sample(by_style[ref_style], 2)
        neg_style = rng.choice([s for s in styles if s != ref_style])
        neg = rng.choice(by_style[neg_style])
        try:
            ref_img = Image.open(ref).convert("RGB")
            pos_img = Image.open(pos).convert("RGB")
            neg_img = Image.open(neg).convert("RGB")
        except Exception:
            continue
        # 2-way MCQ — randomize order. Index 0 is the correct (same-style) candidate.
        correct_pos, ordering = shuffle_choices_with_index(0, 2, rng)
        cands = [pos_img, neg_img]
        a, b = cands[ordering[0]], cands[ordering[1]]
        composite = _stack3(ref_img, a, b)
        out_filename = f"art_style_{len(saved):05d}.png"
        out_path = out_dir / out_filename
        save_image(composite, str(out_path), format="PNG")

        question = (
            "Which painting shares the same art style as the reference painting (top)? (A) (B)"
        )
        formatted_choices = ["(A)", "(B)"]
        rec = make_livr_record(
            question=question,
            ground_truth=OPTION_LETTERS[correct_pos],
            choices=formatted_choices,
            image_path=str(out_path),
            dataset_name=TASK_NAME,
        )
        saved.append((out_path, rec))
        if len(saved) % 200 == 0:
            logger.info("  built %d/%d", len(saved), needed)
    logger.info("Saved %d art_style composites", len(saved))

    keep_set: set[Path] | None = None
    if not args.skip_dedup:
        blink_val = Path(args.blink_val_dir)
        images_dir = blink_val / "Art_Style" / "images"
        if images_dir.exists():
            blink_imgs = sorted(images_dir.glob("*.png"))
        else:
            candidates = list(blink_val.rglob("*Art_Style*/*"))
            candidates += list(blink_val.rglob("*art_style*/*"))
            blink_imgs = [
                p
                for p in candidates
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        logger.info("BLINK Art_Style val images: %d", len(blink_imgs))
        if blink_imgs:
            # Appendix A: CLIP + pHash + SSIM ANDed.
            keep_set = full_dedup(
                candidates=[p for p, _ in saved],
                exclude=blink_imgs,
                clip_sim_thresh=0.95,
                phash_thresh=8,
                ssim_thresh=0.95,
                device=args.clip_device,
            )

    final = [(p, r) for p, r in saved if keep_set is None or p in keep_set]
    logger.info("After dedup: %d", len(final))

    train = [r for _, r in final[:N_TRAIN]]
    val = [r for _, r in final[N_TRAIN : N_TRAIN + N_VAL]]
    test = [r for _, r in final[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]]
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"
    write_jsonl(train, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info("Art_Style done: %d/%d/%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
