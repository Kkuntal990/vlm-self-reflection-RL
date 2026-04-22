"""Upload LIVR mixed GRPO dataset to HuggingFace.

Uploads `livr_perception_mixed.jsonl` and the referenced images from the
PVC to the `Kkuntal990/LIVR_mixed` dataset repo. Only images referenced
by the JSONL are uploaded. Image paths in the uploaded JSONL are rewritten
to be repo-relative (e.g., `images/livr/counting/counting_0304.jpg`).

Intended to run on the helper pod `vlm-jupyter-eval2`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="/outputs/livr_data/livr_perception_mixed.jsonl")
    parser.add_argument("--image_base", default="/outputs/image_base")
    parser.add_argument("--repo_id", default="Kkuntal990/LIVR_mixed")
    parser.add_argument("--staging_dir", default="/outputs/hf_staging/livr_mixed")
    parser.add_argument("--jsonl_name", default="livr_perception_mixed.jsonl")
    parser.add_argument("--image_prefix", default="images")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_staging", action="store_true", help="Reuse existing staging dir")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        raise RuntimeError("HF_TOKEN env var not set")

    staging = Path(args.staging_dir)
    src_jsonl = Path(args.jsonl)
    image_base = Path(args.image_base)
    new_jsonl_path = staging / args.jsonl_name

    if args.skip_staging and staging.exists():
        logger.info("Reusing existing staging dir %s", staging)
        cache_dir = staging / ".cache"
        if cache_dir.exists():
            logger.info("Removing stale .cache in staging")
            shutil.rmtree(cache_dir)
    else:
        if staging.exists():
            logger.info("Clearing existing staging dir %s", staging)
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        referenced: set[Path] = set()
        n_rows = 0
        with src_jsonl.open() as fin, new_jsonl_path.open("w") as fout:
            for line in fin:
                row = json.loads(line)
                images = row.get("images") or []
                new_images: list[str] = []
                for p in images:
                    abs_src = Path(p)
                    try:
                        rel = abs_src.relative_to(image_base)
                    except ValueError:
                        raise RuntimeError(f"Image path {p} not under image_base {image_base}")
                    new_rel = f"{args.image_prefix}/{rel.as_posix()}"
                    new_images.append(new_rel)
                    referenced.add(abs_src)
                row["images"] = new_images
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

        logger.info("Wrote %d rows to %s", n_rows, new_jsonl_path)
        logger.info("Unique referenced images: %d", len(referenced))

        missing = [p for p in referenced if not p.exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} referenced images missing. First: {missing[:5]}")

        logger.info("Staging %d images via hardlink/symlink into %s", len(referenced), staging)
        for abs_src in referenced:
            rel = abs_src.relative_to(image_base)
            dst = staging / args.image_prefix / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(abs_src, dst)
            except OSError:
                os.symlink(abs_src, dst)

        logger.info("Staging complete. Root: %s", staging)
        total_bytes = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        logger.info("Total staged size: %.2f GB", total_bytes / 1e9)

    if args.dry_run:
        logger.info("Dry run, skipping upload")
        return

    api = HfApi(token=token)
    logger.info("Creating/ensuring repo %s (dataset)", args.repo_id)
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, token=token)

    logger.info("Uploading jsonl to %s", args.repo_id)
    api.upload_file(
        path_or_fileobj=str(new_jsonl_path),
        path_in_repo=args.jsonl_name,
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    per_task_dirs = sorted((staging / args.image_prefix / "livr").iterdir())
    logger.info("Uploading images in %d task batches to avoid rate limits", len(per_task_dirs))
    for task_dir in per_task_dirs:
        rel_prefix = f"{args.image_prefix}/livr/{task_dir.name}"
        n = sum(1 for _ in task_dir.iterdir())
        logger.info("Uploading %s (%d files)", rel_prefix, n)
        api.upload_folder(
            folder_path=str(task_dir),
            path_in_repo=rel_prefix,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {rel_prefix}",
        )
    logger.info("Upload complete: https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
