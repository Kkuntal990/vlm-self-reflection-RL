"""Convert LIVR mixed jsonl to ShareGPT-style format and upload to HF.

Reads the repo-relative jsonl staged at /outputs/hf_staging/livr_mixed/
and writes a ShareGPT variant where each row is:

    {
      "conversations": [
        {"from": "human", "value": "<image>\\n{question}"},
        {"from": "gpt",   "value": "{ground_truth}"}
      ],
      "images": ["images/livr/<task>/<file>.jpg"],
      "answer_type": "...",
      "dataset_name": "..."
    }

Uploads as `livr_perception_mixed_sharegpt.jsonl` to Kkuntal990/LIVR_mixed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/outputs/hf_staging/livr_mixed/livr_perception_mixed.jsonl")
    parser.add_argument("--dst", default="/outputs/hf_staging/livr_mixed/livr_perception_mixed_sharegpt.jsonl")
    parser.add_argument("--repo_id", default="Kkuntal990/LIVR_mixed")
    parser.add_argument("--path_in_repo", default="livr_perception_mixed_sharegpt.jsonl")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    src = Path(args.src)
    dst = Path(args.dst)

    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            question = row["question"]
            images = row.get("images") or []
            image_tags = "".join("<image>\n" for _ in images)
            human = f"{image_tags}{question}"
            out = {
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": row["ground_truth"]},
                ],
                "images": images,
                "answer_type": row.get("answer_type"),
                "dataset_name": row.get("dataset_name"),
            }
            if "choices" in row:
                out["choices"] = row["choices"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    logger.info("Wrote %d rows to %s", n, dst)

    api = HfApi(token=token)
    logger.info("Uploading %s to %s", args.path_in_repo, args.repo_id)
    api.upload_file(
        path_or_fileobj=str(dst),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add ShareGPT-format jsonl",
    )
    logger.info("Upload complete: https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
