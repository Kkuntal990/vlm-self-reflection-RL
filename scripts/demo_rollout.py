#!/usr/bin/env python3
"""
Demo rollout: generate K=8 trajectories (A1→F1→A2) for 10 samples
(2 per category) using Qwen2.5-VL-7B-Instruct, and show rewards.

Usage:
    python3 /tmp/demo_rollout.py
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

os.environ["VLM_USE_LLM_JUDGE"] = "1"

sys.path.insert(0, "/tmp/vlm_grpo_test")

DATASET_PATH = "/outputs/grpo_data/balanced_70k.jsonl"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_BASE_DIR = "/outputs/image_base"
K_SAMPLES = 8
SAMPLES_PER_CATEGORY = 2
SEED = 42


def load_samples_by_category(path: str, n_per_cat: int, seed: int) -> list[dict]:
    """Load n samples per category from the balanced dataset."""
    random.seed(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            sample = json.loads(line.strip())
            buckets[sample["answer_type"]].append(sample)

    selected = []
    for cat in ["mcq", "yesno", "counting", "short", "open"]:
        pool = buckets.get(cat, [])
        if len(pool) >= n_per_cat:
            selected.extend(random.sample(pool, n_per_cat))
        else:
            selected.extend(pool)
    return selected


def load_image(sample: dict) -> "Image.Image | None":
    """Load image from sample."""
    from PIL import Image

    image_path = sample.get("image_path", "")
    if not image_path:
        images = sample.get("images", [])
        if images:
            image_path = images[0]
    if image_path and not os.path.isabs(image_path):
        image_path = os.path.join(IMAGE_BASE_DIR, image_path)
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        max_px = 401408
        if w * h > max_px:
            scale = (max_px / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  [WARN] Failed to load image: {e}")
        return None


def main() -> None:
    """Run demo rollout."""
    print("=" * 90)
    print("DEMO ROLLOUT: A1 → F1 → A2 with Qwen2.5-VL-7B-Instruct (K=8)")
    print("=" * 90)

    # Load samples
    samples = load_samples_by_category(DATASET_PATH, SAMPLES_PER_CATEGORY, SEED)
    print(f"Selected {len(samples)} samples: {', '.join(s['answer_type'] for s in samples)}\n")

    # Load model
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading model: {MODEL_ID}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=200704, max_pixels=401408)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Import reward functions
    from vlm_grpo.config import FeedbackRewardWeights, ResponseRewardWeights
    from vlm_grpo.prompts import (
        build_critic_prompt,
        build_initial_answer_prompt,
        build_refiner_prompt,
    )
    from vlm_grpo.rewards.composition import (
        compute_feedback_reward_breakdown,
        compute_response_reward_breakdown,
    )

    resp_weights = ResponseRewardWeights()
    fb_weights = FeedbackRewardWeights()

    def generate(messages, image, temp, max_tokens):
        """Generate one completion."""
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=text, return_tensors="pt").to(model.device)

        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temp > 0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
            )
        return processor.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    # Run rollout for each sample
    for si, sample in enumerate(samples):
        question = sample.get("question", "").replace("<image>", "").strip()
        gt = sample["ground_truth"]
        answer_type = sample["answer_type"]
        choices = sample.get("choices", "")
        image = load_image(sample)

        print("=" * 90)
        print(f"SAMPLE {si + 1}/{len(samples)} | Type: {answer_type}")
        print(f"Question: {question[:120]}")
        print(f"GT: {gt[:100]}")
        print(f"Image: {'loaded' if image else 'MISSING'}")
        print("=" * 90)

        for k in range(K_SAMPLES):
            # A1: Initial answer
            a1_prompt = build_initial_answer_prompt(question)
            a1 = generate(a1_prompt, image, temp=1.0, max_tokens=200)

            # F1: Feedback
            f1_prompt = build_critic_prompt(question, a1, model_type="qwen2vl")
            f1 = generate(f1_prompt, image, temp=1.0, max_tokens=512)

            # A2: Refined answer
            a2_prompt = build_refiner_prompt(question, a1, f1)
            a2 = generate(a2_prompt, image, temp=0.3, max_tokens=200)

            # Compute rewards
            resp_bd = compute_response_reward_breakdown(
                a1_text=a1,
                a2_text=a2,
                ground_truth=gt,
                answer_type=answer_type,
                choices=choices,
                weights=resp_weights,
            )
            fb_bd = compute_feedback_reward_breakdown(
                feedback_text=f1,
                a1_text=a1,
                a2_text=a2,
                ground_truth=gt,
                answer_type=answer_type,
                choices=choices,
                weights=fb_weights,
            )

            # Transition label
            if resp_bd.a1_correct and resp_bd.a2_correct:
                trans = "RR"
            elif resp_bd.a1_correct and not resp_bd.a2_correct:
                trans = "RW"
            elif not resp_bd.a1_correct and resp_bd.a2_correct:
                trans = "WR"
            else:
                trans = "WW"

            print(f"\n  --- Trajectory {k + 1}/{K_SAMPLES} [{trans}] ---")
            print(f"  A1: {a1[:120]}")
            print(f"  F1: {f1[:150]}")
            print(f"  A2: {a2[:120]}")
            print(
                f"  A1_correct={resp_bd.a1_correct} | "
                f"A2_correct={resp_bd.a2_correct} | "
                f"format_valid={resp_bd.a2_format_valid}"
            )
            print(
                f"  Response reward: {resp_bd.total_reward:+.2f} "
                f"(a1={resp_bd.components['a1_correctness']:+.1f} "
                f"a2={resp_bd.components['a2_correctness']:+.1f} "
                f"noreg={resp_bd.components['no_regression']:+.1f} "
                f"fmt={resp_bd.components['a2_format']:+.1f} "
                f"edit={resp_bd.components['minimal_edit']:.2f})"
            )
            print(
                f"  Feedback reward: {fb_bd.total_reward:+.2f} "
                f"(downstream={fb_bd.components['downstream']:+.1f} "
                f"fmt={fb_bd.components['format']:+.1f})"
            )

        print()


if __name__ == "__main__":
    main()
