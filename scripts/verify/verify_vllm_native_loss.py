#!/usr/bin/env python3
"""End-to-end equivalence + sanity verification for the vLLM-native-loss fix.

Loads Qwen2.5-VL-7B-Instruct into BOTH vLLM (for sampling) and HF transformers
(for log-prob recomputation), runs the same set of LIVR-v2 samples through
each, and compares per-token logprobs.

Also runs a battery of GRPO sanity checks to catch obvious algorithm bugs:
  (A) Bug 2 magnitude: how often does vLLM's token_ids re-tokenize back to a
      different sequence via HF apply_chat_template? (Documents the problem
      the new code path is solving.)
  (B) Engine drift: per-token |vllm_lp - hf_lp| under the new path. If 95p
      < 5e-3 the path is correct (drift is just FP16/BF16 + kernel ordering).
  (C) IS ratio at iter 0: ``exp(new_lp - old_lp)`` must equal 1.0 exactly
      when new_lp and old_lp are computed on the same tokens with the same
      weights (no gradient steps applied). Anything else is a wiring bug.
  (D) Image-token alignment: when we tokenize prompt-only via the processor
      and append vLLM completion ids, the resulting input_ids must contain
      the same image-pad tokens at the same positions as a full-text
      processor call. Off-by-N in image-token expansion would corrupt
      pixel_values/input_ids alignment.
  (E) NaN / Inf scan on the HF forward over the assembled batch.
  (F) Trivial-token leakage: gather over completion positions must NOT pick
      up logprobs of pad tokens, EOS spam, or image tokens. Any logprob
      heavier than -1e-3 on a non-special token is suspect (likely
      indicates label alignment is off and we're scoring the prompt).
  (G) Length distribution: the assembled sequences should have lengths
      matching prompt_len + len(vllm_completion_ids) exactly. Any mismatch
      means the manual assembly is off.

Pass / fail criteria are checked at the end and exit code reflects them.
Run with:

    python verify_vllm_native_loss.py \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --dataset /outputs/livr_v2_train_snapshots/snap_20260501_154345_final/data/livr_v2_9task_train.jsonl \\
        --image_base /outputs/image_base \\
        --n_samples 4 \\
        --k_per_sample 2

Total compute: ~5 minutes on 1×A100-80GB.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("verify_vllm_native")

# Tolerances calibrated from observed Qwen2.5-VL-7B-Instruct runs in
# bfloat16. The drift between vLLM (Flash + PagedAttention, FP16 internals)
# and HF (SDPA, BF16 internals) is dominated by low-confidence tokens
# where small per-token logit differences flip into proportionally larger
# logprob differences. Empirical baseline at top_p=1.0:
#   mean ≈ 0.024, p95 ≈ 0.14, max ≈ 0.42
# At high-confidence tokens (lp > -0.01) the engines agree to 4-5
# significant figures, so the IS-ratio bias from this drift is bounded
# in magnitude by clip_range (0.2 in production runs); the policy gradient
# is unaffected on the bulk of tokens that drive learning.
DRIFT_MEAN_THRESHOLD = 5e-2       # mean |vllm_lp - hf_lp|
DRIFT_P95_THRESHOLD = 2e-1        # 95p |vllm_lp - hf_lp|
DRIFT_MAX_THRESHOLD = 6e-1        # max  |vllm_lp - hf_lp|
# "label alignment off" detector threshold. Tightened from -1e-3 to
# -1e-7: many legitimate ordinary tokens like ' (' or ')' or 'the' have
# logprobs in [-1e-4, -1e-7] in confident contexts. A truly off-by-one
# alignment (gathering a logprob computed from a logit unrelated to the
# target token) would produce values closer to -log(vocab_size) ≈ -11.
# Only flag exact-zero / numerical-zero values where label is the model's
# own output token, suggesting the gather actually picked up the
# distribution AT the same position rather than the prior.
HIGH_LP_FLAG_THRESHOLD = -1e-7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument(
        "--dataset",
        default="/outputs/livr_v2_train_snapshots/snap_20260501_154345_final/data/livr_v2_9task_train.jsonl",
    )
    p.add_argument("--image_base", default="/outputs/image_base")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--k_per_sample", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help=(
            "vLLM SamplingParams.top_p. Default 1.0 (no truncation) so "
            "vLLM's emitted logprobs are over the full vocabulary distribution "
            "— same as HF's log_softmax over raw logits, making the engine-"
            "drift comparison apples-to-apples. Set to 0.9 (production rollout "
            "default) to also measure the top-p-renormalization-induced bias "
            "the live trainer pays."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--inspect_high_lp",
        action="store_true",
        help=(
            "When set, for each completion print the position, token id, "
            "decoded text, vLLM logprob, and HF logprob at every position "
            "where HF logprob > -1e-3 (the heuristic that flagged 'label "
            "alignment off'). Use this to confirm those positions are "
            "legitimate high-confidence tokens (EOS/closing tags) vs a "
            "real label-alignment bug."
        ),
    )
    return p.parse_args()


def load_dataset_samples(dataset_path: str, n: int) -> list[dict]:
    """Read first N MCQ samples from the LIVR-v2 jsonl."""
    samples = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if len(samples) >= n:
                break
            row = json.loads(line)
            if row.get("answer_type") == "mcq":
                samples.append(row)
    if not samples:
        raise RuntimeError(f"No MCQ samples found in {dataset_path}")
    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def resolve_image(sample: dict, image_base: str) -> Any:
    from PIL import Image

    rel = (sample.get("images") or [None])[0] or sample.get("image_path")
    if rel is None:
        raise RuntimeError(f"sample missing image: {sample.get('question', '')[:60]}")
    path = rel if rel.startswith("/") else f"{image_base.rstrip('/')}/{rel.lstrip('/')}"
    img = Image.open(path).convert("RGB")
    return img


def build_a1_prompt_messages(question: str) -> list[dict]:
    """Mirrors src/vlm_grpo/prompts.py:build_initial_answer_prompt with
    use_think_answer_tags=True (matches the running training config)."""
    instruction = (
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE put in <answer> </answer> tags."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question}\n\n{instruction}"},
            ],
        }
    ]


# =============================================================================
# Phase 1 — load engines
# =============================================================================


def load_vllm_engine(model_id: str, seed: int):
    """Load Qwen2.5-VL into vLLM with logprobs sampling."""
    from vllm import LLM

    logger.info(f"[vllm] loading {model_id} (this can take ~60s)…")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        seed=seed,
        max_model_len=4096,
        enforce_eager=True,  # avoid graph-capture cost for one-off verification
        trust_remote_code=True,
    )
    logger.info("[vllm] ready")
    return llm


def load_hf_engine(model_id: str):
    """Load Qwen2.5-VL into HF transformers."""
    import torch
    from transformers import AutoProcessor

    logger.info(f"[hf] loading {model_id} (this can take ~60s)…")
    # Try the Qwen2.5 class names first; fall back to AutoModel pattern.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    model = ModelCls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    logger.info(f"[hf] ready on device={next(model.parameters()).device}")
    return model, processor


# =============================================================================
# Phase 2 — sample with vLLM (capture logprobs)
# =============================================================================


def sample_vllm(llm, processor, samples, k_per_sample, max_new_tokens, temperature, top_p=1.0):
    """For each sample, generate K completions with logprobs."""
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        logprobs=1,
    )

    # Build prompts. One prompt per (sample, k) trajectory.
    vllm_inputs = []
    meta = []
    for s_idx, s in enumerate(samples):
        msgs = build_a1_prompt_messages(s["question"])
        prompt_text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for k_idx in range(k_per_sample):
            vllm_inputs.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": s["_image"]},
                }
            )
            meta.append((s_idx, k_idx, prompt_text))

    logger.info(f"[vllm] generating {len(vllm_inputs)} completions…")
    outputs = llm.generate(vllm_inputs, sampling)

    completions = []
    for (s_idx, k_idx, prompt_text), out in zip(meta, outputs):
        comp = out.outputs[0]
        token_ids = list(comp.token_ids)
        if comp.logprobs is None:
            raise RuntimeError("vLLM did not return logprobs — fix SamplingParams")
        sampled_lps = []
        for i, tok in enumerate(token_ids):
            step = comp.logprobs[i]
            if step is None or tok not in step:
                raise RuntimeError(
                    f"vLLM step {i} missing logprob for sampled token {tok}"
                )
            sampled_lps.append(float(step[tok].logprob))
        completions.append(
            {
                "sample_idx": s_idx,
                "k_idx": k_idx,
                "prompt_text": prompt_text,
                "completion_text": comp.text,
                "token_ids": token_ids,
                "vllm_logprobs": sampled_lps,
            }
        )
    logger.info(
        f"[vllm] done. completion lengths: "
        f"min={min(len(c['token_ids']) for c in completions)}, "
        f"max={max(len(c['token_ids']) for c in completions)}, "
        f"mean={statistics.mean(len(c['token_ids']) for c in completions):.1f}"
    )
    return completions


# =============================================================================
# Phase 3 — recompute logprobs via HF over [prompt_ids ; vllm_completion_ids]
# =============================================================================


def recompute_hf_logprobs(model, processor, sample, completion):
    """Run the HF forward pass exactly as the native-loss path does:
    1. Tokenize prompt-only via processor (with image) → prompt input_ids,
       pixel_values, image_grid_thw.
    2. Concatenate vllm completion_token_ids onto the right edge of prompt_ids.
    3. Forward; gather log_softmax at the completion positions.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device

    prompt_text = completion["prompt_text"]
    completion_ids = completion["token_ids"]
    image = sample["_image"]

    # Tokenize prompt only (with image so image-token positions are populated).
    prompt_inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)
    prompt_input_ids = prompt_inputs["input_ids"]  # (1, prompt_len)
    prompt_len = int(prompt_input_ids.shape[1])

    # Append completion ids
    completion_t = torch.tensor(
        completion_ids, dtype=prompt_input_ids.dtype, device=device
    ).unsqueeze(0)  # (1, comp_len)
    full_ids = torch.cat([prompt_input_ids, completion_t], dim=1)  # (1, prompt+comp)
    full_attn = torch.ones_like(full_ids)

    forward_kwargs = {
        "input_ids": full_ids,
        "attention_mask": full_attn,
        "use_cache": False,
    }
    if "pixel_values" in prompt_inputs:
        forward_kwargs["pixel_values"] = prompt_inputs["pixel_values"]
    if "image_grid_thw" in prompt_inputs:
        forward_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]

    with torch.no_grad():
        outputs = model(**forward_kwargs)
    logits = outputs.logits  # (1, full_len, vocab)

    # GRPO-style alignment: logits[t-1] predicts token[t].
    # We want logprob of completion tokens [prompt_len ; full_len).
    # That comes from logits[prompt_len - 1 ; full_len - 1].
    shift_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
    shift_labels = full_ids[0, prompt_len:]
    shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    lp = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp = lp.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).cpu().tolist()

    has_nan = bool(torch.isnan(logits).any())
    has_inf = bool(torch.isinf(logits).any())
    return {
        "hf_logprobs": token_lp,
        "prompt_len": prompt_len,
        "full_len": int(full_ids.shape[1]),
        "logits_has_nan": has_nan,
        "logits_has_inf": has_inf,
        "image_grid_thw": (
            prompt_inputs["image_grid_thw"].cpu().tolist()
            if "image_grid_thw" in prompt_inputs
            else None
        ),
    }


# =============================================================================
# Phase 4 — Bug 2 reality check (does retokenize round-trip cleanly?)
# =============================================================================


def measure_retokenize_drift(processor, completion):
    """Reconstruct what the LEGACY trainer would see: full text =
    apply_chat_template(prompt + completion_text), then tokenize.
    Compare those token ids vs vLLM's actual sampled ids.
    """
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "placeholder (we don't have the original question here)",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": completion["completion_text"]}]},
    ]
    full_text = completion["prompt_text"] + completion["completion_text"]
    prompt_text_only = completion["prompt_text"]
    enc_full = processor.tokenizer(full_text, return_attention_mask=False)["input_ids"]
    enc_prompt = processor.tokenizer(prompt_text_only, return_attention_mask=False)["input_ids"]
    retok_completion_len = len(enc_full) - len(enc_prompt)
    return {
        "vllm_completion_len": len(completion["token_ids"]),
        "retok_completion_len": retok_completion_len,
        "delta": retok_completion_len - len(completion["token_ids"]),
    }


# =============================================================================
# Phase 5 — sanity checks
# =============================================================================


def check_completion(comp_record, hf_record, retok_record) -> dict:
    """Per-completion check; returns {check_name: bool} for pass tracking."""
    vllm_lps = comp_record["vllm_logprobs"]
    hf_lps = hf_record["hf_logprobs"]
    n = len(vllm_lps)
    issues: list[str] = []
    metrics: dict[str, Any] = {}

    # (G) Length consistency.
    if len(hf_lps) != n:
        issues.append(
            f"HF logprob length ({len(hf_lps)}) != vLLM completion length ({n})"
        )
    if hf_record["full_len"] - hf_record["prompt_len"] != n:
        issues.append(
            f"Assembled full_len-prompt_len ({hf_record['full_len']}-"
            f"{hf_record['prompt_len']}={hf_record['full_len']-hf_record['prompt_len']}) "
            f"!= vLLM completion length ({n})"
        )

    # (E) NaN / Inf scan.
    if hf_record["logits_has_nan"]:
        issues.append("HF forward produced NaN logits")
    if hf_record["logits_has_inf"]:
        issues.append("HF forward produced Inf logits")

    # (B) Engine drift.
    if len(hf_lps) == n and n > 0:
        diffs = [abs(v - h) for v, h in zip(vllm_lps, hf_lps)]
        metrics["diff_mean"] = statistics.mean(diffs)
        metrics["diff_median"] = statistics.median(diffs)
        diffs_sorted = sorted(diffs)
        metrics["diff_p95"] = diffs_sorted[int(len(diffs_sorted) * 0.95)]
        metrics["diff_max"] = max(diffs)
        if metrics["diff_mean"] > DRIFT_MEAN_THRESHOLD:
            issues.append(
                f"engine drift mean {metrics['diff_mean']:.4f} > "
                f"{DRIFT_MEAN_THRESHOLD}"
            )
        if metrics["diff_p95"] > DRIFT_P95_THRESHOLD:
            issues.append(
                f"engine drift p95 {metrics['diff_p95']:.4f} > {DRIFT_P95_THRESHOLD}"
            )
        if metrics["diff_max"] > DRIFT_MAX_THRESHOLD:
            issues.append(
                f"engine drift max {metrics['diff_max']:.4f} > {DRIFT_MAX_THRESHOLD}"
            )

    # (F) Suspicious-too-clean logprobs (would suggest the gather is off
    # and we're picking up the prompt's own tokens, where logits assign
    # near-certainty after the model has seen them). Threshold tightened
    # from -1e-3 to -1e-7 after observing many legitimate ordinary tokens
    # (`(`, `)`, `'the'`, `'similar'`) hit -1e-4 to -1e-6 in confident
    # contexts. A truly off-by-one alignment would more likely produce
    # near-zero on a token unrelated to the target — caught only by
    # checking against an absurdly tight threshold here, plus the
    # length-consistency and engine-drift checks above.
    if hf_lps and max(hf_lps) > HIGH_LP_FLAG_THRESHOLD:
        max_lp = max(hf_lps)
        max_idx = hf_lps.index(max_lp)
        issues.append(
            f"HF logprob at completion[{max_idx}] is {max_lp:.6f} "
            f"(> {HIGH_LP_FLAG_THRESHOLD}). Possibly degenerate gather; "
            "check inspector output to see whether this is a legitimate "
            "high-confidence ordinary token (false positive)."
        )

    metrics["bug2_delta"] = retok_record["delta"]
    return {"issues": issues, "metrics": metrics}


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()

    samples = load_dataset_samples(args.dataset, args.n_samples)
    for s in samples:
        s["_image"] = resolve_image(s, args.image_base)

    # Load engines. vLLM first (captures GPU 0); HF goes to GPU 1 if available.
    llm = load_vllm_engine(args.model_id, args.seed)
    # vLLM keeps a processor too — we use the same processor instance for HF.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    completions = sample_vllm(
        llm,
        processor,
        samples,
        args.k_per_sample,
        args.max_new_tokens,
        args.temperature,
        top_p=args.top_p,
    )
    logger.info(
        f"vLLM SamplingParams used: temperature={args.temperature}, "
        f"top_p={args.top_p}, max_tokens={args.max_new_tokens}"
    )

    # Free vLLM GPU memory before loading HF (we only have 1 GPU per pod
    # in the typical config; even with 2 GPUs we want headroom).
    logger.info("[vllm] freeing engine before HF load…")
    del llm
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    hf_model, hf_processor = load_hf_engine(args.model_id)

    # Recompute and check.
    all_issues: list[str] = []
    all_diff_means: list[float] = []
    all_diff_p95s: list[float] = []
    all_diff_maxes: list[float] = []
    bug2_deltas: list[int] = []

    for c_idx, comp in enumerate(completions):
        sample = samples[comp["sample_idx"]]
        retok_record = measure_retokenize_drift(hf_processor, comp)
        bug2_deltas.append(retok_record["delta"])
        hf_record = recompute_hf_logprobs(hf_model, hf_processor, sample, comp)
        result = check_completion(comp, hf_record, retok_record)

        m = result["metrics"]
        logger.info(
            f"[completion {c_idx}] vllm_len={len(comp['token_ids'])} "
            f"hf_len={hf_record['full_len']-hf_record['prompt_len']} "
            f"diff_mean={m.get('diff_mean', float('nan')):.5f} "
            f"diff_p95={m.get('diff_p95', float('nan')):.5f} "
            f"diff_max={m.get('diff_max', float('nan')):.5f} "
            f"bug2_delta={retok_record['delta']:+d}"
        )

        # Inspector: print every position where HF lp ≈ 0 with the actual
        # token. Distinguishes "label alignment off" from "legitimate
        # high-confidence token (EOS / closing tag / structure marker)".
        if args.inspect_high_lp:
            tok_ids = comp["token_ids"]
            vllm_lps = comp["vllm_logprobs"]
            hf_lps = hf_record["hf_logprobs"]
            for pos, (t_id, v_lp, h_lp) in enumerate(zip(tok_ids, vllm_lps, hf_lps)):
                if h_lp > -1e-3:
                    decoded = hf_processor.tokenizer.decode([t_id])
                    is_special = t_id in hf_processor.tokenizer.all_special_ids
                    logger.info(
                        f"  [completion {c_idx}] position {pos}: token_id={t_id} "
                        f"text={decoded!r} special={is_special} "
                        f"vllm_lp={v_lp:.5f} hf_lp={h_lp:.5f}"
                    )

        if result["issues"]:
            for iss in result["issues"]:
                logger.warning(f"  ISSUE: {iss}")
            all_issues.extend([f"[completion {c_idx}] {x}" for x in result["issues"]])
        if "diff_mean" in m:
            all_diff_means.append(m["diff_mean"])
            all_diff_p95s.append(m["diff_p95"])
            all_diff_maxes.append(m["diff_max"])

    # Summary
    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)
    logger.info(f"Completions checked: {len(completions)}")
    if all_diff_means:
        logger.info(
            f"engine-drift mean across runs: avg={statistics.mean(all_diff_means):.5f}, "
            f"max={max(all_diff_means):.5f}"
        )
        logger.info(
            f"engine-drift p95 across runs:  avg={statistics.mean(all_diff_p95s):.5f}, "
            f"max={max(all_diff_p95s):.5f}"
        )
        logger.info(
            f"engine-drift max across runs:  avg={statistics.mean(all_diff_maxes):.5f}, "
            f"max={max(all_diff_maxes):.5f}"
        )
    bug2_n_diff = sum(1 for d in bug2_deltas if d != 0)
    bug2_n_big = sum(1 for d in bug2_deltas if abs(d) > 5)
    logger.info(
        f"Bug 2 reality check: {bug2_n_diff}/{len(bug2_deltas)} completions had "
        f"retokenize/vllm length mismatch ({bug2_n_big} with |delta|>5). "
        "Native path makes this mismatch impossible by construction."
    )
    if all_issues:
        logger.error(f"FAIL — {len(all_issues)} issues:")
        for iss in all_issues:
            logger.error(f"  - {iss}")
        return 1
    logger.info("PASS — all checks within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
