# Architecture

How the multi-turn rollout, data, prompts, and codebase are organised. Read
this when you need to understand the *shape* of the pipeline — what each turn
sees, how images get to vLLM, where the code lives. For the actual training
math (loss, KL, advantage), see [`training.md`](training.md). For the per-
component reward landscape, see [`rewards.md`](rewards.md).

## Data

### GRPO training set — LIVR-v2 9K (active)
- **Path**: `/outputs/livr_v2_train_snapshots/snap_20260501_154345_final/data/livr_v2_9task_train.jsonl`
- **9 tasks (1000 samples each)**: Counting, Jigsaw, ObjLoc, ArtStyle,
  RelReflect, VisCorr, SemCorr, FunCorr, VisSim
- All MCQ. Ground truth in `(A)` format. `answer_type=mcq`, deterministic
  matching (`VLM_USE_LLM_JUDGE=0`).
- **Image format**: a SINGLE pre-composited PNG per sample with all
  candidate sub-images + region labels drawn in. The composing is done
  offline rather than at inference time (LIVR paper composes inline).
  Driven by `scripts/data/livr_v2/livr_common.py`
  (`create_reference_and_options`, `create_side_by_side`,
  `create_grid_image`).
- Curriculum subset
  `/outputs/livr_v2/data/livr_v2_9k_curriculum_filtered.jsonl` is the
  difficulty-filtered variant (see `scripts/data/profile_difficulty_a1.py`
  → `scripts/data/filter_by_difficulty.py`).

### Required input fields
`question`, `ground_truth`, `answer_type`, `images` (array — first entry is
the image path). The `messages` field that ships in some legacy JSONLs is
**ignored** by GRPO — it carries an SFT-format system prompt that does NOT
match the prompts we use. `data.py` only falls back to parsing `messages`
when `question` is not present as a top-level field.

## Pattern-A self-reflection (all three turns)

A1, F1, A2 all use a **single user message** with **no custom system
prompt** — Qwen's default `"You are a helpful assistant."` system prompt
fires from the chat template. The candidate answer A1 and the feedback F1
are quoted as text **inside the next user message**, not chained as
assistant/user turns. Code: `src/vlm_grpo/prompts.py`.

Why: matches LLaVA-Critic-R1 (arXiv:2509.00676), Critique-GRPO
(arXiv:2506.03106), CriticGPT, and Volcano — none use role-flipping for
same-model critique. The role-flipped variant we used earlier (assistant
gets image + question, user gets A1) is deprecated; see git history if you
need to revive it.

| Turn | Builder | What it gets | Expected output |
|---|---|---|---|
| A1 | `build_initial_answer_prompt(question)` | `[image] + question + tag instruction` | `<think>…</think><answer>(X)</answer>` |
| F1 | `build_critic_prompt(question, answer1)` | `[image] + Q + "Candidate answer: <full A1 text>" + verifier instruction` | `<think>…</think> \boxed{CORRECT\|INCORRECT}` |
| A2 | `build_refiner_prompt(question, answer1, feedback1)` | `[image] + Q + "Your previous answer: <A1>" + "Feedback: <F1>" + tag instruction` | `<think>…</think><answer>(X)</answer>` |

F1 sees the **raw full A1 text** — every byte the policy emitted, tags and
all. No tag-stripping happens on the prompt construction side. F1 makes its
content judgement; the reward calls A1 right or wrong via strict-tag
extraction (see [`rewards.md`](rewards.md)).

## Output format (think + answer / boxed)

Tag-format prompting is on **unconditionally** — the `--use_think_answer_tags`
/ `--use_answer_tag_only` toggles were removed in commit `c825376`.

- A1/A2 user message ends with the `THINK_ANSWER_INSTRUCTION` env var
  (set in the YAML). Expected: `<think>reasoning</think><answer>(A)</answer>`.
- F1 user message ends with `F1_VERIFIER_INSTRUCTION`. Expected:
  `<think>reasoning</think> \boxed{CORRECT|INCORRECT}`. F1 uses
  `\boxed{}` rather than `<answer>` because verdict ≠ answer.
- Format rewards (`R_a1_format`, `R_a2_format`, `R_fb_format`) are all
  binary {0, +1}. +1 only when both tag pairs are present AND the inner
  content is a clean atomic answer (or `CORRECT`/`INCORRECT` for F1).
  No partial credit.

## Rollout pipeline

1. **A1 rollout**: vLLM samples K trajectories at `temperature=1.0` from
   `build_initial_answer_prompt`. Image preprocessed by vLLM's
   `mm_processor` with `min_pixels=200704, max_pixels=401408`.
2. **F1 rollout**: per A1, build `build_critic_prompt(question, a1_text)`.
   Sample at `temperature=1.0`. Extract `\boxed{CORRECT|INCORRECT}`.
3. **A2 rollout**: per A1+F1, build `build_refiner_prompt(…)`. Sample at
   `temperature=0.7` (sharper, since A2 is a refinement). **Selective
   revision gate** (`--use_selective_revision`): if F1 emits
   `\boxed{CORRECT}`, A2 is skipped — the breakdown records
   `gated=True, r_a2=None`.
4. Rewards computed per trajectory in `compute_pag_response_breakdown` +
   `compute_pag_feedback_breakdown` (PAG path) or
   `compute_response_reward_breakdown` + `compute_feedback_reward_breakdown`
   (non-PAG raw / `_01` rescaled paths). See [`rewards.md`](rewards.md).

vLLM is in sleep mode between rollouts so its KV cache + weights are released
back to the cumem pool while training runs the HF backward pass. See
[`training.md`](training.md) for the sleep/wake lifecycle and the vLLM cumem
fragmentation backport that keeps physical pages from leaking.

## Repository layout

```
.
├── train_self_reflection.py            # Main training entry point
├── pyproject.toml                      # uv + ruff/mypy/pytest config
├── uv.lock                             # locked deps (vllm pinned via YAML pip-install)
├── CLAUDE.md                           # operational handbook (this file)
├── experiments.md                      # experiment log (v1 → latest)
│
├── docs/
│   ├── architecture.md                 # this file
│   ├── training.md                     # algorithm, KL, loss, refs
│   ├── rewards.md                      # per-component reward walkthrough
│   └── troubleshooting.md              # diagnostics (LoRA probe, KL, fragmentation, etc.)
│
├── src/vlm_grpo/                       # installable package
│   ├── config.py                       # SelfReflectionConfig + reward weight dataclasses
│   ├── critic_grpo.py                  # SelfReflectionGRPOTrainer (GRPO loop, KL, policy update)
│   ├── data.py                         # dataset loading + answer_type detection
│   ├── prompts.py                      # A1 / F1 / A2 prompt builders
│   ├── rollout.py                      # HF generate() rollout + metrics aggregation
│   ├── vllm_rollout.py                 # vLLM rollout engine (sleep mode + cumem patch hook)
│   ├── trajectory.py                   # tag parsing, MCQ letter normalisation
│   ├── multi_adapter.py                # per-turn LoRA routing config
│   ├── utils.py                        # seeding, env setup, normalised edit distance
│   ├── _vllm_cumem_patch.py            # runtime backport of vLLM PR #40812 for 0.12.x
│   └── rewards/
│       ├── composition.py              # response + feedback breakdowns (raw, _01, PAG)
│       ├── correctness.py              # binary / continuous correctness rewards
│       ├── deterministic.py            # MCQ / yesno / numeric matchers
│       ├── judge_llm.py                # optional LLM judge (VLM_USE_LLM_JUDGE=1)
│       └── verifier.py                 # top-level verify_answer dispatcher
│
├── tests/                              # pytest suite — ~490 tests, see `pytest tests/ -v`
│
├── scripts/
│   ├── analysis/                       # difficulty bucketing
│   ├── data/                           # LIVR-v2 builders + BLINK compositing
│   └── verify/                         # numerical-gradient + vLLM↔HF equivalence harnesses
│
└── k8s/                                # active training YAMLs + eval shards + helper pod
    ├── job-qwen-grpo-livr-v2-9k-pag-faithful.yaml          # active: PAG single-adapter
    ├── job-qwen-grpo-livr-v2-9k-pag-1a-k8-base.yaml        # A/B: same but base-Qwen init
    ├── job-qwen-grpo-livr-v2-9k-two-adapters-kl-fix-no-bonus.yaml  # active: two-adapter rescaled+GDPO
    ├── job-build-livr-v2*.yaml                              # dataset builders
    ├── job-eval-vlmevalkit-blink-*.yaml                     # BLINK PatternA eval shards
    ├── jupyter-1gpu-test.yaml                               # helper pod (PVC reads)
    ├── multi_gpu.yaml / deepspeed_zero3.yaml                # accelerate configs
    └── ds_zero3_config.json                                 # DeepSpeed engine config
```
