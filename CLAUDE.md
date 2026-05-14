# VLM Self-Reflection GRPO

Multi-turn GRPO pipeline for Qwen2.5-VL-7B with a critique-then-refine
loop. Three-turn rollout (A1 → F1 → A2), two-reward design (response vs
feedback), shared LoRA adapter, vLLM colocate with sleep mode.

**This file is the operational handbook.** For deeper material:
- [`docs/architecture.md`](docs/architecture.md) — data, prompts, repo layout
- [`docs/training.md`](docs/training.md) — algorithm, KL, loss, references, literature audit
- [`docs/rewards.md`](docs/rewards.md) — per-component reward walkthrough
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — diagnostics
- [`experiments.md`](experiments.md) — experiment log

## Current goal

Validate the PAG architectural pivot (selective revision + per-segment
rewards, arXiv:2506.10406) on top of the post-audit codebase. Both active
runs warm-start from `baseline-a1` ckpt-1000 (61.4% BLINK avg, +5.3pp over
base) and KL-anchor against the same checkpoint via the shared-base
`kl_ref` LoRA adapter. Success = exceed the baseline-A1 BLINK score on
LIVR-v2 → BLINK transfer.

The 2026-05 literature audit (see [`docs/training.md`](docs/training.md))
found that same-model in-loop self-critique under RL has **no positive
precedent** at our scale for visual reasoning. The architectural pivots
(Critic-V, DPO-over-correction-pairs, PRM, external judge, two-stage SFT
critic) remain open follow-ups if PAG-faithful underperforms.

## Active training runs

| Job | YAML | Distinguishing config |
|---|---|---|
| `pag-1a-k8` | `job-qwen-grpo-livr-v2-9k-pag-faithful.yaml` | Single shared adapter (A1/F1/A2). Selective revision gate. PAG per-segment binary rewards (0.9 corr + 0.1 fmt). `pag_shaping_alpha=1.0`. SFT init from baseline-a1 ckpt-1000. |
| `pag-1a-k8-base` | `job-qwen-grpo-livr-v2-9k-pag-1a-k8-base.yaml` | A/B variant of `pag-1a-k8` — base-Qwen init, otherwise identical. |
| `kl-fix-no-bonus` | `job-qwen-grpo-livr-v2-9k-two-adapters-kl-fix-no-bonus.yaml` | Two-adapter routing (response + feedback). Rescaled rewards + GDPO normalisation. A1 frozen via `W_A1_correctness=0`. r=64 α=128 on both adapters. |

Shared across all three: K=8 (PAG) / K=12 (kl-fix), batch=2, grad_acc=8,
LR=1e-5 with 56-step warmup, vanilla GRPO loss, symmetric PPO clip 0.2,
uniform per-turn KL coefficients (220), LoRA r=64 α=128, completions
capped at 384 tokens, `feedback_temp=1.0`, `a2_temp=0.7`,
`VLLM_GPU_MEM=0.30`, vLLM colocate with sleep mode.

## Recent fixes (last 30 days)

| Commit | Change | Impact |
|---|---|---|
| `c825376` | **Bug #1** A2 PPO ratio temperature divisor; **Bug #3** `reward/a2_mean` denominator under PAG gating | Fixes silent gradient attenuation + headline metric dilution |
| `5d3ad9e` | **KL ref refactor**: `--ref_model_init_from_checkpoint` now loads a frozen `kl_ref` LoRA adapter on the policy (not a duplicate 7B base) | −15 GB / rank, restored batch=2 + 384 working config |
| `5d3ad9e` | **Bug B**: `max_num_batched_tokens` parameterised + sentinel counter on `0.0` logprob fallback | Eliminates chunked-prefill PPO-ratio bias |
| `021f57a` | **vLLM cumem PR #40812 backport** for 0.12.x — `expandable_segments` coexistence with sleep mode | Sleep-mode physical pages released cleanly; no step-30 fragmentation OOM |
| `8b169a8` | **Bug A** strict-everywhere: F1 verification reward uses `strict=True` A1 truth (same as response head). New `sr/strict_liberal_a1_disagree_rate` wandb metric | Eliminates contradictory gradients on untagged trajectories |

Detail in commit messages and `docs/training.md` "Bug-fix audit notes".

## How to run

```bash
# Install
uv sync
uv sync --extra dev          # for testing

# Lint
ruff format src/ tests/ train_self_reflection.py scripts/
ruff check  src/ tests/ train_self_reflection.py scripts/

# Tests (~490 cases)
uv run pytest tests/ -v

# Training (4× A100, Qwen2.5-VL base)
accelerate launch --config_file k8s/multi_gpu.yaml --num_processes=4 \
    train_self_reflection.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2vl \
    --dataset_path /outputs/livr_v2_train_snapshots/snap_20260501_154345_final/data/livr_v2_9task_train.jsonl \
    --output_dir /outputs/grpo_qwen_livr_v2_9k_<run_name> \
    --image_base_dir /outputs/image_base \
    --use_vllm --freeze_vision_tower
```

For active-run deployment, use the YAMLs above with `kubectl apply -f
<yaml>`. Each YAML pip-installs `vllm==0.12.0` at pod start; the cumem
PR #40812 backport (in `src/vlm_grpo/_vllm_cumem_patch.py`) is applied
inside `VLLMRolloutEngine.__init__` before the first `LLM()` construction.

## Cluster operations

**Helper pod**: all `kubectl exec` for reading logs, listing checkpoints
on `/outputs/`, or running scripts against PVC contents must use the
helper pod `vlm-jupyter-eval2` — **not** the training-job pods (which
don't have all the mounts you'd expect). Redeploy if the pod is in
`Completed` or `Failed`:

```bash
kubectl delete pod vlm-jupyter-eval2 --ignore-not-found
kubectl apply -f /Users/kuntalkokate/svcl-projects/vlm-self-reflection/k8s/jupyter-1gpu-test.yaml
kubectl wait --for=condition=Ready pod/vlm-jupyter-eval2 --timeout=120s
```

## Experiment logging

After each experiment, add an entry to `experiments.md`:

1. Final metrics (early 10% vs late 10%): A1/A2 acc, WR, RW, entropy,
   rewards, F1 tag-leak rate, gated_rate, strict/liberal divergence.
2. What worked / what failed / root cause (1–2 sentences each).
3. Artifacts: output dir, wandb run, k8s YAML, commit.
4. Concise — no config tables if same as prior experiment (just note diffs).

## Coding conventions (MUST follow)

1. **Imports**: stdlib first, third-party next, local last. Alphabetised
   within groups.
2. **Type hints**: required on all function parameters and return types.
3. **Docstrings**: Google-style with Args/Returns sections.
4. **Naming**: `snake_case` functions/variables, `PascalCase` classes,
   `UPPER_SNAKE_CASE` constants.
5. **Logging**: `logging.basicConfig()` and
   `logger = logging.getLogger(__name__)`.
6. **Entry points**: every script ends with
   `if __name__ == "__main__": main()`.
7. **Dataclasses**: `@dataclass` for structured results with `to_dict()`.

## MUST NOT change

1. **Lazy imports in model classes**: do NOT move imports from `__init__`
   methods to module level in the model wrapper classes (vllm rollout
   engine, etc.) — vLLM's import side-effects depend on the order.

## Known gaps

- Multi-view_Reasoning + Visual_Correspondence on BLINK score near random
  — these tasks aren't in LIVR training (Multi-view) or use a
  denser-marker composite (Correspondence) that diverges from LIVR's
  Source/Target layout.
- BLINK IQ_Test under `vlmevalkit` exact-matching is unscorable for
  verbose base predictions; needs the GPT judge enabled to extract a
  letter from prose.
