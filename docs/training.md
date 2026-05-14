# Training Algorithm

Math and implementation notes for the GRPO trainer. For data flow, prompts,
and code layout, see [`architecture.md`](architecture.md). For per-component
reward derivations, see [`rewards.md`](rewards.md).

## Two-reward GRPO

Each trajectory carries **two separate reward scalars**, driving two
gradients into the same policy:

- `r_response` flows through the A1 + A2 log-probs. SCoRe¹-style outcome
  reward; weights and components in [`rewards.md`](rewards.md).
- `r_feedback` flows through the F1 log-prob only. Critique-GRPO² style
  separate verification head.

Both losses backprop into the same shared LoRA weights. The PAG variant
splits the response side further into per-segment `r_a1`, `r_a2` so the
A1 and A2 turns get independent K-group baselines and their own α-shaped
transition signal. See the PAG section below.

## KL term

**Estimator**: Schulman k3⁸ — `exp(ref_lp − new_lp) − (ref_lp − new_lp) − 1`.
Unbiased, non-negative, ratio-friendly.

**Aggregation**: `sum / max_completion_length` (Dr. GRPO⁷ convention).
This avoids the low-variance bias of dividing by actual completion length,
which would underweight short completions.

**Per-turn coefficients**: `a1_kl_coeff`, `a2_kl_coeff`, `fb_kl_coeff`,
each multiplied by the global `kl_coeff` (default 0.001). The active runs
use **uniform** `220` across all three turns. The older 2500× anchor on
A1 was tuned for an anchored-A1-only regime and over-suppressed A1 updates
when A1 was actively trained.

**Reference distribution**: a frozen LoRA adapter on the policy model,
loaded via `--ref_model_init_from_checkpoint <ckpt>` → `model.load_adapter(
ckpt, "kl_ref", is_trainable=False)`. The trainer's ref-forward block
temporarily swaps the active adapter to `kl_ref`, runs the forward under
`torch.no_grad()`, and restores the previous active adapter on exit.
Per-rank memory: ~50 MB for the adapter vs ~15 GB for the prior
duplicate-base approach. When `--ref_model_init_from_checkpoint` is empty,
falls back to `disable_adapter_layers()` (anchors KL against raw base —
correct only when the policy adapter is also a fresh init).

Both active runs use `baseline-a1` ckpt-1000 (61.4% BLINK avg) as both the
policy init AND the KL ref → step-0 KL is exactly 0.

## Policy update

**Loss**: `--loss_type grpo` (vanilla GRPO³) in both active runs. Dr. GRPO⁷
(`--loss_type dr_grpo`) removes std normalisation but isn't exercised.

**Advantages**: per-K-group `(reward − group_mean) / group_std`. PAG splits
this into per-segment baselines — `r_a1` over all K, `r_a2` over the
non-gated subset only.

**PPO clipping**: symmetric `--clip_range 0.2`, `--clip_high 0.2`. DAPO's
asymmetric clip-higher (0.28 upper) is available via `--clip_high` but **off**
in both active runs.

**Per-turn temperature divisor** (Bug #1 fix, commit `c825376`): vLLM samples
A2 at `T=0.7` and returns `log softmax(logits / 0.7)`. The HF forward pass
divides `shift_logits` by the per-turn `sampling_temperature` before
`log_softmax`, so `exp(new_lp − old_lp) = 1.0` at step 0. A1/F1 are
no-ops (`T=1.0`). Verified by `tests/test_temperature_consistency.py`
and `scripts/verify/verify_temperature_consistency.py`.

**Empty completions** (immediate-EOS trajectories): emit empty tensors for
both `old_lp` and `new_lp`. The `sum / max_completion_length` aggregation
cleanly evaluates to 0. Earlier code used a `[0.0]` sentinel that injected
a spurious `A / max_len` term — fixed in commit `95f091b`.

## PAG segment rewards (active path)

`compute_pag_response_breakdown` (arXiv:2506.10406) emits a
`PAGSegmentRewardBreakdown` with `r_a1` and `r_a2` as **separate scalars**
rather than a pooled response reward. Binary {0, 1} components per the
released PAG implementation:

```
r_a1 = w_a1_corr · R_a1_corr_01 + w_a1_fmt · R_a1_fmt_01
r_a2 = w_a2_corr · R_a2_corr_01 + w_a2_fmt · R_a2_fmt_01
                + α · (R_a2_corr_01 − R_a1_corr_01)
```

The α shaping bonus is added **to A2 only** and uses the *raw binary
accuracies*, not the weighted rewards. Paper default `α = 1.0`
(`pag_shaping_alpha=1.0`, `rs_coef=1`). With weights 0.9/0.1, `r_a2`
reaches `1.0 + α = 2.0` on a WR transition and dips to `−α = −1.0` on a
RW.

**Selective revision gate** (`--use_selective_revision`): when F1 emits
`\boxed{CORRECT}`, A2 is skipped. The breakdown sets `r_a2=None` and
`gated=True`. The trainer's PAG branch excludes gated trajectories from
the A2 K-group baseline and contributes 0 A2 policy loss for them.

`compute_pag_feedback_breakdown` emits binary {0, 1} verification + format
rewards. PAG has no downstream component (γ=0 turn-independent), so
`w_downstream` is ignored on this path.

## Other reward modes

- **Rescaled rewards** (`--use_rescaled_rewards`): each raw component is
  rescaled to [0, 1] via `_to_unit(raw, lo, hi)` before weighting.
  Per-unit-weight gradient magnitude is equalised across components and
  `total_reward ∈ [0, 1]` for convex weights. Used by `kl-fix-no-bonus`.
- **Single-turn A1 baseline** (`--single_turn_a1`): F1+A2 skipped;
  reward = `0.9·R_a1_correct_01 + 0.1·R_a1_format_01`. Code path retained
  for the baseline-A1 run that produced ckpt-1000.

## GDPO normalisation

`--use_gdpo_normalization` (Liu 2026, arXiv:2601.05242): per-component
K-group advantage normalisation, then weighted sum, then batch-renormalise.
Equalises per-component gradient contribution. Used by `kl-fix-no-bonus`.
The PAG run uses single-adapter routing for role-isolation at the parameter
level and leaves GDPO off.

## DAPO (off in active runs)

`--use_dynamic_sampling`: drop zero-variance K-groups (advantage = 0,
gradient = 0). `--clip_high <upper>`: asymmetric clip. Paper defaults
(arXiv:2503.14476): clip-higher 0.28, dynamic sampling on. Both active
runs leave them off.

## Bug-fix audit notes

- **Bug #1** — A2 PPO temperature mismatch (commit `c825376`): see "Per-turn
  temperature divisor" above.
- **Bug #3** — `reward/a2_mean` denominator (commit `c825376`): under PAG
  selective revision, gated trajectories carried `r_a2=0.0` placeholders.
  Fixed by denominating over `a2_active_mask.sum()` (PAG) or the full
  tensor (non-PAG); cross-rank reduce sums `(a2_sum_local, a2_count_local)`
  separately rather than averaging per-rank means. Verified by
  `tests/test_a2_reward_mean_dilution.py`.
- **Bug A** — strict-vs-liberal A1 truth asymmetry between response and
  feedback breakdowns (commit `8b169a8`): both heads now use `strict=True`
  for the A1-correctness target. F1's verification reward is anchored to
  the same strict-tag A1 truth the response head sees. The new
  `sr/strict_liberal_a1_disagree_rate` wandb metric tracks how often the
  two criteria disagreed; useful for monitoring whether the policy
  internalises the tag contract.
- **Bug B** — vLLM `0.0` logprob sentinel (commit `5d3ad9e`):
  `vllm_rollout.py` parameterised `max_num_batched_tokens` (default
  `max_model_len`, was hardcoded 4096). Added a per-call counter +
  `logger.warning` when the fallback fires.
- **KL ref memory regression** (commit `5d3ad9e`): replaced
  duplicate-base ref model with frozen LoRA adapter on the policy. ~15 GB
  saved per rank.
- **vLLM cumem fragmentation** (commit `021f57a`): runtime backport of
  vLLM PR #40812 in `src/vlm_grpo/_vllm_cumem_patch.py`. Lets PyTorch's
  expandable_segments allocator coexist with the cumem sleep pool on
  vLLM 0.12.x without a CUDA-13 image rebuild.

## References

1. **SCoRe** — Kumar et al., "Training Language Models to Self-Correct via
   Reinforcement Learning" (2024). [arXiv:2409.12917](https://arxiv.org/abs/2409.12917)
   → Shaped reward `α·(r_a2 − r_a1)`, Stage I first-turn KL anchor.
2. **Critique-GRPO** — Zhang et al., "Critique-GRPO" (2025).
   [arXiv:2506.03106](https://arxiv.org/abs/2506.03106)
   → Two-reward design, downstream-aware feedback reward.
3. **GRPO** — Shao et al., "DeepSeekMath" (2024).
   [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
   → Group Relative Policy Optimization.
4. **DeepSeek-R1** — Guo et al. (2025).
   [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
   → `<think>/<answer>` format reward, β=0.001 KL anchor.
5. **PAG** — "Pessimistic Advantage Grouping" (2025).
   [arXiv:2506.10406](https://arxiv.org/abs/2506.10406)
   → Per-segment K-group baselines, α-shaped transition reward, selective
   revision gate.
6. **Self-Refine** — Madaan et al. (2023).
   [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
   → Over-refinement failure mode; stability when already correct.
7. **Dr. GRPO** — Liu et al. (2025).
   [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
   → Remove std normalisation, fixes low-variance-group bias.
8. **Schulman k3 KL estimator** — Schulman (2020).
   [joschu.net/blog/kl-approx](http://joschu.net/blog/kl-approx.html)
9. **DAPO** — "DAPO" (2025).
   [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
   → Clip-higher asymmetric clipping; dynamic K-group filtering.

## Literature audit (2026-05) — the same-model-self-critique question

A multi-agent audit reviewed whether **same-model in-loop self-critique
under RL** has positive precedent at our scale for visual reasoning. Headline:

> The conceptual frame — a single 7B VLM, no external verifier, in-loop
> critique-then-refine trained jointly with RL — is **not supported by
> the 2024–2026 literature for visual reasoning**.

Every comparable paper has an asymmetry we don't: SCoRe has no critique
step; Critique-GRPO uses GPT-4o as the critic; CriticGPT and LLaVA-Critic
use separate trained critics; LLaVA-Critic-R1 applies self-critique at
test time only.

The "honest F1 hurts A2" pattern matches the canonical
**generation–verification-gap** failure mode (Huang 2023 arXiv:2310.01798;
Stechly & Kambhampati 2024 arXiv:2402.08115; Kambhampati survey
arXiv:2406.01297). Recommended architectural pivots with published
positive evidence at our scale:

1. **Critic-V** (Zhang CVPR 2025, arXiv:2411.18203) — separate DPO-trained
   VLM critic + vanilla GRPO reasoner. Most directly applicable.
2. **DPO over (A1_wrong, A2_corrected) pairs** — drop F1 entirely.
3. **Process Reward Models** (PROPA arXiv:2511.10279, VRPRM arXiv:2508.03556).
4. **External LLM judge as F1** — GPT-4o / Claude-class critic.
5. **Two-stage SFT critic → RL** — supervise F1 first, then RL A1+A2.

`pag-faithful` is our first pivot test on this framework (selective
revision + per-segment rewards on the post-audit codebase). The
literature consensus is that the fix is **architectural**, not in the
reward weights — no amount of α / weight tuning closes the gap our 7B
same-model setup produces.
