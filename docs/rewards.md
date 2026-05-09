# GRPO Reward Structure

This doc tracks how every reward component is computed for the
self-reflection GRPO pipeline. All values are taken from
`src/vlm_grpo/rewards/composition.py` and `src/vlm_grpo/trajectory.py`;
walkthrough numbers are reproduced by `uv run python -c "from vlm_grpo..."`.

There are three reward modes in the code:

- **Multi-turn (raw)** — `compute_response_reward_breakdown` /
  `compute_feedback_reward_breakdown`. Components in their natural ranges
  (a1/a2 corr ±1, no_regression up to ±3, downstream up to ±3 or ±2α). All
  walkthrough tables below describe this mode.
- **Multi-turn (rescaled)** — `compute_response_reward_breakdown_01` /
  `compute_feedback_reward_breakdown_01`, enabled via
  `--use_rescaled_rewards`. Each component is mapped to [0, 1] before
  weighting so per-unit-weight gradient magnitude is equalised. Used by all
  recent `frozen-a1-mt-*` runs except `frozen-a1-mt-full-raw`. See
  [Rescaled mode](#rescaled-mode-use_rescaled_rewards).
- **Single-turn A1 baseline** — `compute_baseline_a1_reward_breakdown`,
  enabled via `--single_turn_a1`. F1 + A2 are skipped, reward is
  `0.9·R_a1_correct_01 + 0.1·R_a1_format_01` ∈ [0, 1]. Used by
  `job-qwen-grpo-livr-v2-9k-baseline-a1.yaml`.

## Two-trajectory design

Each sample produces three model outputs:

| Turn | Role | Output shape | Drives policy update for |
|---|---|---|---|
| **A1** | initial answer | `<think>...</think><answer>(X)</answer>` | A1 log-probs (via response reward) |
| **F1** | verification | `<think>...</think> \boxed{CORRECT|INCORRECT}` | F1 log-probs (via feedback reward) |
| **A2** | refined answer | `<think>...</think><answer>(X)</answer>` | A2 log-probs (via response reward) |

**Two separate reward signals** are computed per trajectory:
- **Response reward** → drives A1 + A2 update
- **Feedback reward** → drives F1 update

Implementation:
- `compute_response_reward_breakdown(...)` in composition.py
- `compute_feedback_reward_breakdown(...)` in composition.py

## Three guarantees the reward structure enforces

1. **Extraction depends only on `<answer>` / `\boxed{}` tags.** No tag → no answer → wrong (handled naturally by `extract_answer_from_text(strict=True)` returning `""`).
2. **`<think>` tags are only relevant to format reward.** They never affect correctness extraction.
3. **No short-circuit overrides.** Format compliance is binary `{0, +1}`. Correctness fails naturally when extraction fails. The reward is monotonic: missing tags ≤ tagged outcomes.

---

## Response reward components

```
r_response = w_a1_corr · r_a1_correctness
           + w_a1_fmt  · r_a1_format
           + w_a2_corr · r_a2_correctness
           + w_a2_fmt  · r_a2_format
           + w_nor     · r_no_regression
```

**Code defaults** (`ResponseRewardWeights`): per-turn sub-reward split is
0.9·corr + 0.1·fmt for both A1 and A2, turn weight 0.3 each, no_regression 0.4:

| weight | code default |
|---|---:|
| `w_a1_correctness` | 0.27 |
| `w_a1_format` | 0.03 |
| `w_a2_correctness` | 0.27 |
| `w_a2_format` | 0.03 |
| `w_no_regression` | 0.40 |
| **sum** | **1.00** |

**Active YAML overrides** (all `frozen-a1-mt-*` runs freeze A1 → zero
A1 weights and shift mass to A2 + no_regression):

| weight | full (`mt`, `r01`, `full`, `full-raw`) | simplified (`full-simple`, `gdpo`, `gdpo-warmup`, `vanilla-warmup`, `vanilla-tokid`) |
|---|---:|---:|
| `w_a1_correctness` | 0.00 | 0.00 |
| `w_a1_format` | 0.00 | 0.00 |
| `w_a2_correctness` | 0.45 | 0.45 |
| `w_a2_format` | 0.05 | 0.05 |
| `w_no_regression` | 0.50 | 0.00 |
| **sum** | **1.00** | **0.50** (intentional — `_validate_weight_sum` warning fires) |

### Turn decoupling: `separate_turn_loss=True` (SCoRe convention)

When `separate_turn_loss` is on, the response reward is **not** a single
scalar applied to both A1 and A2 log-probs. It's split:

```python
# critic_grpo.py
a1_reward = weighted(a1_correctness) + weighted(a1_format)
a2_reward = weighted(a2_correctness) + weighted(a2_format) + weighted(no_regression)
```

Separate group-advantages are computed for each turn; A1 token gradients
only see `a1_advantages`, A2 tokens only see `a2_advantages`. Matches
SCoRe Eq. 4 (arXiv:2409.12917): the `α·(r_a2 − r_a1)` shaping bonus
(which we call `no_regression`) flows only to A2's gradient.

### `a1_correctness` ∈ {−1, +1}

```python
a1_result = verify_answer(a1_text, ground_truth, answer_type, strict=tag_mode)
r_a1 = 1.0 if a1_result.is_correct else -1.0
```

`strict=True` requires the `<answer>` tag and accepts only atomic answers (or
`(X) descriptor` — leading letter at position 0).

| A1 text | Inner | Extracted | GT | Reward |
|---|---|---|---|---|
| `<think>r</think><answer>(A)</answer>` | `(A)` | `A` | `A` | **+1** |
| `<think>r</think><answer>(A) Different style</answer>` | `(A) Different style` | `A` | `A` | **+1** (LIVR option text recovered) |
| `<think>r</think><answer>(B) is wrong, (A) is right</answer>` | mid-prose | `B` | `A` | **−1** (mismatch via leading letter) |
| `Just plain (A)` | (no tag) | `""` | `A` | **−1** (no `<answer>` → no extraction) |

### `a2_correctness` ∈ [−1, +1]

Same logic as `a1_correctness` but applied to A2. For deterministic types
(MCQ/yesno/numeric) it's binary `±1`. For counting/open it's continuous
`2·score − 1` based on the verifier's similarity score.

### `no_regression` (transition reward)

```python
# reward_shaping_alpha=0 (transition table):
if answer_type in DETERMINISTIC_TYPES:
    if a1_correct:     r_nor = +1 if a2_correct else -2  # RR=+1 RW=-2
    else:              r_nor = +3 if a2_correct else  0  # WR=+3 WW=0
else:  # open/counting
    if a1_correct:     r_nor = +1 if a2_correct else -3
    else:              r_nor = +2 if a2_correct else  0

# reward_shaping_alpha > 0 (SCoRe-style shaped, used when α env > 0):
r_nor = α · (r_a2 − r_a1)
```

Asymmetric: WR (correction) > RR (stability) > WW (no change) > RW (regression).
Encourages the model to fix wrong A1s, not to rewrite correct ones.

### `a2_format` ∈ {0, +1}  (binary, structural + clean inner)

```python
def _compute_tag_format_reward(a2_text, answer_type, ...):
    if not has_think_answer_tags(a2_text):
        return 0.0
    inner = extract_from_answer_tags(a2_text).strip()
    if not _is_clean_atomic_answer(inner, answer_type):
        return 0.0
    return 1.0
```

`_is_clean_atomic_answer`:
- **MCQ**: `(A)` or `A` or `A.` (no extra content)
- **counting**: pure integer (`6`, `42`)
- **numeric**: parseable float (`3.14`, `1/2`, `50%`)
- **yesno**: `yes` or `no` (with optional `.,;:`)
- **open**: any non-empty content

| A2 text | format reward |
|---|---|
| `<think>r</think><answer>(A)</answer>` | **+1** (both tags + clean `(A)`) |
| `<think>r</think><answer>A</answer>` | **+1** (bare letter still atomic) |
| `<think>r</think><answer>(A) Different style</answer>` | **0** (extra descriptor — correctness still works) |
| `<answer>(A)</answer>` (no `<think>`) | **0** (think tag missing) |
| `Plain prose with (A)` | **0** (no tags at all) |
| `<think>r</think><answer></answer>` | **0** (empty inner) |

Format reward is **independent** of correctness. A clean `(B)` when GT is `A`
gets `format=+1, a2_correctness=−1`. Correctness handles the wrongness;
format just rewards compliance.

---

## Feedback reward components

```
r_feedback = w_ds · r_downstream_gated
           + w_ver · r_verification
           + w_fmt · r_format
```

**Code defaults** (`FeedbackRewardWeights`): `w_ds=0.45, w_ver=0.45,
w_fmt=0.1` (sum = 1.0). Active YAML overrides:

| weight | full | simplified |
|---|---:|---:|
| `w_downstream` | 0.45 | 0.00 |
| `w_verification_accuracy` | 0.45 | 0.45 |
| `w_fb_format` | 0.10 | 0.05 |
| **sum** | **1.00** | **0.50** (warning fires; intentional) |

### `verification` ∈ {−1, +1}

```python
def compute_verification_accuracy_reward(feedback_text, a1_is_correct):
    boxed = extract_from_boxed(feedback_text).upper()
    if boxed == "INCORRECT":  return 1.0 if not a1_is_correct else -1.0
    if boxed == "CORRECT":    return 1.0 if a1_is_correct else -1.0
    return -1.0  # missing or unparseable boxed verdict → wrong
```

Verdict is extracted **only** from `\boxed{...}`. No keyword fallback on
`<think>` prose (that path was a noise source).

| F1 text | A1 truth | Verification |
|---|---|---|
| `<think>r</think> \boxed{CORRECT}` | A1 right | **+1** (calibrated) |
| `<think>r</think> \boxed{INCORRECT}` | A1 wrong | **+1** (calibrated) |
| `<think>r</think> \boxed{CORRECT}` | A1 wrong | **−1** (sycophantic) |
| `<think>r</think> \boxed{Correct}` | A1 right | **+1** (case-insensitive `.upper()`) |
| `<think>r</think> \boxed{(A)}` | any | **−1** (not a verdict) |
| `Plain INCORRECT prose` | any | **−1** (no `\boxed{}`) |
| `\boxed{INCORRECT}` (no `<think>`) | A1 wrong | **+1** (verdict still extractable) |

### `downstream` (asymmetric gate by verification)

```python
# Shaped reward (active runs use α=1; α=5 was the v10 setting):
r_a1 = 1.0 if a1_correct else -1.0
r_a2 = 1.0 if a2_correct else -1.0
r_downstream = r_a2 + α · (r_a2 − r_a1)

# Asymmetric gate:
#   - Calibrated verdict  (verification > 0): full bidirectional signal
#   - Miscalibrated       (verification ≤ 0): only NEGATIVE downstream flows
if r_verification > 0:
    r_downstream_gated = r_downstream
else:
    r_downstream_gated = min(r_downstream, 0.0)
```

α=1 is the active value across all `frozen-a1-mt-*` runs (SCoRe paper
default). α=5 is shown in some older walkthroughs and produces a wider
±(1+2α)=±11 envelope; α=1 produces ±3.

Two properties the asymmetric gate enforces:

1. **Sycophancy prevention (positive side)** — `\boxed{CORRECT}` on a
   wrong A1 cannot farm the WR bonus when A2 variance-flips right.
2. **Harm-causation penalty (negative side)** — `\boxed{INCORRECT}` on a
   right A1 that pushes A2 off the correct answer (RW transition) still
   incurs the full negative downstream penalty. F1 is credited for causing
   actual damage even when the verdict was wrong.

Numbers below shown for α=1 (active) — multiply by 11/3 to recover α=5
values:

| Verdict | Transition | raw downstream (α=1) | gated downstream |
|---|---|---:|---:|
| calibrated | WR (best) | +3 | +3 |
| calibrated | RR | +1 | +1 |
| calibrated | WW | −1 | −1 |
| calibrated | RW | −3 | −3 |
| wrong INCORRECT on right A1 | RR (ignored advice) | +1 | **0** (positive gated) |
| wrong INCORRECT on right A1 | **RW (F1 caused harm)** | −3 | **−3** (flows through) |
| sycophantic CORRECT on wrong A1 | WR (A2 variance-flip) | +3 | **0** (positive gated) |
| sycophantic CORRECT on wrong A1 | WW (F1 reinforced) | −1 | **−1** (flows through) |

### `format` ∈ {0, +1}  (binary, structural + clean verdict)

```python
def compute_feedback_format_reward(feedback_text):
    if not has_think_boxed(feedback_text):
        return 0.0
    verdict = extract_from_boxed(feedback_text).upper()
    return 1.0 if verdict in ("CORRECT", "INCORRECT") else 0.0
```

| F1 text | Format reward |
|---|---|
| `<think>r</think> \boxed{CORRECT}` | **+1** |
| `<think>r</think> \boxed{Correct}` | **+1** (case-insensitive) |
| `<think>r</think> \boxed{(A)}` | **0** (boxed not a verdict) |
| `<think>r</think> \boxed{maybe}` | **0** (boxed not a verdict) |
| `<think>r</think> INCORRECT` | **0** (no `\boxed{}`) |
| `\boxed{INCORRECT}` (no `<think>`) | **0** (think missing) |

---

## Reward landscape — totals across scenarios

Computed with **active full-mode YAML weights** (`w_a1_corr=0, w_a1_fmt=0,
w_a2_corr=0.45, w_a2_fmt=0.05, w_no_reg=0.50` for response;
`w_ds=0.45, w_ver=0.45, w_fmt=0.1` for feedback), deterministic MCQ,
`α_response=1`, `α_feedback=1`, raw (un-rescaled) breakdown. With α=1, the
shaped no_regression and downstream values are tighter than the α=0 transition
table (no_regression ∈ {−2, 0, +2}; downstream ∈ {−3, −1, +1, +3}).

### Response reward (A1 frozen → a1 weights = 0, α=1)

`r_no_reg = α·(r_a2 − r_a1)` with `r_a1, r_a2 ∈ {±1}`.

| Scenario | a2c | nor | a2_fmt | **TOTAL** |
|---|---:|---:|---:|---:|
| Tagged WR (wrong→right, clean) | +1 | +2 | +1 | **+1.500** |
| Tagged RR (right→right, clean) | +1 |  0 | +1 | **+0.500** |
| Tagged WW (wrong→wrong, clean) | −1 |  0 | +1 | **−0.400** |
| Tagged RW (right→wrong, clean) | −1 | −2 | +1 | **−1.400** |
| Correct + descriptor `(A) text`, RR | +1 |  0 |  0 | **+0.450** |
| Missing tags, RW (extracted as wrong) | −1 | −2 |  0 | **−1.450** |

### Feedback reward (α=1, gated)

`r_downstream = r_a2 + α·(r_a2 − r_a1)`. Numbers shown raw, then after the
asymmetric gate (`min(raw, 0)` when verification ≤ 0).

| Scenario | raw ds | gated ds | verif | fmt | **TOTAL** |
|---|---:|---:|---:|---:|---:|
| WR honest `\boxed{INCORRECT}` | +3 | +3 | +1 | +1 | **+1.900** |
| RR honest `\boxed{CORRECT}` | +1 | +1 | +1 | +1 | **+0.650** |
| WW miscalibrated `\boxed{CORRECT}` | −1 | −1 | −1 | +1 | **−0.800** |
| RW miscalibrated `\boxed{CORRECT}` | −3 | −3 | −1 | +1 | **−1.700** |
| WR sycophantic `\boxed{CORRECT}` | +3 |  0 | −1 | +1 | **−0.350** |
| `\boxed{}` only (no think, RR) | +1 | +1 | +1 |  0 | **+0.550** |
| `<think>` only (no boxed) | 0 (empty F1 path) |  0 | −1 |  0 | **−0.450** |
| Plain text `INCORRECT` | (no `\boxed{}`) | gated | −1 |  0 | **−0.450** |
| Boxed garbage `\boxed{(A)}` | gated to ≤ 0 |  0 | −1 |  0 | **−0.450** |

---

## Rescaled mode (`--use_rescaled_rewards`)

Each component is mapped to `[0, 1]` via `_to_unit(raw, lo, hi)` (clipped to
`[0, 1]`) before weighting:

| Component | Raw range used as `[lo, hi]` |
|---|---|
| a1_correctness, a2_correctness | `[−1, +1]` (deterministic); `[−1, +1]` after `2·score − 1` for continuous |
| a1_format, a2_format, fb_format | `[0, +1]` (binary tag-mode) → pass-through; `[−1, 0]` (bare mode) → rescaled |
| no_regression (α=0, det) | `[−2, +3]` |
| no_regression (α=0, open) | `[−3, +2]` |
| no_regression (α>0) | `[−2α, +2α]` |
| downstream (α=0, det) | `[−1.5, +3]` |
| downstream (α=0, open) | `[−2, +2]` |
| downstream (α>0) | `[−1−2α, +1+2α]` |
| downstream (improvement-mode) | `[−2, +2]` |
| verification | `[−1, +1]` |

The asymmetric downstream gate becomes `min(value, midpoint)` where
`midpoint = _to_unit(0.0, lo, hi)` — the post-rescaling equivalent of
"clamp positive flow when verification fails". With convex weights, the
total reward lives in `[0, 1]` and the per-unit-weight gradient magnitude
is comparable across components.

## End-to-end walkthrough (raw mode, full-weight YAML, α=1)

**Setup:** LIVR ArtStyle MCQ, GT=`(A)`, A1 wrong on first try, F1 honestly
identifies error, A2 corrects. A1 is frozen so `w_a1_corr=w_a1_fmt=0`.

```
A1 = <think>The reference is Renaissance.</think><answer>(B) Same style as reference (romanticism)</answer>
F1 = <think>The reference is Renaissance, but (B) is Romanticism. Wrong.</think> \boxed{INCORRECT}
A2 = <think>Re-examining brushwork — Option A is Renaissance.</think><answer>(A)</answer>
GT = (A)
```

### Response reward trace (α=1)

1. `extract_from_answer_tags(A1)` → `"(B) Same style as reference (romanticism)"`
2. `extract_answer_from_text(A1, 'mcq', strict=True)` → `"B"` (leading `(B)` matches; trailing descriptor allowed)
3. `verify_answer(A1, "(A)", 'mcq', strict=True)` → WRONG (B ≠ A) → `a1_correct=False`, so `r_a1 = −1`
4. Same for A2 → `"A"` → CORRECT → `a2_correct=True`, so `r_a2 = +1`
5. Shaped no_regression: `α·(r_a2 − r_a1) = 1·(+1 − (−1)) = +2`
6. Format: A1 inner has descriptor → not clean atomic → `r_a1_format = 0`. A2 inner = `"(A)"` → clean atomic → `r_a2_format = +1`
7. Components: `a1c=−1, a1_fmt=0, a2c=+1, a2_fmt=+1, no_reg=+2`
8. Weighted (full-mode YAML): `0·(−1) + 0·0 + 0.45·(+1) + 0.05·(+1) + 0.50·(+2) = 0.45 + 0.05 + 1.00 = +1.500`
9. **Response total: +1.500** (best possible WR with α=1)

### Feedback reward trace (α=1)

1. `extract_from_boxed(F1)` → `"INCORRECT"`
2. `compute_verification_accuracy_reward(F1, a1_is_correct=False)` → boxed = `INCORRECT`, A1 wrong → `+1`
3. Shaped downstream: `r_a1 = −1, r_a2 = +1`, `r_ds = +1 + 1·(+1 − (−1)) = +3`
4. Gate: `verification = +1 > 0` → downstream NOT zeroed → `r_downstream_gated = +3`
5. Format: `<think>...</think>...\boxed{INCORRECT}` matches structure AND verdict is clean → `+1`
6. Weighted (full-mode YAML): `0.45·3 + 0.45·1 + 0.1·1 = 1.35 + 0.45 + 0.10 = +1.900`
7. **Feedback total: +1.900** (best possible WR scenario)

### Why this is the desired training signal

- A2 is rewarded for landing on the correct answer (+0.45).
- The WR transition is heavily rewarded (+1.00 from shaped no_regression).
- F1 is rewarded for an honest assessment (+0.45) AND for the downstream improvement it enabled (+1.35).
- Format compliance gives a small bonus on each side (+0.05 / +0.10).

If F1 had said `\boxed{CORRECT}` instead (sycophantic on a wrong A1), the
gate would zero downstream and the verification penalty would kick in:
total feedback = `0·0.45 − 1·0.45 + 1·0.1 = −0.35`. F1 cannot exploit a
lucky A2 flip without being honest first.

---

## Implementation entry points

| Function | File | Purpose |
|---|---|---|
| `compute_response_reward_breakdown` | `src/vlm_grpo/rewards/composition.py` | Top-level response reward (raw) |
| `compute_feedback_reward_breakdown` | `src/vlm_grpo/rewards/composition.py` | Top-level feedback reward (raw) |
| `compute_response_reward_breakdown_01` | `src/vlm_grpo/rewards/composition.py` | Top-level response reward ([0, 1] rescaled) |
| `compute_feedback_reward_breakdown_01` | `src/vlm_grpo/rewards/composition.py` | Top-level feedback reward ([0, 1] rescaled) |
| `compute_baseline_a1_reward_breakdown` | `src/vlm_grpo/rewards/composition.py` | Single-turn A1 baseline reward |
| `compute_verification_accuracy_reward` | `src/vlm_grpo/rewards/composition.py` | F1 verdict ↔ A1 truth |
| `compute_feedback_format_reward` | `src/vlm_grpo/rewards/composition.py` | F1 format check |
| `_compute_tag_format_reward` | `src/vlm_grpo/rewards/composition.py` | A1/A2 format check (think+answer) |
| `_is_clean_atomic_answer` | `src/vlm_grpo/rewards/composition.py` | Per-type atomic-answer test |
| `verify_answer` | `src/vlm_grpo/rewards/verifier.py` | Answer correctness with strict mode |
| `extract_answer_from_text` | `src/vlm_grpo/trajectory.py` | Strict extraction (requires `<answer>` tag) |
| `extract_from_boxed` | `src/vlm_grpo/trajectory.py` | F1 verdict extraction |
| `has_think_answer_tags`, `has_think_boxed` | `src/vlm_grpo/trajectory.py` | Tag-pair structural checks |

---

## Reproducing the reward landscape

Active full-mode YAML weights, A1 frozen, α=1:

```bash
uv run python3 -c "
from vlm_grpo.config import ResponseRewardWeights, FeedbackRewardWeights
from vlm_grpo.rewards.composition import (
    compute_response_reward_breakdown, compute_feedback_reward_breakdown,
)

rw = ResponseRewardWeights(
    w_a1_correctness=0.0, w_a1_format=0.0,
    w_a2_correctness=0.45, w_a2_format=0.05,
    w_no_regression=0.50,
)
fw = FeedbackRewardWeights(
    w_downstream=0.45, w_verification_accuracy=0.45, w_format=0.1,
)

# Tagged WR (best response transition)
bd = compute_response_reward_breakdown(
    a1_text='<think>r</think><answer>(B)</answer>',
    a2_text='<think>r</think><answer>(A)</answer>',
    ground_truth='(A)', answer_type='mcq', choices='', weights=rw,
    use_think_answer_tags=True,
    reward_shaping_alpha=1.0,
)
print('WR response:', bd.total_reward, bd.components)

# WR honest feedback (α=1)
bd = compute_feedback_reward_breakdown(
    feedback_text='<think>r</think> \\\\boxed{INCORRECT}',
    a1_text='<think>r</think><answer>(B)</answer>',
    a2_text='<think>r</think><answer>(A)</answer>',
    ground_truth='(A)',
    answer_type='mcq', choices='', weights=fw,
    reward_shaping_alpha=1.0,
)
print('WR feedback:', bd.total_reward, bd.components)
"
```

Expected output (α=1, active weights):
```
WR response: 1.5  {a1_correctness: -1.0, a1_format: 1.0, a2_correctness: 1.0, a2_format: 1.0, no_regression: 2.0}
WR feedback: 1.9  {downstream: 3.0, verification: 1.0, format: 1.0}
```
