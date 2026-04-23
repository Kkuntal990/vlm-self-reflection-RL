# GRPO Reward Structure (v10)

This doc tracks how every reward component is computed for the
self-reflection GRPO pipeline. All values are taken from
`src/vlm_grpo/rewards/composition.py` and `src/vlm_grpo/trajectory.py`;
walkthrough numbers are reproduced by `uv run python -c "from vlm_grpo..."`.

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
r_response = w_a1 · r_a1_correctness
           + w_a2 · r_a2_correctness
           + w_nor · r_no_regression
           + w_fmt · r_a2_format
```

Default weights (v10): `w_a1=0.3, w_a2=0.3, w_nor=0.3, w_fmt=0.1` (sum = 1.0).

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

Default weights (v10): `w_ds=0.45, w_ver=0.45, w_fmt=0.1` (sum = 1.0).

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

### `downstream` (gated by verification)

```python
# Shaped reward (α=5 default for v10):
r_a1 = 1.0 if a1_correct else -1.0
r_a2 = 1.0 if a2_correct else -1.0
r_downstream = r_a2 + α · (r_a2 − r_a1)

# Gate: F1 only earns downstream credit if its verdict is calibrated.
r_downstream_gated = r_downstream if r_verification > 0 else 0.0
```

The gate prevents sycophancy: a `\boxed{CORRECT}` on a wrong A1 cannot earn
downstream credit even if A2 happens to variance-flip to correct. F1 must
be honest about A1 to be credited for downstream improvements.

| Transition | r_a1 | r_a2 | shaped (α=5) | If verification > 0 | If verification ≤ 0 |
|---|---|---|---|---|---|
| WR (best) | −1 | +1 | **+11** | +11 | 0 (gated) |
| RR | +1 | +1 | **+1** | +1 | 0 |
| WW | −1 | −1 | **−1** | −1 | 0 |
| RW (regression) | +1 | −1 | **−11** | −11 | 0 |

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

Computed with default weights, deterministic MCQ, `α_response=0`, `α_feedback=5`.

### Response reward

| Scenario | a1c | a2c | nor | fmt | **TOTAL** |
|---|---:|---:|---:|---:|---:|
| Tagged RR (right→right, clean inner) | +1 | +1 | +1 | +1 | **+1.000** |
| Tagged WR (wrong→right, clean) | −1 | +1 | +3 | +1 | **+1.000** |
| Tagged RW (right→wrong, clean) | +1 | −1 | −2 | +1 | **−0.500** |
| Tagged WW (wrong→wrong, clean) | −1 | −1 |  0 | +1 | **−0.500** |
| Correct + descriptor `(A) text` | +1 | +1 | +1 | 0 | **+0.900** |
| Missing tags, A1 right | +1 | −1 | −2 |  0 | **−0.600** |
| Missing tags, A1 wrong | −1 | −1 |  0 |  0 | **−0.600** |
| `<answer>` only, no `<think>` (correct) | +1 | +1 | +1 |  0 | **+0.900** |

### Feedback reward (α=5)

| Scenario | downstream | verification | format | **TOTAL** |
|---|---:|---:|---:|---:|
| WR honest `\boxed{INCORRECT}` | +11.00 | +1 | +1 | **+5.500** |
| RR honest `\boxed{CORRECT}` | +1.00 | +1 | +1 | **+1.000** |
| WW miscalibrated `\boxed{CORRECT}` | 0 (gated) | −1 | +1 | **−0.350** |
| WR sycophantic `\boxed{CORRECT}` | 0 (gated) | −1 | +1 | **−0.350** |
| `\boxed{}` only (no think) | +11.00 | +1 |  0 | **+5.400** |
| `<think>` only (no boxed) | 0 (gated) | −1 |  0 | **−0.450** |
| Plain text `INCORRECT` | 0 (gated) | −1 |  0 | **−0.450** |
| Boxed garbage `\boxed{(A)}` | 0 (gated) | −1 |  0 | **−0.450** |

---

## End-to-end walkthrough

**Setup:** LIVR ArtStyle MCQ, GT=`(A)`, A1 wrong on first try, F1 honestly
identifies error, A2 corrects.

```
A1 = <think>The reference is Renaissance.</think><answer>(B) Same style as reference (romanticism)</answer>
F1 = <think>The reference is Renaissance, but (B) is Romanticism. Wrong.</think> \boxed{INCORRECT}
A2 = <think>Re-examining brushwork — Option A is Renaissance.</think><answer>(A)</answer>
GT = (A)
```

### Response reward trace

1. `extract_from_answer_tags(A1)` → `"(B) Same style as reference (romanticism)"`
2. `extract_answer_from_text(A1, 'mcq', strict=True)` → `"B"` (leading `(B)` matches; trailing descriptor allowed)
3. `verify_answer(A1, "(A)", 'mcq', strict=True)` → WRONG (B ≠ A) → `a1_correct=False`
4. Same for A2 → `"A"` → CORRECT → `a2_correct=True`
5. Transition: WR → `no_regression = +3` (deterministic table)
6. Format: A1 inner has descriptor → not clean atomic → A1 doesn't have its own format reward; A2 inner = `"(A)"` → clean atomic → `a2_format = +1`
7. Components: `a1c=−1, a2c=+1, nor=+3, fmt=+1`
8. Weighted: `−0.3 + 0.3 + 0.9 + 0.1 = +1.0`
9. **Response total: +1.000** (tied for best possible)

### Feedback reward trace

1. `extract_from_boxed(F1)` → `"INCORRECT"`
2. `compute_verification_accuracy_reward(F1, a1_is_correct=False)` → boxed = `INCORRECT`, A1 wrong → `+1`
3. Shaped downstream: `r_a1=−1, r_a2=+1`, `r_ds = +1 + 5·(+1−(−1)) = +11`
4. Gate: `verification=+1 > 0` → downstream NOT zeroed → `r_downstream_gated = +11`
5. Format: `<think>...</think>...\boxed{INCORRECT}` matches structure AND verdict is clean → `+1`
6. Weighted: `0.45·11 + 0.45·1 + 0.1·1 = 4.95 + 0.45 + 0.1 = +5.5`
7. **Feedback total: +5.500** (best possible WR scenario)

### Why this is the desired training signal

- A1 is penalized for being wrong (−0.3).
- A2 is rewarded for being correct (+0.3).
- The WR transition is heavily rewarded (+0.9 from no_regression).
- F1 is rewarded for an honest assessment (+0.45) AND for the downstream improvement it enabled (+4.95).
- Format compliance gives a small bonus (+0.1 each).

If F1 had said `\boxed{CORRECT}` instead (sycophantic on a wrong A1), the
gate would zero downstream and the verification penalty would kick in:
total feedback = `0 − 0.45 + 0.1 = −0.35`. F1 cannot exploit a lucky A2
flip without being honest first.

---

## Implementation entry points

| Function | File | Purpose |
|---|---|---|
| `compute_response_reward_breakdown` | `src/vlm_grpo/rewards/composition.py` | Top-level response reward |
| `compute_feedback_reward_breakdown` | `src/vlm_grpo/rewards/composition.py` | Top-level feedback reward |
| `compute_verification_accuracy_reward` | `src/vlm_grpo/rewards/composition.py` | F1 verdict ↔ A1 truth |
| `compute_feedback_format_reward` | `src/vlm_grpo/rewards/composition.py` | F1 format check |
| `_compute_tag_format_reward` | `src/vlm_grpo/rewards/composition.py` | A2 format check (think+answer) |
| `_is_clean_atomic_answer` | `src/vlm_grpo/rewards/composition.py` | Per-type atomic-answer test |
| `verify_answer` | `src/vlm_grpo/rewards/verifier.py` | Answer correctness with strict mode |
| `extract_answer_from_text` | `src/vlm_grpo/trajectory.py` | Strict extraction (requires `<answer>` tag) |
| `extract_from_boxed` | `src/vlm_grpo/trajectory.py` | F1 verdict extraction |

---

## Reproducing the reward landscape

```bash
uv run python3 -c "
from vlm_grpo.config import ResponseRewardWeights, FeedbackRewardWeights
from vlm_grpo.rewards.composition import (
    compute_response_reward_breakdown, compute_feedback_reward_breakdown,
)

rw = ResponseRewardWeights(0.3, 0.3, 0.3, 0.1)
fw = FeedbackRewardWeights(0.45, 0.45, 0.1)

# Tagged WR (best response transition)
bd = compute_response_reward_breakdown(
    a1_text='<think>r</think><answer>(B)</answer>',
    a2_text='<think>r</think><answer>(A)</answer>',
    ground_truth='(A)', answer_type='mcq', choices='', weights=rw,
    use_think_answer_tags=True,
)
print('WR response:', bd.total_reward, bd.components)

# WR honest feedback
bd = compute_feedback_reward_breakdown(
    feedback_text='<think>r</think> \\\\boxed{INCORRECT}',
    a1_text='(B)', a2_text='(A)', ground_truth='(A)',
    answer_type='mcq', choices='', weights=fw,
    reward_shaping_alpha=5.0,
)
print('WR feedback:', bd.total_reward, bd.components)
"
```

Expected output:
```
WR response: 1.0  {a1_correctness: -1.0, a2_correctness: 1.0, no_regression: 3.0, a2_format: 1.0}
WR feedback: 5.5  {downstream: 11.0, verification: 1.0, format: 1.0}
```
