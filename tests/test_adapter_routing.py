#!/usr/bin/env python3
"""Tests for the AdapterRoutingConfig / multi-adapter pipeline.

Covers the config layer, validation, JSON round-trip, and the CLI-flag
wiring path (parsing ``--adapter_routing_json`` into a working
``SelfReflectionConfig``). Heavier integration tests that require an
actual base model + PEFT wrap live under k8s smoke jobs, not here.
"""

import json

import pytest

from vlm_grpo.config import (
    AdapterRoutingConfig,
    AdapterSpec,
    SelfReflectionConfig,
)


def test_default_routing_disabled() -> None:
    """Empty AdapterRoutingConfig defaults to disabled / single-adapter mode."""
    cfg = AdapterRoutingConfig()
    assert cfg.enabled is False
    assert cfg.adapter_for_turn("a1") == "default"
    assert cfg.adapter_for_turn("f1") == "default"
    assert cfg.adapter_for_turn("a2") == "default"
    assert cfg.trainable_adapter_names() == []


def test_response_feedback_split_routing() -> None:
    """The headline two-adapter routing parses correctly and resolves per turn."""
    cfg = AdapterRoutingConfig(
        turns={"a1": "response", "f1": "feedback", "a2": "response"},
        adapters=[
            AdapterSpec(name="response", trainable=True),
            AdapterSpec(name="feedback", trainable=True),
        ],
    )
    cfg.validate()
    assert cfg.enabled
    assert cfg.adapter_for_turn("a1") == "response"
    assert cfg.adapter_for_turn("f1") == "feedback"
    assert cfg.adapter_for_turn("a2") == "response"
    assert cfg.trainable_adapter_names() == ["response", "feedback"]


def test_validate_missing_turn() -> None:
    """validate() rejects routing that doesn't cover all three turns."""
    cfg = AdapterRoutingConfig(
        turns={"a1": "x"},
        adapters=[AdapterSpec(name="x")],
    )
    with pytest.raises(ValueError, match="missing entries"):
        cfg.validate()


def test_validate_unknown_adapter() -> None:
    """validate() rejects a turn that points at an adapter not in the list."""
    cfg = AdapterRoutingConfig(
        turns={"a1": "missing", "f1": "x", "a2": "x"},
        adapters=[AdapterSpec(name="x")],
    )
    with pytest.raises(ValueError, match="unknown adapters"):
        cfg.validate()


def test_validate_duplicate_adapter_name() -> None:
    cfg = AdapterRoutingConfig(
        turns={"a1": "x", "f1": "x", "a2": "x"},
        adapters=[AdapterSpec(name="x"), AdapterSpec(name="x")],
    )
    with pytest.raises(ValueError, match="Duplicate adapter name"):
        cfg.validate()


def test_validate_warm_start_ordering() -> None:
    """A warm-start source must appear in the list before its consumer."""
    cfg = AdapterRoutingConfig(
        turns={"a1": "a", "f1": "b", "a2": "a"},
        adapters=[
            AdapterSpec(name="a", warm_start_from_adapter="b"),  # b not yet loaded
            AdapterSpec(name="b"),
        ],
    )
    with pytest.raises(ValueError, match="not loaded before it"):
        cfg.validate()


def test_validate_mutually_exclusive_init_options() -> None:
    cfg = AdapterRoutingConfig(
        turns={"a1": "x", "f1": "y", "a2": "x"},
        adapters=[
            AdapterSpec(
                name="x",
                init_from_checkpoint="/tmp/fake",
                warm_start_from_adapter="y",
            ),
            AdapterSpec(name="y"),
        ],
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.validate()


def test_validate_at_least_one_trainable() -> None:
    cfg = AdapterRoutingConfig(
        turns={"a1": "x", "f1": "x", "a2": "x"},
        adapters=[AdapterSpec(name="x", trainable=False)],
    )
    with pytest.raises(ValueError, match="no trainable adapter"):
        cfg.validate()


def test_from_dict_round_trip() -> None:
    """to_dict / from_dict preserves all spec fields including None init paths."""
    original = AdapterRoutingConfig(
        turns={"a1": "r", "f1": "f", "a2": "r"},
        adapters=[
            AdapterSpec(name="r", trainable=True),
            AdapterSpec(
                name="f",
                trainable=True,
                warm_start_from_adapter="r",
            ),
        ],
    )
    blob = json.dumps(original.to_dict())
    rebuilt = AdapterRoutingConfig.from_dict(json.loads(blob))
    assert rebuilt.turns == original.turns
    assert [a.name for a in rebuilt.adapters] == ["r", "f"]
    assert rebuilt.adapters[1].warm_start_from_adapter == "r"
    assert rebuilt.adapters[0].init_from_checkpoint is None


def test_from_dict_empty_returns_disabled() -> None:
    """Empty dict and None both produce a disabled routing config."""
    assert AdapterRoutingConfig.from_dict({}).enabled is False
    assert AdapterRoutingConfig.from_dict(None).enabled is False  # type: ignore[arg-type]


def test_self_reflection_config_accepts_routing_field() -> None:
    """SelfReflectionConfig accepts adapter_routing and exposes it via to_dict."""
    routing = AdapterRoutingConfig(
        turns={"a1": "r", "f1": "f", "a2": "r"},
        adapters=[
            AdapterSpec(name="r", trainable=True),
            AdapterSpec(name="f", trainable=True),
        ],
    )
    cfg = SelfReflectionConfig(adapter_routing=routing)
    assert cfg.adapter_routing.enabled
    serialized = cfg.to_dict()
    assert serialized["adapter_routing"]["turns"]["a1"] == "r"
    assert {a["name"] for a in serialized["adapter_routing"]["adapters"]} == {"r", "f"}


def test_routing_is_optional_on_self_reflection_config() -> None:
    """A SelfReflectionConfig without adapter_routing keeps the disabled default."""
    cfg = SelfReflectionConfig()
    assert cfg.adapter_routing.enabled is False
    assert cfg.adapter_routing.adapter_for_turn("a1") == "default"


def test_apply_trainable_flags_sets_both_directions() -> None:
    """``_apply_trainable_flags`` must FORCE requires_grad to match each spec.

    Regression test for the first deploy of two-adapters: PEFT's
    ``set_adapter("response")`` froze the "feedback" adapter's params at
    ``requires_grad=False`` (it isn't the active adapter), and the original
    ``_apply_trainable_flags`` only re-enforced the ``False`` side — it
    never lifted ``requires_grad`` back to ``True`` for trainable specs.
    The multi-adapter init then raised ``Adapter 'feedback' has 0
    trainable params`` because every feedback leaf was still frozen.
    """
    import torch

    from vlm_grpo.multi_adapter import _apply_trainable_flags

    class _FakePeftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.response_A = torch.nn.Parameter(torch.zeros(2))
            self.response_B = torch.nn.Parameter(torch.zeros(2))
            self.feedback_A = torch.nn.Parameter(torch.zeros(2))
            self.feedback_B = torch.nn.Parameter(torch.zeros(2))

        def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
            yield "base.lora_A.response.weight", self.response_A
            yield "base.lora_B.response.weight", self.response_B
            yield "base.lora_A.feedback.weight", self.feedback_A
            yield "base.lora_B.feedback.weight", self.feedback_B

    model = _FakePeftModel()
    # Simulate post-set_adapter("response"): response is True, feedback frozen.
    model.response_A.requires_grad = True
    model.response_B.requires_grad = True
    model.feedback_A.requires_grad = False
    model.feedback_B.requires_grad = False

    routing = AdapterRoutingConfig(
        turns={"a1": "response", "f1": "feedback", "a2": "response"},
        adapters=[
            AdapterSpec(name="response", trainable=True),
            AdapterSpec(name="feedback", trainable=True),
        ],
    )

    _apply_trainable_flags(model, routing)

    assert model.response_A.requires_grad is True
    assert model.response_B.requires_grad is True
    assert model.feedback_A.requires_grad is True
    assert model.feedback_B.requires_grad is True


def test_apply_trainable_flags_freezes_non_trainable_specs() -> None:
    """Frozen specs must end up with requires_grad=False even after PEFT toggles."""
    import torch

    from vlm_grpo.multi_adapter import _apply_trainable_flags

    class _FakePeftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a1_A = torch.nn.Parameter(torch.zeros(2))
            self.a1_B = torch.nn.Parameter(torch.zeros(2))
            self.f1_A = torch.nn.Parameter(torch.zeros(2))
            self.f1_B = torch.nn.Parameter(torch.zeros(2))

        def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
            yield "base.lora_A.a1_expert.weight", self.a1_A
            yield "base.lora_B.a1_expert.weight", self.a1_B
            yield "base.lora_A.f1_a2_expert.weight", self.f1_A
            yield "base.lora_B.f1_a2_expert.weight", self.f1_B

    model = _FakePeftModel()
    # Simulate PEFT freshly setting all to True (e.g. add_adapter result).
    for p in (model.a1_A, model.a1_B, model.f1_A, model.f1_B):
        p.requires_grad = True

    routing = AdapterRoutingConfig(
        turns={"a1": "a1_expert", "f1": "f1_a2_expert", "a2": "f1_a2_expert"},
        adapters=[
            AdapterSpec(name="a1_expert", trainable=False),
            AdapterSpec(name="f1_a2_expert", trainable=True),
        ],
    )

    _apply_trainable_flags(model, routing)

    assert model.a1_A.requires_grad is False
    assert model.a1_B.requires_grad is False
    assert model.f1_A.requires_grad is True
    assert model.f1_B.requires_grad is True


def test_apply_trainable_flags_anchored_match_avoids_prefix_collision() -> None:
    """Adapter names that share a prefix must not cross-match.

    Latent bug pre-fix: ``_apply_trainable_flags`` matched the adapter
    marker as ``f".{adapter_name}."``. For routing with adapters
    ``"response"`` (trainable) and ``"response_v2"`` (frozen), every
    param of ``response_v2`` ALSO contains ``.response.`` as a substring
    so the inner loop matched on the FIRST entry of the dict (``response``)
    and silently flipped ``response_v2``'s requires_grad to True.

    Fix: match anchored by the PEFT LoRA tensor prefix
    (``.lora_A.<name>.`` / ``.lora_B.<name>.``) so ``response`` does
    not appear inside the marker for ``response_v2``.
    """
    import torch

    from vlm_grpo.multi_adapter import _apply_trainable_flags

    class _FakePeftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.r_A = torch.nn.Parameter(torch.zeros(2))
            self.r_B = torch.nn.Parameter(torch.zeros(2))
            self.r_v2_A = torch.nn.Parameter(torch.zeros(2))
            self.r_v2_B = torch.nn.Parameter(torch.zeros(2))

        def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
            # PEFT-style names: ``base_model...lora_A.<adapter>.weight``
            yield "base.lora_A.response.weight", self.r_A
            yield "base.lora_B.response.weight", self.r_B
            yield "base.lora_A.response_v2.weight", self.r_v2_A
            yield "base.lora_B.response_v2.weight", self.r_v2_B

    model = _FakePeftModel()
    for p in (model.r_A, model.r_B, model.r_v2_A, model.r_v2_B):
        p.requires_grad = True

    routing = AdapterRoutingConfig(
        turns={"a1": "response", "f1": "response_v2", "a2": "response"},
        adapters=[
            AdapterSpec(name="response", trainable=True),
            AdapterSpec(name="response_v2", trainable=False),
        ],
    )

    _apply_trainable_flags(model, routing)

    # response (prefix of response_v2) must NOT freeze response_v2 by
    # accident, and response_v2 (frozen spec) must NOT trickle False into
    # response.
    assert model.r_A.requires_grad is True
    assert model.r_B.requires_grad is True
    assert model.r_v2_A.requires_grad is False
    assert model.r_v2_B.requires_grad is False


# =============================================================================
# Regression: gradient-checkpointing + multi-adapter backward
# =============================================================================


def test_multi_adapter_backward_routes_grad_to_each_adapters_text_lora() -> None:
    """Regression for the dead-feedback-text-LoRA bug.

    Reproduces the gradient-checkpoint mis-attribution discovered in the
    two-adapters runs: a single ``mb_loss.backward()`` lands ALL text-decoder
    gradient on whichever adapter is active at backward time, regardless of
    which adapter ran the original forward. The fix (``_multi_adapter_backward``)
    splits the loss per adapter and calls backward separately with the right
    adapter active before each.

    Production observation: after 250 steps both adapters' text-side
    ``lora_B`` should be non-zero, but feedback's was exactly 0 across all
    196 text-decoder tensors (only its merger LoRA — which is not grad-
    checkpointed — had moved). The optimizer state confirmed only 400 of
    1432 LoRA params ever received a gradient (the response head's 296 +
    feedback's 4 merger; no feedback text).
    """
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    from vlm_grpo.config import AdapterRoutingConfig
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    # Tiny stand-in for the trainer's text decoder: two linear layers,
    # the SECOND wrapped in gradient checkpointing. Only the second layer's
    # gradient flow is affected by the bug — the first is not checkpointed
    # and serves as a sanity baseline.
    class MiniDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(8, 8)
            self.second = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.first(x)
            h = torch.utils.checkpoint.checkpoint(self.second, h, use_reentrant=True)
            return h

        # PEFT looks for this on the inner model.
        def prepare_inputs_for_generation(self, *args, **kwargs):  # noqa: D401
            return {}

    model = MiniDecoder()
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["first", "second"])
    peft_model = get_peft_model(model, lora_cfg, adapter_name="response")
    peft_model.add_adapter("feedback", lora_cfg)
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

    # Build a stub trainer that owns just the routing config + the helper
    # methods involved in the bug + fix.
    trainer = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    trainer._routing = AdapterRoutingConfig.from_dict(
        {
            "turns": {"a1": "response", "f1": "feedback", "a2": "response"},
            "adapters": [
                {"name": "response", "trainable": True},
                {"name": "feedback", "trainable": True},
            ],
        }
    )

    # Reproduce the routing pattern: A1+A2 batched forward with response
    # active, F1 forward with feedback active.
    x = torch.randn(2, 8, requires_grad=True)

    peft_model.set_adapter("response")
    out_response = peft_model(x).sum()

    peft_model.set_adapter("feedback")
    out_feedback = peft_model(x).sum()

    # Per-trajectory traj_resp_loss / traj_fb_loss accumulators feed the
    # per-adapter buckets we hand to ``_multi_adapter_backward``.
    mb_response_loss = out_response
    mb_feedback_loss = out_feedback
    mb_loss = mb_response_loss + mb_feedback_loss  # legacy combined loss

    # Apply the fix.
    trainer._multi_adapter_backward(
        inner_model=peft_model,
        mb_loss=mb_loss,
        mb_response_loss=mb_response_loss,
        mb_feedback_loss=mb_feedback_loss,
    )

    # Collect gradient norms on the grad-checkpointed second linear's
    # ``lora_B`` for each adapter — this is the tensor that stayed at
    # exactly 0 in production.
    grads: dict[str, float | None] = {"response": None, "feedback": None}
    for name, param in peft_model.named_parameters():
        if "second.lora_B" in name:
            for adapter in grads:
                if f".lora_B.{adapter}." in name:
                    grads[adapter] = (
                        None if param.grad is None else param.grad.float().norm().item()
                    )

    assert grads["response"] is not None and grads["response"] > 0.0, (
        f"Response adapter's grad-checkpointed text-LoRA still has zero / None "
        f"gradient after the fix (got {grads['response']!r})"
    )
    assert grads["feedback"] is not None and grads["feedback"] > 0.0, (
        f"Feedback adapter's grad-checkpointed text-LoRA still has zero / None "
        f"gradient after the fix (got {grads['feedback']!r}) — this is the "
        "exact production bug; the per-adapter backward split is not working."
    )


def test_multi_adapter_backward_falls_back_for_single_turn_a1() -> None:
    """Single-turn A1 path leaves the per-adapter accumulators at None.

    ``_multi_adapter_backward`` must detect that case and use the legacy
    combined ``mb_loss.backward()`` instead of skipping backward entirely.
    """
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    from vlm_grpo.config import AdapterRoutingConfig
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l = nn.Linear(8, 8)

        def forward(self, x):
            return self.l(x)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}

    model = M()
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["l"]), adapter_name="response"
    )
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)

    trainer = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    trainer._routing = AdapterRoutingConfig.from_dict(
        {
            "turns": {"a1": "response", "f1": "response", "a2": "response"},
            "adapters": [{"name": "response", "trainable": True}],
        }
    )

    x = torch.randn(2, 8, requires_grad=True)
    out = peft_model(x).sum()

    # Single-turn-A1 path: only mb_loss populated, per-adapter accumulators None.
    trainer._multi_adapter_backward(
        inner_model=peft_model,
        mb_loss=out,
        mb_response_loss=None,
        mb_feedback_loss=None,
    )

    # Some LoRA param must have received a gradient (proves backward fired).
    n_with_grad = sum(1 for p in peft_model.parameters() if p.grad is not None)
    assert n_with_grad > 0, "Single-turn-A1 fallback did not call backward at all"


# =============================================================================
# frozen_lora_patterns: cross-adapter module-family freeze
# =============================================================================


def test_frozen_lora_patterns_from_json() -> None:
    """``frozen_lora_patterns`` round-trips through from_dict/to_dict."""
    raw = {
        "turns": {"a1": "response", "f1": "feedback", "a2": "response"},
        "adapters": [
            {"name": "response", "trainable": True},
            {"name": "feedback", "trainable": True},
        ],
        "frozen_lora_patterns": ["visual"],
    }
    cfg = AdapterRoutingConfig.from_dict(raw)
    assert cfg.frozen_lora_patterns == ["visual"]
    # round-trip
    rt = AdapterRoutingConfig.from_dict(cfg.to_dict())
    assert rt.frozen_lora_patterns == ["visual"]


def test_frozen_lora_patterns_overrides_adapter_trainable() -> None:
    """A LoRA param matching a frozen pattern must have requires_grad=False
    even on an adapter whose ``trainable=True``.

    Regression: the Job-A two-adapter setup loads the baseline-A1 ckpt for
    ``response``, which carries merger LoRA. Without this override the
    merger LoRA would silently continue training even with
    ``--freeze_vision_tower`` set — exactly what happened in the prior runs.
    """
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    from vlm_grpo.multi_adapter import _apply_trainable_flags

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # ``visual.merger.fc`` simulates the vision-merger module.
            self.visual_merger_fc = nn.Linear(8, 8)
            self.language_fc = nn.Linear(8, 8)

        def forward(self, x):
            return self.language_fc(self.visual_merger_fc(x))

        def prepare_inputs_for_generation(self, *a, **k):
            return {}

    model = M()
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["visual_merger_fc", "language_fc"])
    peft_model = get_peft_model(model, lora_cfg, adapter_name="response")
    peft_model.add_adapter("feedback", lora_cfg)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)

    routing = AdapterRoutingConfig.from_dict(
        {
            "turns": {"a1": "response", "f1": "feedback", "a2": "response"},
            "adapters": [
                {"name": "response", "trainable": True},
                {"name": "feedback", "trainable": True},
            ],
            "frozen_lora_patterns": ["visual"],
        }
    )
    _apply_trainable_flags(peft_model, routing)

    # Snapshot every LoRA param's requires_grad bucketed by module + adapter.
    state: dict[str, bool] = {}
    for name, p in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            state[name] = p.requires_grad

    # Visual-merger LoRA: frozen on BOTH adapters.
    visual_merger_keys = [k for k in state if "visual_merger_fc" in k]
    assert visual_merger_keys, "test setup did not produce any visual-merger LoRA params"
    for k in visual_merger_keys:
        assert state[k] is False, (
            f"frozen_lora_patterns=['visual'] should have set requires_grad=False "
            f"on {k}, but it is True"
        )

    # Language-decoder LoRA: trainable on BOTH adapters.
    lang_keys = [k for k in state if "language_fc" in k]
    assert lang_keys
    for k in lang_keys:
        assert state[k] is True, (
            f"language LoRA should remain trainable, but {k} has requires_grad=False"
        )


def test_frozen_lora_patterns_default_empty() -> None:
    """When ``frozen_lora_patterns`` is missing from JSON, the field defaults
    to an empty list (i.e. no extra freezing)."""
    raw = {
        "turns": {"a1": "response", "f1": "feedback", "a2": "response"},
        "adapters": [
            {"name": "response", "trainable": True},
            {"name": "feedback", "trainable": True},
        ],
    }
    cfg = AdapterRoutingConfig.from_dict(raw)
    assert cfg.frozen_lora_patterns == []


def test_frozen_lora_patterns_translate_to_exclude_modules_regex() -> None:
    """When ``frozen_lora_patterns`` is non-empty, ``train_self_reflection.py``
    builds a regex that PEFT's ``LoraConfig.exclude_modules`` understands as
    ``fullmatch``-able, so the matching modules are SKIPPED entirely (not
    LoRA-wrapped). Save memory on fresh adapters in Job-A-style runs.

    PEFT's exclude_modules with a substring LIST does NOT match by substring
    (it uses suffix matching), so the trainer translates ``["visual"]`` →
    ``".*visual.*"``. This test pins that translation and verifies PEFT
    actually excludes the matching modules.
    """
    import re

    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_qkv = nn.Linear(8, 8)
            self.visual_merger = nn.Linear(8, 8)
            self.language_q = nn.Linear(8, 8)

        def forward(self, x):
            return self.language_q(x)

        def prepare_inputs_for_generation(self, *a, **k):
            return {}

    frozen_patterns = ["visual"]
    # Same translation the trainer applies.
    exclude_regex = "|".join(".*" + re.escape(p) + ".*" for p in frozen_patterns)
    assert exclude_regex == ".*visual.*"

    cfg = LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", exclude_modules=exclude_regex)
    pm = get_peft_model(M(), cfg, adapter_name="response")

    # Every LoRA-wrapped module's name should NOT contain "visual".
    lora_module_names = {
        n.split(".lora_A.")[0]
        for n, _ in pm.named_parameters()
        if ".lora_A." in n and ".weight" in n
    }
    assert lora_module_names, "no LoRA wrapping happened at all — bad test setup"
    assert all("visual" not in n for n in lora_module_names), (
        f"exclude_modules regex did not exclude visual modules; LoRA names: {lora_module_names}"
    )


def test_multi_adapter_backward_loss_scale() -> None:
    """``loss_scale`` divides each adapter's loss before backward, so the
    accumulated gradient over N micro-steps reproduces the unscaled single-
    step magnitude. Used by the grad-accumulation wrapper.
    """
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    from vlm_grpo.config import AdapterRoutingConfig
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l = nn.Linear(8, 8)

        def forward(self, x):
            return self.l(x)

        def prepare_inputs_for_generation(self, *a, **k):
            return {}

    # ONE model — we'll run two backwards on it with different loss_scale
    # values, zeroing grads in between, and compare. Using one model avoids
    # the Kaiming-init randomness that would otherwise differ between two
    # separate ``get_peft_model`` calls.
    model = M()
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["l"]), adapter_name="response"
    )
    peft_model.add_adapter("feedback", LoraConfig(r=4, lora_alpha=8, target_modules=["l"]))
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
    trainer = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    trainer._routing = AdapterRoutingConfig.from_dict(
        {
            "turns": {"a1": "response", "f1": "feedback", "a2": "response"},
            "adapters": [
                {"name": "response", "trainable": True},
                {"name": "feedback", "trainable": True},
            ],
        }
    )

    x = torch.randn(2, 8, requires_grad=True)

    def run_backward(loss_scale: float) -> dict:
        for p in peft_model.parameters():
            p.grad = None
        peft_model.set_adapter("response")
        out_resp = peft_model(x).sum()
        peft_model.set_adapter("feedback")
        out_fb = peft_model(x).sum()
        trainer._multi_adapter_backward(
            inner_model=peft_model,
            mb_loss=out_resp + out_fb,
            mb_response_loss=out_resp,
            mb_feedback_loss=out_fb,
            loss_scale=loss_scale,
        )
        return {
            name: p.grad.detach().clone()
            for name, p in peft_model.named_parameters()
            if "lora_B" in name and p.grad is not None
        }

    baseline_grads = run_backward(loss_scale=1.0)
    assert baseline_grads, "baseline backward produced no gradients"

    scaled_grads = run_backward(loss_scale=0.25)

    # Each parameter's scaled gradient should equal 0.25 * baseline.
    for name in baseline_grads:
        assert name in scaled_grads
        assert torch.allclose(scaled_grads[name], baseline_grads[name] * 0.25, atol=1e-6), (
            f"loss_scale=0.25 should produce 0.25x gradient for {name}; "
            f"got max abs diff "
            f"{(scaled_grads[name] - baseline_grads[name] * 0.25).abs().max().item()}"
        )
