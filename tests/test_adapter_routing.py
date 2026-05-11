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
