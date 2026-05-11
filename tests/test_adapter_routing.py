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
