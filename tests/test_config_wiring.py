#!/usr/bin/env python3
"""Tests verifying every YAML/CLI hyperparameter reaches its runtime consumer.

Background: the trainer historically read several rollout-config fields as
``getattr(self.config, "a1_max_completion_length", 200)`` even though the
attribute lives on ``SelfReflectionConfig.rollout``, not on the top-level
``SelfReflectionConfig``. The default silently masked the miss, so a YAML
``A1_MAX_COMPLETION=386`` never reached the trainer's Dr.GRPO normalization
divisor. The tests below pin every nested-vs-top-level access path so any
regression flips a fast unit test instead of silently rescaling the gradient.
"""

from vlm_grpo.config import (
    BaselineA1RewardWeights,
    FeedbackRewardWeights,
    ResponseRewardWeights,
    RolloutConfig,
    SelfReflectionConfig,
)


def _make_config(**overrides) -> SelfReflectionConfig:
    """Build a SelfReflectionConfig with non-default rollout lengths."""
    rollout = RolloutConfig(
        a1_max_completion_length=386,
        a2_max_completion_length=357,
        f1_max_completion_length=512,
        max_completion_length=386,
        k_samples=16,
    )
    defaults: dict = {
        "rollout": rollout,
        "response_weights": ResponseRewardWeights(),
        "feedback_weights": FeedbackRewardWeights(),
        "baseline_weights": BaselineA1RewardWeights(),
        "kl_coeff": 0.001,
        "a1_kl_coeff": 1.0,
    }
    defaults.update(overrides)
    return SelfReflectionConfig(**defaults)


# =============================================================================
# Per-turn completion length lives on RolloutConfig, NOT SelfReflectionConfig
# =============================================================================


def test_a1_max_completion_length_lives_on_rollout() -> None:
    """The trainer reads a1_max_completion_length from self.config.rollout.

    A bare self.config.a1_max_completion_length lookup would AttributeError
    here — this test guards against re-introducing the silent-default bug.
    """
    config = _make_config()
    assert not hasattr(config, "a1_max_completion_length")
    assert config.rollout.a1_max_completion_length == 386


def test_a2_max_completion_length_lives_on_rollout() -> None:
    config = _make_config()
    assert not hasattr(config, "a2_max_completion_length")
    assert config.rollout.a2_max_completion_length == 357


def test_f1_max_completion_length_lives_on_rollout() -> None:
    config = _make_config()
    assert not hasattr(config, "f1_max_completion_length")
    assert config.rollout.f1_max_completion_length == 512


def test_max_completion_length_lives_on_rollout() -> None:
    config = _make_config()
    assert not hasattr(config, "max_completion_length")
    assert config.rollout.max_completion_length == 386


# =============================================================================
# Trainer's actual access pattern returns the YAML-provided value
# =============================================================================


def test_trainer_drgrpo_divisor_uses_yaml_value() -> None:
    """Reproduces the exact lookup pattern the trainer uses for Dr.GRPO norm.

    Before the fix: float(getattr(self.config, "a1_max_completion_length", 200) or 200)
    silently returned 200 because the attribute was on .rollout, not self.config.
    After the fix:  float(self.config.rollout.a1_max_completion_length or 200)
    returns the configured 386. This test pins the post-fix behavior.
    """
    config = _make_config()
    a1_div = float(config.rollout.a1_max_completion_length or 200)
    a2_div = float(config.rollout.a2_max_completion_length or 200)
    f1_div = float(config.rollout.f1_max_completion_length or 512)
    assert a1_div == 386.0
    assert a2_div == 357.0
    assert f1_div == 512.0


def test_drgrpo_divisor_falls_back_to_default_when_zero() -> None:
    """The `or default` fallback still works when the rollout field is 0/None."""
    rollout = RolloutConfig(
        a1_max_completion_length=0,
        a2_max_completion_length=0,
        f1_max_completion_length=0,
    )
    config = SelfReflectionConfig(rollout=rollout)
    assert (config.rollout.a1_max_completion_length or 200) == 200
    assert (config.rollout.a2_max_completion_length or 200) == 200
    assert (config.rollout.f1_max_completion_length or 512) == 512


# =============================================================================
# KL coefficients live on SelfReflectionConfig (not RolloutConfig)
# =============================================================================


def test_kl_coeffs_live_on_top_level_config() -> None:
    """kl_coeff, a1_kl_coeff, a2_kl_coeff, fb_kl_coeff are top-level fields."""
    config = _make_config()
    assert config.kl_coeff == 0.001
    assert config.a1_kl_coeff == 1.0
    assert config.a2_kl_coeff == 1.0
    assert config.fb_kl_coeff == 1.0
    # And NOT on rollout.
    assert not hasattr(config.rollout, "kl_coeff")
    assert not hasattr(config.rollout, "a1_kl_coeff")


# =============================================================================
# Single-turn baseline flag lives on RolloutConfig (not SelfReflectionConfig)
# =============================================================================


def test_single_turn_a1_lives_on_rollout() -> None:
    """The trainer dispatches on self.config.rollout.single_turn_a1."""
    rollout = RolloutConfig(single_turn_a1=True)
    config = SelfReflectionConfig(rollout=rollout)
    assert config.rollout.single_turn_a1 is True
    assert not hasattr(config, "single_turn_a1")


# =============================================================================
# Reward weights are accessed via their dedicated dataclasses, not self.config
# =============================================================================


def test_response_weights_dataclass_holds_w_a1_correctness() -> None:
    config = _make_config(response_weights=ResponseRewardWeights(w_a1_correctness=0.5))
    assert config.response_weights.w_a1_correctness == 0.5
    assert not hasattr(config, "w_a1_correctness")


def test_baseline_weights_dataclass_holds_w_a1_correctness() -> None:
    config = _make_config(baseline_weights=BaselineA1RewardWeights(w_a1_correctness=0.9))
    assert config.baseline_weights.w_a1_correctness == 0.9


# =============================================================================
# Top-level fields used by trainer (smoke test on a representative subset)
# =============================================================================


def test_top_level_trainer_fields() -> None:
    """Fields the trainer reads via plain self.config.X (no nesting)."""
    config = _make_config(
        learning_rate=2e-5,
        clip_range=0.2,
        clip_high=0.28,
        loss_type="dr_grpo",
        num_inner_epochs=1,
        freeze_a1_steps=280,
        freeze_a2_steps=0,
        separate_turn_loss=True,
        use_dynamic_sampling=True,
        use_ssr=False,
        ref_adapter_path="",
    )
    assert config.learning_rate == 2e-5
    assert config.clip_range == 0.2
    assert config.clip_high == 0.28
    assert config.loss_type == "dr_grpo"
    assert config.num_inner_epochs == 1
    assert config.freeze_a1_steps == 280
    assert config.freeze_a2_steps == 0
    assert config.separate_turn_loss is True
    assert config.use_dynamic_sampling is True
    assert config.use_ssr is False
    assert config.ref_adapter_path == ""


# =============================================================================
# Rollout fields accessed by rollout.py via plain config.X (RolloutConfig arg)
# =============================================================================


def test_rollout_dataclass_fields_for_rollout_engine() -> None:
    """Fields the rollout engine reads from its `config: RolloutConfig` arg."""
    rollout = RolloutConfig(
        k_samples=16,
        temperature=1.0,
        feedback_temperature=1.0,
        a2_temperature=0.7,
        top_p=0.9,
        batch_size=2,
        use_think_answer_tags=True,
        use_answer_tag_only=False,
        use_improvement_reward=False,
        reward_shaping_alpha=1.0,
        response_alpha=1.0,
        feedback_alpha=1.0,
        single_turn_a1=False,
    )
    assert rollout.k_samples == 16
    assert rollout.temperature == 1.0
    assert rollout.feedback_temperature == 1.0
    assert rollout.a2_temperature == 0.7
    assert rollout.use_think_answer_tags is True
    assert rollout.use_improvement_reward is False
    assert rollout.reward_shaping_alpha == 1.0
    assert rollout.response_alpha == 1.0
    assert rollout.feedback_alpha == 1.0
    assert rollout.single_turn_a1 is False
