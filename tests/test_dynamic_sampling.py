#!/usr/bin/env python3
"""Tests for DAPO Dynamic Sampling drop-zero-var helpers.

Covers the pure helpers that decide which K-groups to drop from a batch
because their reward variance is zero (and so their GRPO advantage —
and gradient — is zero). These helpers underpin both the SSR replay
path and the standalone Dynamic Sampling path; both must agree on
which groups are degenerate.
"""

from types import SimpleNamespace

from vlm_grpo.critic_grpo import (
    _filter_kept_groups,
    _identify_zero_var_groups,
    _mean_abs_advantage,
)


def _make_result(resp: list[float], fb: list[float]) -> SimpleNamespace:
    """Build a minimal rollout-result stub with the three fields the
    drop logic reads. Length of `resp`/`fb` is K; `response_breakdowns`
    is required to match length but its values are unused by the
    drop logic itself.
    """
    k = len(resp)
    return SimpleNamespace(
        response_rewards=list(resp),
        feedback_rewards=list(fb),
        response_breakdowns=[None] * k,
        feedback_breakdowns=[None] * k,
        question="dummy",
        image_path="/tmp/dummy.png",
        answer1s=["A"] * k,
        feedbacks=["fb"] * k,
        answer2s=["A"] * k,
        ground_truth="A",
        answer_type="mcq",
        dataset_name="test",
    )


class TestMeanAbsAdvantage:
    """`_mean_abs_advantage` must return 0 on zero-variance groups (the
    drop predicate) and a finite positive value otherwise."""

    def test_all_equal_returns_zero(self) -> None:
        # All rewards equal → std=0 → degenerate (would yield zero advantage).
        assert _mean_abs_advantage([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert _mean_abs_advantage([]) == 0.0

    def test_below_eps_returns_zero(self) -> None:
        # tiny float noise still counts as zero variance under 1e-6 floor
        assert _mean_abs_advantage([1.0, 1.0 + 1e-9, 1.0, 1.0]) == 0.0

    def test_nonzero_variance_positive(self) -> None:
        # Symmetric ±1 group: GRPO-normalized |Â| = 1.0 for all elements.
        val = _mean_abs_advantage([1.0, -1.0, 1.0, -1.0])
        # Bounded: GRPO normalization (r-mean)/std then absolute → ≈1.0
        assert val > 0.5
        assert val < 2.0

    def test_single_outlier(self) -> None:
        val = _mean_abs_advantage([1.0, 1.0, 1.0, -1.0])
        assert val > 0.0


class TestIdentifyZeroVarGroups:
    """`_identify_zero_var_groups` must drop only groups where BOTH
    response and feedback heads are degenerate. A non-zero-var
    feedback head must keep the group alive (it still produces
    feedback-side gradient) and vice-versa."""

    def test_no_drop_when_both_heads_have_variance(self) -> None:
        results = [
            _make_result([1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]),
            _make_result([1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0]),
        ]
        kept_mask, n_dropped = _identify_zero_var_groups(results, k=4)
        assert kept_mask == [True, True]
        assert n_dropped == 0

    def test_drop_only_when_both_heads_zero_var(self) -> None:
        results = [
            _make_result([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),  # both zero → DROP
            _make_result([1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]),  # resp non-zero → keep
            _make_result([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]),  # fb non-zero → keep
        ]
        kept_mask, n_dropped = _identify_zero_var_groups(results, k=4)
        assert kept_mask == [False, True, True]
        assert n_dropped == 1

    def test_drop_all_when_all_groups_degenerate(self) -> None:
        results = [
            _make_result([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]),
            _make_result([0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]),
        ]
        kept_mask, n_dropped = _identify_zero_var_groups(results, k=4)
        assert kept_mask == [False, False]
        assert n_dropped == 2

    def test_malformed_group_dropped(self) -> None:
        # K-mismatch → defensive drop (avoid downstream shape errors).
        bad = _make_result([1.0, -1.0], [1.0, -1.0])  # only 2, but k=4
        good = _make_result([1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0])
        kept_mask, n_dropped = _identify_zero_var_groups([bad, good], k=4)
        assert kept_mask == [False, True]
        assert n_dropped == 1

    def test_empty_input(self) -> None:
        kept_mask, n_dropped = _identify_zero_var_groups([], k=4)
        assert kept_mask == []
        assert n_dropped == 0


class TestFilterKeptGroups:
    """`_filter_kept_groups` must filter rollout_results, the per-trajectory
    reward arrays, and the per-trajectory trajectory_data list ATOMICALLY
    by the same mask — otherwise downstream tensors and prompt batches go
    out of alignment and silently train on the wrong data."""

    def test_filter_consistent_across_parallel_arrays(self) -> None:
        # Two groups of K=2 trajectories each; group 0 will be dropped.
        results = [
            _make_result([1.0, 1.0], [1.0, 1.0]),  # zero-var → drop
            _make_result([1.0, -1.0], [-1.0, 1.0]),  # keep
        ]
        all_resp = [10.0, 11.0, 20.0, 21.0]
        all_fb = [100.0, 101.0, 200.0, 201.0]
        traj = [
            {"id": "g0_t0"},
            {"id": "g0_t1"},
            {"id": "g1_t0"},
            {"id": "g1_t1"},
        ]
        slices = [(0, 2), (2, 4)]
        kept_mask = [False, True]

        kept_results, kept_resp, kept_fb, kept_traj = _filter_kept_groups(
            rollout_results=results,
            all_resp_rewards=all_resp,
            all_fb_rewards=all_fb,
            trajectory_data=traj,
            group_slices=slices,
            kept_mask=kept_mask,
        )

        assert len(kept_results) == 1
        assert kept_results[0] is results[1]
        assert kept_resp == [20.0, 21.0]
        assert kept_fb == [200.0, 201.0]
        assert kept_traj == [{"id": "g1_t0"}, {"id": "g1_t1"}]

    def test_keep_all_returns_originals_unchanged(self) -> None:
        results = [
            _make_result([1.0, -1.0], [1.0, -1.0]),
            _make_result([1.0, -1.0], [-1.0, 1.0]),
        ]
        all_resp = [1.0, 2.0, 3.0, 4.0]
        all_fb = [10.0, 20.0, 30.0, 40.0]
        traj = [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]
        slices = [(0, 2), (2, 4)]
        kept_mask = [True, True]

        kept_results, kept_resp, kept_fb, kept_traj = _filter_kept_groups(
            rollout_results=results,
            all_resp_rewards=all_resp,
            all_fb_rewards=all_fb,
            trajectory_data=traj,
            group_slices=slices,
            kept_mask=kept_mask,
        )

        assert kept_results == results
        assert kept_resp == all_resp
        assert kept_fb == all_fb
        assert kept_traj == traj

    def test_drop_all_returns_empty(self) -> None:
        # All-zero-var batch → caller should later see empty all_resp_rewards
        # and short-circuit the policy update (no degenerate gradient).
        results = [
            _make_result([1.0, 1.0], [1.0, 1.0]),
            _make_result([0.0, 0.0], [0.0, 0.0]),
        ]
        slices = [(0, 2), (2, 4)]
        kept_mask = [False, False]

        kept_results, kept_resp, kept_fb, kept_traj = _filter_kept_groups(
            rollout_results=results,
            all_resp_rewards=[1.0, 2.0, 3.0, 4.0],
            all_fb_rewards=[10.0, 20.0, 30.0, 40.0],
            trajectory_data=[{"i": i} for i in range(4)],
            group_slices=slices,
            kept_mask=kept_mask,
        )

        assert kept_results == []
        assert kept_resp == []
        assert kept_fb == []
        assert kept_traj == []


class TestConfigFlag:
    """`use_dynamic_sampling` must default to False so existing experiments
    are bit-identical, and the flag must be plumbed through SelfReflectionConfig."""

    def test_default_disabled(self) -> None:
        from vlm_grpo.config import SelfReflectionConfig

        cfg = SelfReflectionConfig()
        assert cfg.use_dynamic_sampling is False

    def test_can_enable(self) -> None:
        from vlm_grpo.config import SelfReflectionConfig

        cfg = SelfReflectionConfig(use_dynamic_sampling=True)
        assert cfg.use_dynamic_sampling is True

    def test_independent_of_use_ssr(self) -> None:
        # Enabling DS must NOT enable SSR, and vice-versa.
        from vlm_grpo.config import SelfReflectionConfig

        cfg_ds = SelfReflectionConfig(use_dynamic_sampling=True)
        assert cfg_ds.use_ssr is False
        cfg_ssr = SelfReflectionConfig(use_ssr=True)
        assert cfg_ssr.use_dynamic_sampling is False
