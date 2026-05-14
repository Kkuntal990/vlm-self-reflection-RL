#!/usr/bin/env python3
"""Tests for the runtime backport of vLLM PR #40812 in
``vlm_grpo._vllm_cumem_patch``.

We can't import the real vLLM here (the trl:0.29.0 image installs it
at pod-start; the local dev env doesn't have it). Instead, each test
materialises a fake ``vllm.device_allocator.cumem`` module on disk
(so ``inspect.getsource`` can read it) and registers it via
``sys.modules`` before calling ``apply()``.

The fake class mirrors the surface area of the real
``CuMemAllocator``:

- ``__init__`` asserts against
  ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` exactly as
  vLLM 0.12.x does. The error message must contain the marker string
  the patch's source-string detection looks for.
- ``use_memory_pool`` is a ``@contextmanager`` we wrap.

Each test resets the ``_APPLIED`` module-level flag so successive
tests can re-run ``apply()`` cleanly.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest


_FAKE_CUMEM_SOURCE = textwrap.dedent(
    '''
    """Test double for vllm.device_allocator.cumem."""
    import contextlib
    import os


    class CuMemAllocator:
        """Stand-in for the real CuMemAllocator. Asserts against
        expandable_segments:True in __init__ exactly like vLLM 0.12.x.
        """

        default_tag = "default"
        instance = None

        def __init__(self):
            conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            assert "expandable_segments:True" not in conf, (
                "Expandable segments are not compatible with memory pool. "
                "Please track issue 147851."
            )
            self.inited = True
            self.pool_entries = []

        @contextlib.contextmanager
        def use_memory_pool(self, tag=None):
            self.pool_entries.append(("enter", tag))
            try:
                yield "pool"
            finally:
                self.pool_entries.append(("exit", tag))
    '''
).lstrip()


def _install_fake_vllm(tmp_path: Path, with_assert: bool = True) -> None:
    """Write a fake ``vllm.device_allocator.cumem`` to disk and register
    it under ``sys.modules`` so the patch's ``import vllm.device_allocator.cumem``
    and ``inspect.getsource`` calls both succeed.
    """
    if with_assert:
        src = _FAKE_CUMEM_SOURCE
    else:
        # Drop the assert — simulates vLLM 0.20.1+ (PR #40812 already merged).
        src = _FAKE_CUMEM_SOURCE.replace(
            'assert "expandable_segments:True" not in conf, (\n'
            '            "Expandable segments are not compatible with memory pool. "\n'
            '            "Please track issue 147851."\n'
            "        )",
            "pass",
        )

    vllm_pkg = tmp_path / "vllm"
    vllm_pkg.mkdir(parents=True, exist_ok=True)
    (vllm_pkg / "__init__.py").write_text("")
    da_pkg = vllm_pkg / "device_allocator"
    da_pkg.mkdir(exist_ok=True)
    (da_pkg / "__init__.py").write_text("")
    (da_pkg / "cumem.py").write_text(src)

    sys.path.insert(0, str(tmp_path))
    # Drop any cached vllm.* so we pick up the fake on next import.
    for mod_name in list(sys.modules):
        if mod_name == "vllm" or mod_name.startswith("vllm."):
            del sys.modules[mod_name]


@pytest.fixture
def fake_vllm_with_assert(tmp_path: Path) -> Iterator[None]:
    _install_fake_vllm(tmp_path, with_assert=True)
    # Reset patch state and the env var so each test starts clean.
    import vlm_grpo._vllm_cumem_patch as patch_mod

    patch_mod._APPLIED = False
    prev_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        yield
    finally:
        if prev_env is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prev_env
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name == "vllm" or mod_name.startswith("vllm."):
                del sys.modules[mod_name]
        importlib.reload(patch_mod)


@pytest.fixture
def fake_vllm_no_assert(tmp_path: Path) -> Iterator[None]:
    _install_fake_vllm(tmp_path, with_assert=False)
    import vlm_grpo._vllm_cumem_patch as patch_mod

    patch_mod._APPLIED = False
    try:
        yield
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name == "vllm" or mod_name.startswith("vllm."):
                del sys.modules[mod_name]
        importlib.reload(patch_mod)


class TestPatchAppliesOnVLLM012:
    """On vLLM 0.12.x (assert present), the patch should detect it,
    monkey-patch ``__init__`` and ``use_memory_pool``, and let
    instantiation succeed with ``expandable_segments:True`` set.
    """

    def test_apply_returns_true(self, fake_vllm_with_assert: None) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from vlm_grpo._vllm_cumem_patch import apply

        assert apply() is True, "patch should fire on a 0.12.x-shaped vLLM"

    def test_init_no_longer_asserts(self, fake_vllm_with_assert: None) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from vlm_grpo._vllm_cumem_patch import apply

        apply()
        from vllm.device_allocator.cumem import CuMemAllocator

        inst = CuMemAllocator()
        assert inst.inited is True, (
            "patched __init__ must complete without raising even when "
            "expandable_segments:True is set"
        )

    def test_env_var_restored_after_init(self, fake_vllm_with_assert: None) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from vlm_grpo._vllm_cumem_patch import apply

        apply()
        from vllm.device_allocator.cumem import CuMemAllocator

        CuMemAllocator()
        assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", (
            "patched __init__ must restore the user's env var on exit"
        )

    def test_env_var_preserves_siblings(self, fake_vllm_with_assert: None) -> None:
        """Other PYTORCH_CUDA_ALLOC_CONF segments must survive the
        temporary strip-and-restore around __init__.
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        from vlm_grpo._vllm_cumem_patch import apply

        apply()
        from vllm.device_allocator.cumem import CuMemAllocator

        CuMemAllocator()
        assert (
            os.environ["PYTORCH_CUDA_ALLOC_CONF"]
            == "expandable_segments:True,max_split_size_mb:128"
        )


class TestSkipOnVLLM020PlusOrMissing:
    """The patch is a no-op when not needed."""

    def test_skip_when_vllm_already_patched(self, fake_vllm_no_assert: None) -> None:
        from vlm_grpo._vllm_cumem_patch import apply

        assert apply() is False, (
            "if installed vLLM has no expandable_segments assert in __init__ "
            "(0.20.1+ or already-patched), apply() should return False"
        )

    def test_skip_when_vllm_not_installed(self, tmp_path: Path) -> None:
        # Ensure vllm is NOT importable.
        for mod_name in list(sys.modules):
            if mod_name == "vllm" or mod_name.startswith("vllm."):
                del sys.modules[mod_name]
        import vlm_grpo._vllm_cumem_patch as patch_mod

        patch_mod._APPLIED = False
        try:
            assert patch_mod.apply() is False, (
                "apply() must return False (not raise) when vLLM is missing"
            )
        finally:
            importlib.reload(patch_mod)


class TestIdempotence:
    def test_second_apply_returns_false(self, fake_vllm_with_assert: None) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from vlm_grpo._vllm_cumem_patch import apply

        assert apply() is True
        assert apply() is False, "second apply() must be a no-op"

    def test_init_still_works_after_double_apply(self, fake_vllm_with_assert: None) -> None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from vlm_grpo._vllm_cumem_patch import apply

        apply()
        apply()
        from vllm.device_allocator.cumem import CuMemAllocator

        inst = CuMemAllocator()
        assert inst.inited is True


class TestOptOutEnvVar:
    def test_disable_via_env_skips_patch(self, fake_vllm_with_assert: None) -> None:
        os.environ["VLM_GRPO_DISABLE_VLLM_CUMEM_PATCH"] = "1"
        try:
            from vlm_grpo._vllm_cumem_patch import apply

            assert apply() is False
        finally:
            os.environ.pop("VLM_GRPO_DISABLE_VLLM_CUMEM_PATCH", None)
