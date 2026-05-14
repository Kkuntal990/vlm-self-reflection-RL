"""Runtime backport of vLLM PR #40812 for vLLM 0.12.x.

Background
----------
vLLM's sleep-mode uses a CuMemAllocator backed by virtual-address
reservations (cuMemAddressReserve / cuMemAddressFree). On vLLM 0.12.x
the allocator's ``__init__`` asserts against
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` and refuses to run
when that PyTorch allocator setting is on. The two allocators ARE
incompatible inside the cumem memory pool itself (see
https://github.com/pytorch/pytorch/issues/147851), but the hard assert
also forces *everything outside* the cumem pool to use PyTorch's older
non-expandable allocator — and that's what produces our step-30
fragmentation OOM: sleep/wake cycles fragment the PyTorch pool, and
without expandable_segments PyTorch can't grow/coalesce regions.

vLLM PR #40812 (released in 0.20.1, Apr 2026) fixes this by:

  1. Removing the hard assert in ``CuMemAllocator.__init__``.
  2. Wrapping ``use_memory_pool`` to toggle expandable_segments off
     for the duration of the pool context and restore on exit.

We can't bump to 0.20.x without rebuilding the docker image on CUDA 13
(torch 2.11+cu130 requirement). This module backports the same patch
at runtime: it loads ``vllm.device_allocator.cumem``, replaces the two
methods on ``CuMemAllocator`` with patched versions, and leaves the
rest of vLLM untouched.

Usage
-----
Call ``apply()`` exactly once before constructing the first
``vllm.LLM(enable_sleep_mode=True, ...)``. The function is idempotent
and a no-op on vLLM 0.20.1+ (which already includes PR #40812).

After applying, set the env var so PyTorch enables the expandable
allocator outside the cumem pool:

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

The patch can be disabled via
``VLM_GRPO_DISABLE_VLLM_CUMEM_PATCH=1`` if it ever causes problems.

References
----------
- vLLM PR #40812: https://github.com/vllm-project/vllm/pull/40812
- PyTorch issue:  https://github.com/pytorch/pytorch/issues/147851
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False
_ENV_DISABLE = "VLM_GRPO_DISABLE_VLLM_CUMEM_PATCH"


def _set_allocator_setting(setting: str) -> None:
    """Apply ``setting`` to torch's CUDA caching allocator at runtime.

    ``torch.cuda.memory._set_allocator_settings`` is the PR #40812 call
    site. On torch 2.11+ it has been deprecated in favour of
    ``torch._C._accelerator_setAllocatorSettings`` but the old name
    still works. We prefer the old name to match PR #40812 byte-for-byte.
    """
    import torch  # local import: keep this module cheap to import.

    set_fn = getattr(torch.cuda.memory, "_set_allocator_settings", None)
    if set_fn is None:
        # Fall back to the new accelerator API (torch 2.11+).
        set_fn = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if set_fn is None:
        logger.warning(
            "No allocator-settings setter found on this PyTorch build; "
            "expandable_segments toggle inside cumem pool will be a no-op."
        )
        return
    set_fn(setting)


def apply() -> bool:
    """Backport vLLM PR #40812 onto an already-installed vLLM 0.12.x.

    Returns:
        True if the patch was applied this call. False if it was already
        applied, or if it isn't needed (vLLM 0.20.1+), or if vLLM isn't
        installed at all, or if the user opted out via the env var.
    """
    global _APPLIED
    if _APPLIED:
        return False
    if os.environ.get(_ENV_DISABLE, "").strip() not in ("", "0", "false", "False"):
        logger.info("%s set; skipping vLLM CuMemAllocator monkey-patch.", _ENV_DISABLE)
        _APPLIED = True
        return False

    try:
        from vllm.device_allocator import cumem
    except ImportError:
        logger.info(
            "vllm.device_allocator.cumem not importable; skipping patch "
            "(vLLM not installed yet — this is expected at unit-test time)."
        )
        return False

    CuMemAllocator: Any = cumem.CuMemAllocator

    # If __init__ no longer contains the offending assert, PR #40812 (or
    # an equivalent) already landed in the installed vLLM — leave it
    # alone. Conservative detection by source-string match so we don't
    # double-wrap on 0.20.1+.
    import inspect

    try:
        src = inspect.getsource(CuMemAllocator.__init__)
    except (OSError, TypeError):
        src = ""
    if "Expandable segments are not compatible with memory pool" not in src:
        logger.info(
            "Installed vLLM CuMemAllocator already includes PR #40812 "
            "(no expandable_segments assert in __init__); skipping monkey-patch."
        )
        _APPLIED = True
        return False

    _orig_init = CuMemAllocator.__init__
    _orig_use_memory_pool = CuMemAllocator.use_memory_pool

    def _patched_init(self: Any) -> None:
        """Replacement ``__init__`` that bypasses the expandable_segments assert.

        We trick the original ``__init__`` by temporarily stripping
        ``expandable_segments:True`` from ``PYTORCH_CUDA_ALLOC_CONF``
        for the duration of the constructor call, then restoring the
        original value. The actual torch runtime allocator setting is
        left as the user configured it.
        """
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        had_expandable = "expandable_segments:True" in conf
        if had_expandable:
            # Drop the offending segment while keeping any siblings.
            kept = [p for p in conf.split(",") if p.strip() and "expandable_segments:True" not in p]
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(kept)
        try:
            _orig_init(self)
        finally:
            if had_expandable:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf

    @contextlib.contextmanager
    def _patched_use_memory_pool(self: Any, tag: str | None = None) -> Any:
        """Wrap ``use_memory_pool`` with the expandable_segments toggle.

        Mirrors the diff added by PR #40812: while inside the cumem
        memory pool, expandable_segments must be off (the two
        allocators are incompatible). When the context exits we
        re-enable expandable_segments so PyTorch's outer caching
        allocator can grow/coalesce regions and dodge sleep-cycle
        fragmentation.
        """
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        had_expandable = "expandable_segments:True" in conf
        if had_expandable:
            _set_allocator_setting("expandable_segments:False")
        try:
            with _orig_use_memory_pool(self, tag):
                yield
        finally:
            if had_expandable:
                _set_allocator_setting("expandable_segments:True")

    CuMemAllocator.__init__ = _patched_init
    CuMemAllocator.use_memory_pool = _patched_use_memory_pool

    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    has_expandable = "expandable_segments:True" in conf
    logger.info(
        "Applied vLLM CuMemAllocator PR #40812 backport. "
        "PYTORCH_CUDA_ALLOC_CONF=%r expandable_segments=%s. "
        "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at process "
        "start to actually benefit from the fix.",
        conf,
        "ON" if has_expandable else "OFF (set the env var to enable)",
    )
    _APPLIED = True
    return True
