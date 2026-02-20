#!/usr/bin/env python3
"""
Utility functions for GRPO RW training.

Provides seeding, environment setup, and string distance computation.

Usage:
    from vlm_grpo.utils import set_seed, setup_environment, normalized_edit_distance

    setup_environment()
    set_seed(42)
    dist = normalized_edit_distance("hello", "hallo")
"""

import hashlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment() -> None:
    """Set default environment variables for training.

    Sets HuggingFace timeouts, disables tokenizer parallelism warnings,
    and configures CUDA memory allocation.
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance normalized to [0.0, 1.0], where 0.0 means identical
        and 1.0 means completely different.
    """
    if s1 == s2:
        return 0.0
    if not s1 or not s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)

    # Dynamic programming for Levenshtein distance
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[len2] / max_len


def hash_sample(question: str, image_path: str) -> str:
    """Create a deterministic hash for a sample.

    Useful for caching judge results across runs.

    Args:
        question: Question text
        image_path: Image path

    Returns:
        Hex digest string (first 16 chars of SHA-256)
    """
    content = f"{question}|{image_path}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
