#!/usr/bin/env python3
"""Image deduplication utilities for LIVR-v2 builders.

Two flavors:

  perceptual_dedup(candidates, exclude, ...)
      pHash + SSIM combined. Used by counting (per Appendix A:
      "CLIP embeddings together with perceptual hashing and SSIM-based
      image similarity"). Cheap to compute, no GPU needed.

  clip_dedup(candidates, exclude, ...)
      CLIP-ViT-L/14 cosine similarity. Used by object_localization,
      art_style, semantic_correspondence (pair-aware), visual_similarity.
      Requires a GPU (or a long CPU run).

Both take a list of candidate paths, a list of paths-to-exclude, and a
similarity threshold. Return the subset of candidate paths whose
similarity to ANY exclude path is BELOW the threshold (i.e. "kept").
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Perceptual dedup (pHash + SSIM)
# =============================================================================


def perceptual_dedup(
    candidates: list[Path],
    exclude: list[Path],
    phash_thresh: int = 8,
    ssim_thresh: float = 0.95,
) -> set[Path]:
    """Keep candidate paths whose pHash distance >= phash_thresh AND
    SSIM <= ssim_thresh against ALL exclude paths.

    Args:
        candidates: paths to potentially keep.
        exclude: paths defining the "no-overlap-with" set (e.g. test
            split, BLINK val).
        phash_thresh: Hamming distance threshold on 64-bit pHash.
            Smaller = stricter (must differ in more bits).
        ssim_thresh: Structural similarity threshold.
            Larger = stricter (must be less similar).

    Returns:
        Set of candidate paths to keep.
    """
    import imagehash
    import numpy as np
    from PIL import Image as PILImage
    from skimage.metrics import structural_similarity as ssim

    if not exclude:
        return set(candidates)

    logger.info(
        "Perceptual dedup: %d candidates vs %d exclude (phash_thresh=%d, ssim_thresh=%.2f)",
        len(candidates),
        len(exclude),
        phash_thresh,
        ssim_thresh,
    )

    # Pre-hash exclude images.
    exclude_hashes = []
    exclude_thumbs = []
    for p in exclude:
        try:
            img = PILImage.open(p).convert("RGB")
            exclude_hashes.append(imagehash.phash(img))
            exclude_thumbs.append(np.asarray(img.resize((64, 64), PILImage.LANCZOS).convert("L")))
        except Exception as e:
            logger.warning("Failed to hash exclude image %s: %s", p, e)

    kept: set[Path] = set()
    for i, p in enumerate(candidates):
        try:
            img = PILImage.open(p).convert("RGB")
            ph = imagehash.phash(img)
            thumb = np.asarray(img.resize((64, 64), PILImage.LANCZOS).convert("L"))
        except Exception as e:
            logger.warning("Failed to hash candidate %s: %s", p, e)
            continue
        # Compare to every exclude image; reject on first match.
        too_similar = False
        for eh, eth in zip(exclude_hashes, exclude_thumbs):
            if ph - eh < phash_thresh:
                # SSIM check as the second filter.
                s = ssim(thumb, eth, data_range=255.0)
                if s > ssim_thresh:
                    too_similar = True
                    break
        if not too_similar:
            kept.add(p)
        if (i + 1) % 500 == 0:
            logger.info("  perceptual_dedup: %d/%d candidates checked", i + 1, len(candidates))

    logger.info("Perceptual dedup kept %d/%d candidates", len(kept), len(candidates))
    return kept


# =============================================================================
# CLIP dedup
# =============================================================================


_CLIP_MODEL = None


def _get_clip(device: str = "cuda"):
    """Lazy-load CLIP-ViT-L/14."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        import clip  # type: ignore
        import torch  # type: ignore

        model, preprocess = clip.load("ViT-L/14", device=device)
        model.eval()
        _CLIP_MODEL = (model, preprocess, device, torch)
    return _CLIP_MODEL


def _clip_embed(paths: Iterable[Path], batch: int = 32):
    """Compute L2-normalized CLIP embeddings for a list of image paths."""
    import numpy as np
    from PIL import Image as PILImage

    model, preprocess, device, torch = _get_clip()

    embs = []
    batch_imgs = []
    for p in paths:
        try:
            img = PILImage.open(p).convert("RGB")
            batch_imgs.append(preprocess(img))
        except Exception as e:
            logger.warning("Failed to load %s for CLIP embed: %s", p, e)
            batch_imgs.append(None)
        if len(batch_imgs) >= batch:
            embs.append(_run_clip_batch(batch_imgs, model, device, torch))
            batch_imgs = []
    if batch_imgs:
        embs.append(_run_clip_batch(batch_imgs, model, device, torch))
    if not embs:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(embs, axis=0)


def _run_clip_batch(batch_imgs, model, device, torch):
    import numpy as np

    valid = [t for t in batch_imgs if t is not None]
    if not valid:
        return np.zeros((len(batch_imgs), 768), dtype=np.float32)
    stacked = torch.stack(valid).to(device)
    with torch.no_grad():
        feats = model.encode_image(stacked)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    out = np.zeros((len(batch_imgs), feats.shape[-1]), dtype=np.float32)
    j = 0
    for i, t in enumerate(batch_imgs):
        if t is not None:
            out[i] = feats[j].cpu().numpy()
            j += 1
    return out


def clip_dedup(
    candidates: list[Path],
    exclude: list[Path],
    sim_thresh: float = 0.95,
    device: str = "cuda",
) -> set[Path]:
    """Keep candidates whose max CLIP cosine similarity to ALL exclude
    images is below `sim_thresh`.

    Args:
        candidates: paths to potentially keep.
        exclude: paths defining the no-overlap target (e.g. BLINK val).
        sim_thresh: cosine similarity threshold (larger = stricter).
        device: "cuda" or "cpu".

    Returns:
        Set of paths to keep.
    """
    import numpy as np

    if not exclude:
        return set(candidates)

    logger.info(
        "CLIP dedup: %d candidates vs %d exclude (sim_thresh=%.2f)",
        len(candidates),
        len(exclude),
        sim_thresh,
    )
    cand_emb = _clip_embed(candidates)
    excl_emb = _clip_embed(exclude)
    if cand_emb.size == 0 or excl_emb.size == 0:
        return set(candidates)

    # max similarity for each candidate
    sims = cand_emb @ excl_emb.T  # (n_cand, n_excl)
    max_sim = sims.max(axis=1)
    keep_mask = max_sim < sim_thresh
    kept = {p for p, keep in zip(candidates, keep_mask) if keep}
    logger.info(
        "CLIP dedup kept %d/%d (max sim distribution: median=%.3f, p95=%.3f)",
        len(kept),
        len(candidates),
        float(np.median(max_sim)),
        float(np.percentile(max_sim, 95)),
    )
    return kept


# =============================================================================
# Composite dedup pipelines (Appendix A says "CLIP + pHash + SSIM")
# =============================================================================


def full_dedup(
    candidates: list[Path],
    exclude: list[Path],
    clip_sim_thresh: float = 0.95,
    phash_thresh: int = 8,
    ssim_thresh: float = 0.95,
    device: str = "cuda",
    blurred_grayscale: bool = False,
) -> set[Path]:
    """Appendix-A dedup: keep only candidates rejected by ANY of the three
    similarity checks (CLIP cosine, perceptual hash, SSIM).

    A candidate is dropped if a near-duplicate is found by ALL THREE checks
    simultaneously (i.e. a candidate is kept if AT LEAST ONE check finds it
    distinct enough). This matches the paper's "CLIP embeddings TOGETHER WITH
    perceptual hashing AND SSIM-based image similarity" — three confirming
    signals required to declare a duplicate.

    Args:
        candidates: paths to potentially keep.
        exclude: held-out target images (e.g. BLINK val) we must not duplicate.
        clip_sim_thresh, phash_thresh, ssim_thresh: per-check thresholds.
        device: CLIP device.
        blurred_grayscale: also run pHash + SSIM on a blurred-grayscale
            version of each image (Appendix A: object_localization runs
            "on both raw and blurred grayscale images").

    Returns:
        Set of paths to keep (i.e. NOT confirmed duplicates by all checks).
    """
    if not exclude:
        return set(candidates)

    candidate_set = set(candidates)
    clip_keep = clip_dedup(candidates, exclude, clip_sim_thresh, device)
    phash_keep = perceptual_dedup(candidates, exclude, phash_thresh, ssim_thresh)

    # A candidate is rejected only if BOTH CLIP and pHash+SSIM call it a dup.
    rejected = (candidate_set - clip_keep) & (candidate_set - phash_keep)

    if blurred_grayscale:
        cand_blurred = _make_blurred_grayscale_copies(candidates)
        excl_blurred = _make_blurred_grayscale_copies(exclude)
        try:
            phash_keep_blur = perceptual_dedup(
                cand_blurred, excl_blurred, phash_thresh, ssim_thresh
            )
            # Map blurred temp paths back to originals.
            blur_to_orig = dict(zip(cand_blurred, candidates))
            blur_rejected_set = set(cand_blurred) - phash_keep_blur
            rejected_blur = {blur_to_orig[b] for b in blur_rejected_set if b in blur_to_orig}
            # Augment rejection set with blurred-grayscale duplicates.
            rejected |= rejected_blur & (candidate_set - clip_keep)
        finally:
            for p in cand_blurred:
                p.unlink(missing_ok=True)
            for p in excl_blurred:
                p.unlink(missing_ok=True)

    kept = candidate_set - rejected
    logger.info(
        "full_dedup: kept %d / %d (rejected %d as confirmed duplicates%s)",
        len(kept),
        len(candidates),
        len(rejected),
        " incl. blurred-grayscale" if blurred_grayscale else "",
    )
    return kept


def _make_blurred_grayscale_copies(paths: list[Path]) -> list[Path]:
    """Save grayscale + Gaussian-blurred copies of each image to a temp
    directory; return paths to the copies.

    Used for the "blurred grayscale" pass in object_localization dedup.
    """
    import tempfile

    from PIL import Image as PILImage
    from PIL import ImageFilter

    temp_root = Path(tempfile.mkdtemp(prefix="livr_v2_blur_"))
    out: list[Path] = []
    for i, p in enumerate(paths):
        try:
            img = PILImage.open(p).convert("L").filter(ImageFilter.GaussianBlur(radius=3))
            tp = temp_root / f"blur_{i:06d}.png"
            img.save(tp)
            out.append(tp)
        except Exception as e:
            logger.warning("Failed to make blurred-grayscale for %s: %s", p, e)
    return out


def pair_aware_dedup(
    candidate_pairs: list[tuple[Path, Path]],
    exclude_pairs: list[tuple[Path, Path]],
    clip_sim_thresh: float = 0.95,
    phash_thresh: int = 8,
    ssim_thresh: float = 0.95,
    device: str = "cuda",
) -> set[int]:
    """Pair-aware dedup for tasks that work on (source, target) image pairs
    (Semantic_Correspondence). A candidate pair (cs, ct) is rejected if
    ANY exclude pair (es, et) matches in EITHER orientation:

      aligned : (cs ~ es) AND (ct ~ et)
      swapped : (cs ~ et) AND (ct ~ es)

    Returns indices of pairs to keep (positions in the input list).

    A pair-side image-pair "matches" if ALL THREE checks (CLIP + pHash +
    SSIM) call it a duplicate, mirroring `full_dedup` for single images.
    """
    if not exclude_pairs:
        return set(range(len(candidate_pairs)))

    # Flatten + embed all unique candidate and exclude images once.
    cand_images: list[Path] = []
    for cs, ct in candidate_pairs:
        cand_images.append(cs)
        cand_images.append(ct)
    excl_images: list[Path] = []
    for es, et in exclude_pairs:
        excl_images.append(es)
        excl_images.append(et)

    import imagehash
    import numpy as np
    from PIL import Image as PILImage
    from skimage.metrics import structural_similarity as ssim

    cand_emb = _clip_embed(cand_images)
    excl_emb = _clip_embed(excl_images)
    if cand_emb.size == 0 or excl_emb.size == 0:
        return set(range(len(candidate_pairs)))

    # CLIP similarity matrix.
    clip_sims = cand_emb @ excl_emb.T  # (2*n_cand, 2*n_excl)

    # pHash + SSIM thumbnails for verification.
    def _hash_and_thumb(p: Path):
        img = PILImage.open(p).convert("RGB")
        thumb = np.asarray(img.resize((64, 64), PILImage.LANCZOS).convert("L"))
        return imagehash.phash(img), thumb

    cand_hashes_thumbs = [_hash_and_thumb(p) for p in cand_images]
    excl_hashes_thumbs = [_hash_and_thumb(p) for p in excl_images]

    def _full_match(ci: int, ei: int) -> bool:
        if clip_sims[ci, ei] < clip_sim_thresh:
            return False
        ch, ct = cand_hashes_thumbs[ci]
        eh, et = excl_hashes_thumbs[ei]
        if (ch - eh) >= phash_thresh:
            return False
        s = ssim(ct, et, data_range=255.0)
        return s > ssim_thresh

    keep: set[int] = set()
    for pi, _ in enumerate(candidate_pairs):
        cs_idx, ct_idx = 2 * pi, 2 * pi + 1
        is_dup = False
        for ei in range(len(exclude_pairs)):
            es_idx, et_idx = 2 * ei, 2 * ei + 1
            aligned = _full_match(cs_idx, es_idx) and _full_match(ct_idx, et_idx)
            if aligned:
                is_dup = True
                break
            swapped = _full_match(cs_idx, et_idx) and _full_match(ct_idx, es_idx)
            if swapped:
                is_dup = True
                break
        if not is_dup:
            keep.add(pi)

    logger.info("pair_aware_dedup: kept %d / %d pairs", len(keep), len(candidate_pairs))
    return keep
