#!/bin/bash
# Targeted re-run after the main pipeline finishes:
#   - rebuilds relative_reflectance with sRGB->linear gamma + disk-mean
#     luminance (audit Discrepancies #5 + #6)
#   - rebuilds visual_similarity with full-triad dedup (Discrepancy #7)
#   - re-merges per-task JSONLs into a TRAIN-ONLY corpus (drops val/test
#     since user is focused on training only)
#
# Usage (run inside the helper pod):
#   bash /workspace/vlm-self-reflection-RL/scripts/data/livr_v2/rerun_after_pipeline.sh

set -uo pipefail
LOG=/outputs/livr_v2/rerun.log
exec > >(tee -a "$LOG") 2>&1

echo "=================================================="
echo "livr-v2 targeted RERUN started at $(date)"
echo "=================================================="

cd /workspace/vlm-self-reflection-RL
git fetch origin feature/two-stage-critic-first 2>&1 | tail -2
git reset --hard origin/feature/two-stage-critic-first 2>&1 | tail -2

# Wipe the two task outputs we're rebuilding (deterministic seed will
# regenerate the composite images; existing v1 PNGs are stale anyway).
rm -rf /outputs/livr_v2/image_base/relative_reflectance
rm -rf /outputs/livr_v2/image_base/visual_similarity
rm -f  /outputs/livr_v2/data/relative_reflectance_*.jsonl
rm -f  /outputs/livr_v2/data/visual_similarity_*.jsonl

echo ""
echo "=== task=relative_reflectance (sRGB->linear + disk-mean fix) ==="
python scripts/data/livr_v2/build_relative_reflectance.py \
    --mid-dir /outputs/livr_v2_sources/mid \
    --output-dir /outputs/livr_v2/image_base/relative_reflectance \
    --output-jsonl-prefix /outputs/livr_v2/data/relative_reflectance
echo "build_relative_reflectance exit code: $?"

echo ""
echo "=== task=visual_similarity (full-triad dedup) ==="
python scripts/data/livr_v2/build_visual_similarity.py \
    --nights-dir /outputs/livr_v2_sources/nights \
    --blink-val-dir /outputs/livr_v2_sources/blink_val \
    --output-dir /outputs/livr_v2/image_base/visual_similarity \
    --output-jsonl-prefix /outputs/livr_v2/data/visual_similarity
echo "build_visual_similarity exit code: $?"

echo ""
echo "=== Train-only merge ==="
# Drop val + test JSONLs from each task so merge.py only sees train.
for f in /outputs/livr_v2/data/*_val.jsonl /outputs/livr_v2/data/*_test.jsonl; do
    [ -f "$f" ] && rm -f "$f"
done
python scripts/data/livr_v2/merge.py \
    --data-dir /outputs/livr_v2/data \
    --output-prefix /outputs/livr_v2/data/livr_v2

echo ""
echo "=================================================="
echo "RERUN DONE at $(date)"
echo "=================================================="
ls -lh /outputs/livr_v2/data/*.jsonl 2>/dev/null
