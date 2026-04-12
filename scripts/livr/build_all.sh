#!/bin/bash
set -euo pipefail

# LIVR Perception MCQ Dataset Construction Pipeline
# Run on pod with /outputs/ mounted
#
# Usage:
#   bash scripts/livr/build_all.sh               # full pipeline
#   bash scripts/livr/build_all.sh --skip-download  # skip downloads
#   bash scripts/livr/build_all.sh --only counting jigsaw  # specific tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_DOWNLOAD=false
ONLY_TASKS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_TASKS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "LIVR Dataset Construction Pipeline"
echo "========================================="
echo "Script dir: ${SCRIPT_DIR}"
echo "Skip download: ${SKIP_DOWNLOAD}"
echo "Only tasks: ${ONLY_TASKS[*]:-all}"
echo ""

# Create output directories
mkdir -p /outputs/livr_data /outputs/livr_sources
mkdir -p /outputs/image_base/livr/{counting,jigsaw,object_localization,visual_correspondence,art_style,semantic_correspondence,functional_correspondence,relative_reflectance,visual_similarity}

# Helper: check if task should run
should_run() {
    local task=$1
    if [ ${#ONLY_TASKS[@]} -eq 0 ]; then
        return 0  # Run all
    fi
    for t in "${ONLY_TASKS[@]}"; do
        if [ "$t" = "$task" ]; then
            return 0
        fi
    done
    return 1
}

# Step 1: Download source datasets
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo ">>> Step 1: Downloading source datasets..."
    python3 "${SCRIPT_DIR}/download_sources.py"
else
    echo ""
    echo ">>> Step 1: Skipping downloads (--skip-download)"
fi

# Step 2: Build per-task datasets
echo ""
echo ">>> Step 2: Building per-task datasets..."

if should_run "counting"; then
    echo ""
    echo "--- Building counting ---"
    python3 "${SCRIPT_DIR}/build_counting.py" || echo "WARNING: counting build failed"
fi

if should_run "art_style"; then
    echo ""
    echo "--- Building art_style ---"
    python3 "${SCRIPT_DIR}/build_art_style.py" || echo "WARNING: art_style build failed"
fi

if should_run "object_localization"; then
    echo ""
    echo "--- Building object_localization ---"
    python3 "${SCRIPT_DIR}/build_object_localization.py" || echo "WARNING: object_localization build failed"
fi

if should_run "jigsaw"; then
    echo ""
    echo "--- Building jigsaw ---"
    python3 "${SCRIPT_DIR}/build_jigsaw.py" || echo "WARNING: jigsaw build failed"
fi

if should_run "visual_similarity"; then
    echo ""
    echo "--- Building visual_similarity ---"
    python3 "${SCRIPT_DIR}/build_visual_similarity.py" || echo "WARNING: visual_similarity build failed"
fi

if should_run "visual_correspondence"; then
    echo ""
    echo "--- Building visual_correspondence ---"
    python3 "${SCRIPT_DIR}/build_visual_correspondence.py" || echo "WARNING: visual_correspondence build failed"
fi

if should_run "semantic_correspondence"; then
    echo ""
    echo "--- Building semantic_correspondence ---"
    python3 "${SCRIPT_DIR}/build_semantic_correspondence.py" || echo "WARNING: semantic_correspondence build failed"
fi

if should_run "relative_reflectance"; then
    echo ""
    echo "--- Building relative_reflectance ---"
    python3 "${SCRIPT_DIR}/build_relative_reflectance.py" || echo "WARNING: relative_reflectance build failed"
fi

if should_run "functional_correspondence"; then
    echo ""
    echo "--- Building functional_correspondence ---"
    python3 "${SCRIPT_DIR}/build_functional_correspondence.py" || echo "WARNING: functional_correspondence build failed"
fi

# Step 3: Merge all tasks
echo ""
echo ">>> Step 3: Merging all tasks..."
python3 "${SCRIPT_DIR}/merge_all.py" --max-per-task 1000

# Step 4: Verify
echo ""
echo ">>> Step 4: Verifying dataset..."
python3 "${SCRIPT_DIR}/verify_dataset.py"

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "Output: /outputs/livr_data/livr_perception_mcq.jsonl"
echo "========================================="
