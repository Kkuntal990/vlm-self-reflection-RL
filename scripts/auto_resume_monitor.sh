#!/bin/bash
# =============================================================================
# GRPO Auto-Resume Monitor
# Checks every hour. If a job crashed, finds the latest checkpoint
# via a helper pod and redeploys with --resume_from_checkpoint.
# =============================================================================
#
# Usage: bash scripts/auto_resume_monitor.sh
#
# The ground truth is the POD LOGS, not the job/pod status.
# A pod can show "Completed" but have crashed mid-training.
# We verify by checking if the training bar reached 100%.

set -uo pipefail

INTERVAL=1800  # 30 minutes
JUPYTER_YAML="/Users/kuntalkokate/svcl-projects/vlm-self-reflection/k8s/jupyter-1gpu-test.yaml"
BASE_YAML="/Users/kuntalkokate/svcl-projects/vlm-self-reflection-RL/k8s/job-qwen-grpo-35k.yaml"
SFTV2_YAML="/Users/kuntalkokate/svcl-projects/vlm-self-reflection-RL/k8s/job-qwen-grpo-35k-from-v2.yaml"
HELPER_POD="vlm-jupyter-eval2"
BASE_OUTPUT="/outputs/grpo_qwen_sr_v9_20k"
SFTV2_OUTPUT="/outputs/grpo_qwen_sr_sftv2_v10_35k"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Ensure the helper pod is running (for PVC access to find checkpoints)
# ---------------------------------------------------------------------------
ensure_helper_pod() {
    local phase
    phase=$(kubectl get pod $HELPER_POD -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")

    if [ "$phase" = "Running" ]; then
        return 0
    fi

    log "Helper pod not running (status=$phase). Starting it..."
    kubectl delete pod $HELPER_POD --ignore-not-found 2>/dev/null
    sleep 5
    kubectl apply -f "$JUPYTER_YAML" 2>/dev/null

    # Wait up to 3 minutes for it to start
    for i in $(seq 1 18); do
        sleep 10
        phase=$(kubectl get pod $HELPER_POD -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
        if [ "$phase" = "Running" ]; then
            log "Helper pod ready."
            return 0
        fi
    done
    log "WARNING: Helper pod didn't start in 3 minutes. Checkpoint lookup may fail."
    return 1
}

# ---------------------------------------------------------------------------
# Find the latest checkpoint in an output dir via the helper pod
# ---------------------------------------------------------------------------
find_latest_checkpoint() {
    local output_dir="$1"
    local latest
    latest=$(kubectl exec $HELPER_POD -- bash -c "ls -d ${output_dir}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1" 2>/dev/null)
    echo "$latest"
}

# ---------------------------------------------------------------------------
# Check if training genuinely completed (100% in the progress bar)
# ---------------------------------------------------------------------------
check_training_complete() {
    local pod="$1"

    # The ONLY reliable signal is "Training: 100%" in the tqdm progress bar.
    # "Training complete" text appears even during crashes (from bash echo).
    # "Processed prompts: 100%" is vLLM output, NOT training completion.
    #
    # Match: "Training: 100%|██████████|" (tqdm format)
    local final_progress
    final_progress=$(kubectl logs "$pod" --tail=5000 2>/dev/null | grep -E "Training:\s+100%" | grep -v "Processed" | tail -1)
    if [ -n "$final_progress" ]; then
        echo "GENUINELY_COMPLETE"
    else
        echo "CRASHED"
    fi
}

# ---------------------------------------------------------------------------
# Get the error type from pod logs
# ---------------------------------------------------------------------------
get_error() {
    local pod="$1"
    local error
    error=$(kubectl logs "$pod" --tail=200 2>/dev/null | grep -E "Fatal|none_dealloc|OOM|OutOfMemoryError|ValueError|AttributeError|SIGABRT" | tail -1)
    echo "${error:-UNKNOWN}"
}

# ---------------------------------------------------------------------------
# Get the progress (samples completed / total)
# ---------------------------------------------------------------------------
get_progress() {
    local pod="$1"
    kubectl logs "$pod" --tail=500 2>/dev/null | grep "sample" | grep "Training:" | tail -1 | sed 's/.*Training://'
}

# ---------------------------------------------------------------------------
# Redeploy a job with resume from checkpoint
# ---------------------------------------------------------------------------
redeploy_job() {
    local job_name="$1"
    local yaml_file="$2"
    local checkpoint="$3"

    log "Redeploying $job_name from $checkpoint"

    # Delete the old job
    kubectl delete job "$job_name" --ignore-not-found 2>/dev/null
    sleep 5

    # Increment the version number in the yaml job name
    local old_name
    old_name=$(grep "^  name:" "$yaml_file" | head -1 | awk '{print $2}')
    local version_num
    version_num=$(echo "$old_name" | grep -o 'v[0-9]*' | head -1 | sed 's/v//')
    local new_version=$((version_num + 1))
    local new_name
    new_name=$(echo "$old_name" | sed "s/v${version_num}/v${new_version}/")

    sed -i '' "s/$old_name/$new_name/" "$yaml_file"
    log "Job name: $old_name → $new_name"

    # Update or add RESUME_CHECKPOINT env var
    if grep -q "RESUME_CHECKPOINT" "$yaml_file"; then
        # Update existing value
        local old_ckpt_line
        old_ckpt_line=$(grep -A1 "name: RESUME_CHECKPOINT" "$yaml_file" | tail -1 | sed 's/.*value: "//' | sed 's/"//')
        sed -i '' "s|${old_ckpt_line}|${checkpoint}|" "$yaml_file"
    else
        # Add RESUME_CHECKPOINT env var before "# Generation" comment
        sed -i '' "/INNER_EPOCHS/{N;s|value: \"1\"|value: \"1\"\n        - name: RESUME_CHECKPOINT\n          value: \"${checkpoint}\"|}" "$yaml_file"
    fi

    # Deploy
    kubectl apply -f "$yaml_file"
    log "Deployed $new_name resuming from $checkpoint"
}

# ---------------------------------------------------------------------------
# Check a single job
# ---------------------------------------------------------------------------
check_job() {
    local job_label="$1"  # e.g. "base" or "sftv2"
    local yaml_file="$2"
    local output_dir="$3"

    # Find the pod for this job
    local pod
    pod=$(kubectl get pods -l app=grpo-trainer -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep "$job_label" | tail -1)

    if [ -z "$pod" ]; then
        log "[$job_label] No pod found."
        return
    fi

    local phase
    phase=$(kubectl get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
    local job_name
    job_name=$(echo "$pod" | sed 's/-[a-z0-9]*$//')

    log "[$job_label] Pod: $pod | Phase: $phase"

    if [ "$phase" = "Running" ]; then
        local progress
        progress=$(get_progress "$pod")
        log "[$job_label] Progress: $progress"

        # Check stability metrics
        local metrics
        metrics=$(kubectl logs "$pod" --tail=10 2>/dev/null | grep "Inner epochs" | tail -1)
        if [ -n "$metrics" ]; then
            log "[$job_label] $(echo $metrics | sed 's/.*Inner/Inner/')"
        fi
        return
    fi

    # Pending/ContainerCreating — still starting up, skip
    if [ "$phase" = "Pending" ] || [ "$phase" = "Unknown" ]; then
        log "[$job_label] Still starting up, skipping."
        return
    fi

    # Pod is NOT running — check if genuinely complete or crashed
    local status
    status=$(check_training_complete "$pod")

    if [ "$status" = "GENUINELY_COMPLETE" ]; then
        log "[$job_label] GENUINELY COMPLETED. Training finished successfully."
        return
    fi

    # CRASHED — find checkpoint and redeploy
    local error
    error=$(get_error "$pod")
    local progress
    progress=$(get_progress "$pod")
    log "[$job_label] CRASHED at $progress"
    log "[$job_label] Error: $error"

    # Get latest checkpoint
    ensure_helper_pod
    local checkpoint
    checkpoint=$(find_latest_checkpoint "$output_dir")

    if [ -z "$checkpoint" ]; then
        log "[$job_label] No checkpoint found in $output_dir. Cannot resume."
        return
    fi

    log "[$job_label] Latest checkpoint: $checkpoint"
    redeploy_job "$job_name" "$yaml_file" "$checkpoint"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
log "============================================="
log "GRPO Auto-Resume Monitor started"
log "  Interval: ${INTERVAL}s (1 hour)"
log "  Base yaml: $BASE_YAML"
log "  SFT-v2 yaml: $SFTV2_YAML"
log "  Base output: $BASE_OUTPUT"
log "  SFT-v2 output: $SFTV2_OUTPUT"
log "============================================="

while true; do
    echo ""
    log "========== Hourly Check =========="

    check_job "base" "$BASE_YAML" "$BASE_OUTPUT"
    check_job "sftv2" "$SFTV2_YAML" "$SFTV2_OUTPUT"

    log "========== Check Complete =========="
    sleep $INTERVAL
done
