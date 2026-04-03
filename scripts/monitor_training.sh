#!/bin/bash
# GRPO Training Monitor — checks stability and detects crashes
# Run: bash scripts/monitor_training.sh [--once] [--interval 3600]
#
# Checks:
#   1. Pod status (Running/Error/Completed)
#   2. Training stability (entropy, grad_norm, frac_zero_std, tok lengths)
#   3. Progress stalls (no advancement in last check)
#   4. Crash detection with error classification
#
# Thresholds (from GTPO, DAPO, TRL literature):
#   entropy:        WARN < 0.5,  CRITICAL < 0.3
#   grad_norm:      WARN > 10,   CRITICAL > 50
#   frac_zero_std:  WARN > 0.6,  CRITICAL > 0.9
#   tok (a1/a2):    WARN < 10,   CRITICAL < 5

set -uo pipefail

INTERVAL="${2:-3600}"
ONCE="${1:-}"
PREV_PROGRESS_FILE="/tmp/grpo_monitor_progress.txt"

# Thresholds
ENTROPY_WARN=0.5
ENTROPY_CRIT=0.3
GRADNORM_WARN=10
GRADNORM_CRIT=50
ZEROSTD_WARN=0.6
ZEROSTD_CRIT=0.9
TOK_WARN=10
TOK_CRIT=5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*"; }
critical() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] CRITICAL: $*"; }

check_stability() {
    local pod="$1"
    local flags=""

    # Get last 5 "Inner epochs" lines
    local metrics
    metrics=$(kubectl logs "$pod" --tail=500 2>/dev/null | grep "Inner epochs" | tail -5)
    if [ -z "$metrics" ]; then
        log "  No training metrics yet (still initializing)"
        return
    fi

    # Parse last line
    local last_line
    last_line=$(echo "$metrics" | tail -1)

    local entropy grad_norm frac_resp frac_fb tok_a1 tok_a2 resp_adv fb_adv
    entropy=$(echo "$last_line" | sed 's/.*entropy=//' | sed 's/,.*//')
    grad_norm=$(echo "$last_line" | sed 's/.*grad_norm=//' | sed 's/,.*//')
    frac_resp=$(echo "$last_line" | sed 's/.*frac_zero_std=//' | sed 's/\/.*//')
    frac_fb=$(echo "$last_line" | sed 's/.*frac_zero_std=[0-9.]*\///' | sed 's/,.*//')
    tok_a1=$(echo "$last_line" | sed 's/.*tok=//' | sed 's/\/.*//')
    tok_a2=$(echo "$last_line" | sed 's/.*tok=[0-9]*\/[0-9]*\///' | sed 's/[^0-9].*//')
    resp_adv=$(echo "$last_line" | sed 's/.*resp_adv=//' | sed 's/(.*//')
    fb_adv=$(echo "$last_line" | sed 's/.*fb_adv=//' | sed 's/(.*//')

    log "  Metrics: entropy=$entropy grad_norm=$grad_norm frac_std=$frac_resp/$frac_fb tok=$tok_a1/../$tok_a2 adv=$resp_adv/$fb_adv"

    # Check entropy
    if [ -n "$entropy" ] && awk "BEGIN{exit !($entropy < $ENTROPY_CRIT)}"; then
        critical "Entropy COLLAPSED ($entropy < $ENTROPY_CRIT) — model degenerated!"
        flags="$flags ENTROPY_COLLAPSE"
    elif [ -n "$entropy" ] && awk "BEGIN{exit !($entropy < $ENTROPY_WARN)}"; then
        warn "Entropy low ($entropy < $ENTROPY_WARN) — monitor closely"
        flags="$flags ENTROPY_LOW"
    fi

    # Check grad_norm
    if [ -n "$grad_norm" ] && awk "BEGIN{exit !($grad_norm > $GRADNORM_CRIT)}"; then
        critical "Gradient explosion ($grad_norm > $GRADNORM_CRIT) — training unstable!"
        flags="$flags GRAD_EXPLOSION"
    elif [ -n "$grad_norm" ] && awk "BEGIN{exit !($grad_norm > $GRADNORM_WARN)}"; then
        warn "Gradient norm high ($grad_norm > $GRADNORM_WARN)"
        flags="$flags GRAD_HIGH"
    fi

    # Check frac_zero_std
    if [ -n "$frac_resp" ] && awk "BEGIN{exit !($frac_resp > $ZEROSTD_CRIT)}"; then
        critical "Response rewards dead (frac_zero_std=$frac_resp > $ZEROSTD_CRIT)"
        flags="$flags RESP_DEAD"
    fi
    if [ -n "$frac_fb" ] && awk "BEGIN{exit !($frac_fb > $ZEROSTD_CRIT)}"; then
        critical "Feedback rewards dead (frac_zero_std=$frac_fb > $ZEROSTD_CRIT)"
        flags="$flags FB_DEAD"
    fi

    # Check token lengths (completion collapse)
    if [ -n "$tok_a1" ] && awk "BEGIN{exit !($tok_a1 < $TOK_CRIT)}"; then
        critical "Token length collapsed (tok_a1=$tok_a1 < $TOK_CRIT) — degenerate outputs!"
        flags="$flags TOK_COLLAPSE"
    elif [ -n "$tok_a1" ] && awk "BEGIN{exit !($tok_a1 < $TOK_WARN)}"; then
        warn "Token length low (tok_a1=$tok_a1 < $TOK_WARN)"
        flags="$flags TOK_LOW"
    fi

    # Check for entropy collapse trend (last 5 steps all below threshold)
    local n_low
    n_low=$(echo "$metrics" | awk -F'entropy=' '{print $2}' | awk -F',' '{print $1}' | awk "BEGIN{c=0} {if(\$1+0 < $ENTROPY_CRIT) c++} END{print c}")
    if [ "$n_low" -ge 4 ]; then
        critical "Sustained entropy collapse ($n_low/5 recent steps below $ENTROPY_CRIT)"
        flags="$flags SUSTAINED_COLLAPSE"
    fi

    if [ -z "$flags" ]; then
        log "  Status: HEALTHY"
    else
        log "  Flags:$flags"
    fi
}

check_progress() {
    local pod="$1"

    # Get current progress (samples done)
    local progress
    progress=$(kubectl logs "$pod" --tail=500 2>/dev/null | grep "Training:" | tail -1 | sed 's/.*| //' | grep -o '[0-9]*/[0-9]*' | head -1 || echo "")
    if [ -z "$progress" ]; then
        log "  Progress: initializing"
        return
    fi

    local current_step
    current_step=$(echo "$progress" | cut -d/ -f1)
    local total_step
    total_step=$(echo "$progress" | cut -d/ -f2)
    local pct=$((current_step * 100 / total_step))

    # Get speed
    local speed
    speed=$(kubectl logs "$pod" --tail=200 2>/dev/null | grep "Training:" | tail -1 | grep -o '[0-9.]*s/sample' | tail -1)

    log "  Progress: $current_step/$total_step ($pct%) | Speed: ${speed:-unknown}"

    # Check for stall (compare with previous check)
    local prev
    prev=$(grep "^$pod " "$PREV_PROGRESS_FILE" 2>/dev/null | awk '{print $2}' || echo "")
    if [ -n "$prev" ] && [ "$prev" = "$current_step" ]; then
        warn "Training STALLED — no progress since last check ($current_step/$total_step)"
    fi

    # Save current progress
    grep -v "^$pod " "$PREV_PROGRESS_FILE" 2>/dev/null > "${PREV_PROGRESS_FILE}.tmp" || true
    echo "$pod $current_step" >> "${PREV_PROGRESS_FILE}.tmp"
    mv "${PREV_PROGRESS_FILE}.tmp" "$PREV_PROGRESS_FILE"
}

classify_error() {
    local pod="$1"
    local logs
    logs=$(kubectl logs "$pod" --tail=200 2>/dev/null)

    if echo "$logs" | grep -q "none_dealloc"; then
        echo "REFCOUNT_CRASH"
    elif echo "$logs" | grep -q "AttributeError"; then
        echo "ATTRIBUTE_ERROR"
    elif echo "$logs" | grep -q "OutOfMemoryError\|OOMKilled"; then
        echo "OOM"
    elif echo "$logs" | grep -q "SIGABRT"; then
        echo "SIGABRT"
    elif echo "$logs" | grep -q "NCCL\|timeout"; then
        echo "NCCL_TIMEOUT"
    elif echo "$logs" | grep -q "SIGTERM\|SIGKILL"; then
        echo "KILLED"
    else
        echo "UNKNOWN"
    fi
}

find_latest_checkpoint() {
    local pod="$1"
    local output_dir="$2"

    # Try to find checkpoints via any running pod or the failed pod itself
    local running_pod
    running_pod=$(kubectl get pods -l app=grpo-trainer --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    local check_pod="${running_pod:-$pod}"
    local ckpts
    ckpts=$(kubectl exec "$check_pod" -- ls "$output_dir" 2>/dev/null | grep "checkpoint-" | sort -t- -k2 -n | tail -1 || echo "")

    if [ -n "$ckpts" ]; then
        echo "$output_dir/$ckpts"
    else
        echo ""
    fi
}

handle_failure() {
    local pod="$1"
    local job_name
    job_name=$(echo "$pod" | sed 's/-[a-z0-9]*$//')

    local error_type
    error_type=$(classify_error "$pod")

    critical "Pod $pod FAILED — error type: $error_type"

    # Show the actual error
    local error_line
    error_line=$(kubectl logs "$pod" --tail=100 2>/dev/null | grep -E "Fatal|Error|OOM|Traceback|AttributeError" | tail -3)
    log "  Error details: $error_line"

    # Find output dir from pod env
    local output_dir
    output_dir=$(kubectl get pod "$pod" -o jsonpath='{.spec.containers[0].env[?(@.name=="OUTPUT_DIR")].value}' 2>/dev/null || echo "")
    log "  Output dir: ${output_dir:-unknown}"

    # Find latest checkpoint
    if [ -n "$output_dir" ]; then
        local ckpt
        ckpt=$(find_latest_checkpoint "$pod" "$output_dir")
        if [ -n "$ckpt" ]; then
            log "  Latest checkpoint: $ckpt"
            log "  ACTION: Redeploy with --resume_from_checkpoint $ckpt"
        else
            log "  No checkpoint found — must restart from scratch"
        fi
    fi

    log "  Error classification: $error_type"
    case "$error_type" in
        REFCOUNT_CRASH)
            log "  Known issue: vLLM refcount leak. Periodic restart should handle it."
            log "  If recurring, reduce restart interval from 800 to 500 steps."
            ;;
        ATTRIBUTE_ERROR)
            log "  Code bug — check the traceback and fix the attribute access."
            ;;
        OOM)
            log "  Out of memory — reduce batch_size or gpu_memory_utilization."
            ;;
        NCCL_TIMEOUT)
            log "  NCCL timeout — likely a hung process. Safe to restart."
            ;;
        *)
            log "  Unknown error — check full logs: kubectl logs $pod"
            ;;
    esac
}

run_check() {
    echo ""
    echo "================================================================"
    log "GRPO Training Monitor Check"
    echo "================================================================"

    local pods
    pods=$(kubectl get pods -l app=grpo-trainer -o jsonpath='{range .items[*]}{.metadata.name}:{.status.phase}{"\n"}{end}' 2>/dev/null)

    if [ -z "$pods" ]; then
        log "No GRPO training pods found."
        return
    fi

    echo "$pods" | while IFS=: read -r pod phase; do
        [ -z "$pod" ] && continue
        echo ""
        log "--- $pod [$phase] ---"

        case "$phase" in
            Running)
                check_progress "$pod"
                check_stability "$pod"
                ;;
            Succeeded|Completed)
                # Check if it completed at 100% or crashed early
                local final_progress
                final_progress=$(kubectl logs "$pod" --tail=200 2>/dev/null | grep "Training:" | tail -1 | grep -o '[0-9]*/[0-9]*' | head -1)
                local current total
                current=$(echo "$final_progress" | cut -d/ -f1)
                total=$(echo "$final_progress" | cut -d/ -f2)

                if [ -n "$current" ] && [ -n "$total" ] && [ "$current" -lt "$total" ]; then
                    warn "Pod shows Completed but only reached $current/$total — likely crashed"
                    handle_failure "$pod"
                else
                    log "  COMPLETED successfully ($final_progress)"
                fi
                ;;
            Failed|Error)
                handle_failure "$pod"
                ;;
            Pending)
                log "  Waiting for resources (GPU/PVC allocation)"
                # Check how long it's been pending
                local age
                age=$(kubectl get pod "$pod" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null)
                log "  Created at: $age"
                ;;
            *)
                log "  Status: $phase"
                ;;
        esac
    done

    echo ""
    echo "================================================================"
    echo ""
}

# Main loop
touch "$PREV_PROGRESS_FILE"

if [ "$ONCE" = "--once" ]; then
    run_check
    exit 0
fi

log "Starting GRPO monitor (interval: ${INTERVAL}s)"
run_check

while true; do
    sleep "$INTERVAL"
    run_check
done
