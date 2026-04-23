#!/bin/bash
# =============================================================================
# Watcher: wait for the syncnet-trained DF run's 0005000.pth to land,
# then launch the mouthweight-14B SF training run.
# =============================================================================
# Intended to run in a detached tmux session so it survives the controlling
# terminal dying. Polls the DF checkpoint directory every 5 minutes;
# launches SF once the final ckpt exists AND its byte size matches the
# previous 500-step ckpt (ensures the write is complete and atomic).
#
# Safety: the DF run has produced 9 ckpts so far, all exactly 8529367455
# bytes. Matching on that exact size catches partial writes.
#
# Usage (inside tmux):
#   tmux new-session -d -s sf_trigger \
#     'bash /home/work/.local/hyunbin/FastGen-redmd/scripts/run_sf_after_df.sh'
#   tmux attach -t sf_trigger
#
# To abort the waiter:
#   tmux kill-session -t sf_trigger
# =============================================================================

set -uo pipefail   # not -e: want the exit line below to run even if SF crashes

# Pin CWD to FastGen-redmd so the SF script's relative paths resolve correctly
# regardless of which tmux session / shell spawned this watcher.
cd /home/work/.local/hyunbin/FastGen-redmd

DF_DIR="/home/work/.local/hyunbin/FastGen-redmd/FASTGEN_OUTPUT/OmniAvatar-FastGen/omniavatar_df_audiofix/df_audiofix_syncnet_trained_shift_5_4gpu_bs16_lr1e5_5000iter/checkpoints"
DF_CKPT="${DF_DIR}/0005000.pth"
REF_CKPT="${DF_DIR}/0004500.pth"
SF_SCRIPT="/home/work/.local/hyunbin/FastGen-redmd/scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.sh"
SF_LOG="/tmp/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight.log"
WAIT_LOG="/tmp/run_sf_after_df_wait.log"

POLL_INTERVAL_SEC=300          # 5 min while waiting for the file
STABLE_CHECK_INTERVAL_SEC=30   # 30 s once the file appears (check it stopped growing)
STABLE_CYCLES_REQUIRED=2       # 2 × 30 s = 60 s of stable size at expected bytes
TIMEOUT_HOURS=14               # abort if DF hasn't finished within 14h (run-time ~10h; ample buffer)

log() {
    # Write to both stdout (for tmux attach) and a persistent log.
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$WAIT_LOG"
}

log "=========================================="
log "  DF -> SF watcher starting"
log "=========================================="
log "  Waiting for: $DF_CKPT"
log "  SF script:   $SF_SCRIPT"
log "  SF log:      $SF_LOG"
log "  Wait log:    $WAIT_LOG"

if [[ ! -f "$REF_CKPT" ]]; then
    log "ERROR: reference ckpt $REF_CKPT missing — cannot determine expected size."
    log "       (DF run may have been reset; please verify manually before re-launching.)"
    exit 1
fi

if [[ ! -x "$SF_SCRIPT" ]]; then
    log "ERROR: SF script $SF_SCRIPT is not executable or missing."
    exit 1
fi

EXPECTED_SIZE=$(stat -c%s "$REF_CKPT")
log "  Expected ckpt size: $EXPECTED_SIZE bytes (from $REF_CKPT)"

DEADLINE=$(($(date +%s) + TIMEOUT_HOURS * 3600))

# Phase 1: wait for the file to exist at all.
log "Phase 1: polling for file existence (every ${POLL_INTERVAL_SEC}s)..."
while [[ ! -f "$DF_CKPT" ]]; do
    if [[ $(date +%s) -ge $DEADLINE ]]; then
        log "TIMEOUT after ${TIMEOUT_HOURS}h — DF ckpt never arrived. Aborting."
        exit 1
    fi
    sleep "$POLL_INTERVAL_SEC"
done
log "File appeared: $DF_CKPT"

# Phase 2: wait for its size to stabilize at the expected size.
log "Phase 2: waiting for write to finish (size must equal ${EXPECTED_SIZE} bytes, stable for $((STABLE_CHECK_INTERVAL_SEC * STABLE_CYCLES_REQUIRED))s)..."
last_size=-1
stable=0
while true; do
    current_size=$(stat -c%s "$DF_CKPT" 2>/dev/null || echo 0)
    if [[ "$current_size" == "$EXPECTED_SIZE" && "$current_size" == "$last_size" ]]; then
        stable=$((stable + 1))
        log "  size stable at $current_size (cycle $stable/$STABLE_CYCLES_REQUIRED)"
        if [[ $stable -ge $STABLE_CYCLES_REQUIRED ]]; then
            break
        fi
    else
        if [[ $stable -gt 0 ]]; then
            log "  size changed from $last_size to $current_size — resetting stability counter"
        fi
        stable=0
    fi
    last_size=$current_size
    sleep "$STABLE_CHECK_INTERVAL_SEC"
done
log "Ckpt ready: ${current_size} bytes, matches reference."

# Phase 3: launch SF training.
log "Phase 3: launching SF training..."
log "       bash $SF_SCRIPT 2>&1 | tee $SF_LOG"
log "=========================================="
bash "$SF_SCRIPT" 2>&1 | tee "$SF_LOG"
rc=${PIPESTATUS[0]}
log "SF training exited with code $rc"
exit "$rc"
