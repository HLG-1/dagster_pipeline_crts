#!/usr/bin/env bash
# Reprend le FT SAM3 masques depuis checkpoint.pt (epochs 2–3 restants).
# Fix dtype val : config building_ft_seg.yaml avec amp bf16 activé.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/sam3_logs"
CKPT="$LOGDIR/building_ft_seg/checkpoints/checkpoint.pt"
STATUS="$LOGDIR/building_ft_seg_STATUS.txt"
RESUME_LOG="$LOGDIR/building_ft_seg_resume.log"

if [[ ! -f "$CKPT" ]]; then
  echo "❌ Checkpoint introuvable : $CKPT"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -f "$LOGDIR/.hf_token" ]]; then
  export HF_TOKEN="$(tr -d '\n\r' < "$LOGDIR/.hf_token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

cat > "$STATUS" <<EOF
updated: $(date -Is)
state: starting
detail: resume epochs 2-3 from checkpoint.pt
checkpoint: $CKPT
tail: n/a
EOF

echo "=== Reprise FT seg $(date -Is) ===" | tee -a "$RESUME_LOG"
echo "Checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))" | tee -a "$RESUME_LOG"

cd "$ROOT"
nohup bash scripts/launch_sam3_finetune.sh --seg-resume >> "$RESUME_LOG" 2>&1 &
echo "PID=$!"
echo "$!" > "$LOGDIR/building_ft_seg_resume.pid"
