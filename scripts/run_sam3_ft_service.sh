#!/usr/bin/env bash
# ============================================================
# run_sam3_ft_service.sh — SAM 3 fine-tuné (building) sur le port 8078
#
# Charge les poids HF + fusionne building_ft_seg (ou building_ft concept-only)
# Usage :
#   bash scripts/run_sam3_ft_service.sh          # premier plan
#   bash scripts/run_sam3_ft_service.sh --tmux   # session tmux
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${SAM3_ENV:-sam3}"
USE_TMUX=0
if [ "${1:-}" = "--tmux" ]; then
  USE_TMUX=1
fi

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ -n "${SAM3_FT_CHECKPOINT:-}" ]; then
  CKPT="$SAM3_FT_CHECKPOINT"
elif [ -f "$ROOT/sam3_logs/building_ft_seg/checkpoints/checkpoint.pt" ]; then
  CKPT="$ROOT/sam3_logs/building_ft_seg/checkpoints/checkpoint.pt"
else
  CKPT="$ROOT/sam3_logs/building_ft/checkpoints/checkpoint.pt"
fi
if [ ! -f "$CKPT" ]; then
  echo "❌ Checkpoint FT introuvable : $CKPT"
  echo "   Lance d'abord le fine-tuning, ou exporte SAM3_FT_CHECKPOINT=..."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAM3_HOST="${SAM3_HOST:-0.0.0.0}"
export SAM3_PORT="${SAM3_PORT:-8078}"
export SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
export SAM3_DTYPE="${SAM3_DTYPE:-float16}"
export SAM3_FT_CHECKPOINT="$CKPT"
export SAM3_USE_FT=1
export SAM3_LABEL="ft"

# Base Meta locale (évite un téléchargement HF si le token est invalide / « hf_… »)
HF_SAM3_PT="${SAM3_BASE_CHECKPOINT:-}"
if [ -z "$HF_SAM3_PT" ]; then
  HF_SAM3_PT="$(find "$HOME/.cache/huggingface/hub/models--facebook--sam3" -name 'sam3.pt' 2>/dev/null | head -1 || true)"
fi
if [ -n "$HF_SAM3_PT" ] && [ -f "$HF_SAM3_PT" ]; then
  export SAM3_CHECKPOINT="$HF_SAM3_PT"
  echo "  Base HF locale : $SAM3_CHECKPOINT"
else
  unset SAM3_CHECKPOINT || true
fi

# Neutralise un token placeholder qui casse httpx (caractère …)
_bad_tok() {
  local t="${1:-}"
  [[ -z "$t" || "$t" == *…* || "$t" == *'...' ]]
}
if _bad_tok "${HF_TOKEN:-}"; then unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true; fi

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda introuvable"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || true

# Repli : binaire explicite de l'env (conda activate parfois inerte sous nohup)
PY="${SAM3_PYTHON:-$HOME/miniconda3/envs/${ENV_NAME}/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python)"
fi
echo "  Python : $PY"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  SAM3 Meta FINE-TUNÉ — micro-service HTTP         ║"
echo "╚═══════════════════════════════════════════════════╝"
"$PY" - <<'PY'
import torch
print(f"  Device : {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  PyTorch: {torch.__version__}")
PY
echo "  URL    : http://${SAM3_HOST}:${SAM3_PORT}"
echo "  FT ckpt: $CKPT"
echo "  Health : curl http://127.0.0.1:${SAM3_PORT}/health"
echo ""

_run() {
  cd "$ROOT"
  exec "$PY" -m services.sam3_service.server
}

if [ "$USE_TMUX" = "1" ]; then
  SESSION="sam3_ft_service"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session tmux '$SESSION' déjà active. Attach : tmux attach -t $SESSION"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash -lc '$(declare -f _run); _run'"
  sleep 2
  if curl -sf -m 5 "http://127.0.0.1:${SAM3_PORT}/health" >/dev/null; then
    echo "✓ Service SAM3 FT démarré dans tmux session '$SESSION'"
  else
    echo "⏳ Service en cours de chargement (modèle ~1-3 min). Voir : tmux attach -t $SESSION"
  fi
else
  _run
fi
