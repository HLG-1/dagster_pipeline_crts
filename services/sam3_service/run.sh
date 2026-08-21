#!/usr/bin/env bash
# Lance le service SAM3 (env conda sam3, cf scripts/run_sam3_service.sh
# du collegue - a adapter ici avec les chemins de ce repo).
set -euo pipefail

#!/usr/bin/env bash
# ============================================================
# run_sam3_service.sh — Lance le micro-service SAM3 Meta (port 8077)
#
# Usage :
#   bash scripts/run_sam3_service.sh          # premier plan
#   bash scripts/run_sam3_service.sh --tmux   # session tmux persistante
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAM3_HOST="${SAM3_HOST:-0.0.0.0}"
export SAM3_PORT="${SAM3_PORT:-8077}"
export SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
export SAM3_DTYPE="${SAM3_DTYPE:-float16}"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda introuvable"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  SAM3 Meta — micro-service HTTP                   ║"
echo "╚═══════════════════════════════════════════════════╝"
python - <<'PY'
import torch
print(f"  Device : {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  PyTorch: {torch.__version__}")
PY
echo "  URL    : http://${SAM3_HOST}:${SAM3_PORT}"
echo "  Health : curl http://127.0.0.1:${SAM3_PORT}/health"
echo ""

_run() {
  cd "$ROOT"
  exec python -m services.sam3_service.server
}

if [ "$USE_TMUX" = "1" ]; then
  SESSION="sam3_service"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session tmux '$SESSION' déjà active. Attach : tmux attach -t $SESSION"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash -lc '$(declare -f _run); _run'"
  sleep 2
  if curl -sf -m 5 "http://127.0.0.1:${SAM3_PORT}/health" >/dev/null; then
    echo "✓ Service SAM3 démarré dans tmux session '$SESSION'"
  else
    echo "⏳ Service en cours de chargement (modèle ~1-3 min). Voir : tmux attach -t $SESSION"
  fi
else
  _run
fi
