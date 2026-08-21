#!/usr/bin/env bash
# ============================================================
# run.sh — Lance le micro-service SAM3 Meta (port 8077)
#
# Repris de run_sam3_service.sh du collegue. SEUL changement : ROOT
# remonte de 2 niveaux (services/sam3_service/ -> racine du repo) au
# lieu d'1 seul, meme correctif que install.sh.
#
# Usage :
#   bash services/sam3_service/run.sh          # premier plan
#   bash services/sam3_service/run.sh --tmux   # session tmux persistante
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
# Detection conda pour Windows/Git Bash et Linux
if command -v conda >/dev/null 2>&1; then
  CONDA_EXE=$(command -v conda)
elif [ -f "$HOME/anaconda3/Scripts/conda.exe" ]; then
  CONDA_EXE="$HOME/anaconda3/Scripts/conda.exe"
elif [ -f "$HOME/miniconda3/Scripts/conda.exe" ]; then
  CONDA_EXE="$HOME/miniconda3/Scripts/conda.exe"
elif [ -f "/c/ProgramData/Anaconda3/Scripts/conda.exe" ]; then
  CONDA_EXE="/c/ProgramData/Anaconda3/Scripts/conda.exe"
elif [ -f "/c/Users/$USER/anaconda3/Scripts/conda.exe" ]; then
  CONDA_EXE="/c/Users/$USER/anaconda3/Scripts/conda.exe"
else
  echo "❌ conda introuvable"
  echo "   Chemins verifies :"
  echo "   - \$HOME/anaconda3/Scripts/conda.exe"
  echo "   - \$HOME/miniconda3/Scripts/conda.exe"
  echo "   - /c/ProgramData/Anaconda3/Scripts/conda.exe"
  echo "   - /c/Users/\$USER/anaconda3/Scripts/conda.exe"
  exit 1
fi
# Activation conda pour Windows
if [[ "$CONDA_EXE" == *.exe ]]; then
  if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    CONDA_ROOT=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_ROOT/Scripts/activate" "$ENV_NAME"
  fi
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    conda activate "$ENV_NAME"
  fi
fi
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
  if [ -f "$ROOT/services/sam3_service/server.py" ]; then
    exec python -m services.sam3_service.server
  else
    echo "❌ Fichier server.py introuvable dans services/sam3_service/"
    echo "   Verifiez que le service SAM3 est installe"
    exit 1
  fi
}
if [ "$USE_TMUX" = "1" ]; then
  SESSION="sam3_service"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session tmux '$SESSION' deja active. Attach : tmux attach -t $SESSION"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash -lc '$(declare -f _run); _run'"
  sleep 2
  if curl -sf -m 5 "http://127.0.0.1:${SAM3_PORT}/health" >/dev/null; then
    echo "✓ Service SAM3 demarre dans tmux session '$SESSION'"
  else
    echo "⏳ Service en cours de chargement (modele ~1-3 min). Voir : tmux attach -t $SESSION"
  fi
else
  _run
fi