#!/usr/bin/env bash
# ============================================================
# run_venv.sh — Lance le micro-service SAM3, variante SANS conda
# (utilise le venv .venv-sam3 avec --system-site-packages).
#
# Usage : bash services/sam3_service/run_venv.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAM3_HOST="${SAM3_HOST:-0.0.0.0}"
export SAM3_PORT="${SAM3_PORT:-8077}"
export SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
export SAM3_DTYPE="${SAM3_DTYPE:-float16}"

VENV_PATH="${SAM3_VENV_PATH:-$ROOT/.venv-sam3}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "❌ Venv introuvable : $VENV_PATH"
  echo "   Creez-le d'abord : python3 -m venv .venv-sam3 --system-site-packages"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  SAM3 Meta — micro-service HTTP (venv, sans conda)║"
echo "╚═══════════════════════════════════════════════════╝"
python - <<'PY'
import torch
print(f"  Device : {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  PyTorch: {torch.__version__}")
PY
echo "  URL    : http://${SAM3_HOST}:${SAM3_PORT}"
echo "  Health : curl http://127.0.0.1:${SAM3_PORT}/health"
echo ""

if [ ! -f "$ROOT/services/sam3_service/server.py" ]; then
  echo "❌ Fichier server.py introuvable dans services/sam3_service/"
  exit 1
fi
exec python -m services.sam3_service.server
