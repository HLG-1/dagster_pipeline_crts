#!/usr/bin/env bash
# ============================================================
# install_sam3_meta.sh — Environnement conda `sam3` + SAM3 Meta officiel
#
# Prérequis :
#   - Accès Hugging Face approuvé pour facebook/sam3
#   - hf auth login (après installation)
#
# Usage : bash scripts/install_sam3_meta.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${SAM3_ENV:-sam3}"
VENDOR="$ROOT/third_party/sam3"
SAM3_REPO="${SAM3_GIT_URL:-https://github.com/facebookresearch/sam3.git}"

# Détection conda pour Windows/Git Bash et WSL/Linux
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
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
  CONDA_EXE="$HOME/anaconda3/bin/conda"
elif [ -f "/opt/conda/bin/conda" ]; then
  CONDA_EXE="/opt/conda/bin/conda"
else
  echo "❌ conda introuvable"
  echo "   Chemins vérifiés :"
  echo "   - \$HOME/anaconda3/Scripts/conda.exe (Windows)"
  echo "   - \$HOME/miniconda3/Scripts/conda.exe (Windows)"
  echo "   - /c/ProgramData/Anaconda3/Scripts/conda.exe (Windows)"
  echo "   - /c/Users/\$USER/anaconda3/Scripts/conda.exe (Windows)"
  echo "   - \$HOME/miniconda3/bin/conda (Linux/WSL)"
  echo "   - \$HOME/anaconda3/bin/conda (Linux/WSL)"
  echo "   - /opt/conda/bin/conda (Linux/WSL)"
  exit 1
fi

echo "✓ Conda trouvé : $CONDA_EXE"

# Activation conda pour Windows
if [[ "$CONDA_EXE" == *.exe ]]; then
  # Windows conda
  CONDA_ROOT=$(dirname $(dirname "$CONDA_EXE"))
  source "$CONDA_ROOT/Scripts/activate"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "▶ Env conda SAM3 : $ENV_NAME"
# Désactiver SSL temporairement pour Git Bash
export SSL_VERIFY=0
conda config --set ssl_verify false
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda activate "$ENV_NAME"
else
  conda create -n "$ENV_NAME" python=3.12 -y
  conda activate "$ENV_NAME"
fi
conda config --set ssl_verify true

# PyTorch récent (requis SAM3 officiel — voir README Meta)
TORCH_INDEX="${SAM3_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
echo "  PyTorch (index $TORCH_INDEX)"
pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url "$TORCH_INDEX"

# Clone SAM3 dans third_party/ (dans le projet uniquement)
mkdir -p "$ROOT/third_party"
if [ ! -d "$VENDOR/.git" ]; then
  echo "  Clone $SAM3_REPO → $VENDOR"
  git clone --depth 1 "$SAM3_REPO" "$VENDOR"
else
  echo "  Repo SAM3 déjà présent : $VENDOR"
fi

echo "  pip install -e $VENDOR"
pip install -e "$VENDOR"
# Installer les extras manuellement
pip install jupyter notebook tensorboard

# Dépendances parfois absentes du resolver pip (import runtime SAM3)
pip install einops pycocotools psutil

# Dépendances micro-service HTTP
if [ -f "$ROOT/services/sam3_service/requirements-sam3.txt" ]; then
  pip install -r "$ROOT/services/sam3_service/requirements-sam3.txt"
fi

# clip Ultralytics (requis par certains chemins SAM3)
pip uninstall clip -y 2>/dev/null || true
pip install "git+https://github.com/ultralytics/CLIP.git" 2>/dev/null || true

python - <<'PY'
import sys
print("Python", sys.version)
import torch
print("PyTorch", torch.__version__, "| CUDA", torch.cuda.is_available())
try:
    from sam3.model_builder import build_sam3_image_model
    print("sam3 package : OK")
except Exception as e:
    print("sam3 package :", e)
    print("→ Après hf auth login, relancez ce script ou testez le service.")
PY

echo ""
echo "✓ Env $ENV_NAME installé."
echo ""
echo "  IMPORTANT — authentification Hugging Face (checkpoints SAM3) :"
echo "    conda activate $ENV_NAME"
echo "    hf auth login"
echo ""
echo "  Puis démarrer le service :"
echo "    bash scripts/run_sam3_service.sh"
