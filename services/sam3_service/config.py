"""Configuration du micro-service SAM 3 (variables d'environnement)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAM3_HOST = os.environ.get("SAM3_HOST", "0.0.0.0")
SAM3_PORT = int(os.environ.get("SAM3_PORT", "8077"))
SAM3_DEVICE = os.environ.get("SAM3_DEVICE", "cuda")
SAM3_DTYPE = os.environ.get("SAM3_DTYPE", "float16")  # float16 | bfloat16 | float32
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "")  # vide = HF auto
# Si aucun chemin local n’est fourni, on évite le téléchargement Hugging Face protégé.
_DEFAULT_LOCAL_CKPT = ROOT / "sam3_weights" / "sam3.pt"
if not SAM3_CHECKPOINT and _DEFAULT_LOCAL_CKPT.is_file():
    SAM3_CHECKPOINT = str(_DEFAULT_LOCAL_CKPT)
# Checkpoint trainer (fine-tune local) : fusionné sur les poids HF (backbone/transformer).
# Ex. sam3_logs/building_ft/checkpoints/checkpoint.pt
_DEFAULT_FT = ROOT / "sam3_logs" / "building_ft_seg" / "checkpoints" / "checkpoint.pt"
SAM3_FT_CHECKPOINT = os.environ.get("SAM3_FT_CHECKPOINT", "")
if not SAM3_FT_CHECKPOINT and os.environ.get("SAM3_USE_FT", "").strip() in ("1", "true", "yes"):
    SAM3_FT_CHECKPOINT = str(_DEFAULT_FT) if _DEFAULT_FT.is_file() else ""
SAM3_HF_REPO = os.environ.get("SAM3_HF_REPO", "facebook/sam3")
SAM3_CONFIDENCE = float(os.environ.get("SAM3_CONFIDENCE", "0.25"))
SAM3_LABEL = os.environ.get("SAM3_LABEL", "base")  # "base" | "ft" (affiché dans /health)
