"""Configuration du micro-service SAM 3 (variables d'environnement)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAM3_HOST = os.environ.get("SAM3_HOST", "0.0.0.0")
SAM3_PORT = int(os.environ.get("SAM3_PORT", "8077"))
SAM3_DEVICE = os.environ.get("SAM3_DEVICE", "cuda")
SAM3_DTYPE = os.environ.get("SAM3_DTYPE", "bfloat16")  # bfloat16 | float32 | float16
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "")  # vide = HF auto ou auto-détection locale

# Recherche automatique de poids locaux
_LOCAL_BASE_CANDIDATES = [
    ROOT / "sam3_weights" / "sam3.pt",
    ROOT / "checkpoints" / "pretrained" / "sam3.pt",
    Path("/app/sam3_ckpt/sam3.pt"),
    Path("/app/sam3_ckpt/checkpoint.pt"),
    ROOT / "sam3_logs" / "building_ft_seg" / "checkpoints" / "sam3.pt",
    ROOT / "sam3_logs" / "building_ft_seg" / "checkpoints" / "checkpoint.pt",
]

if not SAM3_CHECKPOINT:
    for cand in _LOCAL_BASE_CANDIDATES:
        if cand.is_file():
            SAM3_CHECKPOINT = str(cand)
            break

# Checkpoint trainer (fine-tune local) : fusionné sur les poids HF ou de base
_FT_CANDIDATES = [
    ROOT / "sam3_logs" / "building_ft_seg" / "checkpoints" / "sam3.pt",
    ROOT / "sam3_logs" / "building_ft_seg" / "checkpoints" / "checkpoint.pt",
    ROOT / "sam3_logs" / "building_ft" / "checkpoints" / "sam3.pt",
    ROOT / "sam3_logs" / "building_ft" / "checkpoints" / "checkpoint.pt",
    Path("/app/sam3_ckpt/sam3.pt"),
    Path("/app/sam3_ckpt/checkpoint.pt"),
]

SAM3_FT_CHECKPOINT = os.environ.get("SAM3_FT_CHECKPOINT", "")
if not SAM3_FT_CHECKPOINT and os.environ.get("SAM3_USE_FT", "1").strip() in ("1", "true", "yes"):
    for cand in _FT_CANDIDATES:
        if cand.is_file():
            SAM3_FT_CHECKPOINT = str(cand)
            break

SAM3_HF_REPO = os.environ.get("SAM3_HF_REPO", "facebook/sam3")
SAM3_CONFIDENCE = float(os.environ.get("SAM3_CONFIDENCE", "0.25"))
SAM3_LABEL = os.environ.get("SAM3_LABEL", "building_ft_seg" if SAM3_FT_CHECKPOINT else "base")
