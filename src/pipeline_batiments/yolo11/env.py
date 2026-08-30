"""
Charge les variables d'environnement depuis .env (racine projet).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

def _find_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "checkpoints").is_dir() or (p / "workspace.yaml").is_file() or (p / "config").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()


def load_dotenv() -> None:
    """Idempotent — appelé au import de yolo11.config."""
    try:
        from dotenv import load_dotenv as _load

        _load(ROOT / ".env", override=False)
    except ImportError:
        pass


def env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def resolved_device() -> str:
    """
    YOLO11_DEVICE=auto|cuda|mps|cpu|0
    """
    load_dotenv()
    d = env_str("YOLO11_DEVICE", "auto").lower()
    if d in ("0", "cuda", "gpu"):
        return "cuda" if _cuda_ok() else ("mps" if _mps_ok() else "cpu")
    if d == "mps":
        return "mps" if _mps_ok() else "cpu"
    if d == "cpu":
        return "cpu"
    # auto
    if _cuda_ok():
        return "cuda"
    if _mps_ok():
        return "mps"
    return "cpu"


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _mps_ok() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False
