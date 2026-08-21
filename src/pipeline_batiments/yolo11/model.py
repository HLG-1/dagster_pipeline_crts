"""
Chargement unique YOLO (.pt local ou nom Ultralytics → téléchargement auto).
"""
from __future__ import annotations

import threading
from pathlib import Path

from .config import YOLO11Config, resolve_weights_path


class YOLO11ModelRegistry:
    _lock = threading.Lock()
    _models: dict[str, object] = {}

    @classmethod
    def get(cls, weights: str | Path) -> tuple[object, Path]:
        w = Path(weights)
        key = str(w.resolve()) if w.is_file() else w.name
        with cls._lock:
            if key not in cls._models:
                from ultralytics import YOLO

                cls._models[key] = YOLO(str(w))
            resolved = w.resolve() if w.is_file() else Path(w.name)
            return cls._models[key], resolved

    @classmethod
    def from_config(cls, cfg: YOLO11Config | None = None) -> tuple[object, Path, str]:
        cfg = cfg or YOLO11Config.from_env()
        w = resolve_weights_path(cfg)
        model, path = cls.get(w)
        if cfg.use_finetuned and path == cfg.best_weights.resolve():
            label = "fine-tuné local"
        elif path.is_file() and "building" in path.name.lower():
            label = "bâtiments (local)"
        elif not path.is_file() or "yolo11" in path.name.lower():
            label = "COCO pré-entraîné"
        else:
            label = "pré-entraîné"
        return model, path, label

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._models.clear()
