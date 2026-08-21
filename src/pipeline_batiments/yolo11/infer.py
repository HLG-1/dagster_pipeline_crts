"""
Inférence YOLO11-seg — façade vers YOLO11BuildingService (rétrocompat CLI / scripts).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import YOLO11Config, default_config, resolve_weights_path
from .service import YOLO11BuildingService, get_service


class YOLO11Segmenter:
    """Wrapper historique — délègue au service production."""

    def __init__(self, cfg: YOLO11Config | None = None, weights: str | Path | None = None):
        cfg = cfg or default_config()
        if weights is not None:
            cfg.pretrained_weights = Path(weights)
        self.cfg = cfg
        self._service = YOLO11BuildingService(cfg)
        self.model = self._service._model
        self.weights_path = str(self._service.weights_path)
        self.backend_used = self._service.backend_used

    def predict(
        self,
        source: np.ndarray | str | Path,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> dict:
        if device:
            self._service.device = device
        out = self._service.segment(source, conf=conf, iou=iou, return_ultralytics=True)
        return {
            "prob": out["prob"],
            "mask": out["mask"],
            "overlay": out["overlay"],
            "results": out.get("results"),
            "boxes": out.get("results").boxes if out.get("results") else None,
        }

    def predict_prob(self, tile_rgb: np.ndarray, device: str | None = None) -> np.ndarray:
        if device:
            self._service.device = device
        return self._service.predict_prob(tile_rgb)

    draw_overlay = staticmethod(YOLO11BuildingService.draw_overlay)

    def predict_and_save(self, source: str | Path, out_dir: str | Path, device: str | None = None) -> Path:
        if device:
            self._service.device = device
        return self._service.predict_and_save(source, out_dir)
