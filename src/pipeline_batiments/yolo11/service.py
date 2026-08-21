"""
YOLO11 — inférence Ultralytics simple (COCO pré-entraîné, sans fine-tune).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    COCO_BUILDING_LIKE,
    SANITY_CHECK_NOTICE,
    YOLO11Config,
    default_config,
    is_segmentation_weights,
)
from .env import resolved_device
from .model import YOLO11ModelRegistry
from .preprocess import prepare_tile


def _merge_masks_raw(masks_data, h: int, w: int) -> tuple[np.ndarray, int]:
    prob = np.zeros((h, w), dtype=np.float32)
    if masks_data is None:
        return prob, 0
    n = 0
    for mask_tensor in masks_data:
        m = mask_tensor.cpu().numpy().astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        prob = np.maximum(prob, m)
        n += 1
    return np.clip(prob, 0, 1), n


def _boxes_to_prob(boxes, h: int, w: int, *, allowed_classes: set[int] | None) -> tuple[np.ndarray, int]:
    """Détection yolo11l.pt / yolo11x.pt → carte prob (rectangles remplis)."""
    prob = np.zeros((h, w), dtype=np.float32)
    if boxes is None or len(boxes) == 0:
        return prob, 0
    n = 0
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
        if allowed_classes is not None and cls_id not in allowed_classes:
            continue
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        prob[y1:y2, x1:x2] = np.maximum(prob[y1:y2, x1:x2], float(c))
        n += 1
    return np.clip(prob, 0, 1), n


class YOLO11BuildingService:
    """predict() Ultralytics — segment (-seg.pt) ou detect (.pt)."""

    def __init__(self, cfg: YOLO11Config | None = None):
        self.cfg = cfg or default_config()
        self.device = resolved_device() if self.cfg.device == "auto" else self.cfg.device
        self._model, self._weights_path, self._weight_label = YOLO11ModelRegistry.from_config(self.cfg)
        self._task = getattr(self._model, "task", None) or self.cfg.task
        if self._task not in ("detect", "segment"):
            self._task = "segment" if is_segmentation_weights(self._weights_path) else "detect"

        name = self._weights_path.name
        self._is_coco_generic = (
            "yolo11" in name.lower() or "yolov8" not in name.lower()
        ) and "building" not in name.lower()
        self.coco_sanity = self._is_coco_generic and not self.cfg.use_finetuned
        tag = "COCO · sanity check" if self.coco_sanity else self._weight_label
        self.backend_used = f"YOLO11 · {name} ({tag})"

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    def _yolo_device(self) -> int | str:
        if self.device == "cuda":
            return 0
        if self.device == "mps":
            return "mps"
        return "cpu"

    def _allowed_classes(self) -> set[int] | None:
        if self.cfg.filter_coco_building_like and self.coco_sanity:
            return set(COCO_BUILDING_LIKE)
        return None

    def segment(
        self,
        source: np.ndarray | str | Path,
        *,
        conf: float | None = None,
        iou: float | None = None,
        return_ultralytics: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        prep = prepare_tile(source, tile_size=None)
        h, w = prep.original_size
        conf_v = conf if conf is not None else self.cfg.conf
        iou_v = iou if iou is not None else self.cfg.iou
        allowed = self._allowed_classes()

        results = self._model.predict(
            prep.rgb,
            task=self._task,
            conf=conf_v,
            iou=iou_v,
            imgsz=prep.infer_imgsz,
            max_det=300,
            retina_masks=self._task == "segment",
            verbose=False,
            device=self._yolo_device(),
        )
        r = results[0]

        if self._task == "segment":
            prob, n_inst = _merge_masks_raw(
                r.masks.data if r.masks is not None else None, h, w
            )
        else:
            prob, n_inst = _boxes_to_prob(r.boxes, h, w, allowed_classes=allowed)

        thr = self.cfg.mask_threshold
        mask = (prob >= thr).astype(np.uint8)
        overlay = self.draw_overlay(prep.rgb, mask)
        latency_ms = (time.perf_counter() - t0) * 1000

        payload: dict[str, Any] = {
            "prob": prob,
            "mask": mask,
            "overlay": overlay,
            "n_instances": n_inst,
            "n_buildings": n_inst,
            "coverage_pct": float(mask.mean() * 100),
            "latency_ms": round(latency_ms, 1),
            "weights": str(self._weights_path),
            "device": self.device,
            "coco_sanity": self.coco_sanity,
            "notice": SANITY_CHECK_NOTICE if self.coco_sanity else "",
            "meta": {
                "task": self._task,
                "infer_imgsz": prep.infer_imgsz,
                "conf": conf_v,
                "iou": iou_v,
            },
        }
        if return_ultralytics:
            payload["results"] = r
        return payload

    def predict_prob(self, tile_rgb: np.ndarray) -> np.ndarray:
        return self.segment(tile_rgb)["prob"]

    @staticmethod
    def draw_overlay(
        rgb: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (255, 80, 60),
        alpha: float = 0.45,
    ) -> np.ndarray:
        vis = rgb.copy().astype(np.uint8)
        m = (mask > 0).astype(bool)
        colored = np.zeros_like(vis)
        colored[:, :, 0], colored[:, :, 1], colored[:, :, 2] = color
        blend = vis.astype(np.float32)
        blend[m] = (1 - alpha) * blend[m] + alpha * colored[m].astype(np.float32)
        return np.clip(blend, 0, 255).astype(np.uint8)

    def predict_and_save(
        self,
        source: str | Path,
        out_dir: str | Path,
        *,
        threshold: float | None = None,
    ) -> Path:
        if threshold is not None:
            self.cfg.mask_threshold = threshold
        out = self.segment(source)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem
        cv2.imwrite(str(out_dir / f"{stem}_yolo_mask.png"), out["mask"] * 255)
        np.save(out_dir / f"{stem}_yolo_prob.npy", out["prob"])
        cv2.imwrite(
            str(out_dir / f"{stem}_yolo_overlay.jpg"),
            cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR),
        )
        return out_dir / f"{stem}_yolo_mask.png"


_service: YOLO11BuildingService | None = None


def get_service(cfg: YOLO11Config | None = None) -> YOLO11BuildingService:
    global _service
    if _service is None or cfg is not None:
        _service = YOLO11BuildingService(cfg)
    return _service
