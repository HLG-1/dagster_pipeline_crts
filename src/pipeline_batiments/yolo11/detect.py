"""
YOLO — détection pure (bounding boxes) pour pipeline hybride YOLO → SAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import YOLO11Config, default_config, is_segmentation_weights
from .model import YOLO11ModelRegistry
from .preprocess import prepare_tile


@dataclass
class YOLODetection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str


def parse_results(result, names: dict[int, str]) -> list[YOLODetection]:
    out: list[YOLODetection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clss):
        out.append(
            YOLODetection(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(c),
                cls_id=int(cid),
                cls_name=str(names.get(int(cid), str(cid))),
            )
        )
    return out


def boxes_to_mask(
    detections: list[YOLODetection],
    h: int,
    w: int,
    *,
    fill_value: int = 1,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = fill_value
    return mask


def boxes_to_prob(
    detections: list[YOLODetection],
    h: int,
    w: int,
) -> np.ndarray:
    prob = np.zeros((h, w), dtype=np.float32)
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            prob[y1:y2, x1:x2] = np.maximum(prob[y1:y2, x1:x2], d.conf)
    return np.clip(prob, 0, 1)


def draw_boxes(
    rgb: np.ndarray,
    detections: list[YOLODetection],
    *,
    color: tuple[int, int, int] = (0, 255, 128),
    thickness: int = 2,
) -> np.ndarray:
    vis = rgb.copy()
    if vis.shape[2] == 3:
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    else:
        vis_bgr = vis
    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), color[::-1], thickness)
        label = f"{d.cls_name} {d.conf:.2f}"
        cv2.putText(
            vis_bgr, label, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color[::-1], 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


class YOLODetector:
    """Détection Ultralytics — pas de fusion masque / vectorisation."""

    def __init__(self, cfg: YOLO11Config | None = None):
        self.cfg = cfg or default_config()
        from .env import resolved_device

        self.device = resolved_device() if self.cfg.device == "auto" else self.cfg.device
        self._model, self._weights_path, self._label = YOLO11ModelRegistry.from_config(self.cfg)
        self.names = self._model.names
        self.backend_used = f"Détection · {self._weights_path.name}"

    def _device(self) -> int | str:
        if self.device == "cuda":
            return 0
        if self.device == "mps":
            return "mps"
        return "cpu"

    def detect(
        self,
        source: np.ndarray | str | Path,
        *,
        conf: float | None = None,
        iou: float | None = None,
    ) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        prep = prepare_tile(source, tile_size=None)
        h, w = prep.original_size
        conf_v = self.cfg.conf if conf is None else conf
        iou_v = self.cfg.iou if iou is None else iou

        # Utiliser la tâche du .pt (yolov8n_building.pt = segment sans « seg » dans le nom)
        task = getattr(self._model, "task", None) or "detect"
        if task not in ("detect", "segment"):
            task = "segment" if is_segmentation_weights(self._weights_path) else "detect"

        results = self._model.predict(
            prep.rgb,
            task=task,
            conf=conf_v,
            iou=iou_v,
            imgsz=prep.infer_imgsz,
            max_det=300,
            retina_masks=False,
            verbose=False,
            device=self._device(),
        )
        dets = parse_results(results[0], self.names)
        prob = boxes_to_prob(dets, h, w)
        mask = boxes_to_mask(dets, h, w)
        return {
            "detections": dets,
            "prob": prob,
            "mask": mask,
            "boxes_xyxy": [d.xyxy for d in dets],
            "boxes_for_sam": dets,
            "n_detections": len(dets),
            "overlay_boxes": draw_boxes(prep.rgb, dets),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "weights": str(self._weights_path),
            "task": task,
            "coco_no_building_class": "building" not in [
                str(v).lower() for v in self.names.values()
            ],
        }

    def detect_instances(
        self,
        source: np.ndarray | str | Path,
        *,
        conf: float | None = None,
        iou: float | None = None,
    ) -> dict[str, Any]:
        """Détection + masques instance (tête segment de yolov8n_building.pt)."""
        import time

        t0 = time.perf_counter()
        prep = prepare_tile(source, tile_size=None)
        h, w = prep.original_size
        conf_v = self.cfg.conf if conf is None else conf
        iou_v = self.cfg.iou if iou is None else iou
        task = getattr(self._model, "task", None) or "detect"
        if task not in ("detect", "segment"):
            task = "segment" if is_segmentation_weights(self._weights_path) else "detect"
        if is_segmentation_weights(self._weights_path):
            task = "segment"

        results = self._model.predict(
            prep.rgb,
            task=task,
            conf=conf_v,
            iou=iou_v,
            imgsz=prep.infer_imgsz,
            max_det=300,
            retina_masks=True,
            verbose=False,
            device=self._device(),
        )
        result = results[0]
        dets = parse_results(result, self.names)
        instances = _masks_to_instance_map(result, h, w)
        prob = boxes_to_prob(dets, h, w)
        mask = (instances > 0).astype(np.uint8)
        return {
            "detections": dets,
            "instances": instances,
            "prob": prob,
            "mask": mask,
            "boxes_xyxy": [list(d.xyxy) for d in dets],
            "n_detections": len(dets),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "weights": str(self._weights_path),
            "task": task,
        }

    def predict_prob(self, tile_rgb: np.ndarray) -> np.ndarray:
        return self.detect(tile_rgb)["prob"]


def _mask_nms(
    binaries: list[np.ndarray],
    confs: np.ndarray,
    iom_thr: float = 0.60,
) -> list[int]:
    """NMS au niveau des masques : supprime les doublons sur un même toit.

    Critère IoM (intersection / aire du plus petit) : deux boîtes YOLO sur le
    même bâtiment produisent des masques quasi identiques → on garde le plus
    confiant. Renvoie les indices conservés, triés par confiance décroissante.
    """
    order = list(np.argsort(-confs))
    kept: list[int] = []
    for idx in order:
        m = binaries[idx]
        a = float(m.sum())
        if a == 0:
            continue
        duplicate = False
        for k in kept:
            inter = float((m & binaries[k]).sum())
            if inter / min(a, float(binaries[k].sum())) >= iom_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(idx)
    return kept


def _clean_instance(binary: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Nettoie un masque d'instance : lissage, plus grande composante, trous remplis."""
    b = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n > 2:  # fond + >1 fragment → ne garder que le plus grand
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        b = (lbl == biggest).astype(np.uint8)
    # Remplir les trous internes (flood fill depuis le bord)
    ff = b.copy()
    hh, ww = b.shape
    mask_ff = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 1)
    b = b | (1 - ff)
    return b


def _masks_to_instance_map(result, h: int, w: int) -> np.ndarray:
    """Masques YOLO segment → carte d'instances propre.

    Stratégie retenue (meilleure séparation mesurée sur les tuiles test :
    inst_F1 0.353 vs 0.322 brut, IoU pixel 0.402 vs 0.322) :
      1. NMS masques (IoM ≥ 0.6) → supprime les doublons sur un même bâtiment ;
      2. nettoyage par instance (fermeture morpho, plus grande CC, trous remplis) ;
      3. masque restreint à sa boîte de détection (+6 px) → coupe les
         débordements sur les toits voisins ;
      4. attribution par pixel au masque le plus confiant → frontières nettes
         entre bâtiments collés, sans fragment au milieu d'un toit.
    """
    inst = np.zeros((h, w), dtype=np.int32)
    if result.masks is None or len(result.masks) == 0:
        return inst
    masks = result.masks.data.cpu().numpy()
    n = len(masks)
    confs = (
        result.boxes.conf.cpu().numpy()
        if result.boxes is not None and len(result.boxes)
        else np.ones(n, dtype=np.float32)
    )
    boxes = (
        result.boxes.xyxy.cpu().numpy()
        if result.boxes is not None and len(result.boxes)
        else None
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binaries: list[np.ndarray] = []
    for m in masks:
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binaries.append((m > 0.5).astype(np.uint8))

    kept = _mask_nms(binaries, confs, iom_thr=0.60)

    pad = 6
    nid = 0
    claimed = np.zeros((h, w), dtype=bool)
    for idx in kept:  # déjà triés par confiance décroissante
        b = _clean_instance(binaries[idx], k)
        if boxes is not None and idx < len(boxes):
            x1, y1, x2, y2 = boxes[idx]
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(w, int(x2) + pad)
            y2 = min(h, int(y2) + pad)
            clip = np.zeros_like(b)
            clip[y1:y2, x1:x2] = 1
            b = b & clip
        region = (b > 0) & ~claimed
        if int(region.sum()) == 0:
            continue
        nid += 1
        inst[region] = nid
        claimed |= region
    return inst
