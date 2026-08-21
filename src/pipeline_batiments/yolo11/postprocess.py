"""
Post-traitement des masques instance YOLO11 → carte probabilité + masque binaire.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationOutput:
    prob: np.ndarray       # H×W float32 [0,1]
    mask: np.ndarray        # H×W uint8 {0,1}
    n_instances: int
    class_ids: list[int]


def merge_instance_masks(
    masks_data,
    h: int,
    w: int,
    *,
    class_ids: list[int] | None = None,
    allowed_classes: set[int] | None = None,
) -> tuple[np.ndarray, int, list[int]]:
    """Fusionne les masques instance Ultralytics en carte de probabilité."""
    prob = np.zeros((h, w), dtype=np.float32)
    seen_classes: list[int] = []
    n = 0

    if masks_data is None:
        return prob, 0, seen_classes

    for i, mask_tensor in enumerate(masks_data):
        cls_id = class_ids[i] if class_ids and i < len(class_ids) else -1
        if allowed_classes is not None and cls_id >= 0 and cls_id not in allowed_classes:
            continue
        m = mask_tensor.cpu().numpy().astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        prob = np.maximum(prob, m)
        n += 1
        if cls_id >= 0:
            seen_classes.append(cls_id)

    return np.clip(prob, 0, 1), n, seen_classes


def prob_to_binary_mask(
    prob: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area_px: int = 50,
    morph_open: int = 3,
    morph_close: int = 3,
) -> np.ndarray:
    """Seuillage + nettoyage morphologique (production)."""
    mask = (prob >= threshold).astype(np.uint8)
    if mask.max() == 0:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(morph_open, 1), max(morph_open, 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(morph_close, 1), max(morph_close, 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            clean[labels == lbl] = 1
    return clean


def build_output(
    prob: np.ndarray,
    n_instances: int,
    class_ids: list[int],
    *,
    threshold: float,
    min_area_px: int,
) -> SegmentationOutput:
    mask = prob_to_binary_mask(
        prob,
        threshold=threshold,
        min_area_px=min_area_px,
    )
    return SegmentationOutput(
        prob=prob,
        mask=mask,
        n_instances=n_instances,
        class_ids=class_ids,
    )
