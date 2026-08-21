"""
Generation de prompts « multiprompt » (boite + point) pour SAM3, d'apres
Seo & Kim (2025).
Pour chaque boite YOLO, on calcule un point interieur au batiment par
traitement d'image (Canny + contours + erosion/dilatation), utilise comme
prompt positif supplementaire par SAM3.

Utilise par : assets/segmentation.py (etape 04)
"""
from __future__ import annotations

import cv2
import numpy as np


def building_mask_from_crop(
    crop_rgb: np.ndarray,
    canny_lo: int = 50,
    canny_hi: int = 150,
    blur_ksize: int = 5,
    close_iter: int = 2,
    min_area_frac: float = 0.05,
) -> np.ndarray:
    """Masque approximatif du batiment dans une vignette."""
    if crop_rgb.size == 0:
        return np.zeros((1, 1), np.uint8)
    if crop_rgb.ndim == 2:
        crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    val = hsv[..., 2]
    edges = cv2.Canny(val, canny_lo, canny_hi)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=close_iter)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(val.shape, np.uint8)
    cv2.drawContours(mask, cnts, -1, 255, thickness=cv2.FILLED)

    if blur_ksize >= 3:
        bk = blur_ksize | 1
        mask = cv2.GaussianBlur(mask, (bk, bk), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    mask = cv2.erode(mask, k, iterations=1)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = mask.shape[0] * mask.shape[1]
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * area:
            out[lbl == i] = 255
    out = cv2.dilate(out, k, iterations=1)
    return out


def point_in_crop(crop_rgb: np.ndarray) -> tuple[float, float]:
    """Point central du batiment dans la vignette (coords vignette)."""
    h, w = crop_rgb.shape[:2]
    mask = building_mask_from_crop(crop_rgb)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return (w / 2.0, h / 2.0)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[biggest]
    if lbl[int(round(cy)), int(round(cx))] != biggest:
        ys, xs = np.where(lbl == biggest)
        cy, cx = float(ys.mean()), float(xs.mean())
        if lbl[int(round(cy)), int(round(cx))] != biggest:
            j = len(xs) // 2
            cx, cy = float(xs[j]), float(ys[j])
    return (float(cx), float(cy))


def point_for_box(image_rgb: np.ndarray, box_xyxy, pad: int = 2) -> list[float]:
    """Point central (coords image pleine) pour une boite [x1,y1,x2,y2] px."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return [float((x1 + x2) / 2), float((y1 + y2) / 2)]
    crop = image_rgb[y1:y2, x1:x2]
    px, py = point_in_crop(crop)
    return [float(x1 + px), float(y1 + py)]


def points_for_boxes(image_rgb: np.ndarray, boxes_xyxy) -> list[list[float]]:
    """Un point central par boite (multiprompt complet)."""
    return [point_for_box(image_rgb, b) for b in boxes_xyxy]




"""
Orthophoto
      │
      ▼
YOLO
      │
      ▼
Bounding Box
      │
      ▼
Découpage de la boîte (crop)
      │
      ▼
Canny
      │
      ▼
Contours
      │
      ▼
Morphologie (fermeture, érosion, dilatation)
      │
      ▼
Masque approximatif du bâtiment
      │
      ▼
Calcul du centroïde
      │
      ▼
Point positif
      │
      ▼
Boîte + Point
      │
      ▼
SAM3
      │
      ▼
Segmentation plus précise

"""