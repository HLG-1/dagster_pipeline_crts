"""
Quick metrics computation for the UI (no PyTorch needed).
All functions work on numpy uint8/float32 arrays.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def compute_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Binary segmentation metrics from probability map + ground-truth mask.

    Args:
        pred : (H, W) float32 probabilities [0,1] or binary uint8
        gt   : (H, W) uint8 binary mask (0/1 or 0/255)
        threshold: binarisation threshold for pred

    Returns:
        dict with iou, dice, precision, recall, f1, accuracy, n_buildings_pred
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin   = (gt   > 127).astype(np.uint8) if gt.max() > 1 else gt.astype(np.uint8)

    tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (gt_bin == 0)).sum())

    eps = 1e-8
    iou       = tp / (tp + fp + fn + eps)
    dice      = 2*tp / (2*tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2*precision*recall / (precision + recall + eps)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)

    # Count predicted buildings
    n_lbl, _ = cv2.connectedComponents(pred_bin, connectivity=8)
    n_buildings = n_lbl - 1

    return {
        "IoU":          round(iou, 4),
        "Dice":         round(dice, 4),
        "Precision":    round(precision, 4),
        "Recall":       round(recall, 4),
        "F1":           round(f1, 4),
        "Accuracy":     round(accuracy, 4),
        "Buildings predicted": n_buildings,
        "TP px": tp, "FP px": fp, "FN px": fn,
    }


def measure_latency(fn, *args, n: int = 3, **kwargs) -> tuple[any, float]:
    """Run fn(*args) n times, return (result, mean_ms)."""
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return result, float(np.mean(times))
