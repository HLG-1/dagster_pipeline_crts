"""
Visualisation utilities: overlays, side-by-side comparisons, footprint drawing.
All functions return numpy arrays (H, W, 3|4) or matplotlib figures.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ── Colour palette (synced with pretrained_catalog) ─────────────

def _build_model_colors() -> dict[str, str]:
    try:
        from pipeline_batiments.core.pretrained_catalog import models_sorted
        return {spec.display_name: spec.color for spec in models_sorted()}
    except ImportError:
        return {
            "U-Net EfficientNet-B4": "#3498DB",
            "SegFormer MiT-B5":      "#E67E22",
            "UPerNet Swin-B":        "#2ECC71",
            "SAM ViT-B":             "#9B59B6",
            "MSHFormer":             "#E74C3C",
        }


MODEL_COLORS_HEX = _build_model_colors()

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── Core overlay ──────────────────────────────────────────────

def make_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_hex: str = "#FF6600",
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary mask over an RGB image.
    Returns (H, W, 3) uint8.
    """
    overlay = rgb.copy().astype(np.float32)
    color   = np.array(hex_to_rgb(color_hex), dtype=np.float32)
    region  = mask > 0
    overlay[region] = overlay[region] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_error_overlay(
    rgb: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    TP=green, FP=red, FN=blue overlay.
    Returns (H, W, 3) uint8.
    """
    overlay = rgb.copy().astype(np.float32)
    tp = (pred > 0) & (gt > 0)
    fp = (pred > 0) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    overlay[tp] = overlay[tp] * (1-alpha) + np.array([68, 255, 68])  * alpha
    overlay[fp] = overlay[fp] * (1-alpha) + np.array([255, 68, 68])  * alpha
    overlay[fn] = overlay[fn] * (1-alpha) + np.array([68, 68, 255])  * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── Contour drawing ───────────────────────────────────────────

def draw_contours(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_hex: str = "#FF6600",
    thickness: int = 2,
) -> np.ndarray:
    """Draw building contours on the RGB image."""
    out  = rgb.copy()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color   = hex_to_rgb(color_hex)[::-1]  # BGR for cv2
    cv2.drawContours(out, cnts, -1, color, thickness)
    return out


# ── Mask colourisation ────────────────────────────────────────

def colorise_mask(mask: np.ndarray, color_hex: str = "#FF6600") -> np.ndarray:
    """Binary mask → (H, W, 3) coloured uint8."""
    color = np.array(hex_to_rgb(color_hex), dtype=np.uint8)
    out   = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


# ── Vue couches (Image / Masque / Label) — rendu unifié ───────

PRED_FILL_HEX = "#F97316"
PRED_LINE_HEX = "#EF4444"
GT_FILL_HEX = "#22C55E"
GT_LINE_HEX = "#EF4444"


def compose_layer_view(
    rgb: np.ndarray,
    pred_inst: np.ndarray | None,
    gt_inst: np.ndarray | None,
    *,
    show_image: bool = True,
    show_pred: bool = True,
    show_label: bool = False,
    alpha: float = 0.45,
    thickness: int = 2,
    pred_borders: bool = False,
    label_borders: bool = True,
) -> np.ndarray:
    """Compose une vue à partir de cases à cocher — même design pour tous les modèles."""
    from pipeline_batiments.core.instances import render_instances_outlined

    h, w = rgb.shape[:2]
    has_pred = pred_inst is not None and int(pred_inst.max()) > 0
    has_gt = gt_inst is not None and int(gt_inst.max()) > 0

    if not show_image and not show_pred and not show_label:
        return np.zeros((h, w, 3), dtype=np.uint8)

    canvas: np.ndarray | None = None

    if show_image:
        canvas = rgb.copy()

    if show_pred and has_pred:
        if pred_borders:
            canvas = render_instances_outlined(
                pred_inst,
                fill_hex=PRED_FILL_HEX,
                line_hex=PRED_LINE_HEX,
                base=canvas,
                alpha=alpha,
                thickness=thickness,
            )
        else:
            canvas = make_overlay(
                canvas if canvas is not None else rgb,
                (pred_inst > 0).astype(np.uint8),
                PRED_FILL_HEX,
                alpha=alpha,
            )

    if show_label and has_gt:
        if label_borders:
            canvas = render_instances_outlined(
                gt_inst,
                fill_hex=GT_FILL_HEX,
                line_hex=GT_LINE_HEX,
                base=canvas,
                alpha=alpha,
                thickness=thickness,
            )
        else:
            canvas = make_overlay(
                canvas if canvas is not None else rgb,
                (gt_inst > 0).astype(np.uint8),
                GT_FILL_HEX,
                alpha=alpha,
            )

    if canvas is None:
        if show_pred and has_pred:
            if pred_borders:
                canvas = render_instances_outlined(
                    pred_inst,
                    fill_hex=PRED_FILL_HEX,
                    line_hex=PRED_LINE_HEX,
                    base=None,
                    alpha=1.0,
                    thickness=thickness,
                )
            else:
                canvas = colorise_mask((pred_inst > 0).astype(np.uint8), PRED_FILL_HEX)
        elif show_label and has_gt:
            if label_borders:
                canvas = render_instances_outlined(
                    gt_inst,
                    fill_hex=GT_FILL_HEX,
                    line_hex=GT_LINE_HEX,
                    base=None,
                    alpha=1.0,
                    thickness=thickness,
                )
            else:
                canvas = colorise_mask((gt_inst > 0).astype(np.uint8), GT_FILL_HEX)
        else:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

    return canvas


# ── Side-by-side figure ───────────────────────────────────────

def side_by_side(
    rgb: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    title: str = "",
    color_hex: str = "#FF6600",
) -> plt.Figure:
    """3-panel figure: Original | Mask | Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb);     axes[0].set_title("Original",   fontweight="bold")
    axes[1].imshow(colorise_mask(mask, color_hex), cmap=None)
    axes[1].set_title("Predicted Mask", fontweight="bold")
    axes[2].imshow(overlay); axes[2].set_title("Overlay",    fontweight="bold")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Model comparison figure ───────────────────────────────────

def comparison_figure(
    rgb: np.ndarray,
    results: list[dict],   # [{"name": str, "mask": ndarray, "color": str}]
    max_cols: int = 3,
) -> plt.Figure:
    """
    Grid: Original + one panel per model.
    """
    n     = len(results) + 1
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes_flat = np.array(axes).flatten()

    axes_flat[0].imshow(rgb)
    axes_flat[0].set_title("Original", fontweight="bold", fontsize=12)
    axes_flat[0].axis("off")

    for i, res in enumerate(results, start=1):
        ov = make_overlay(rgb, res["mask"], res["color"])
        axes_flat[i].imshow(ov)
        axes_flat[i].set_title(res["name"], fontweight="bold", fontsize=11,
                                color=res["color"])
        axes_flat[i].axis("off")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


# ── Footprint drawing ─────────────────────────────────────────

def draw_footprints(rgb: np.ndarray, gdf, transform=None) -> np.ndarray:
    """
    Draw GeoDataFrame polygon footprints on RGB image.
    Works even without a geo-transform (pixel coordinates assumed).
    """
    import rasterio.transform
    out = rgb.copy()

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.exterior.coords)
            if transform is not None:
                # Project geo-coords → pixel coords
                pixels = [
                    rasterio.transform.rowcol(transform, x, y)
                    for x, y in coords
                ]
                pts = np.array([[c, r] for r, c in pixels], dtype=np.int32)
            else:
                pts = np.array([[int(x), int(y)] for x, y in coords], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=(255, 165, 0), thickness=2)
        except Exception:
            continue
    return out


# ── Heatmap ───────────────────────────────────────────────────

def prob_to_heatmap(prob: np.ndarray) -> np.ndarray:
    """
    Convert (H, W) float32 probability map → (H, W, 3) uint8 heatmap (jet colormap).
    """
    norm = np.clip(prob, 0, 1)
    heat = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR→RGB


# ── PIL conversion helper ─────────────────────────────────────

def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
