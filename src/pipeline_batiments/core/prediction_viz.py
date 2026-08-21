"""Rendu unifié des prédictions (avec ou sans post-traitement instances)."""
from __future__ import annotations

import cv2
import numpy as np

from pipeline_batiments.core.instances import (
    instance_boundaries,
    render_gdf_outlined,
    render_instances_outlined,
)
from pipeline_batiments.core.viz import make_error_overlay, make_overlay

GT_FILL, GT_LINE = "#22C55E", "#EF4444"


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def render_gt_label(
    rgb: np.ndarray,
    meta: dict,
    gt_gdf,
    gt_inst: np.ndarray | None,
) -> np.ndarray:
    """Label vectoriel lissé (GPKG prioritaire)."""
    if gt_gdf is not None and len(gt_gdf) and meta.get("transform") is not None:
        return render_gdf_outlined(
            rgb, gt_gdf, meta["transform"],
            fill_hex=GT_FILL, line_hex=GT_LINE, alpha=0.45, line_thickness=2,
        )
    if gt_inst is not None and int(gt_inst.max()) > 0:
        return render_instances_outlined(
            gt_inst, fill_hex=GT_FILL, line_hex=GT_LINE,
            base=rgb, alpha=0.45, thickness=2,
        )
    return rgb.copy()


def _instance_map_for_viz(
    result: dict,
    meta: dict,
    *,
    use_postprocess: bool,
    min_area: int = 50,
) -> np.ndarray | None:
    """Carte d'instances pour affichage (résultat modèle ou séparation viz)."""
    inst = result.get("instances")
    if inst is not None and int(inst.max()) > 0:
        return inst.astype(np.int32)

    mask = result.get("mask")
    prob = result.get("prob")
    if mask is None or mask.max() == 0:
        return None

    # Séparation viz : même sans post-traitement actif au run (calque UI)
    from pipeline_batiments.core.instances import separate_building_instances

    p = prob.astype(np.float32) if prob is not None else None
    inst_viz = separate_building_instances(
        (mask > 0).astype(np.uint8),
        prob=p,
        min_area=min_area,
    )
    return inst_viz if inst_viz.max() > 0 else None


def render_prediction_layers(
    rgb: np.ndarray,
    result: dict,
    meta: dict,
    *,
    color_hex: str = "#F97316",
    use_postprocess: bool = False,
    min_area: int = 50,
    sep_color: str = "#FFFFFF",
    sep_thickness: int = 3,
) -> dict[str, np.ndarray]:
    """
    Calques indépendants :
      - fill : remplissage uniforme (sans traits)
      - separation : contours entre instances uniquement
    """
    h, w = rgb.shape[:2]
    fill = np.zeros((h, w, 3), dtype=np.uint8)
    sep = np.zeros((h, w, 3), dtype=np.uint8)

    mask = result.get("mask")
    inst = _instance_map_for_viz(result, meta, use_postprocess=use_postprocess, min_area=min_area)
    line = np.array(_hex_rgb(sep_color), dtype=np.uint8)

    # Remplissage = masque binaire lisse (pas les traits déjà dessinés)
    binary = None
    if inst is not None and inst.max() > 0:
        binary = (inst > 0).astype(np.uint8)
    elif mask is not None and mask.max() > 0:
        binary = (mask > 0).astype(np.uint8)

    if binary is not None:
        overlay = make_overlay(rgb, binary, color_hex, alpha=0.48)
        fill[binary > 0] = overlay[binary > 0]

    # Séparation = traits entre instances (visible seulement si ≥ 2 bâtiments)
    if inst is not None and int(inst.max()) > 1:
        bounds = instance_boundaries(inst, sep_thickness)
        sep[bounds] = line
        # Contour sombre pour contraste sur toits clairs
        k = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(bounds.astype(np.uint8), k, iterations=1).astype(bool)
        sep[dilated & ~bounds] = np.array(_hex_rgb("#111827"), dtype=np.uint8)

    n_inst = int(inst.max()) if inst is not None else (1 if binary is not None else 0)
    return {
        "fill": fill,
        "separation": sep,
        "n_instances": n_inst,
        "has_separation": n_inst > 1,
    }


def compose_layered_view(
    rgb: np.ndarray,
    *,
    label_img: np.ndarray,
    pred_fill: np.ndarray,
    pred_separation: np.ndarray,
    layers: dict[str, bool],
) -> np.ndarray:
    """Assemble les calques cochés sur fond noir ou image."""
    h, w = rgb.shape[:2]
    if layers.get("base", True):
        out = rgb.astype(np.float32).copy()
    else:
        out = np.zeros((h, w, 3), dtype=np.float32)

    alpha_pred = 0.55
    if layers.get("prediction", True):
        m = np.any(pred_fill > 0, axis=-1)
        out[m] = out[m] * (1 - alpha_pred) + pred_fill[m].astype(np.float32) * alpha_pred

    alpha_label = 0.52
    if layers.get("label", False):
        diff = np.any(label_img.astype(np.int16) != rgb.astype(np.int16), axis=-1)
        if diff.any():
            out[diff] = out[diff] * (1 - alpha_label) + label_img[diff].astype(np.float32) * alpha_label

    if layers.get("separation", True):
        m = np.any(pred_separation > 0, axis=-1)
        if m.any():
            out[m] = pred_separation[m].astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def render_model_prediction(
    rgb: np.ndarray,
    result: dict,
    meta: dict,
    *,
    color_hex: str = "#F97316",
    use_postprocess: bool = False,
    line_hex: str = "#FBBF24",
) -> np.ndarray:
    """
    Affichage adapté au mode post-traitement.

    - **Sans post-traitement** : overlay sémantique lisse (pas de traits artificiels).
    - **Avec post-traitement** : polygones vectoriels lisses si GPKG dispo,
      sinon instances raster.
    """
    mask = result.get("mask")
    inst = result.get("instances")
    gdf = result.get("gdf")
    transform = meta.get("transform")

    if not use_postprocess:
        if mask is not None and mask.max() > 0:
            return make_overlay(rgb, mask, color_hex, alpha=0.45)
        return rgb.copy()

    if gdf is not None and len(gdf) > 0 and transform is not None:
        return render_gdf_outlined(
            rgb, gdf, transform,
            fill_hex=color_hex, line_hex=line_hex,
            alpha=0.45, line_thickness=2,
        )

    if inst is not None and int(inst.max()) > 0:
        return render_instances_outlined(
            inst, fill_hex=color_hex, line_hex=line_hex,
            base=rgb, alpha=0.45, thickness=2,
        )

    if mask is not None and mask.max() > 0:
        return make_overlay(rgb, mask, color_hex, alpha=0.45)
    return rgb.copy()


def render_error_view(
    rgb: np.ndarray,
    result: dict,
    gt_bin: np.ndarray | None,
    *,
    use_postprocess: bool = False,
) -> np.ndarray:
    if gt_bin is None:
        return rgb.copy()
    if not use_postprocess and result.get("mask") is not None:
        return make_error_overlay(rgb, result["mask"], gt_bin)
    pred_bin = result.get("mask")
    if result.get("instances") is not None and result["instances"].max() > 0:
        pred_bin = (result["instances"] > 0).astype(np.uint8)
    if pred_bin is None:
        return rgb.copy()
    return make_error_overlay(rgb, pred_bin, gt_bin)
