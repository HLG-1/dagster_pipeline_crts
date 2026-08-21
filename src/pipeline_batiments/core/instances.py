"""
Segmentation par instance (bâtiment par bâtiment).

Le jeu vectoriel `dataset_bati_vector` fournit la vérité-terrain sous forme de
polygones séparés (un par bâtiment, GPKG). Ce module :

  - rasterise les polygones GT en carte d'instances (1..N) ;
  - sépare les masques prédits (sémantiques) en instances via marqueurs
    adaptatifs sur la transformée de distance (par composante connexe) ;
  - vectorise une carte d'instances en polygones (contour + régularisation optionnelle) ;
  - calcule des métriques d'instance (matching par IoU : P/R/F1@IoU, PQ, comptage).

Toutes les cartes d'instances sont des int32 : 0 = fond, 1..N = bâtiments.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import rasterio.features
from shapely.geometry import shape
import geopandas as gpd

from .geo_io import safe_crs

_EPS = 1e-8


# ── Séparation en instances ─────────────────────────────────────
def _estimate_peak_distance(area_px: float, min_area: int) -> int:
    """Distance min entre centres de bâtiments (~1/3 du rayon équivalent)."""
    if area_px <= 0:
        return 8
    r = (area_px / 3.14159) ** 0.5
    return int(max(6, min(28, r / 3.0)))


def _markers_from_distance(
    dist: np.ndarray,
    mask: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    """Marqueurs stables via h-maxima + pics locaux (centres de toits)."""
    from skimage.morphology import h_maxima

    m = mask.astype(bool)
    if not m.any() or float(dist[m].max()) <= 0:
        return np.zeros(dist.shape, dtype=np.int32)

    dmax = float(dist[m].max())
    h = max(1.5, 0.15 * dmax)
    peaks = h_maxima(dist.astype(np.float64), h)
    peaks &= m
    markers, n = ndi.label(peaks.astype(np.uint8))
    if n >= 2:
        return markers.astype(np.int32)

    coords = peak_local_max(
        dist,
        min_distance=min_distance,
        labels=m.astype(np.int32),
        exclude_border=True,
    )
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    return markers


def _split_component(
    comp: np.ndarray,
    prob: np.ndarray | None,
    min_area: int,
    min_distance: int | None,
    opening_px: int,
) -> np.ndarray:
    """Sépare les bâtiments collés à l'intérieur d'un blob binaire."""
    m = (comp > 0).astype(np.uint8)
    area = int(m.sum())
    if area < min_area:
        return np.zeros_like(m, dtype=np.int32)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_px, opening_px))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    if int(opened.sum()) >= max(min_area, area // 3):
        sub = connected_instances(opened, min_area=max(10, min_area // 2))
        if int(sub.max()) > 1:
            return sub

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    md = min_distance or _estimate_peak_distance(float(area), min_area)
    markers = _markers_from_distance(dist, m, md)
    n_markers = int(markers.max())
    if n_markers <= 1:
        out = np.zeros_like(m, dtype=np.int32)
        out[m > 0] = 1
        return out

    ws_surface = -dist.astype(np.float64)
    if prob is not None:
        p = prob.astype(np.float64)
        if p.max() > 1:
            p = p / max(p.max(), 1e-6)
        ws_surface = -dist * (0.35 + 0.65 * np.clip(p, 0, 1))
        ws_surface[~m.astype(bool)] = 0

    labels = watershed(ws_surface, markers, mask=m).astype(np.int32)
    out = np.zeros_like(labels, dtype=np.int32)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        region = labels == lab
        if int(region.sum()) >= min_area:
            nid += 1
            out[region] = nid
    if nid == 0:
        out[m > 0] = 1
    return out


def separate_building_instances(
    binary_mask: np.ndarray,
    prob: np.ndarray | None = None,
    min_area: int = 50,
    min_distance: int | None = None,
    opening_px: int = 3,
) -> np.ndarray:
    """
    Séparation robuste bâtiment par bâtiment (tuiles satellite).

    Traite **chaque composante connexe** séparément (évite les coupes diagonales
    globales). Étapes : ouverture morphologique → marqueurs h-maxima sur la
    transformée de distance → watershed local guidé par la proba.
    """
    m = (binary_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.int32)

    n, lbl = cv2.connectedComponents(m, connectivity=8)
    out = np.zeros(m.shape, dtype=np.int32)
    next_id = 0
    for lab in range(1, n):
        comp = lbl == lab
        if int(comp.sum()) < min_area:
            continue
        sub = _split_component(
            comp.astype(np.uint8),
            prob,
            min_area=min_area,
            min_distance=min_distance,
            opening_px=opening_px,
        )
        for sid in range(1, int(sub.max()) + 1):
            region = sub == sid
            if int(region.sum()) >= min_area:
                next_id += 1
                out[region] = next_id
    return out


def split_large_instance_map(
    instances: np.ndarray,
    prob: np.ndarray | None = None,
    min_area: int = 50,
    area_factor: float = 2.2,
) -> np.ndarray:
    """Re-sépare les grosses instances natives (YOLO/SAM) en plusieurs bâtiments."""
    if instances is None or int(instances.max()) == 0:
        return instances

    out = np.zeros_like(instances, dtype=np.int32)
    next_id = 0
    for lab in range(1, int(instances.max()) + 1):
        comp = (instances == lab).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        md = _estimate_peak_distance(float(area), min_area)
        n_peaks = int(_markers_from_distance(dist, comp, md).max())
        needs_split = area >= min_area * area_factor and n_peaks >= 2
        if needs_split:
            sub = _split_component(comp, prob, min_area, md, opening_px=3)
        else:
            sub = np.zeros_like(comp, dtype=np.int32)
            sub[comp > 0] = 1
        for sid in range(1, int(sub.max()) + 1):
            region = sub == sid
            if int(region.sum()) >= min_area:
                next_id += 1
                out[region] = next_id
    return out


def separate_instances(
    binary_mask: np.ndarray,
    min_area: int = 50,
    min_distance: int = 7,
    boundary: np.ndarray | None = None,
) -> np.ndarray:
    """Masque binaire → carte d'instances int32 (watershed distance transform).

    Args:
        binary_mask : (H,W) 0/1 — masque sémantique « bâtiment ».
        min_area    : aire min (px) d'une instance conservée.
        min_distance: distance min entre deux pics (sépare les bâtiments collés).
        boundary    : (H,W) optionnel — proba/masque de contour pour creuser
                      les frontières entre bâtiments avant watershed.
    """
    m = (binary_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.int32)

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if boundary is not None:
        b = boundary.astype(np.float32)
        if b.max() > 1:
            b = b / 255.0
        dist = np.clip(dist - dist.max() * 0.5 * b, 0, None)

    # Marqueurs = maxima locaux de la distance (centres de bâtiments)
    coords = peak_local_max(
        dist, min_distance=min_distance, labels=m, exclude_border=False,
    )
    peaks = np.zeros(dist.shape, dtype=bool)
    if len(coords):
        peaks[tuple(coords.T)] = True
    markers, n_markers = ndi.label(peaks)

    if n_markers == 0:
        n, lbl = cv2.connectedComponents(m, connectivity=8)
        labels = lbl.astype(np.int32)
    else:
        labels = watershed(-dist, markers, mask=m).astype(np.int32)

    # Filtre des petites instances + renumérotation compacte
    out = np.zeros_like(labels, dtype=np.int32)
    nid = 0
    for lab in range(1, int(labels.max()) + 1):
        comp = labels == lab
        if comp.sum() >= min_area:
            nid += 1
            out[comp] = nid
    return out


def connected_instances(binary_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Instances = simples composantes connexes (pas de séparation)."""
    m = (binary_mask > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(lbl, dtype=np.int32)
    nid = 0
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            nid += 1
            out[lbl == lab] = nid
    return out


# ── Rasterisation / vectorisation ─────────────────────────────────
def rasterize_polygons(gdf, transform, shape_hw) -> np.ndarray:
    """GeoDataFrame de polygones → carte d'instances int32 (1..N).

    Les petits bâtiments sont peints en premier pour éviter qu'un grand polygone
    écrase entièrement un petit (chevauchement aux bordures / cours intérieures).
    """
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape_hw, dtype=np.int32)

    rows = [
        (i + 1, geom)
        for i, geom in enumerate(gdf.geometry)
        if geom is not None and not geom.is_empty
    ]
    if not rows:
        return np.zeros(shape_hw, dtype=np.int32)

    # Plus petits d'abord → chaque bâtiment garde au moins ses pixels propres
    rows.sort(key=lambda item: item[1].area)

    out = np.zeros(shape_hw, dtype=np.int32)
    for lab, geom in rows:
        layer = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=shape_hw,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        unlabeled = (layer > 0) & (out == 0)
        out[unlabeled] = lab
    return out


def instances_to_polygons_rectfit(
    inst: np.ndarray, transform, crs_str: str, min_area_px: int = 1,
):
    """Carte d'instances → (GeoDataFrame, RectFitStats) via re-fit géométrique.

    Rectangle tourné / rectilinéaire par instance, orientation partagée
    par îlot (rect_regularizer). Colonne ``reg_level`` = fit retenu.
    """
    from shapely.ops import transform as shp_transform
    from shapely.validation import make_valid

    from .rect_regularizer import RectilinearRegularizer, _largest_polygon

    crs = safe_crs(crs_str)
    reg = RectilinearRegularizer()
    pix_polys, stats = reg.process_instance_map(inst)
    levels = {r.building_id: r.level.value for r in stats.results}

    def _pix2geo(poly):
        if transform is None:
            return poly
        from rasterio.transform import xy

        def _tf(x, y, z=None):
            gx, gy = xy(transform, y, x)
            return (gx, gy)

        return _largest_polygon(make_valid(shp_transform(_tf, poly)))

    rows = []
    for lab, poly in sorted(pix_polys.items()):
        area_px = int((inst == lab).sum())
        if area_px < min_area_px:
            continue
        geo = _pix2geo(poly)
        if geo is None or geo.is_empty:
            continue
        rows.append({
            "geometry": geo,
            "building_id": lab,
            "area_m2": round(geo.area, 2),
            "area_px": area_px,
            "reg_level": levels.get(lab, ""),
        })
    if not rows:
        gdf = gpd.GeoDataFrame(
            columns=["geometry", "building_id", "area_m2", "area_px", "reg_level"],
            crs=crs)
    else:
        try:
            gdf = gpd.GeoDataFrame(rows, crs=crs)
        except Exception:
            gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf, stats


def instances_to_polygons(
    inst: np.ndarray, transform, crs_str: str,
    simplify_tol: float = 0.5, min_area_px: int = 1,
    regularize: bool = True,
    method: str = "rectfit",
) -> gpd.GeoDataFrame:
    """Carte d'instances → GeoDataFrame (un polygone par bâtiment).

    Si ``regularize=True`` (défaut), ``method`` choisit le régularisateur :
      - ``"rectfit"``  (défaut) : re-fit géométrique — rectangle tourné /
        rectilinéaire, orientation partagée par îlot (rect_regularizer).
      - ``"topology"`` : partition topologique globale (topology_regularizer).
    """
    crs = safe_crs(crs_str)
    if regularize and method == "topology":
        from .topology_regularizer import TopologicalBuildingRegularizer, TopologyConfig
        topo = TopologicalBuildingRegularizer(TopologyConfig(enable_shape_refiner=False))
        gdf, _ = topo.process_instance_map(
            inst, transform=transform, crs_str=crs_str, min_area_px=min_area_px,
        )
        return gdf
    if regularize:
        gdf, _ = instances_to_polygons_rectfit(
            inst, transform, crs_str, min_area_px=min_area_px,
        )
        return gdf

    polys = []
    for lab in range(1, int(inst.max()) + 1):
        comp = (inst == lab).astype(np.uint8)
        area_px = int(comp.sum())
        if area_px < min_area_px:
            continue
        best = None
        for geom_dict, val in rasterio.features.shapes(
            comp, mask=comp, transform=transform,
        ):
            if val == 0:
                continue
            poly = shape(geom_dict).simplify(simplify_tol, preserve_topology=True)
            if poly.is_empty or not poly.is_valid:
                continue
            if best is None or poly.area > best.area:
                best = poly

        if best is not None and not best.is_empty:
            polys.append({
                "geometry": best,
                "building_id": lab,
                "area_m2": round(best.area, 2),
                "area_px": area_px,
            })
    if not polys:
        return gpd.GeoDataFrame(columns=["geometry", "building_id", "area_m2", "area_px"], crs=crs)
    try:
        return gpd.GeoDataFrame(polys, crs=crs)
    except Exception:
        return gpd.GeoDataFrame(polys, crs="EPSG:4326")


# ── Métriques d'instance (matching par IoU) ───────────────────────
def _iou_pairs(gt: np.ndarray, pred: np.ndarray):
    """Renvoie (iou, gt_id, pred_id) pour chaque paire qui se recouvre."""
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    gt_area = np.bincount(gt.ravel())
    pred_area = np.bincount(pred.ravel())
    both = (gt > 0) & (pred > 0)
    if not both.any():
        return []
    stride = int(pred.max()) + 1
    keys = gt[both] * stride + pred[both]
    uniq, cnt = np.unique(keys, return_counts=True)
    g_ids = uniq // stride
    p_ids = uniq % stride
    pairs = []
    for gi, pi, inter in zip(g_ids, p_ids, cnt):
        union = gt_area[gi] + pred_area[pi] - inter
        pairs.append((inter / (union + _EPS), int(gi), int(pi)))
    return pairs


def instance_metrics(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> dict:
    """Métriques d'instance par matching glouton (IoU décroissant).

    Renvoie comptages GT/pred, TP/FP/FN, Precision/Recall/F1 @ IoU,
    IoU moyen des paires appariées et Panoptic Quality (PQ).
    """
    n_gt = int(len(np.unique(gt)) - (1 if (gt == 0).any() else 0))
    n_pred = int(len(np.unique(pred)) - (1 if (pred == 0).any() else 0))

    pairs = sorted(_iou_pairs(gt, pred), reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, gi, pi in pairs:
        if iou < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append(iou)

    tp = len(matched)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    f1 = 2 * precision * recall / (precision + recall + _EPS)
    mean_iou = float(np.mean(matched)) if matched else 0.0
    pq = (sum(matched) / (tp + 0.5 * fp + 0.5 * fn + _EPS)) if (tp + fp + fn) else 0.0

    return {
        "n_GT": n_gt,
        "n_pred": n_pred,
        "count_err": n_pred - n_gt,
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "mean_IoU": round(mean_iou, 4),
        "PQ": round(pq, 4),
    }


# ── Visualisation ────────────────────────────────────────────────
def _label_colors(n: int, seed: int = 12345) -> np.ndarray:
    """Palette déterministe de n couleurs distinctes (uint8 RGB)."""
    rng = np.random.default_rng(seed)
    cols = rng.integers(40, 256, size=(max(n, 1), 3), dtype=np.uint8)
    return cols


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def instance_boundaries(inst: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Masque des traits de séparation entre bâtiments (bord interne).

    Un pixel est « frontière » s'il appartient à un bâtiment et touche un pixel
    d'étiquette différente (autre bâtiment OU fond). Cela trace donc à la fois le
    contour des bâtiments isolés et la ligne entre deux bâtiments collés.
    """
    inst = inst.astype(np.int32)
    diff = np.zeros(inst.shape, dtype=bool)
    diff[:, :-1] |= inst[:, :-1] != inst[:, 1:]
    diff[:, 1:] |= inst[:, 1:] != inst[:, :-1]
    diff[:-1, :] |= inst[:-1, :] != inst[1:, :]
    diff[1:, :] |= inst[1:, :] != inst[:-1, :]
    b = diff & (inst > 0)
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        b = cv2.dilate(b.astype(np.uint8), k) > 0
    return b


def render_instances_outlined(
    inst: np.ndarray,
    fill_hex: str = "#22C55E",
    line_hex: str = "#EF4444",
    base: np.ndarray | None = None,
    alpha: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """Bâtiments en **une seule couleur** + **trait** de séparation entre collés.

    - `base=None`  → fond noir, bâtiments pleins (vue « Label/masque seul ».
    - `base=image` → remplissage semi-transparent sur l'image (vue overlay).
    Les traits (line_hex) délimitent chaque bâtiment : on distingue deux bâtiments
    accolés même s'ils sont de la même couleur.
    """
    h, w = inst.shape
    fill = np.array(_hex_rgb(fill_hex), dtype=np.uint8)
    line = np.array(_hex_rgb(line_hex), dtype=np.uint8)
    m = inst > 0
    if base is None:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[m] = fill
    else:
        out = base.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        out[m] = (out[m] * (1 - alpha) + fill * alpha).astype(np.uint8)
    out[instance_boundaries(inst, thickness)] = line
    return out


def _iter_polygon_rings(geom):
    """Anneaux extérieurs d'un Polygon / MultiPolygon."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom.exterior.coords
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            if not poly.is_empty:
                yield poly.exterior.coords


def render_gdf_outlined(
    rgb: np.ndarray,
    gdf,
    transform,
    fill_hex: str = "#22C55E",
    line_hex: str = "#EF4444",
    alpha: float = 0.45,
    line_thickness: int = 2,
) -> np.ndarray:
    """Dessine les polygones GPKG/SHP en traits vectoriels lisses (anti-alias).

    Contrairement à ``render_instances_outlined`` (carte raster 512×512), les
    contours suivent la géométrie SIG — comme dans QGIS.
    """
    import rasterio.transform

    if gdf is None or len(gdf) == 0:
        return rgb.copy()

    out = rgb.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    h, w = out.shape[:2]
    fill = np.array(_hex_rgb(fill_hex), dtype=np.float32)
    line = tuple(int(c) for c in _hex_rgb(line_hex))

    fill_mask = np.zeros((h, w), dtype=np.uint8)
    for geom in gdf.geometry:
        for ring in _iter_polygon_rings(geom):
            coords = np.asarray(ring, dtype=np.float64)
            if len(coords) < 3:
                continue
            rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
            pts = np.stack([cols, rows], axis=1).astype(np.int32)
            cv2.fillPoly(fill_mask, [pts], 255)
            cv2.polylines(out, [pts], True, line, line_thickness, cv2.LINE_AA)

    m = fill_mask > 0
    blended = out.astype(np.float32)
    blended[m] = blended[m] * (1 - alpha) + fill * alpha
    out = np.clip(blended, 0, 255).astype(np.uint8)

    for geom in gdf.geometry:
        for ring in _iter_polygon_rings(geom):
            coords = np.asarray(ring, dtype=np.float64)
            if len(coords) < 3:
                continue
            rows, cols = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])
            pts = np.stack([cols, rows], axis=1).astype(np.int32)
            cv2.polylines(out, [pts], True, line, line_thickness, cv2.LINE_AA)
    return out


def colorize_instances(inst: np.ndarray) -> np.ndarray:
    """Carte d'instances → image RGB (une couleur par bâtiment, fond noir)."""
    h, w = inst.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    n = int(inst.max())
    if n == 0:
        return out
    cols = _label_colors(n)
    for lab in range(1, n + 1):
        out[inst == lab] = cols[lab - 1]
    return out


def overlay_instances(
    rgb: np.ndarray, inst: np.ndarray, alpha: float = 0.45, outline: bool = True,
) -> np.ndarray:
    """Superpose les instances colorées sur l'image + contours blancs (séparation)."""
    base = rgb.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    color = colorize_instances(inst)
    mask = inst > 0
    out = base.copy()
    out[mask] = (base[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    if outline:
        for lab in range(1, int(inst.max()) + 1):
            comp = (inst == lab).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out
