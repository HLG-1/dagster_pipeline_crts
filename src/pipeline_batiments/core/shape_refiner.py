"""
Affinage géométrique par classification de forme.

Philosophie : généralisation cartographique (comme les labels), pas fidélité
pixel-par-pixel au masque. Les bâtiments rectangulaires → 4 sommets (rectangle
orienté). Formes L/U significatives → orthogonal complexe. Sinon simplifié.

Intégré après la régularisation topologique globale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from .building_regularizer import (
    BuildingPolygonRegularizer,
    RegularizerConfig,
    _angle_diff_deg,
    _deg2rad,
    _largest_polygon,
    _norm_angle_deg,
    _rad2deg,
)


class ShapeClass(str, Enum):
    RECTANGLE = "rectangle"
    ORTHOGONAL_COMPLEX = "orthogonal_complex"
    IRREGULAR = "irregular"


@dataclass
class ShapeRefinerConfig:
    """Seuils calibrables (512×512 px)."""
    # Rectangle probable
    min_rectangularity: float = 0.82
    min_rectangle_iou: float = 0.80
    max_area_change: float = 0.15
    max_mean_contour_dist: float = 4.0
    max_concavity_ratio: float = 0.06
    min_l_shape_concavity: float = 0.10
    min_l_shape_iou_gain: float = 0.04

    # Décrochements
    max_notch_depth_px: float = 4.0
    max_notch_area_ratio: float = 0.015
    max_notch_width_px: float = 5.0

    # Optimisation rectangle orienté
    angle_search_deg: float = 5.0
    angle_step_deg: float = 1.0
    scale_search: tuple[float, ...] = (0.97, 1.0, 1.03)
    translate_px: tuple[float, ...] = (-1.5, 0.0, 1.5)

    # Score parcimonie
    lambda_area: float = 0.35
    lambda_distance: float = 0.08
    lambda_vertices: float = 0.06

    # Orthogonal complexe
    orthogonal_snap_deg: float = 12.0
    min_complex_vertices: int = 6
    max_complex_vertices: int = 10
    simplify_ratio: float = 0.008

    # Règle décision explicite
    rule_rectangularity: float = 0.85
    rule_rectangle_iou: float = 0.82


@dataclass
class ShapeMetrics:
    n_vertices: int = 0
    rectangularity: float = 0.0
    rectangle_iou: float = 0.0
    area_change_ratio: float = 0.0
    mean_contour_dist: float = 0.0
    concavity_ratio: float = 0.0
    max_notch_depth: float = 0.0
    dominant_angle_deg: float = 0.0
    mask_iou_initial: float = 0.0
    mask_iou_final: float = 0.0


@dataclass
class ShapeRefineResult:
    building_id: int
    polygon: Polygon | None
    shape_class: ShapeClass
    n_vertices_before: int = 0
    n_vertices_after: int = 0
    metrics: ShapeMetrics = field(default_factory=ShapeMetrics)
    rectangle_candidate: Polygon | None = None
    complex_candidate: Polygon | None = None
    message: str = ""


@dataclass
class ShapeRefineStats:
    n_rectangle: int = 0
    n_orthogonal: int = 0
    n_irregular: int = 0
    pct_4_vertices: float = 0.0
    mean_vertices: float = 0.0
    results: list[ShapeRefineResult] = field(default_factory=list)


def _verts(poly: Polygon | None) -> int:
    if poly is None or poly.is_empty:
        return 0
    return len(list(poly.exterior.coords)) - 1


def _box_poly_from_rect(rect) -> Polygon | None:
    pts = cv2.boxPoints(rect)
    ring = [(float(x), float(y)) for x, y in pts]
    ring.append(ring[0])
    return _repair(Polygon(ring))


def _internal_angles(coords: list[tuple[float, float]]) -> list[float]:
    angles = []
    n = len(coords)
    for i in range(n):
        p0, p1, p2 = coords[i - 1], coords[i], coords[(i + 1) % n]
        a1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        a2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        diff = abs(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1)))
        angles.append(_rad2deg(diff))
    return angles


def _repair(poly) -> Polygon | None:
    if poly is None or poly.is_empty:
        return None
    g = make_valid(poly)
    return _largest_polygon(g)


class BuildingShapeRefiner:
    def __init__(self, config: ShapeRefinerConfig | None = None):
        self.config = config or ShapeRefinerConfig()

    # ── Métriques ─────────────────────────────────────────────────
    def mask_iou(self, poly: Polygon, mask: np.ndarray) -> float:
        if poly is None or poly.is_empty:
            return 0.0
        H, W = mask.shape
        pred = self._poly_to_mask(poly, (H, W))
        orig = (mask > 0).astype(np.uint8)
        inter = int((pred & orig).sum())
        union = int(((pred | orig) > 0).sum())
        return inter / max(union, 1)

    @staticmethod
    def _poly_to_mask(poly: Polygon, shape_hw: tuple[int, int]) -> np.ndarray:
        import rasterio.features
        from affine import Affine
        from shapely.geometry import mapping

        return rasterio.features.rasterize(
            [(mapping(poly), 1)],
            out_shape=shape_hw,
            transform=Affine.identity(),
            fill=0,
            dtype=np.uint8,
        )

    def compute_metrics(self, poly: Polygon, mask: np.ndarray) -> ShapeMetrics:
        m = ShapeMetrics()
        poly = _repair(poly)
        if poly is None:
            return m
        m.n_vertices = _verts(poly)
        rect = _repair(poly.minimum_rotated_rectangle)
        if rect is None or rect.is_empty:
            return m
        m.rectangularity = poly.area / max(rect.area, 1e-6)
        inter = poly.intersection(rect).area
        union = poly.union(rect).area
        m.rectangle_iou = inter / max(union, 1e-6)
        m.area_change_ratio = abs(rect.area - poly.area) / max(poly.area, 1e-6)
        m.mean_contour_dist = poly.boundary.distance(rect.boundary)
        hull = poly.convex_hull
        m.concavity_ratio = (hull.area - poly.area) / max(hull.area, 1e-6)
        m.max_notch_depth = self._max_notch_depth(poly, rect)
        m.mask_iou_initial = self.mask_iou(poly, mask)

        pts = np.array(list(poly.exterior.coords)[:-1], dtype=np.float32)
        if len(pts) >= 3:
            (_, (_, _), angle) = cv2.minAreaRect(pts)
            if angle < -45:
                angle += 90
            m.dominant_angle_deg = float(angle)
        return m

    def _max_notch_depth(self, poly: Polygon, rect: Polygon) -> float:
        """Profondeur max des encoches/excrorescences vs le rectangle orienté."""
        diff = poly.symmetric_difference(rect)
        if diff.is_empty:
            return 0.0
        parts = list(diff.geoms) if hasattr(diff, "geoms") else [diff]
        depths = []
        for part in parts:
            if part.is_empty:
                continue
            w = part.bounds[2] - part.bounds[0]
            h = part.bounds[3] - part.bounds[1]
            depths.append(min(w, h))
        return max(depths) if depths else 0.0

    def _notch_significant(self, poly: Polygon) -> bool:
        """True si au moins un décrochement dépasse les seuils configurables."""
        cfg = self.config
        poly = _repair(poly)
        rect = _repair(poly.minimum_rotated_rectangle) if poly else None
        if poly is None or rect is None:
            return False
        area_ref = max(poly.area, 1.0)
        diff = poly.symmetric_difference(rect)
        parts = list(diff.geoms) if hasattr(diff, "geoms") else ([diff] if not diff.is_empty else [])
        for part in parts:
            if part.is_empty:
                continue
            w = part.bounds[2] - part.bounds[0]
            h = part.bounds[3] - part.bounds[1]
            depth = min(w, h)
            width = max(w, h)
            if (depth >= cfg.max_notch_depth_px
                    or part.area / area_ref >= cfg.max_notch_area_ratio
                    or width >= cfg.max_notch_width_px):
                return True
        return False

    def _strip_insignificant_notches(self, poly: Polygon, mask: np.ndarray) -> Polygon | None:
        """Supprime les petits décrochements en favorisant le rectangle orienté."""
        poly = _repair(poly)
        if poly is None:
            return None
        if self._notch_significant(poly):
            return poly
        rect = self.fit_oriented_rectangle(poly, mask)
        if rect is None:
            return poly
        iou_rect = self.mask_iou(rect, mask)
        iou_poly = self.mask_iou(poly, mask)
        if iou_rect >= iou_poly - 0.02:
            return rect
        return poly

    def _is_significant_l_shape(
        self, poly: Polygon, mask: np.ndarray, met: ShapeMetrics,
    ) -> bool:
        """Vrai L/U seulement si concavité, encoche et gain IoU significatifs."""
        cfg = self.config
        if met.concavity_ratio < cfg.min_l_shape_concavity:
            return False
        if met.max_notch_depth < cfg.max_notch_depth_px:
            return False
        rect = _repair(poly.minimum_rotated_rectangle)
        if rect is None:
            return False
        missing = rect.difference(poly)
        if missing.is_empty or missing.area / max(poly.area, 1) < cfg.max_notch_area_ratio * 2:
            return False
        rect_fit = self.fit_oriented_rectangle(poly, mask)
        cx = self.build_orthogonal_complex(poly, mask, require_gain=False)
        if rect_fit is None or cx is None:
            return False
        return self.mask_iou(cx, mask) - self.mask_iou(rect_fit, mask) >= cfg.min_l_shape_iou_gain

    def _orth_reg(self) -> BuildingPolygonRegularizer:
        cfg = self.config
        return BuildingPolygonRegularizer(RegularizerConfig(
            collinear_angle_threshold=8.0,
            orthogonal_angle_threshold=cfg.orthogonal_snap_deg,
            min_edge_length=2.0,
            enable_orthogonalization=True,
            preserve_non_orthogonal_shapes=False,
        ))

    def _approx_target_vertices(
        self, poly: Polygon, target: int, mask: np.ndarray,
    ) -> Polygon | None:
        """Approximation Douglas-Peucker visant un nombre cible de sommets."""
        poly = _repair(poly)
        if poly is None:
            return None
        peri = poly.length
        best: Polygon | None = None
        best_score = -1e9
        for frac in np.linspace(0.001, 0.09, 45):
            cand = _repair(poly.simplify(frac * peri, preserve_topology=True))
            if cand is None:
                continue
            nv = _verts(cand)
            if nv > target + 2:
                continue
            sc = self._parsimony_score(cand, mask, n_target=target)
            if sc > best_score:
                best_score = sc
                best = cand
            if nv <= target:
                break
        return best

    # ── Classification ────────────────────────────────────────────
    def classify_building_shape(
        self, poly: Polygon, mask: np.ndarray,
    ) -> tuple[ShapeClass, ShapeMetrics]:
        cfg = self.config
        poly = _repair(poly)
        if poly is None:
            return ShapeClass.IRREGULAR, ShapeMetrics()

        met = self.compute_metrics(poly, mask)

        # Règle explicite utilisateur
        if (met.rectangularity >= cfg.rule_rectangularity
                and met.rectangle_iou >= cfg.rule_rectangle_iou):
            return ShapeClass.RECTANGLE, met

        if (met.concavity_ratio < cfg.max_concavity_ratio
                and met.max_notch_depth < cfg.max_notch_depth_px
                and met.rectangularity >= cfg.min_rectangularity):
            return ShapeClass.RECTANGLE, met

        # L / U significatif ?
        if self._is_significant_l_shape(poly, mask, met):
            return ShapeClass.ORTHOGONAL_COMPLEX, met

        if (met.concavity_ratio >= cfg.min_l_shape_concavity
                and met.n_vertices >= 6):
            # Décrochement présent mais pas assez significatif → rectangle
            if (met.rectangularity >= cfg.min_rectangularity
                    or met.rectangle_iou >= cfg.min_rectangle_iou):
                return ShapeClass.RECTANGLE, met

        if (met.rectangularity >= cfg.min_rectangularity
                and met.rectangle_iou >= cfg.min_rectangle_iou
                and met.area_change_ratio <= cfg.max_area_change
                and met.mean_contour_dist <= cfg.max_mean_contour_dist):
            return ShapeClass.RECTANGLE, met

        if met.concavity_ratio >= cfg.max_concavity_ratio:
            return ShapeClass.ORTHOGONAL_COMPLEX, met

        return ShapeClass.IRREGULAR, met

    # ── Rectangle orienté optimisé ────────────────────────────────
    def fit_oriented_rectangle(
        self, poly: Polygon, mask: np.ndarray,
    ) -> Polygon | None:
        poly = _repair(poly)
        if poly is None:
            return None
        cfg = self.config
        pts = np.array(list(poly.exterior.coords)[:-1], dtype=np.float32)
        if len(pts) < 3:
            return None

        base_rect = cv2.minAreaRect(pts)
        best_score = -1e9
        best_poly: Polygon | None = None
        (cx, cy), (w, h), angle = base_rect

        angles = np.arange(
            angle - cfg.angle_search_deg,
            angle + cfg.angle_search_deg + 0.01,
            cfg.angle_step_deg,
        )
        for a in angles:
            for sw in cfg.scale_search:
                for sh in cfg.scale_search:
                    for dx in cfg.translate_px:
                        for dy in cfg.translate_px:
                            cand = ((cx + dx, cy + dy), (w * sw, h * sh), a)
                            box_pts = cv2.boxPoints(cand)
                            ring = [(float(x), float(y)) for x, y in box_pts]
                            ring.append(ring[0])
                            p = _repair(Polygon(ring))
                            if p is None:
                                continue
                            sc = self._parsimony_score(p, mask, n_target=4)
                            if sc > best_score:
                                best_score = sc
                                best_poly = p
        return best_poly

    def _parsimony_score(
        self, poly: Polygon, mask: np.ndarray, n_target: int = 4,
    ) -> float:
        cfg = self.config
        iou = self.mask_iou(poly, mask)
        orig_area = max(int(mask.sum()), 1)
        area_chg = abs(poly.area - orig_area) / orig_area
        pts = np.array(list(poly.exterior.coords)[:-1], dtype=np.float32)
        if len(pts) >= 3:
            base_poly = _box_poly_from_rect(cv2.minAreaRect(pts))
            dist = poly.boundary.distance(base_poly.boundary) if base_poly else 0.0
        else:
            dist = 0.0
        nv = _verts(poly)
        return (
            iou
            - cfg.lambda_area * area_chg
            - cfg.lambda_distance * min(dist, 10.0) / 10.0
            - cfg.lambda_vertices * max(0, nv - n_target) / max(n_target, 1)
        )

    # ── Forme orthogonale complexe (L, U…) ────────────────────────
    def build_orthogonal_complex(
        self, poly: Polygon, mask: np.ndarray, *, require_gain: bool = True,
    ) -> Polygon | None:
        poly = _repair(poly)
        if poly is None:
            return None
        cfg = self.config
        tol = max(cfg.simplify_ratio * math.sqrt(poly.area), 1.5)
        simp = poly.simplify(tol, preserve_topology=True)
        simp = _repair(simp)
        if simp is None:
            return None

        orth = self._orthogonalize_strict(simp)
        orth = _repair(orth)
        if orth is None:
            return None

        nv = _verts(orth)
        if nv > cfg.max_complex_vertices:
            orth2 = orth.simplify(tol * 1.5, preserve_topology=True)
            orth = _repair(orth2) or orth
        elif nv < cfg.min_complex_vertices:
            orth6 = self._approx_target_vertices(simp, 6, mask)
            if orth6 is not None and _verts(orth6) >= cfg.min_complex_vertices:
                orth = self._orthogonalize_strict(orth6) or orth6
                orth = _repair(orth) or orth

        if require_gain:
            rect = self.fit_oriented_rectangle(poly, mask)
            if rect is not None:
                if self.mask_iou(orth, mask) - self.mask_iou(rect, mask) < cfg.min_l_shape_iou_gain:
                    return None
        return orth

    def _dominant_angle(self, poly: Polygon) -> tuple[float, float]:
        pts = np.array(list(poly.exterior.coords)[:-1], dtype=np.float32)
        if len(pts) < 3:
            return 0.0, 0.0
        rect = cv2.minAreaRect(pts)
        (_, (w, h), ang) = rect
        if w < h:
            ang += 90
        return _norm_angle_deg(ang), 1.0

    def _orthogonalize_strict(self, poly: Polygon) -> Polygon | None:
        """Orthogonalisation stricte : snap 90°/180°, fusion collinéaire, intersections."""
        reg = self._orth_reg()
        out = reg.regularize_edges(poly)
        return _repair(out) if out is not None else None

    def _orthogonalize_polygon(self, poly: Polygon, dom_deg: float) -> Polygon | None:
        return self._orthogonalize_strict(poly)

    def _select_best_candidate(
        self,
        poly: Polygon,
        mask: np.ndarray,
        shape_class: ShapeClass,
        rect_cand: Polygon | None,
        cx_cand: Polygon | None,
    ) -> tuple[Polygon | None, ShapeClass]:
        """Parcimonie : choisir la forme la plus simple à qualité comparable."""
        cfg = self.config
        candidates: list[tuple[Polygon, ShapeClass, int]] = []

        if rect_cand is not None:
            candidates.append((rect_cand, ShapeClass.RECTANGLE, 4))

        if cx_cand is not None:
            candidates.append((cx_cand, ShapeClass.ORTHOGONAL_COMPLEX, _verts(cx_cand)))

        orth4 = self._approx_target_vertices(poly, 4, mask)
        if orth4 is not None:
            orth4 = self._orthogonalize_strict(orth4) or orth4
            orth4 = _repair(orth4)
            if orth4 is not None:
                candidates.append((orth4, ShapeClass.RECTANGLE, _verts(orth4)))

        orth6 = self._approx_target_vertices(poly, 6, mask)
        if orth6 is not None:
            orth6 = self._orthogonalize_strict(orth6) or orth6
            orth6 = _repair(orth6)
            if orth6 is not None:
                candidates.append((orth6, ShapeClass.ORTHOGONAL_COMPLEX, _verts(orth6)))

        tol = max(cfg.simplify_ratio * math.sqrt(poly.area), 1.0)
        simp = _repair(poly.simplify(tol, preserve_topology=True))
        if simp is not None:
            candidates.append((simp, ShapeClass.IRREGULAR, _verts(simp)))

        if not candidates:
            return poly, shape_class

        iou_rect = self.mask_iou(rect_cand, mask) if rect_cand else 0.0
        scored: list[tuple[float, Polygon, ShapeClass]] = []
        for cand, cls, _nv in candidates:
            sc = self._parsimony_score(cand, mask, n_target=4 if cls == ShapeClass.RECTANGLE else 6)
            iou_c = self.mask_iou(cand, mask)
            if cls != ShapeClass.RECTANGLE and rect_cand is not None:
                if iou_c - iou_rect < cfg.min_l_shape_iou_gain:
                    continue
            if abs(cand.area - poly.area) / max(poly.area, 1) > cfg.max_area_change * 1.5:
                continue
            scored.append((sc, cand, cls))

        if not scored:
            if shape_class == ShapeClass.RECTANGLE and rect_cand is not None:
                return rect_cand, ShapeClass.RECTANGLE
            return poly, shape_class

        scored.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_poly, best_cls = scored[0]

        if shape_class == ShapeClass.RECTANGLE and rect_cand is not None:
            return rect_cand, ShapeClass.RECTANGLE
        if shape_class == ShapeClass.ORTHOGONAL_COMPLEX and cx_cand is not None:
            return cx_cand, ShapeClass.ORTHOGONAL_COMPLEX

        # Fallback parcimonie : rectangle si score proche
        if rect_cand is not None:
            sc_rect = self._parsimony_score(rect_cand, mask, n_target=4)
            if sc_rect >= best_sc - 0.02:
                return rect_cand, ShapeClass.RECTANGLE

        return best_poly, best_cls

    # ── Pipeline par bâtiment ─────────────────────────────────────
    def refine_building(
        self,
        building_id: int,
        poly: Polygon,
        mask: np.ndarray,
    ) -> ShapeRefineResult:
        poly = _repair(poly)
        if poly is None:
            return ShapeRefineResult(building_id, None, ShapeClass.IRREGULAR)

        n_before = _verts(poly)
        poly = self._strip_insignificant_notches(poly, mask) or poly
        poly = _repair(poly)
        if poly is None:
            return ShapeRefineResult(building_id, None, ShapeClass.IRREGULAR)

        met = self.compute_metrics(poly, mask)
        shape_class, met = self.classify_building_shape(poly, mask)

        rect_cand = self.fit_oriented_rectangle(poly, mask)
        cx_cand = self.build_orthogonal_complex(poly, mask)

        final, shape_class = self._select_best_candidate(
            poly, mask, shape_class, rect_cand, cx_cand,
        )

        final = _repair(final)
        if shape_class == ShapeClass.RECTANGLE and rect_cand is not None:
            final = rect_cand
        elif final and shape_class == ShapeClass.ORTHOGONAL_COMPLEX:
            final = self._orthogonalize_strict(final) or final
            final = _repair(final)

        met.mask_iou_final = self.mask_iou(final, mask) if final else 0

        return ShapeRefineResult(
            building_id=building_id,
            polygon=final,
            shape_class=shape_class,
            n_vertices_before=n_before,
            n_vertices_after=_verts(final),
            metrics=met,
            rectangle_candidate=rect_cand,
            complex_candidate=cx_cand,
        )

    def refine_all(
        self,
        polys: dict[int, Polygon],
        inst: np.ndarray,
    ) -> tuple[dict[int, Polygon], ShapeRefineStats]:
        stats = ShapeRefineStats()
        out: dict[int, Polygon] = {}
        for lab, poly in polys.items():
            mask = (inst == lab).astype(np.uint8)
            if mask.sum() == 0:
                continue
            res = self.refine_building(lab, poly, mask)
            stats.results.append(res)
            if res.polygon is not None:
                out[lab] = res.polygon
            if res.shape_class == ShapeClass.RECTANGLE:
                stats.n_rectangle += 1
            elif res.shape_class == ShapeClass.ORTHOGONAL_COMPLEX:
                stats.n_orthogonal += 1
            else:
                stats.n_irregular += 1

        n = len(stats.results)
        if n:
            verts = [r.n_vertices_after for r in stats.results if r.polygon]
            stats.mean_vertices = sum(verts) / len(verts)
            stats.pct_4_vertices = 100.0 * sum(v == 4 for v in verts) / len(verts)
        return out, stats


def polygon_geometry_features(poly: Polygon) -> dict[str, Any]:
    """Features géométriques d'un polygone (pour analyse labels/prédictions)."""
    poly = _repair(poly)
    if poly is None:
        return {}
    nv = _verts(poly)
    rect = poly.minimum_rotated_rectangle
    rect_iou = poly.intersection(rect).area / max(poly.union(rect).area, 1e-6)
    rectangularity = poly.area / max(rect.area, 1e-6)
    hull = poly.convex_hull
    concavity = (hull.area - poly.area) / max(hull.area, 1e-6)

    angles = []
    coords = list(poly.exterior.coords)[:-1]
    for i in range(len(coords)):
        p0, p1, p2 = coords[i - 1], coords[i], coords[(i + 1) % len(coords)]
        a1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        a2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        diff = abs(math.degrees(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1))))
        angles.append(diff)

    near_90 = sum(1 for a in angles if abs(a - 90) <= 15 or abs(a - 180) <= 15)

    return {
        "n_vertices": nv,
        "rectangularity": round(rectangularity, 4),
        "rectangle_iou": round(rect_iou, 4),
        "concavity_ratio": round(concavity, 4),
        "pct_angles_near_90": round(100.0 * near_90 / max(len(angles), 1), 1),
        "is_quad": nv == 4,
        "is_6plus": nv >= 6,
    }
