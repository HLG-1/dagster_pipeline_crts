"""
Post-traitement géométrique des empreintes bâtiment prédites.

Transforme chaque masque d'instance en polygone vectoriel propre :
contours simplifiés, segments droits, angles régularisés (90° si justifié),
sans fusionner les instances voisines.

Pipeline par instance :
  clean mask → contour → repair → simplify → merge collinear →
  snap orientations → reconstruct → validate (fallback si dégradation)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid

try:
    from shapely import affinity
except ImportError:
    affinity = None  # type: ignore


class RegularizationLevel(str, Enum):
    REGULARIZED = "regularized"
    SIMPLIFIED = "simplified"
    ORIGINAL = "original"
    REJECTED = "rejected"


@dataclass
class RegularizerConfig:
    """Seuils adaptés aux tuiles 512×512 (coords pixel)."""
    min_component_area: int = 25
    max_hole_area: int = 64
    morph_close_px: int = 0
    simplify_tolerance: float = 0.0
    simplify_tolerance_ratio: float = 0.004
    collinear_angle_threshold: float = 8.0
    orthogonal_angle_threshold: float = 14.0
    min_edge_length: float = 2.5
    max_area_change_ratio: float = 0.22
    min_polygon_iou: float = 0.72
    max_centroid_shift_ratio: float = 0.12
    max_hausdorff_ratio: float = 0.06
    enable_orthogonalization: bool = True
    preserve_non_orthogonal_shapes: bool = True
    dominant_angle_bins: int = 36


@dataclass
class RegularizationResult:
    building_id: int
    polygon: Polygon | None
    level: RegularizationLevel
    n_vertices_before: int = 0
    n_vertices_after: int = 0
    area_before: float = 0.0
    area_after: float = 0.0
    iou_vs_original: float = 0.0
    area_change_ratio: float = 0.0
    centroid_shift: float = 0.0
    hausdorff_dist: float = 0.0
    elapsed_ms: float = 0.0
    message: str = ""


@dataclass
class RegularizationStats:
    n_input: int = 0
    n_regularized: int = 0
    n_simplified: int = 0
    n_original: int = 0
    n_rejected: int = 0
    elapsed_ms: float = 0.0
    results: list[RegularizationResult] = field(default_factory=list)


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def _norm_angle_deg(a: float) -> float:
    """Angle dans [0, 180)."""
    a = a % 180.0
    if a < 0:
        a += 180.0
    return a


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(_norm_angle_deg(a) - _norm_angle_deg(b))
    return min(d, 180.0 - d)


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        return max(polys, key=lambda p: p.area) if polys else None
    return None


class BuildingPolygonRegularizer:
    def __init__(self, config: RegularizerConfig | None = None):
        self.config = config or RegularizerConfig()

    # ── Masque ────────────────────────────────────────────────────
    def clean_instance_mask(self, mask: np.ndarray) -> np.ndarray:
        """Nettoyage léger par instance (sans fusionner les voisins)."""
        cfg = self.config
        m = (mask > 0).astype(np.uint8)
        if int(m.sum()) < cfg.min_component_area:
            return m

        try:
            from skimage.morphology import remove_small_holes
            m_bool = m.astype(bool)
            m_bool = remove_small_holes(m_bool, area_threshold=max(1, cfg.max_hole_area))
            m = m_bool.astype(np.uint8)
        except ImportError:
            pass

        if cfg.morph_close_px > 0:
            k = cfg.morph_close_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        return m

    def mask_to_polygon(self, mask: np.ndarray) -> Polygon | None:
        """Contour externe OpenCV → Polygon (coords pixel x=col, y=row)."""
        m = self.clean_instance_mask(mask)
        if m.max() == 0:
            return None
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.config.min_component_area:
            return None
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        if len(pts) < 3:
            return None
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        poly = Polygon(pts)
        return _largest_polygon(self.repair_geometry(poly))

    def repair_geometry(self, geom) -> Polygon | None:
        if geom is None or geom.is_empty:
            return None
        g = make_valid(geom)
        poly = _largest_polygon(g)
        if poly is None:
            return None
        if not poly.is_valid:
            poly = poly.buffer(0)
            poly = _largest_polygon(poly)
        return poly

    def simplify_polygon(self, poly: Polygon) -> Polygon | None:
        poly = self.repair_geometry(poly)
        if poly is None:
            return None
        cfg = self.config
        tol = cfg.simplify_tolerance
        if tol <= 0:
            perim = poly.length
            tol = max(cfg.min_edge_length, cfg.simplify_tolerance_ratio * perim)
        simplified = poly.simplify(tol, preserve_topology=True)
        return self.repair_geometry(simplified)

    # ── Sommets / segments ────────────────────────────────────────
    def _exterior_vertices(self, poly: Polygon) -> list[tuple[float, float]]:
        coords = list(poly.exterior.coords)[:-1]
        return [(float(x), float(y)) for x, y in coords]

    def merge_collinear_edges(self, poly: Polygon) -> Polygon | None:
        poly = self.repair_geometry(poly)
        if poly is None:
            return None
        verts = self._exterior_vertices(poly)
        if len(verts) < 3:
            return poly
        thr = _deg2rad(self.config.collinear_angle_threshold)
        min_len = self.config.min_edge_length
        changed = True
        while changed and len(verts) >= 3:
            changed = False
            new_v: list[tuple[float, float]] = []
            n = len(verts)
            for i in range(n):
                prev = verts[(i - 1) % n]
                cur = verts[i]
                nxt = verts[(i + 1) % n]
                v1 = (cur[0] - prev[0], cur[1] - prev[1])
                v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
                l1 = math.hypot(*v1)
                l2 = math.hypot(*v2)
                if l1 < min_len or l2 < min_len:
                    new_v.append(cur)
                    continue
                a1 = math.atan2(v1[1], v1[0])
                a2 = math.atan2(v2[1], v2[0])
                diff = abs(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1)))
                if diff < thr:
                    changed = True
                    continue
                new_v.append(cur)
            if len(new_v) >= 3:
                verts = new_v
            else:
                break
        if len(verts) < 3:
            return poly
        ring = verts + [verts[0]]
        return self.repair_geometry(Polygon(ring))

    def estimate_dominant_orientations(self, poly: Polygon) -> tuple[float, float]:
        """Retourne (angle_principal_deg, confiance 0..1) via minAreaRect."""
        poly = self.repair_geometry(poly)
        if poly is None:
            return 0.0, 0.0
        pts = np.array(self._exterior_vertices(poly), dtype=np.float32)
        if len(pts) < 3:
            return 0.0, 0.0
        rect = cv2.minAreaRect(pts)
        (_, (w, h), angle_cv) = rect
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        # OpenCV angle → direction du côté le plus long
        if w < h:
            angle_cv += 90.0
        dom = _norm_angle_deg(angle_cv)
        ar = max(w, h) / max(min(w, h), 1e-6)
        conf = min(1.0, (ar - 1.0) / 3.0)
        return dom, conf

    def _snap_angle_to_dominant(self, edge_deg: float, dominant_deg: float) -> float:
        """Snap vers dominant ou dominant+90 si proche."""
        cfg = self.config
        if not cfg.enable_orthogonalization:
            return edge_deg
        candidates = [
            dominant_deg,
            dominant_deg + 90.0,
            dominant_deg + 180.0,
            dominant_deg + 270.0,
        ]
        best = edge_deg
        best_diff = 180.0
        for c in candidates:
            d = _angle_diff_deg(edge_deg, c)
            if d < best_diff:
                best_diff = d
                best = c
        if best_diff <= cfg.orthogonal_angle_threshold:
            return _norm_angle_deg(best)
        if cfg.preserve_non_orthogonal_shapes:
            return edge_deg
        return _norm_angle_deg(best)

    def regularize_edges(self, poly: Polygon) -> Polygon | None:
        """Snap des directions de segments + reconstruction par intersections."""
        poly = self.merge_collinear_edges(poly)
        if poly is None:
            return None
        verts = self._exterior_vertices(poly)
        if len(verts) < 3:
            return poly
        dom, conf = self.estimate_dominant_orientations(poly)
        if conf < 0.15 and self.config.preserve_non_orthogonal_shapes:
            return poly

        n = len(verts)
        snapped_dirs: list[float] = []
        lengths: list[float] = []
        for i in range(n):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < self.config.min_edge_length:
                length = self.config.min_edge_length
            edge_deg = _rad2deg(math.atan2(dy, dx))
            snapped = self._snap_angle_to_dominant(edge_deg, dom)
            snapped_dirs.append(_deg2rad(snapped))
            lengths.append(length)

        return self.reconstruct_polygon(verts[0], snapped_dirs, lengths)

    def reconstruct_polygon(
        self,
        origin: tuple[float, float],
        directions_rad: list[float],
        lengths: list[float],
    ) -> Polygon | None:
        """Reconstruction par intersections de droites consécutives."""
        n = len(directions_rad)
        if n < 3:
            return None
        lines: list[tuple[tuple[float, float], float]] = []
        x, y = origin
        for i in range(n):
            ang = directions_rad[i]
            length = lengths[i]
            p1 = (x, y)
            x2 = x + length * math.cos(ang)
            y2 = y + length * math.sin(ang)
            p2 = (x2, y2)
            lines.append((p1, ang))
            x, y = p2

        new_verts: list[tuple[float, float]] = []
        for i in range(n):
            p0, a0 = lines[i]
            p1, a1 = lines[(i + 1) % n]
            inter = self._line_intersection(p0, a0, p1, a1)
            if inter is not None:
                new_verts.append(inter)
            else:
                new_verts.append(p1)

        if len(new_verts) < 3:
            return None
        ring = new_verts + [new_verts[0]]
        poly = self.repair_geometry(Polygon(ring))
        return poly

    @staticmethod
    def _line_intersection(
        p0: tuple[float, float], ang0: float,
        p1: tuple[float, float], ang1: float,
    ) -> tuple[float, float] | None:
        """Intersection de deux droites (point + angle)."""
        dx0, dy0 = math.cos(ang0), math.sin(ang0)
        dx1, dy1 = math.cos(ang1), math.sin(ang1)
        denom = dx0 * dy1 - dy0 * dx1
        if abs(denom) < 1e-10:
            return p1
        t = ((p1[0] - p0[0]) * dy1 - (p1[1] - p0[1]) * dx1) / denom
        return (p0[0] + t * dx0, p0[1] + t * dy0)

    def validate_regularization(
        self,
        original: Polygon,
        candidate: Polygon,
    ) -> tuple[bool, dict[str, float]]:
        cfg = self.config
        original = self.repair_geometry(original)
        candidate = self.repair_geometry(candidate)
        if original is None or candidate is None:
            return False, {"iou": 0.0}

        inter = original.intersection(candidate).area
        union = original.union(candidate).area
        iou = inter / union if union > 0 else 0.0
        area_before = original.area
        area_after = candidate.area
        area_chg = abs(area_after - area_before) / max(area_before, 1e-6)
        c0 = original.centroid
        c1 = candidate.centroid
        shift = math.hypot(c1.x - c0.x, c1.y - c0.y)
        diag = math.sqrt(original.bounds[2] - original.bounds[0]) ** 2 + (
            original.bounds[3] - original.bounds[1]
        ) ** 2
        shift_ratio = shift / max(diag, 1e-6)
        hausdorff = original.hausdorff_distance(candidate)
        haus_ratio = hausdorff / max(diag, 1e-6)

        metrics = {
            "iou": iou,
            "area_change_ratio": area_chg,
            "centroid_shift": shift,
            "centroid_shift_ratio": shift_ratio,
            "hausdorff": hausdorff,
            "hausdorff_ratio": haus_ratio,
        }
        ok = (
            iou >= cfg.min_polygon_iou
            and area_chg <= cfg.max_area_change_ratio
            and shift_ratio <= cfg.max_centroid_shift_ratio
            and haus_ratio <= cfg.max_hausdorff_ratio
        )
        return ok, metrics

    def process_mask(
        self,
        mask: np.ndarray,
        building_id: int = 0,
    ) -> RegularizationResult:
        """Pipeline complet sur un masque binaire d'une instance."""
        t0 = time.perf_counter()
        cfg = self.config

        original = self.mask_to_polygon(mask)
        if original is None:
            return RegularizationResult(
                building_id=building_id,
                polygon=None,
                level=RegularizationLevel.REJECTED,
                message="empty_mask",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        n_before = len(self._exterior_vertices(original))
        simplified = self.simplify_polygon(original)
        regularized = self.regularize_edges(simplified) if simplified else None

        candidates: list[tuple[RegularizationLevel, Polygon | None]] = [
            (RegularizationLevel.REGULARIZED, regularized),
            (RegularizationLevel.SIMPLIFIED, simplified),
            (RegularizationLevel.ORIGINAL, original),
        ]

        chosen_level = RegularizationLevel.REJECTED
        chosen_poly: Polygon | None = None
        metrics: dict[str, float] = {}

        for level, cand in candidates:
            if cand is None:
                continue
            ok, metrics = self.validate_regularization(original, cand)
            if ok or level == RegularizationLevel.ORIGINAL:
                chosen_level = level
                chosen_poly = cand
                break

        if chosen_poly is None:
            chosen_poly = original
            chosen_level = RegularizationLevel.ORIGINAL
            _, metrics = self.validate_regularization(original, original)

        n_after = len(self._exterior_vertices(chosen_poly)) if chosen_poly else 0
        return RegularizationResult(
            building_id=building_id,
            polygon=chosen_poly,
            level=chosen_level,
            n_vertices_before=n_before,
            n_vertices_after=n_after,
            area_before=original.area,
            area_after=chosen_poly.area if chosen_poly else 0.0,
            iou_vs_original=metrics.get("iou", 0.0),
            area_change_ratio=metrics.get("area_change_ratio", 0.0),
            centroid_shift=metrics.get("centroid_shift", 0.0),
            hausdorff_dist=metrics.get("hausdorff", 0.0),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def process_instance_map(
        self,
        inst: np.ndarray,
        min_area_px: int = 1,
    ) -> tuple[list[RegularizationResult], RegularizationStats]:
        """Traite chaque label de la carte d'instances indépendamment."""
        t0 = time.perf_counter()
        results: list[RegularizationResult] = []
        stats = RegularizationStats()
        for lab in range(1, int(inst.max()) + 1):
            comp = (inst == lab).astype(np.uint8)
            if int(comp.sum()) < max(min_area_px, self.config.min_component_area):
                continue
            res = self.process_mask(comp, building_id=lab)
            results.append(res)
            stats.n_input += 1
            if res.level == RegularizationLevel.REGULARIZED:
                stats.n_regularized += 1
            elif res.level == RegularizationLevel.SIMPLIFIED:
                stats.n_simplified += 1
            elif res.level == RegularizationLevel.ORIGINAL:
                stats.n_original += 1
            else:
                stats.n_rejected += 1
        stats.results = results
        stats.elapsed_ms = (time.perf_counter() - t0) * 1000
        return results, stats

    def polygon_to_mask(
        self,
        poly: Polygon,
        shape_hw: tuple[int, int],
    ) -> np.ndarray:
        """Rasterise un polygone pixel → masque uint8."""
        import rasterio.features
        from affine import Affine

        if poly is None or poly.is_empty:
            return np.zeros(shape_hw, dtype=np.uint8)
        tf = Affine.identity()
        mask = rasterio.features.rasterize(
            [(mapping(poly), 1)],
            out_shape=shape_hw,
            transform=tf,
            fill=0,
            dtype=np.uint8,
        )
        return mask

    @staticmethod
    def pixel_polygon_to_geo(poly: Polygon, transform) -> Polygon | None:
        """Convertit un polygone pixel (x=col, y=row) → coords géo."""
        if poly is None or poly.is_empty:
            return None
        from rasterio.transform import xy

        def _tf(x, y, z=None):
            gx, gy = xy(transform, y, x)
            return (gx, gy)

        geo = shp_transform(_tf, poly)
        g = make_valid(geo)
        return _largest_polygon(g)

    def export_polygons(
        self,
        results: list[RegularizationResult],
        transform,
        crs_str: str,
        out_dir: str | Path,
        stem: str = "buildings",
    ) -> dict[str, Path]:
        """Export GeoJSON + GPKG via geo_io."""
        import geopandas as gpd

        from .geo_io import export_geojson, export_gpkg, safe_crs

        rows = []
        for r in results:
            if r.polygon is None:
                continue
            geo = self.pixel_polygon_to_geo(r.polygon, transform)
            if geo is None:
                continue
            rows.append({
                "geometry": geo,
                "building_id": r.building_id,
                "area_m2": round(geo.area, 2),
                "reg_level": r.level.value,
                "n_vertices": r.n_vertices_after,
                "iou_vs_orig": round(r.iou_vs_original, 4),
            })
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if not rows:
            return paths
        crs = safe_crs(crs_str)
        gdf = gpd.GeoDataFrame(rows, crs=crs)
        gj = out_dir / f"{stem}.geojson"
        gp = out_dir / f"{stem}.gpkg"
        export_geojson(gdf, gj)
        export_gpkg(gdf, gp)
        paths["geojson"] = gj
        paths["gpkg"] = gp
        return paths
