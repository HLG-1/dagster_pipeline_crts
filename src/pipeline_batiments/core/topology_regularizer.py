"""
Régularisation topologique globale des empreintes bâtiment.

Philosophie (option 1) :
  1. préserver chaque instance
  2. construire des frontières communes entre voisins
  3. orthogonaliser le réseau de frontières (θ, θ+90)
  4. reconstruire les polygones depuis ce réseau commun

Pas de remodelage indépendant par bâtiment (shape refiner désactivé).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from shapely.validation import make_valid

from .building_regularizer import (
    _angle_diff_deg,
    _deg2rad,
    _largest_polygon,
    _norm_angle_deg,
    _rad2deg,
)

# Type alias : (p0, p1, label_a, label_b)
Seg = tuple[tuple[float, float], tuple[float, float], int, int]


@dataclass
class TopologyConfig:
    """Seuils pour tuiles 512×512 (coords pixel)."""
    snap_vertex_tol: float = 2.5
    collinear_angle_deg: float = 8.0
    orthogonal_snap_deg: float = 12.0
    min_edge_length: float = 2.0
    min_component_area: int = 25
    max_hole_area: int = 64
    gap_fill_max_area: float = 20.0
    overlap_max_area: float = 0.5
    min_polygon_iou: float = 0.82
    max_area_change_ratio: float = 0.18
    preserve_oblique: bool = True
    min_oblique_confidence: float = 0.2

    # Shape refiner désactivé (option 1) — frontières communes d'abord
    enable_shape_refiner: bool = False

    # Sommets presque colinéaires / angles ouverts
    open_angle_min_deg: float = 170.0
    open_angle_max_deg: float = 190.0
    soft_open_angle_deg: float = 150.0
    right_angle_min_deg: float = 80.0
    right_angle_max_deg: float = 100.0
    collinear_distance_tol: float = 1.5

    # Frontières communes entre voisins proches
    shared_boundary_max_distance: float = 3.5
    shared_boundary_angle_threshold: float = 8.0
    shared_boundary_min_overlap_ratio: float = 0.60
    shared_boundary_min_length: float = 6.0


@dataclass
class TopologyMetrics:
    n_instances_in: int = 0
    n_instances_out: int = 0
    n_overlaps: int = 0
    overlap_area: float = 0.0
    n_gaps: int = 0
    gap_area: float = 0.0
    duplicate_boundary_length: float = 0.0
    n_vertices_before: int = 0
    n_vertices_after: int = 0
    pct_angles_near_90: float = 0.0
    mean_iou_vs_raster: float = 0.0
    mean_area_change: float = 0.0
    invalid_geometry_count: int = 0
    shared_boundary_coordinates_match: bool = True
    elapsed_ms: float = 0.0
    accepted: bool = False
    message: str = ""


@dataclass
class TopologyResult:
    polygons: dict[int, Polygon]
    segments: list[LineString]
    shared_segments: list[LineString]
    exterior_segments: list[LineString]
    nodes: list[tuple[float, float]]
    metrics: TopologyMetrics
    raster_regularized: np.ndarray | None = None
    shape_stats: Any = None


def _canonical_pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _seg_key(p0, p1, la: int, lb: int) -> tuple:
    cp = _canonical_pair(la, lb)
    if p0 > p1:
        p0, p1 = p1, p0
    return (round(p0[0], 4), round(p0[1], 4), round(p1[0], 4), round(p1[1], 4), cp)


def _pt_key(p: tuple[float, float], nd: int = 3) -> tuple[float, float]:
    return (round(p[0], nd), round(p[1], nd))


def _seg_angle_deg(p0, p1) -> float:
    return _norm_angle_deg(_rad2deg(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))


def _seg_length(p0, p1) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def _point_line_distance(p, a, b) -> float:
    """Distance de p à la droite AB."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def _turn_angle_deg(a, b, c) -> float:
    """Angle de déviation au sommet B (0 = colinéaire, 90 = coin droit)."""
    a1 = math.atan2(b[1] - a[1], b[0] - a[0])
    a2 = math.atan2(c[1] - b[1], c[0] - b[0])
    diff = abs(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1)))
    return _rad2deg(diff)


class TopologicalBuildingRegularizer:
    def __init__(self, config: TopologyConfig | None = None):
        self.config = config or TopologyConfig()

    # ── 1. Partition raster ───────────────────────────────────────
    def clean_partition_raster(self, inst: np.ndarray) -> np.ndarray:
        """Une étiquette par pixel ; petits trous comblés sans toucher les voisins."""
        out = inst.astype(np.int32).copy()
        try:
            from skimage.morphology import remove_small_holes
        except ImportError:
            remove_small_holes = None

        cfg = self.config
        for lab in range(1, int(out.max()) + 1):
            m = out == lab
            if int(m.sum()) < cfg.min_component_area:
                out[m] = 0
                continue
            if remove_small_holes is not None:
                filled = remove_small_holes(m, area_threshold=max(1, cfg.max_hole_area))
                holes = filled & ~m
                out[holes] = lab
        return out

    # ── 2. Segments de frontière (graphe planaire) ────────────────
    def extract_boundary_segments(self, inst: np.ndarray) -> list[Seg]:
        """Extrait les arêtes de grille entre labels différents (une par frontière)."""
        H, W = inst.shape
        raw: list[Seg] = []

        for r in range(H):
            for c in range(W - 1):
                a, b = int(inst[r, c]), int(inst[r, c + 1])
                if a == b:
                    continue
                x = c + 0.5
                raw.append(((x, float(r)), (x, float(r + 1)), a, b))

        for r in range(H - 1):
            for c in range(W):
                a, b = int(inst[r, c]), int(inst[r + 1, c])
                if a == b:
                    continue
                y = r + 0.5
                raw.append(((float(c), y), (float(c + 1), y), a, b))

        return self._merge_axis_aligned_segments(raw)

    @staticmethod
    def _merge_axis_aligned_segments(raw: list[Seg]) -> list[Seg]:
        """Fusionne les micro-segments colinéaires (même paire de labels)."""
        buckets: dict[tuple, list] = defaultdict(list)
        for p0, p1, la, lb in raw:
            cp = _canonical_pair(la, lb)
            vertical = abs(p0[0] - p1[0]) < 1e-9
            if vertical:
                x = p0[0]
                y0, y1 = sorted((p0[1], p1[1]))
                buckets[("v", x, cp)].append((y0, y1, la, lb))
            else:
                y = p0[1]
                x0, x1 = sorted((p0[0], p1[0]))
                buckets[("h", y, cp)].append((x0, x1, la, lb))

        merged: list[Seg] = []
        for key, segs in buckets.items():
            orient = key[0]
            if orient == "v":
                x = key[1]
                segs.sort()
                cur_y0, cur_y1, la, lb = segs[0]
                for y0, y1, la2, lb2 in segs[1:]:
                    if la2 == la and lb2 == lb and y0 <= cur_y1 + 1e-9:
                        cur_y1 = max(cur_y1, y1)
                    else:
                        merged.append(((x, cur_y0), (x, cur_y1), la, lb))
                        cur_y0, cur_y1, la, lb = y0, y1, la2, lb2
                merged.append(((x, cur_y0), (x, cur_y1), la, lb))
            else:
                y = key[1]
                segs.sort()
                cur_x0, cur_x1, la, lb = segs[0]
                for x0, x1, la2, lb2 in segs[1:]:
                    if la2 == la and lb2 == lb and x0 <= cur_x1 + 1e-9:
                        cur_x1 = max(cur_x1, x1)
                    else:
                        merged.append(((cur_x0, y), (cur_x1, y), la, lb))
                        cur_x0, cur_x1, la, lb = x0, x1, la2, lb2
                merged.append(((cur_x0, y), (cur_x1, y), la, lb))
        return merged

    def build_adjacency_graph(self, segments: list[Seg]) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = defaultdict(set)
        for _, _, la, lb in segments:
            if la > 0 and lb > 0:
                adj[la].add(lb)
                adj[lb].add(la)
        return adj

    # ── 3. Snap global ────────────────────────────────────────────
    def snap_vertices_global(
        self, segments: list[Seg],
    ) -> tuple[list[Seg], list[tuple[float, float]]]:
        tol = self.config.snap_vertex_tol
        points: list[tuple[float, float]] = []
        for p0, p1, _, _ in segments:
            points.append(p0)
            points.append(p1)
        if not points:
            return segments, []

        cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        for x, y in points:
            cells[(int(round(x / tol)), int(round(y / tol)))].append((x, y))

        remap: dict[tuple[float, float], tuple[float, float]] = {}
        nodes: list[tuple[float, float]] = []
        for pts in cells.values():
            node = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            nodes.append(node)
            for p in pts:
                remap[p] = node

        snapped = [(remap.get(p0, p0), remap.get(p1, p1), la, lb) for p0, p1, la, lb in segments]
        return self._merge_axis_aligned_segments(snapped), nodes

    # ── 4–5. Frontières communes entre voisins proches ────────────
    def build_shared_boundary(
        self,
        segment_a: tuple[tuple[float, float], tuple[float, float]],
        segment_b: tuple[tuple[float, float], tuple[float, float]],
    ) -> LineString | None:
        """Ligne médiane entre deux segments parallèles face-à-face."""
        a0, a1 = segment_a
        b0, b1 = segment_b
        if _seg_length(a0, a1) < 1e-6 or _seg_length(b0, b1) < 1e-6:
            return None

        # Projection paramétrique sur A pour trouver le recouvrement
        ax, ay = a0
        dx, dy = a1[0] - a0[0], a1[1] - a0[1]
        L2 = dx * dx + dy * dy

        def _t(p):
            return ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2

        ts = sorted([_t(b0), _t(b1), 0.0, 1.0])
        t0, t1 = max(0.0, min(_t(b0), _t(b1))), min(1.0, max(_t(b0), _t(b1)))
        if t1 - t0 < 1e-6:
            return None

        def _on_a(t):
            return (ax + t * dx, ay + t * dy)

        def _on_b_near(pa):
            # point de B le plus proche de pa le long de B
            bx, by = b0
            bdx, bdy = b1[0] - b0[0], b1[1] - b0[1]
            bL2 = bdx * bdx + bdy * bdy
            u = max(0.0, min(1.0, ((pa[0] - bx) * bdx + (pa[1] - by) * bdy) / max(bL2, 1e-12)))
            return (bx + u * bdx, by + u * bdy)

        p_start_a = _on_a(t0)
        p_end_a = _on_a(t1)
        p_start_b = _on_b_near(p_start_a)
        p_end_b = _on_b_near(p_end_a)
        mid0 = ((p_start_a[0] + p_start_b[0]) / 2, (p_start_a[1] + p_start_b[1]) / 2)
        mid1 = ((p_end_a[0] + p_end_b[0]) / 2, (p_end_a[1] + p_end_b[1]) / 2)
        if _seg_length(mid0, mid1) < self.config.shared_boundary_min_length * 0.5:
            return None
        return LineString([mid0, mid1])

    def _segment_pair_candidates(
        self, segments: list[Seg],
    ) -> list[tuple[Seg, Seg, float, float]]:
        """Paires de segments extérieurs face-à-face candidats à fusion."""
        cfg = self.config
        exterior = []
        for p0, p1, la, lb in segments:
            if la > 0 and lb == 0:
                exterior.append((p0, p1, la, 0))
            elif lb > 0 and la == 0:
                exterior.append((p0, p1, lb, 0))

        candidates = []
        for i, sa in enumerate(exterior):
            for sb in exterior[i + 1:]:
                if sa[2] == sb[2]:
                    continue  # même bâtiment
                ang_a = _seg_angle_deg(sa[0], sa[1])
                ang_b = _seg_angle_deg(sb[0], sb[1])
                # parallèle (y compris sens opposé)
                d_ang = min(_angle_diff_deg(ang_a, ang_b), _angle_diff_deg(ang_a, ang_b + 180))
                if d_ang > cfg.shared_boundary_angle_threshold:
                    continue
                # distance moyenne entre milieux / points
                mid_a = ((sa[0][0] + sa[1][0]) / 2, (sa[0][1] + sa[1][1]) / 2)
                dist = _point_line_distance(mid_a, sb[0], sb[1])
                if dist > cfg.shared_boundary_max_distance or dist < 0.15:
                    continue
                len_a = _seg_length(sa[0], sa[1])
                len_b = _seg_length(sb[0], sb[1])
                short = min(len_a, len_b)
                if short < cfg.shared_boundary_min_length:
                    continue
                # recouvrement longitudinal approx
                ax, ay = sa[0]
                dx, dy = sa[1][0] - sa[0][0], sa[1][1] - sa[0][1]
                L2 = max(dx * dx + dy * dy, 1e-12)

                def _t(p):
                    return ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2

                t0 = max(0.0, min(_t(sb[0]), _t(sb[1])))
                t1 = min(1.0, max(_t(sb[0]), _t(sb[1])))
                overlap = max(0.0, t1 - t0) * len_a
                if overlap / short < cfg.shared_boundary_min_overlap_ratio:
                    continue
                candidates.append((sa, sb, dist, overlap))
        return candidates

    def merge_close_parallel_boundaries(self, segments: list[Seg]) -> list[Seg]:
        """Remplace deux murs parallèles proches par une frontière médiane commune."""
        cfg = self.config
        candidates = self._segment_pair_candidates(segments)
        if not candidates:
            return segments

        # Traiter les paires les plus proches d'abord
        candidates.sort(key=lambda x: x[2])
        used_keys: set[tuple] = set()
        to_remove: set[tuple] = set()
        to_add: list[Seg] = []

        for sa, sb, dist, overlap in candidates:
            ka = _seg_key(sa[0], sa[1], sa[2], 0)
            kb = _seg_key(sb[0], sb[1], sb[2], 0)
            if ka in used_keys or kb in used_keys:
                continue
            mid = self.build_shared_boundary((sa[0], sa[1]), (sb[0], sb[1]))
            if mid is None or mid.length < cfg.shared_boundary_min_length * 0.5:
                continue
            coords = list(mid.coords)
            m0, m1 = (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])
            to_add.append((m0, m1, sa[2], sb[2]))
            to_remove.add(ka)
            to_remove.add(kb)
            used_keys.add(ka)
            used_keys.add(kb)

        if not to_add:
            return segments

        out: list[Seg] = []
        for p0, p1, la, lb in segments:
            # segments extérieurs remplacés
            if lb == 0 and _seg_key(p0, p1, la, 0) in to_remove:
                continue
            if la == 0 and _seg_key(p0, p1, lb, 0) in to_remove:
                continue
            out.append((p0, p1, la, lb))
        out.extend(to_add)
        return out

    # ── 6–7. Orientation + orthogonalisation ──────────────────────
    def estimate_dominant_orientations(
        self, segments: list[Seg],
    ) -> tuple[float, float]:
        lengths: list[float] = []
        angles: list[float] = []
        for p0, p1, _, _ in segments:
            L = _seg_length(p0, p1)
            if L < self.config.min_edge_length:
                continue
            lengths.append(L)
            angles.append(_seg_angle_deg(p0, p1) % 180)

        if not lengths:
            return 0.0, 0.0

        hist = np.zeros(36, dtype=np.float64)
        for a, w in zip(angles, lengths):
            hist[min(35, int(a / 5.0))] += w
        dom = float(np.argmax(hist)) * 5.0
        conf = float(hist.max() / max(sum(hist), 1e-6))
        return dom, conf

    def _snap_edge_angle(self, p0, p1, dom: float, conf: float):
        L = _seg_length(p0, p1)
        if L < self.config.min_edge_length:
            return p0, p1
        ang = _seg_angle_deg(p0, p1)
        cfg = self.config
        if not cfg.preserve_oblique or conf >= cfg.min_oblique_confidence:
            best, best_d = ang, 180.0
            for c in (dom, dom + 90, dom + 180, dom + 270):
                d = _angle_diff_deg(ang, c)
                if d < best_d:
                    best_d, best = d, c
            if best_d <= cfg.orthogonal_snap_deg:
                ang = _norm_angle_deg(best)
        rad = _deg2rad(ang)
        cx = (p0[0] + p1[0]) / 2
        cy = (p0[1] + p1[1]) / 2
        h = L / 2
        return (cx - h * math.cos(rad), cy - h * math.sin(rad)), (
            cx + h * math.cos(rad), cy + h * math.sin(rad),
        )

    def regularize_segments(self, segments: list[Seg]) -> list[Seg]:
        dom, conf = self.estimate_dominant_orientations(segments)
        reg = []
        for p0, p1, la, lb in segments:
            np0, np1 = self._snap_edge_angle(p0, p1, dom, conf)
            reg.append((np0, np1, la, lb))
        reg, _ = self.snap_vertices_global(reg)
        return self._merge_collinear_segments(reg)

    def _merge_collinear_segments(self, segments: list[Seg]) -> list[Seg]:
        """Fusionne segments consécutifs quasi colinéaires (même paire de labels)."""
        cfg = self.config
        thr = cfg.collinear_angle_deg
        dist_tol = cfg.collinear_distance_tol

        # Index: endpoint → list of segment indices
        by_end: dict[tuple[float, float], list[int]] = defaultdict(list)
        segs = list(segments)
        for i, (p0, p1, la, lb) in enumerate(segs):
            by_end[_pt_key(p0)].append(i)
            by_end[_pt_key(p1)].append(i)

        used = [False] * len(segs)
        out: list[Seg] = []

        for i, (p0, p1, la, lb) in enumerate(segs):
            if used[i]:
                continue
            used[i] = True
            cp = _canonical_pair(la, lb)
            # étendre dans les deux directions
            chain = [p0, p1]
            # forward from p1
            tip = p1
            while True:
                tip_k = _pt_key(tip)
                found = None
                for j in by_end.get(tip_k, []):
                    if used[j]:
                        continue
                    q0, q1, qa, qb = segs[j]
                    if _canonical_pair(qa, qb) != cp:
                        continue
                    # orienter pour partir de tip
                    if _pt_key(q0) == tip_k:
                        nxt = q1
                    elif _pt_key(q1) == tip_k:
                        nxt = q0
                    else:
                        continue
                    if len(chain) < 2:
                        continue
                    prev = chain[-2]
                    ang = _turn_angle_deg(prev, tip, nxt)
                    # quasi-colinéaire si angle ~0 ou ~180
                    if ang <= thr or ang >= (180 - thr):
                        if _point_line_distance(tip, prev, nxt) <= dist_tol:
                            found = (j, nxt)
                            break
                if found is None:
                    break
                j, nxt = found
                used[j] = True
                chain.append(nxt)
                tip = nxt

            # backward from p0
            tip = p0
            while True:
                tip_k = _pt_key(tip)
                found = None
                for j in by_end.get(tip_k, []):
                    if used[j]:
                        continue
                    q0, q1, qa, qb = segs[j]
                    if _canonical_pair(qa, qb) != cp:
                        continue
                    if _pt_key(q0) == tip_k:
                        nxt = q1
                    elif _pt_key(q1) == tip_k:
                        nxt = q0
                    else:
                        continue
                    if len(chain) < 2:
                        continue
                    prev = chain[1]
                    ang = _turn_angle_deg(nxt, tip, prev)
                    if ang <= thr or ang >= (180 - thr):
                        if _point_line_distance(tip, nxt, prev) <= dist_tol:
                            found = (j, nxt)
                            break
                if found is None:
                    break
                j, nxt = found
                used[j] = True
                chain.insert(0, nxt)
                tip = nxt

            if _seg_length(chain[0], chain[-1]) >= cfg.min_edge_length * 0.5:
                out.append((chain[0], chain[-1], la, lb))

        return self._merge_axis_aligned_segments(out) if out else out

    # ── 8. Suppression sommets presque colinéaires (polygones) ────
    def remove_nearly_collinear_vertices(
        self,
        polygon: Polygon,
        angle_threshold: float | None = None,
        distance_threshold: float | None = None,
    ) -> Polygon | None:
        """
        Supprime les sommets inutiles :
        - 80–100° → snap conceptuel (coin conservé)
        - 170–190° → mur continu, supprimer B
        - >150° + dist(B, AC) faible → supprimer B
        - zigzags courts → approximer par une droite
        """
        cfg = self.config
        ang_thr = angle_threshold if angle_threshold is not None else cfg.collinear_angle_deg
        dist_thr = distance_threshold if distance_threshold is not None else cfg.collinear_distance_tol

        poly = _largest_polygon(make_valid(polygon))
        if poly is None or poly.is_empty:
            return None
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return poly

        # ang = déviation de direction (0° = mur droit, 90° = coin)
        # angle interne ~180° ⇔ déviation ~0°
        open_turn_max = 180.0 - cfg.open_angle_min_deg  # 170° interne → 10° turn
        soft_turn_max = 180.0 - cfg.soft_open_angle_deg  # 150° interne → 30° turn

        changed = True
        while changed and len(coords) >= 3:
            changed = False
            new_coords: list[tuple[float, float]] = []
            n = len(coords)
            skip = set()
            for i in range(n):
                if i in skip:
                    continue
                a = coords[i - 1]
                b = coords[i]
                c = coords[(i + 1) % n]
                turn = _turn_angle_deg(a, b, c)
                dist_ac = _point_line_distance(b, a, c)
                len_ab = _seg_length(a, b)
                len_bc = _seg_length(b, c)

                # mur continu : déviation faible (angle interne ~180°)
                if turn <= max(ang_thr, open_turn_max) and dist_ac <= dist_thr * 2:
                    changed = True
                    continue
                # angle très ouvert + B proche de AC
                if turn <= soft_turn_max and dist_ac <= dist_thr:
                    changed = True
                    continue
                # zigzag : micro-segments avec petite déviation
                if (len_ab < cfg.min_edge_length * 2 and len_bc < cfg.min_edge_length * 2
                        and turn <= soft_turn_max):
                    changed = True
                    continue
                # coin ~90° : conserver (80–100° interne ⇔ turn 80–100°)
                new_coords.append(b)

            if len(new_coords) >= 3 and len(new_coords) < n:
                coords = new_coords
            elif len(new_coords) >= 3:
                break
            else:
                break

        if len(coords) < 3:
            return poly
        ring = coords + [coords[0]]
        return _largest_polygon(make_valid(Polygon(ring)))

    def orthogonalize_polygon_ring(self, polygon: Polygon) -> Polygon | None:
        """Orthogonalise un polygone isolé (θ, θ+90) + suppression colinéaires."""
        poly = _largest_polygon(make_valid(polygon))
        if poly is None:
            return None
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return poly

        # orientation dominante via minAreaRect
        pts = np.array(coords, dtype=np.float32)
        (_, (w, h), ang) = cv2.minAreaRect(pts)
        if w < h:
            ang += 90
        dom = _norm_angle_deg(ang)

        # Snap chaque arête, reconstruire par intersections
        n = len(coords)
        dirs = []
        lengths = []
        for i in range(n):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % n]
            L = max(_seg_length((x0, y0), (x1, y1)), self.config.min_edge_length)
            edge = _seg_angle_deg((x0, y0), (x1, y1))
            best, best_d = edge, 180.0
            for c in (dom, dom + 90, dom + 180, dom + 270):
                d = _angle_diff_deg(edge, c)
                if d < best_d:
                    best_d, best = d, c
            if best_d <= self.config.orthogonal_snap_deg:
                edge = _norm_angle_deg(best)
            dirs.append(_deg2rad(edge))
            lengths.append(L)

        # reconstruction par intersections de droites
        lines = []
        x, y = coords[0]
        for i in range(n):
            lines.append(((x, y), dirs[i]))
            x += lengths[i] * math.cos(dirs[i])
            y += lengths[i] * math.sin(dirs[i])

        new_verts = []
        for i in range(n):
            p0, a0 = lines[i]
            p1, a1 = lines[(i + 1) % n]
            inter = self._line_intersection(p0, a0, p1, a1)
            new_verts.append(inter if inter is not None else p1)

        if len(new_verts) < 3:
            return poly
        out = _largest_polygon(make_valid(Polygon(new_verts + [new_verts[0]])))
        if out is None:
            return poly
        return self.remove_nearly_collinear_vertices(out) or out

    @staticmethod
    def _line_intersection(p0, ang0, p1, ang1):
        dx0, dy0 = math.cos(ang0), math.sin(ang0)
        dx1, dy1 = math.cos(ang1), math.sin(ang1)
        denom = dx0 * dy1 - dy0 * dx1
        if abs(denom) < 1e-10:
            return p1
        t = ((p1[0] - p0[0]) * dy1 - (p1[1] - p0[1]) * dx1) / denom
        return (p0[0] + t * dx0, p0[1] + t * dy0)

    # ── 9–10. Polygonize + attribution ────────────────────────────
    def segments_to_lines(self, segments: list[Seg]) -> list[LineString]:
        seen: set[tuple] = set()
        lines: list[LineString] = []
        for p0, p1, la, lb in segments:
            key = _seg_key(p0, p1, la, lb)
            if key in seen:
                continue
            seen.add(key)
            if _seg_length(p0, p1) >= self.config.min_edge_length * 0.5:
                lines.append(LineString([p0, p1]))
        return lines

    def polygonize_partition(
        self, inst: np.ndarray, lines: list[LineString],
    ) -> dict[int, Polygon]:
        if not lines:
            return {}

        # noding via unary_union pour intersections exactes
        try:
            merged = unary_union(lines)
            if isinstance(merged, LineString):
                geoms = [merged]
            elif isinstance(merged, MultiLineString):
                geoms = list(merged.geoms)
            else:
                geoms = lines
            faces = list(polygonize(geoms))
        except Exception:
            faces = list(polygonize(lines))

        if not faces:
            return self._polygons_from_raster_direct(inst)

        by_label: dict[int, list[Polygon]] = defaultdict(list)
        H, W = inst.shape
        for face in faces:
            c = face.centroid
            cx, cy = int(round(c.x)), int(round(c.y))
            if cx < 0 or cy < 0 or cx >= W or cy >= H:
                continue
            lab = int(inst[cy, cx])
            if lab <= 0:
                # vote IoU sur le masque
                lab = self._best_label_for_face(face, inst)
            if lab <= 0:
                continue
            by_label[lab].append(face)

        polys: dict[int, Polygon] = {}
        for lab, parts in by_label.items():
            merged = unary_union(parts)
            poly = _largest_polygon(make_valid(merged))
            if poly is not None and poly.area >= self.config.min_component_area:
                polys[lab] = poly
        return polys

    def _best_label_for_face(self, face: Polygon, inst: np.ndarray) -> int:
        H, W = inst.shape
        mask = self.polygon_to_mask(face, (H, W))
        if mask.sum() == 0:
            return 0
        best_lab, best_ov = 0, 0
        for lab in range(1, int(inst.max()) + 1):
            ov = int(((inst == lab) & (mask > 0)).sum())
            if ov > best_ov:
                best_ov, best_lab = ov, lab
        return best_lab

    @staticmethod
    def _polygons_from_raster_direct(inst: np.ndarray) -> dict[int, Polygon]:
        import rasterio.features
        from shapely.geometry import shape

        polys: dict[int, Polygon] = {}
        for geom, val in rasterio.features.shapes(inst.astype(np.int32), mask=inst > 0):
            lab = int(val)
            if lab <= 0:
                continue
            g = _largest_polygon(make_valid(shape(geom)))
            if g is not None:
                if lab in polys:
                    polys[lab] = _largest_polygon(make_valid(unary_union([polys[lab], g])))
                else:
                    polys[lab] = g
        return polys

    def resolve_overlaps_and_gaps(
        self, polys: dict[int, Polygon], inst: np.ndarray,
    ) -> dict[int, Polygon]:
        """Rasterise → partition stricte → re-vectorise (zéro overlap)."""
        H, W = inst.shape
        canvas = np.zeros((H, W), np.int32)
        order = sorted(polys.keys(), key=lambda k: polys[k].area, reverse=True)
        for lab in order:
            mask = self.polygon_to_mask(polys[lab], (H, W))
            canvas[(mask > 0) & (canvas == 0)] = lab
        missing = (canvas == 0) & (inst > 0)
        canvas[missing] = inst[missing]
        return self._polygons_from_raster_direct(canvas)

    def restore_shared_boundaries(
        self, polys: dict[int, Polygon], segments: list[Seg],
    ) -> dict[int, Polygon]:
        """
        Après simplification locale, force les frontières partagées
        à utiliser exactement les mêmes coordonnées (segments du réseau).
        """
        shared = [(p0, p1, la, lb) for p0, p1, la, lb in segments if la > 0 and lb > 0]
        if not shared:
            return polys

        out = dict(polys)
        for p0, p1, la, lb in shared:
            if la not in out or lb not in out:
                continue
            shared_line = LineString([p0, p1])
            # snap les deux polygones sur cette ligne
            try:
                out[la] = make_valid(out[la].buffer(0))
                out[lb] = make_valid(out[lb].buffer(0))
                # découpe : chaque côté de la ligne médiane
                # approche simple : difference mutuelle + snap
                from shapely.ops import snap as shp_snap
                out[la] = _largest_polygon(shp_snap(out[la], shared_line, 1.0)) or out[la]
                out[lb] = _largest_polygon(shp_snap(out[lb], shared_line, 1.0)) or out[lb]
            except Exception:
                pass

        # zéro overlap final par clip ordonné
        return self.clip_overlaps(out)

    def clip_overlaps(self, polys: dict[int, Polygon]) -> dict[int, Polygon]:
        cfg = self.config
        order = sorted(polys.keys(), key=lambda k: polys[k].area if polys[k] else 0, reverse=True)
        kept: dict[int, Polygon] = {}
        for lab in order:
            poly = _largest_polygon(make_valid(polys[lab]))
            if poly is None or poly.is_empty:
                continue
            for other in kept.values():
                if not poly.intersects(other):
                    continue
                inter = poly.intersection(other)
                if inter.is_empty or inter.area <= cfg.overlap_max_area:
                    continue
                poly = _largest_polygon(make_valid(poly.difference(other)))
                if poly is None or poly.is_empty:
                    break
            if poly is not None and not poly.is_empty:
                kept[lab] = poly
        return kept

    def polygon_to_mask(self, poly: Polygon, shape_hw: tuple[int, int]) -> np.ndarray:
        import rasterio.features
        from affine import Affine
        from shapely.geometry import mapping

        if poly is None or poly.is_empty:
            return np.zeros(shape_hw, dtype=np.uint8)
        return rasterio.features.rasterize(
            [(mapping(poly), 1)],
            out_shape=shape_hw,
            transform=Affine.identity(),
            fill=0,
            dtype=np.uint8,
        )

    # ── 11. Validation topologique ────────────────────────────────
    def validate_topology(
        self, polys: dict[int, Polygon], inst: np.ndarray,
    ) -> TopologyMetrics:
        cfg = self.config
        m = TopologyMetrics()
        m.n_instances_in = int(inst.max())
        m.n_instances_out = len(polys)

        ids = sorted(polys.keys())
        overlap_area = 0.0
        n_over = 0
        invalid = 0
        for i, a in enumerate(ids):
            if not polys[a].is_valid:
                invalid += 1
            for b in ids[i + 1:]:
                inter = polys[a].intersection(polys[b])
                if not inter.is_empty and inter.area > cfg.overlap_max_area:
                    n_over += 1
                    overlap_area += inter.area

        m.n_overlaps = n_over
        m.overlap_area = overlap_area
        m.invalid_geometry_count = invalid

        H, W = inst.shape
        canvas = np.zeros((H, W), np.int32)
        for lab, poly in polys.items():
            canvas[self.polygon_to_mask(poly, (H, W)) > 0] = lab
        gaps = (inst > 0) & (canvas == 0)
        m.n_gaps = int(gaps.sum())
        m.gap_area = float(m.n_gaps)

        ious, area_chgs = [], []
        n_verts = n_90 = n_angles = 0
        for lab, poly in polys.items():
            pred_m = self.polygon_to_mask(poly, (H, W))
            orig_m = (inst == lab).astype(np.uint8)
            pred_sum = int(pred_m.sum())
            oa = int(orig_m.sum())
            inter = int((pred_m & orig_m).sum())
            union = int(((pred_m | orig_m) > 0).sum())
            ious.append(inter / max(union, 1))
            area_chgs.append(abs(pred_sum - oa) / max(oa, 1))
            coords = list(poly.exterior.coords)[:-1]
            n_verts += len(coords)
            for i in range(len(coords)):
                ang = _turn_angle_deg(coords[i - 1], coords[i], coords[(i + 1) % len(coords)])
                n_angles += 1
                if abs(ang - 90) <= cfg.orthogonal_snap_deg or abs(ang - 180) <= cfg.orthogonal_snap_deg:
                    n_90 += 1

        m.mean_iou_vs_raster = float(np.mean(ious)) if ious else 0.0
        m.mean_area_change = float(np.mean(area_chgs)) if area_chgs else 0.0
        m.n_vertices_after = n_verts
        m.pct_angles_near_90 = 100.0 * n_90 / max(n_angles, 1)

        ok_shared, dup_len = self.check_shared_boundaries(polys)
        m.duplicate_boundary_length = dup_len
        m.shared_boundary_coordinates_match = ok_shared

        m.accepted = (
            m.overlap_area <= cfg.overlap_max_area
            and m.invalid_geometry_count == 0
            and m.mean_iou_vs_raster >= cfg.min_polygon_iou
            and m.mean_area_change <= cfg.max_area_change_ratio
            and m.n_instances_out >= max(1, m.n_instances_in - 2)
        )
        if not m.accepted:
            m.message = (
                f"overlap={m.overlap_area:.1f} iou={m.mean_iou_vs_raster:.3f} "
                f"area_chg={m.mean_area_change:.3f} invalid={m.invalid_geometry_count}"
            )
        return m

    def check_shared_boundaries(
        self, polys: dict[int, Polygon], tol: float = 0.5,
    ) -> tuple[bool, float]:
        """
        Vérifie l'absence de double trait entre voisins.
        Retourne (ok, longueur_double_frontière).
        """
        ids = sorted(polys.keys())
        dup_len = 0.0
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = polys[a], polys[b]
                dist = pa.distance(pb)
                if dist > tol:
                    continue
                # voisins proches : doivent toucher avec frontière commune
                shared = pa.boundary.intersection(pb.boundary)
                if not shared.is_empty and shared.length > 0:
                    continue
                # gap ou double trait : length of near-parallel offset
                buffered = pa.boundary.buffer(tol).intersection(pb.boundary)
                if not buffered.is_empty:
                    dup_len += buffered.length if hasattr(buffered, "length") else 0.0
        return dup_len < 1.0, dup_len

    # ── Pipeline principal (option 1) ─────────────────────────────
    def process_instance_map(
        self,
        inst: np.ndarray,
        transform=None,
        crs_str: str = "EPSG:4326",
        min_area_px: int = 1,
    ):
        """
        Ordre :
          1. nettoyage masque
          2. extraction contours
          3. simplification légère
          4. détection voisins + frontières communes
          5–7. snap + orientation + orthogonalisation
          8–9. fusion colinéaire / suppression ondulations
          10. reconstruction polygones
          11. validation
        """
        import geopandas as gpd

        from .geo_io import safe_crs

        t0 = time.perf_counter()
        inst_clean = self.clean_partition_raster(inst)

        # 2–3. extraction + merge axis-aligned (simplification légère)
        raw_segs = self.extract_boundary_segments(inst_clean)
        n_verts_before = len(raw_segs) * 2

        # 4. frontières communes pour petits gaps / doubles murs
        merged_segs = self.merge_close_parallel_boundaries(raw_segs)

        # 5. snap global
        snapped, nodes = self.snap_vertices_global(merged_segs)

        # 6–9. orientation + orthogonalisation + fusion colinéaire
        reg_segs = self.regularize_segments(snapped)
        lines = self.segments_to_lines(reg_segs)

        # 10. reconstruction
        polys = self.polygonize_partition(inst_clean, lines)

        # Compléter les labels manquants depuis le raster
        expected = {
            lab for lab in range(1, int(inst_clean.max()) + 1)
            if int((inst_clean == lab).sum()) >= self.config.min_component_area
        }
        missing = expected - set(polys.keys())
        if missing:
            direct = self._polygons_from_raster_direct(inst_clean)
            for lab in missing:
                if lab in direct:
                    polys[lab] = direct[lab]

        # Orthogonalisation + nettoyage des sommets sur chaque face
        # (les arêtes partagées du réseau restent la référence)
        cleaned: dict[int, Polygon] = {}
        for lab, poly in polys.items():
            # Si le bâtiment a des voisins, ne pas remodeler agressivement :
            # seulement remove_nearly_collinear (conserve la topologie locale)
            adj = self.build_adjacency_graph(reg_segs)
            has_neighbors = lab in adj and len(adj[lab]) > 0
            if has_neighbors:
                cleaned[lab] = self.remove_nearly_collinear_vertices(poly) or poly
            else:
                orth = self.orthogonalize_polygon_ring(poly)
                cleaned[lab] = orth or self.remove_nearly_collinear_vertices(poly) or poly
        polys = cleaned

        # Garantir zéro overlap sans détruire le réseau
        if any(
            not polys[a].intersection(polys[b]).is_empty
            and polys[a].intersection(polys[b]).area > self.config.overlap_max_area
            for i, a in enumerate(sorted(polys))
            for b in sorted(polys)[i + 1:]
        ):
            polys = self.resolve_overlaps_and_gaps(polys, inst_clean)
            # re-nettoyer sommets après re-vectorisation
            polys = {
                lab: self.remove_nearly_collinear_vertices(p) or p
                for lab, p in polys.items()
            }

        metrics = self.validate_topology(polys, inst_clean)
        metrics.n_vertices_before = n_verts_before

        if not metrics.accepted:
            # fallback : partition raster (topologie correcte, géométrie plus brute)
            polys = self._polygons_from_raster_direct(inst_clean)
            polys = {
                lab: self.remove_nearly_collinear_vertices(p) or p
                for lab, p in polys.items()
            }
            polys = self.clip_overlaps(polys)
            metrics = self.validate_topology(polys, inst_clean)
            metrics.n_vertices_before = n_verts_before
            metrics.message += " | fallback=raster_direct"

        metrics.elapsed_ms = (time.perf_counter() - t0) * 1000

        shared_lines: list[LineString] = []
        exterior_lines: list[LineString] = []
        for p0, p1, la, lb in reg_segs:
            ls = LineString([p0, p1])
            if la > 0 and lb > 0:
                shared_lines.append(ls)
            else:
                exterior_lines.append(ls)

        rows = []
        for lab, poly in sorted(polys.items()):
            if int((inst_clean == lab).sum()) < max(min_area_px, self.config.min_component_area):
                continue
            geo = poly
            if transform is not None:
                geo = self._pixel_to_geo(poly, transform)
            if geo is None or geo.is_empty:
                continue
            rows.append({
                "geometry": geo,
                "building_id": lab,
                "area_m2": round(geo.area, 2),
                "area_px": int((inst_clean == lab).sum()),
            })

        crs = safe_crs(crs_str)
        if not rows:
            gdf = gpd.GeoDataFrame(
                columns=["geometry", "building_id", "area_m2", "area_px"], crs=crs,
            )
        else:
            try:
                gdf = gpd.GeoDataFrame(rows, crs=crs)
            except Exception:
                gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

        result = TopologyResult(
            polygons=polys,
            segments=[LineString([p0, p1]) for p0, p1, _, _ in reg_segs],
            shared_segments=shared_lines,
            exterior_segments=exterior_lines,
            nodes=nodes,
            metrics=metrics,
            raster_regularized=self._rasterize_polys(polys, inst.shape),
            shape_stats=None,
        )
        return gdf, result

    @staticmethod
    def _pixel_to_geo(poly: Polygon, transform) -> Polygon | None:
        from rasterio.transform import xy
        from shapely.ops import transform as shp_transform

        def _tf(x, y, z=None):
            gx, gy = xy(transform, y, x)
            return (gx, gy)

        geo = shp_transform(_tf, poly)
        return _largest_polygon(make_valid(geo))

    def _rasterize_polys(self, polys: dict[int, Polygon], shape_hw) -> np.ndarray:
        out = np.zeros(shape_hw, np.int32)
        for lab, poly in polys.items():
            out[self.polygon_to_mask(poly, shape_hw) > 0] = lab
        return out
