"""
Régularisation par re-fit géométrique (rectangle tourné / rectilinéaire).

Contrairement à building_regularizer.py qui corrige le contour existant
(snap arête par arête → hérite du zigzag), on REFITTE une forme idéale :

  1. minAreaRect : si le rectangle minimum couvre le masque (IoU élevé),
     on remplace le polygone par le rectangle → 4 sommets, angles 90°.
  2. Sinon fit rectilinéaire : rotation dans le repère du bâtiment,
     chaque arête forcée horizontale/verticale (moyenne pondérée),
     reconstruction par intersections → formes en L/T propres.
  3. Orientation par îlot : les voisins proches partagent une orientation
     dominante (moyenne circulaire mod 90 pondérée par surface).

Validation IoU vs masque d'origine + fallback simplifié sinon.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from .building_regularizer import _norm_angle_deg


def _largest_polygon(geom) -> Polygon | None:
    """Plus grand polygone de n'importe quelle géométrie (GeometryCollection incluse)."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]
    # GeometryCollection imbriquée (make_valid peut en produire)
    for g in getattr(geom, "geoms", []):
        if not isinstance(g, (Polygon, MultiPolygon)) and hasattr(g, "geoms"):
            sub = _largest_polygon(g)
            if sub is not None:
                polys.append(sub)
        elif isinstance(g, MultiPolygon):
            sub = _largest_polygon(g)
            if sub is not None:
                polys.append(sub)
    return max(polys, key=lambda p: p.area) if polys else None


class FitLevel(str, Enum):
    RECTANGLE = "rectangle"
    RECTILINEAR = "rectilinear"
    SIMPLIFIED = "simplified"
    ORIGINAL = "original"
    REJECTED = "rejected"


@dataclass
class RectRegularizerConfig:
    """Seuils en pixels pour tuiles 512×512."""
    min_component_area: int = 25
    max_hole_area: int = 64
    # 1. Rectangle direct si IoU(masque, minAreaRect) >= ce seuil
    rect_iou_accept: float = 0.78
    # 2. Fit rectilinéaire
    simplify_tolerance: float = 1.8
    min_edge_length: float = 3.0
    # Validation du candidat retenu vs masque d'origine
    min_final_iou: float = 0.70
    max_area_change_ratio: float = 0.30
    # 3. Orientation par îlot
    group_orientation: bool = True
    neighbor_max_dist: float = 12.0
    group_angle_snap_deg: float = 10.0
    # Snap final des sommets entre voisins (murs mitoyens)
    snap_neighbor_vertices: bool = True
    snap_vertex_tol: float = 2.0
    # Murs mitoyens : arêtes parallèles face-à-face fusionnées (un seul trait)
    align_shared_walls: bool = True
    wall_snap_tol: float = 4.0
    wall_min_overlap: float = 5.0
    # Bâtiments coupés par le bord : orientation forcée de l'îlot + clip
    force_border_orientation: bool = True
    clip_to_bounds: bool = True


@dataclass
class RectFitResult:
    building_id: int
    polygon: Polygon | None
    level: FitLevel
    angle_deg: float = 0.0
    iou_vs_mask: float = 0.0
    n_vertices: int = 0
    elapsed_ms: float = 0.0


@dataclass
class RectFitStats:
    n_input: int = 0
    n_rectangle: int = 0
    n_rectilinear: int = 0
    n_simplified: int = 0
    n_original: int = 0
    n_rejected: int = 0
    elapsed_ms: float = 0.0
    results: list[RectFitResult] = field(default_factory=list)


def _mean_angle_mod90(angles_deg: list[float], weights: list[float]) -> float:
    """Moyenne circulaire d'angles équivalents mod 90° (repères de rectangles)."""
    s = c = 0.0
    for a, w in zip(angles_deg, weights):
        rad = math.radians((a % 90.0) * 4.0)
        s += w * math.sin(rad)
        c += w * math.cos(rad)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return angles_deg[0] % 90.0
    return (math.degrees(math.atan2(s, c)) / 4.0) % 90.0


def _angle_diff_mod90(a: float, b: float) -> float:
    d = abs((a - b) % 90.0)
    return min(d, 90.0 - d)


class RectilinearRegularizer:
    def __init__(self, config: RectRegularizerConfig | None = None):
        self.config = config or RectRegularizerConfig()

    # ── Masque → polygone brut ────────────────────────────────────
    def mask_to_polygon(self, mask: np.ndarray) -> Polygon | None:
        m = (mask > 0).astype(np.uint8)
        if int(m.sum()) < self.config.min_component_area:
            return None
        try:
            from skimage.morphology import remove_small_holes
            m = remove_small_holes(m.astype(bool),
                                   area_threshold=max(1, self.config.max_hole_area)
                                   ).astype(np.uint8)
        except ImportError:
            pass
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        if len(pts) < 3:
            return None
        poly = _largest_polygon(make_valid(Polygon(pts + [pts[0]])))
        return poly

    # ── Orientation ───────────────────────────────────────────────
    @staticmethod
    def dominant_angle(poly: Polygon) -> float:
        """Angle (deg) du grand côté du rectangle minimum, dans [0, 180)."""
        pts = np.array(list(poly.exterior.coords)[:-1], dtype=np.float32)
        (_, (w, h), ang) = cv2.minAreaRect(pts)
        if w < h:
            ang += 90.0
        return _norm_angle_deg(ang)

    @staticmethod
    def min_area_rectangle(poly: Polygon) -> Polygon | None:
        pts = np.array(list(poly.exterior.coords)[:-1], dtype=np.float32)
        if len(pts) < 3:
            return None
        box = cv2.boxPoints(cv2.minAreaRect(pts))
        return _largest_polygon(make_valid(Polygon([tuple(map(float, p)) for p in box])))

    @staticmethod
    def _iou(a: Polygon | None, b: Polygon | None) -> float:
        if a is None or b is None or a.is_empty or b.is_empty:
            return 0.0
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0

    # ── Fit rectilinéaire dans le repère tourné ───────────────────
    def rectilinear_fit(self, poly: Polygon, angle_deg: float) -> Polygon | None:
        """Force chaque arête à être H ou V dans le repère tourné de -angle."""
        cfg = self.config
        origin = poly.centroid
        rot = affinity.rotate(poly, -angle_deg, origin=origin)
        simp = rot.simplify(cfg.simplify_tolerance, preserve_topology=True)
        simp = _largest_polygon(make_valid(simp))
        if simp is None:
            return None
        coords = list(simp.exterior.coords)[:-1]
        if len(coords) < 3:
            return None

        # 1. classer chaque arête H ou V, regrouper les runs consécutifs
        n = len(coords)
        orient: list[str] = []
        for i in range(n):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % n]
            orient.append("h" if abs(x1 - x0) >= abs(y1 - y0) else "v")

        runs: list[tuple[str, list[int]]] = []
        for i, o in enumerate(orient):
            if runs and runs[-1][0] == o:
                runs[-1][1].append(i)
            else:
                runs.append((o, [i]))
        # boucle fermée : fusion premier/dernier run si même orientation
        if len(runs) >= 2 and runs[0][0] == runs[-1][0]:
            runs[0] = (runs[0][0], runs[-1][1] + runs[0][1])
            runs.pop()
        if len(runs) < 4:
            return None  # pas assez d'alternances → rectangle fera l'affaire

        # 2. chaque run → une droite (coordonnée = moyenne pondérée par longueur)
        lines: list[tuple[str, float]] = []
        for o, idxs in runs:
            wsum = vsum = 0.0
            for i in idxs:
                x0, y0 = coords[i]
                x1, y1 = coords[(i + 1) % n]
                L = math.hypot(x1 - x0, y1 - y0)
                mid = (y0 + y1) / 2 if o == "h" else (x0 + x1) / 2
                wsum += L
                vsum += L * mid
            if wsum <= 0:
                return None
            lines.append((o, vsum / wsum))

        # 3. sommets = intersections de droites H/V consécutives
        m = len(lines)
        verts: list[tuple[float, float]] = []
        for i in range(m):
            o0, v0 = lines[i]
            o1, v1 = lines[(i + 1) % m]
            if o0 == o1:
                return None  # dégénéré
            x = v1 if o1 == "v" else v0
            y = v1 if o1 == "h" else v0
            verts.append((x, y))

        # arêtes trop courtes → supprimer les paires de sommets, refuser si < 4
        cleaned = self._drop_short_edges(verts)
        if cleaned is None or len(cleaned) < 4:
            return None
        out = _largest_polygon(make_valid(Polygon(cleaned + [cleaned[0]])))
        if out is None or out.is_empty:
            return None
        return affinity.rotate(out, angle_deg, origin=origin)

    def _drop_short_edges(
        self, verts: list[tuple[float, float]],
    ) -> list[tuple[float, float]] | None:
        """Supprime les micro-crans (arêtes < min_edge_length) en re-fusionnant."""
        min_len = self.config.min_edge_length
        pts = list(verts)
        for _ in range(len(verts)):
            n = len(pts)
            if n < 4:
                return None
            short_i = None
            for i in range(n):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % n]
                if math.hypot(x1 - x0, y1 - y0) < min_len:
                    short_i = i
                    break
            if short_i is None:
                return pts
            # retirer l'arête courte : fusionner ses deux extrémités au milieu,
            # puis re-projeter en rectilinéaire (moyenne des voisins)
            i = short_i
            j = (i + 1) % n
            mx = (pts[i][0] + pts[j][0]) / 2
            my = (pts[i][1] + pts[j][1]) / 2
            keep = [pts[k] for k in range(n) if k not in (i, j)]
            prev = pts[(i - 1) % n]
            nxt = pts[(j + 1) % n]
            # nouveau sommet à l'intersection des directions voisines
            if abs(prev[0] - pts[i][0]) < abs(prev[1] - pts[i][1]):
                new_pt = (prev[0], my) if abs(nxt[1] - pts[j][1]) < abs(nxt[0] - pts[j][0]) else (mx, my)
            else:
                new_pt = (mx, prev[1]) if abs(nxt[0] - pts[j][0]) < abs(nxt[1] - pts[j][1]) else (mx, my)
            keep.insert((i - 1) % n + 1 if i > 0 else 0, new_pt)
            pts = keep
        return pts

    # ── Fit d'une instance ────────────────────────────────────────
    def fit_instance(
        self,
        mask: np.ndarray,
        building_id: int = 0,
        forced_angle: float | None = None,
    ) -> RectFitResult:
        t0 = time.perf_counter()
        cfg = self.config
        original = self.mask_to_polygon(mask)
        if original is None:
            return RectFitResult(building_id, None, FitLevel.REJECTED,
                                 elapsed_ms=(time.perf_counter() - t0) * 1000)

        angle = forced_angle if forced_angle is not None else self.dominant_angle(original)

        # candidats, du plus propre au plus fidèle
        rect = (self.min_area_rectangle(original) if forced_angle is None
                else self._oriented_rectangle(original, angle))
        rectl = self.rectilinear_fit(original, angle)
        simp = _largest_polygon(make_valid(
            original.simplify(cfg.simplify_tolerance, preserve_topology=True)))

        candidates: list[tuple[FitLevel, Polygon | None, float]] = []
        iou_rect = self._iou(original, rect)
        if rect is not None and iou_rect >= cfg.rect_iou_accept:
            candidates.append((FitLevel.RECTANGLE, rect, iou_rect))
        if rectl is not None:
            candidates.append((FitLevel.RECTILINEAR, rectl, self._iou(original, rectl)))
        if rect is not None:
            candidates.append((FitLevel.RECTANGLE, rect, iou_rect))
        if simp is not None:
            candidates.append((FitLevel.SIMPLIFIED, simp, self._iou(original, simp)))
        candidates.append((FitLevel.ORIGINAL, original, 1.0))

        for level, cand, iou in candidates:
            if cand is None or cand.is_empty:
                continue
            area_chg = abs(cand.area - original.area) / max(original.area, 1e-6)
            if level != FitLevel.ORIGINAL and (
                iou < cfg.min_final_iou or area_chg > cfg.max_area_change_ratio
            ):
                continue
            return RectFitResult(
                building_id=building_id,
                polygon=cand,
                level=level,
                angle_deg=angle,
                iou_vs_mask=iou,
                n_vertices=len(cand.exterior.coords) - 1,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        return RectFitResult(building_id, original, FitLevel.ORIGINAL,
                             angle_deg=angle, iou_vs_mask=1.0,
                             n_vertices=len(original.exterior.coords) - 1,
                             elapsed_ms=(time.perf_counter() - t0) * 1000)

    def _oriented_rectangle(self, poly: Polygon, angle_deg: float) -> Polygon | None:
        """Rectangle englobant minimal à orientation imposée."""
        origin = poly.centroid
        rot = affinity.rotate(poly, -angle_deg, origin=origin)
        x0, y0, x1, y1 = rot.bounds
        rect = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        return affinity.rotate(rect, angle_deg, origin=origin)

    # ── Carte d'instances complète ────────────────────────────────
    def process_instance_map(
        self, inst: np.ndarray,
    ) -> tuple[dict[int, Polygon], RectFitStats]:
        t0 = time.perf_counter()
        cfg = self.config
        labels = [lab for lab in range(1, int(inst.max()) + 1)
                  if int((inst == lab).sum()) >= cfg.min_component_area]

        # passe 1 : polygones bruts + angles individuels
        raw: dict[int, Polygon] = {}
        angles: dict[int, float] = {}
        for lab in labels:
            p = self.mask_to_polygon((inst == lab).astype(np.uint8))
            if p is not None:
                raw[lab] = p
                angles[lab] = self.dominant_angle(p)

        # bâtiments tronqués par le bord de la tuile → angle propre non fiable
        H, W = inst.shape
        border_labels = {
            lab for lab in raw
            if (inst[0, :] == lab).any() or (inst[-1, :] == lab).any()
            or (inst[:, 0] == lab).any() or (inst[:, -1] == lab).any()
        }

        # passe 2 : orientation partagée par îlot (voisins proches)
        forced: dict[int, float] = {}
        if cfg.group_orientation and len(raw) > 1:
            groups = self._neighbor_groups(raw, cfg.neighbor_max_dist)
            for grp in groups:
                if len(grp) < 2:
                    continue
                # angle de l'îlot estimé sur les bâtiments entiers si possible
                ref = [g for g in grp if g not in border_labels] or grp
                grp_angle = _mean_angle_mod90(
                    [angles[g] for g in ref], [raw[g].area for g in ref])
                for g in grp:
                    is_border = g in border_labels and cfg.force_border_orientation
                    if is_border or _angle_diff_mod90(
                            angles[g], grp_angle) <= cfg.group_angle_snap_deg:
                        # remonter dans [0,180) proche de l'angle individuel
                        cand = min(
                            (grp_angle + k * 90.0 for k in range(2)),
                            key=lambda a: min(abs(angles[g] - a), 180 - abs(angles[g] - a)),
                        )
                        forced[g] = _norm_angle_deg(cand)

        # bâtiment de bord isolé : orientation dominante de la tuile
        if cfg.force_border_orientation and border_labels:
            interior = [g for g in raw if g not in border_labels]
            if interior:
                tile_angle = _mean_angle_mod90(
                    [angles[g] for g in interior], [raw[g].area for g in interior])
                for g in border_labels:
                    if g not in forced:
                        cand = min(
                            (tile_angle + k * 90.0 for k in range(2)),
                            key=lambda a: min(abs(angles[g] - a), 180 - abs(angles[g] - a)),
                        )
                        forced[g] = _norm_angle_deg(cand)

        # passe 3 : fit par instance
        stats = RectFitStats()
        polys: dict[int, Polygon] = {}
        for lab in labels:
            res = self.fit_instance((inst == lab).astype(np.uint8), lab,
                                    forced_angle=forced.get(lab))
            stats.n_input += 1
            stats.results.append(res)
            if res.level == FitLevel.RECTANGLE:
                stats.n_rectangle += 1
            elif res.level == FitLevel.RECTILINEAR:
                stats.n_rectilinear += 1
            elif res.level == FitLevel.SIMPLIFIED:
                stats.n_simplified += 1
            elif res.level == FitLevel.ORIGINAL:
                stats.n_original += 1
            else:
                stats.n_rejected += 1
            if res.polygon is not None:
                polys[lab] = res.polygon

        # passe 4 : clip aux limites de la tuile (bâtiments de bord)
        if cfg.clip_to_bounds:
            from shapely.geometry import box
            bounds = box(-0.5, -0.5, W - 0.5, H - 0.5)
            for lab in list(polys):
                clipped = _largest_polygon(make_valid(polys[lab].intersection(bounds)))
                if clipped is not None and not clipped.is_empty:
                    polys[lab] = clipped

        # passe 5 : murs mitoyens — un seul trait entre voisins
        angles_out = {r.building_id: r.angle_deg for r in stats.results}
        if cfg.align_shared_walls:
            polys = self._align_shared_walls(polys, angles_out)
        if cfg.snap_neighbor_vertices:
            polys = self._snap_shared_vertices(polys, cfg.snap_vertex_tol)

        # passe 6 : zéro chevauchement résiduel
        polys = self._clip_residual_overlaps(polys)

        stats.elapsed_ms = (time.perf_counter() - t0) * 1000
        return polys, stats

    # ── Murs mitoyens ─────────────────────────────────────────────
    @staticmethod
    def _axis_edges(coords: list[tuple[float, float]], eps: float = 0.35):
        """Arêtes ~horizontales/verticales : (orient, coord, t0, t1, i0, i1)."""
        n = len(coords)
        edges = []
        for i in range(n):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % n]
            if abs(x1 - x0) <= eps and abs(y1 - y0) > eps:
                edges.append(("v", (x0 + x1) / 2, min(y0, y1), max(y0, y1), i, (i + 1) % n))
            elif abs(y1 - y0) <= eps and abs(x1 - x0) > eps:
                edges.append(("h", (y0 + y1) / 2, min(x0, x1), max(x0, x1), i, (i + 1) % n))
        return edges

    def _align_shared_walls(
        self, polys: dict[int, Polygon], angles: dict[int, float],
    ) -> dict[int, Polygon]:
        """Fusionne les murs parallèles face-à-face de bâtiments voisins.

        Dans le repère tourné de l'îlot, deux arêtes de même orientation
        distantes de < wall_snap_tol avec recouvrement longitudinal sont
        ramenées à leur ligne médiane → une seule séparation, sans
        chevauchement ni double trait.
        """
        cfg = self.config
        # regrouper par angle (mod 90) quasi identique
        by_angle: dict[float, list[int]] = {}
        for lab in polys:
            key = round((angles.get(lab, 0.0) % 90.0) * 2) / 2
            by_angle.setdefault(key, []).append(lab)

        out = dict(polys)
        for ang_key, labs in by_angle.items():
            if len(labs) < 2:
                continue
            rot = {lab: affinity.rotate(out[lab], -ang_key, origin=(0, 0))
                   for lab in labs}
            coords = {lab: [(float(x), float(y))
                            for x, y in list(rot[lab].exterior.coords)[:-1]]
                      for lab in labs}
            moves: dict[tuple[int, int], tuple[float, float]] = {}

            for ia, a in enumerate(labs):
                ea = self._axis_edges(coords[a])
                ca = rot[a].centroid
                for b in labs[ia + 1:]:
                    if rot[a].distance(rot[b]) > cfg.wall_snap_tol:
                        continue
                    eb = self._axis_edges(coords[b])
                    cb = rot[b].centroid
                    for oa, va, a0, a1, i0a, i1a in ea:
                        for ob, vb, b0, b1, i0b, i1b in eb:
                            if oa != ob or abs(va - vb) > cfg.wall_snap_tol:
                                continue
                            ov = min(a1, b1) - max(a0, b0)
                            if ov < cfg.wall_min_overlap:
                                continue
                            # intérieurs de part et d'autre du mur
                            pa = ca.x if oa == "v" else ca.y
                            pb = cb.x if oa == "v" else cb.y
                            if (pa - va) * (pb - vb) >= 0:
                                continue
                            mid = (va + vb) / 2
                            for lab_i, idx in ((a, i0a), (a, i1a), (b, i0b), (b, i1b)):
                                x, y = moves.get((lab_i, idx), coords[lab_i][idx])
                                moves[(lab_i, idx)] = (mid, y) if oa == "v" else (x, mid)

            if not moves:
                continue
            for (lab, idx), pt in moves.items():
                coords[lab][idx] = pt
            for lab in labs:
                cs = coords[lab]
                p = _largest_polygon(make_valid(Polygon(cs + [cs[0]])))
                if p is not None and not p.is_empty:
                    out[lab] = affinity.rotate(p, ang_key, origin=(0, 0))
        return out

    def _clip_residual_overlaps(self, polys: dict[int, Polygon]) -> dict[int, Polygon]:
        """Différence ordonnée (grands d'abord) → zéro chevauchement garanti."""
        order = sorted(polys, key=lambda k: polys[k].area, reverse=True)
        kept: dict[int, Polygon] = {}
        for lab in order:
            poly = _largest_polygon(make_valid(polys[lab]))
            if poly is None or poly.is_empty:
                continue
            for other in kept.values():
                if not poly.intersects(other):
                    continue
                inter = poly.intersection(other)
                if inter.is_empty or inter.area < 0.5:
                    continue
                diff = _largest_polygon(make_valid(poly.difference(other)))
                if diff is None or diff.is_empty:
                    poly = None
                    break
                poly = diff
            if poly is not None and not poly.is_empty:
                kept[lab] = poly
        return {lab: kept[lab] for lab in sorted(kept)}

    @staticmethod
    def _neighbor_groups(
        polys: dict[int, Polygon], max_dist: float,
    ) -> list[list[int]]:
        """Composantes connexes du graphe « distance < max_dist »."""
        ids = sorted(polys.keys())
        parent = {i: i for i in ids}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                # filtre rapide par bounding box
                ax0, ay0, ax1, ay1 = polys[a].bounds
                bx0, by0, bx1, by1 = polys[b].bounds
                if (ax0 - bx1 > max_dist or bx0 - ax1 > max_dist
                        or ay0 - by1 > max_dist or by0 - ay1 > max_dist):
                    continue
                if polys[a].distance(polys[b]) <= max_dist:
                    parent[find(a)] = find(b)

        groups: dict[int, list[int]] = {}
        for i in ids:
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    @staticmethod
    def _snap_shared_vertices(
        polys: dict[int, Polygon], tol: float,
    ) -> dict[int, Polygon]:
        """Fusionne les sommets de polygones voisins distants de < tol."""
        ids = sorted(polys.keys())
        all_pts: list[tuple[float, float]] = []
        owner: list[tuple[int, int]] = []  # (label, index sommet)
        coords: dict[int, list[tuple[float, float]]] = {}
        for lab in ids:
            cs = [(float(x), float(y)) for x, y in list(polys[lab].exterior.coords)[:-1]]
            coords[lab] = cs
            for k, p in enumerate(cs):
                all_pts.append(p)
                owner.append((lab, k))

        # clustering par grille
        cells: dict[tuple[int, int], list[int]] = {}
        for idx, (x, y) in enumerate(all_pts):
            cells.setdefault((int(round(x / tol)), int(round(y / tol))), []).append(idx)

        for members in cells.values():
            labs = {owner[i][0] for i in members}
            if len(labs) < 2:
                continue  # ne bouge que les sommets partagés entre bâtiments
            cx = sum(all_pts[i][0] for i in members) / len(members)
            cy = sum(all_pts[i][1] for i in members) / len(members)
            for i in members:
                lab, k = owner[i]
                coords[lab][k] = (cx, cy)

        out: dict[int, Polygon] = {}
        for lab in ids:
            cs = coords[lab]
            p = _largest_polygon(make_valid(Polygon(cs + [cs[0]])))
            out[lab] = p if p is not None and not p.is_empty else polys[lab]
        return out
