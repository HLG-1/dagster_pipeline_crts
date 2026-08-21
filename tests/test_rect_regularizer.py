"""Tests du re-fit géométrique (rect_regularizer)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_batiments.core.rect_regularizer import (
    FitLevel,
    RectilinearRegularizer,
    RectRegularizerConfig,
)


def _rasterize(poly: Polygon, shape=(256, 256)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    pts = np.array(list(poly.exterior.coords)[:-1], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m


def _noisy(mask: np.ndarray, seed=0, amp=2) -> np.ndarray:
    """Zigzag artificiel : érosions/dilatations aléatoires du bord."""
    rng = np.random.default_rng(seed)
    noise = (rng.random(mask.shape) > 0.5).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    border = cv2.dilate(mask, k, iterations=amp) - cv2.erode(mask, k, iterations=amp)
    out = mask.copy()
    out[(border > 0) & (noise > 0)] = 1
    out[(border > 0) & (noise == 0)] = 0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    n, lab = cv2.connectedComponents(out)
    if n > 2:
        best = max(range(1, n), key=lambda i: (lab == i).sum())
        out = (lab == best).astype(np.uint8)
    return out


def test_noisy_rotated_rectangle_becomes_4_vertices():
    rect = affinity.rotate(
        Polygon([(60, 80), (180, 80), (180, 150), (60, 150)]), 27, origin="centroid")
    mask = _noisy(_rasterize(rect), seed=1)
    reg = RectilinearRegularizer()
    res = reg.fit_instance(mask)
    assert res.level == FitLevel.RECTANGLE
    assert res.n_vertices == 4
    assert res.iou_vs_mask >= 0.85
    assert abs(((res.angle_deg - 27) % 90 + 45) % 90 - 45) < 5


def test_noisy_L_shape_becomes_rectilinear_6_vertices():
    L = Polygon([(50, 50), (200, 50), (200, 120), (130, 120), (130, 200), (50, 200)])
    L = affinity.rotate(L, 15, origin="centroid")
    mask = _noisy(_rasterize(L), seed=2)
    reg = RectilinearRegularizer()
    res = reg.fit_instance(mask)
    assert res.level in (FitLevel.RECTILINEAR, FitLevel.RECTANGLE)
    if res.level == FitLevel.RECTILINEAR:
        assert res.n_vertices == 6
        # tous les angles ~90°
        cs = list(res.polygon.exterior.coords)[:-1]
        for i in range(len(cs)):
            a, b, c = cs[i - 1], cs[i], cs[(i + 1) % len(cs)]
            v1 = np.array(b) - np.array(a)
            v2 = np.array(c) - np.array(b)
            cosang = abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            assert cosang < 0.1, f"angle non droit au sommet {i}"
    assert res.iou_vs_mask >= 0.80


def test_row_of_buildings_share_orientation():
    """Trois maisons mitoyennes légèrement bruitées → même orientation."""
    inst = np.zeros((256, 256), np.int32)
    base = Polygon([(0, 0), (55, 0), (55, 70), (0, 70)])
    for i, x in enumerate([30, 90, 150]):
        p = affinity.translate(affinity.rotate(base, 20, origin=(0, 0)), x, 80 + i * 18)
        m = _noisy(_rasterize(p), seed=10 + i)
        inst[(m > 0) & (inst == 0)] = i + 1

    reg = RectilinearRegularizer()
    polys, stats = reg.process_instance_map(inst)
    assert len(polys) == 3
    assert stats.n_rectangle >= 2
    angs = [r.angle_deg % 90 for r in stats.results]
    spread = max(angs) - min(angs)
    spread = min(spread, 90 - spread)
    assert spread < 3.0, f"orientations non alignées: {angs}"


def test_iou_fallback_keeps_original_when_fit_bad():
    """Forme très non rectangulaire → pas de rectangle forcé."""
    tri = Polygon([(30, 30), (220, 40), (60, 210)])
    mask = _rasterize(tri)
    cfg = RectRegularizerConfig(rect_iou_accept=0.95)
    res = RectilinearRegularizer(cfg).fit_instance(mask)
    assert res.polygon is not None
    assert res.iou_vs_mask >= cfg.min_final_iou


def test_adjacent_buildings_no_overlap_single_wall():
    """Deux bâtis mitoyens → zéro chevauchement, mur commun unique."""
    inst = np.zeros((256, 256), np.int32)
    a = affinity.rotate(Polygon([(40, 60), (120, 60), (120, 160), (40, 160)]),
                        18, origin=(128, 128))
    b = affinity.rotate(Polygon([(122, 60), (200, 60), (200, 160), (122, 160)]),
                        18, origin=(128, 128))
    ma = _noisy(_rasterize(a), seed=5)
    mb = _noisy(_rasterize(b), seed=6)
    inst[ma > 0] = 1
    inst[(mb > 0) & (inst == 0)] = 2

    polys, _ = RectilinearRegularizer().process_instance_map(inst)
    assert len(polys) == 2
    inter = polys[1].intersection(polys[2]).area
    assert inter < 0.5, f"chevauchement résiduel: {inter:.2f} px²"
    # mur commun : les frontières doivent se toucher (pas de double trait)
    gap = polys[1].distance(polys[2])
    assert gap < 0.5, f"double frontière, écart {gap:.2f} px"


def test_border_truncated_building_uses_neighbor_orientation():
    """Bâtiment coupé par le bord → orientation de l'îlot, pas du minAreaRect."""
    inst = np.zeros((200, 200), np.int32)
    full = affinity.rotate(Polygon([(60, 60), (130, 60), (130, 130), (60, 130)]),
                           25, origin=(100, 100))
    cut = affinity.translate(full, 90, 0)  # dépasse le bord droit
    inst[_rasterize(full, (200, 200)) > 0] = 1
    m2 = _rasterize(cut, (200, 200))
    inst[(m2 > 0) & (inst == 0)] = 2

    polys, stats = RectilinearRegularizer().process_instance_map(inst)
    assert 2 in polys
    angs = {r.building_id: r.angle_deg % 90 for r in stats.results}
    d = abs(angs[1] - angs[2])
    assert min(d, 90 - d) < 2.0, f"bord mal orienté: {angs}"
    # le polygone du bord ne dépasse pas la tuile
    x0, y0, x1, y1 = polys[2].bounds
    assert x1 <= 199.5 + 1e-6 and y1 <= 199.5 + 1e-6


def test_empty_mask_rejected():
    res = RectilinearRegularizer().fit_instance(np.zeros((64, 64), np.uint8))
    assert res.level == FitLevel.REJECTED
    assert res.polygon is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
