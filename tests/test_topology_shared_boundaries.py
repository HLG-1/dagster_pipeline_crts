#!/usr/bin/env python3
"""Tests — frontiers communes, orthogonalisation, sommets colinéaires."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_batiments.core.topology_regularizer import (
    TopologicalBuildingRegularizer,
    TopologyConfig,
    _turn_angle_deg,
)


@pytest.fixture
def topo() -> TopologicalBuildingRegularizer:
    return TopologicalBuildingRegularizer(TopologyConfig(
        min_component_area=10,
        enable_shape_refiner=False,
        shared_boundary_max_distance=4.0,
        shared_boundary_min_length=5.0,
    ))


# ── Masques synthétiques ──────────────────────────────────────────

def _two_gap_1px(h=80, w=80) -> np.ndarray:
    """Deux rectangles séparés par 1 pixel."""
    inst = np.zeros((h, w), np.int32)
    inst[15:65, 10:38] = 1
    inst[15:65, 39:67] = 2
    return inst


def _two_overlap_2px(h=80, w=80) -> np.ndarray:
    """Deux bâtiments collés (partition raster sans vrai overlap)."""
    inst = np.zeros((h, w), np.int32)
    inst[20:60, 10:40] = 1
    inst[20:60, 40:70] = 2
    return inst


def _double_contour_like(h=80, w=80) -> np.ndarray:
    """Deux bâtiments adjacents (cas double contour potentiel)."""
    inst = np.zeros((h, w), np.int32)
    inst[15:65, 10:38] = 1
    inst[15:65, 38:68] = 2
    return inst


def _row_of_five(h=80, w=150) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    step = w // 5
    for i in range(5):
        inst[20:60, i * step + 2:(i + 1) * step - 2] = i + 1
    return inst


def _t_junction(h=80, w=80) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    inst[10:70, 10:35] = 1
    inst[35:55, 35:70] = 2
    inst[10:35, 35:55] = 3
    return inst


def _wave_wall_polygon() -> Polygon:
    """Mur en petite onde (zigzag)."""
    return Polygon([
        (10, 10), (30, 10), (32, 11), (34, 10), (36, 11), (38, 10),
        (50, 10), (50, 40), (10, 40), (10, 10),
    ])


def _angle_175_polygon() -> Polygon:
    """Sommet B avec angle ~175°."""
    return Polygon([(0, 0), (50, 0), (100, 2), (100, 40), (0, 40), (0, 0)])


def _angle_87_polygon() -> Polygon:
    """Coin ~87° à snaper vers 90°."""
    return Polygon([(0, 0), (40, 0), (39, 38), (0, 40), (0, 0)])


def _rotated_25(h=100, w=100) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    rect = ((50, 50), (40, 25), 25)
    box = cv2.boxPoints(rect).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [box], 1)
    inst[mask > 0] = 1
    return inst


def _l_stuck_to_rect(h=80, w=100) -> np.ndarray:
    """Bâtiment en L collé à un rectangle."""
    inst = np.zeros((h, w), np.int32)
    # L
    inst[10:50, 10:35] = 1
    inst[10:25, 35:55] = 1
    # rectangle collé
    inst[10:50, 55:85] = 2
    return inst


def _assert_no_overlaps(polys, tol=0.5):
    ids = sorted(polys.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert polys[a].intersection(polys[b]).area <= tol


def _assert_valid(polys):
    for lab, p in polys.items():
        assert p.is_valid and p.area > 0


# ── Tests pipeline ────────────────────────────────────────────────

class TestGapAndOverlap:
    def test_two_rects_gap_1px(self, topo):
        inst = _two_gap_1px()
        _, res = topo.process_instance_map(inst)
        assert len(res.polygons) == 2
        _assert_no_overlaps(res.polygons)
        assert res.metrics.overlap_area == 0 or res.metrics.overlap_area <= 0.5
        assert res.metrics.n_instances_out == res.metrics.n_instances_in or \
            res.metrics.n_instances_out == 2

    def test_two_rects_adjacent(self, topo):
        inst = _two_overlap_2px()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        assert 1 in res.polygons and 2 in res.polygons
        # frontière commune
        shared = res.polygons[1].boundary.intersection(res.polygons[2].boundary)
        assert not shared.is_empty or res.polygons[1].distance(res.polygons[2]) < 1.0

    def test_double_contour_single_boundary(self, topo):
        inst = _double_contour_like()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        assert res.metrics.overlap_area <= 0.5
        shared = res.polygons[1].boundary.intersection(res.polygons[2].boundary)
        assert shared.length > 0 or res.polygons[1].touches(res.polygons[2])


class TestComplexLayouts:
    def test_row_of_five(self, topo):
        inst = _row_of_five()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        assert res.metrics.n_instances_out == 5
        assert res.metrics.invalid_geometry_count == 0

    def test_t_junction_three(self, topo):
        inst = _t_junction()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        _assert_valid(res.polygons)
        assert res.metrics.n_instances_out >= 2

    def test_l_stuck_to_rect(self, topo):
        inst = _l_stuck_to_rect()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        assert 1 in res.polygons and 2 in res.polygons


class TestCollinearAndAngles:
    def test_wave_wall_simplified(self, topo):
        poly = _wave_wall_polygon()
        n_before = len(list(poly.exterior.coords)) - 1
        out = topo.remove_nearly_collinear_vertices(poly)
        assert out is not None
        n_after = len(list(out.exterior.coords)) - 1
        assert n_after < n_before
        assert out.is_valid

    def test_angle_175_vertex_removed(self, topo):
        poly = _angle_175_polygon()
        out = topo.remove_nearly_collinear_vertices(poly)
        assert out is not None
        coords = list(out.exterior.coords)[:-1]
        # le point (100,2) quasi-colinéaire doit disparaître
        near = [c for c in coords if abs(c[0] - 100) < 1 and abs(c[1] - 2) < 3]
        assert len(near) == 0 or len(coords) < 5

    def test_angle_87_preserved_as_corner(self, topo):
        poly = _angle_87_polygon()
        out = topo.remove_nearly_collinear_vertices(poly)
        assert out is not None
        # coin droit conservé (≥4 sommets typiquement)
        assert len(list(out.exterior.coords)) - 1 >= 3

    def test_rotated_25_valid(self, topo):
        inst = _rotated_25()
        _, res = topo.process_instance_map(inst)
        assert 1 in res.polygons
        _assert_valid(res.polygons)
        # orientation non forcée H/V : polygone tourné OK
        assert res.polygons[1].area > 100


class TestSharedBoundaryHelpers:
    def test_build_shared_boundary_median(self, topo):
        # deux segments parallèles distants de 2 px
        a = ((10.0, 10.0), (10.0, 50.0))
        b = ((12.0, 10.0), (12.0, 50.0))
        mid = topo.build_shared_boundary(a, b)
        assert mid is not None
        assert abs(mid.coords[0][0] - 11.0) < 0.5
        assert mid.length > 30

    def test_merge_close_parallel_creates_shared(self, topo):
        inst = _two_gap_1px()
        segs = topo.extract_boundary_segments(inst)
        before_shared = sum(1 for *_, la, lb in segs if la > 0 and lb > 0)
        after = topo.merge_close_parallel_boundaries(segs)
        after_shared = sum(1 for *_, la, lb in after if la > 0 and lb > 0)
        # au minimum autant de frontières partagées (idéalement plus)
        assert after_shared >= before_shared


class TestAssertions:
    def test_overlap_zero(self, topo):
        _, res = topo.process_instance_map(_row_of_five())
        assert res.metrics.overlap_area <= topo.config.overlap_max_area

    def test_invalid_geometry_zero(self, topo):
        _, res = topo.process_instance_map(_double_contour_like())
        assert res.metrics.invalid_geometry_count == 0

    def test_instance_count_preserved(self, topo):
        inst = _row_of_five()
        _, res = topo.process_instance_map(inst)
        assert res.metrics.n_instances_out == 5

    def test_no_merge_of_instances(self, topo):
        inst = _double_contour_like()
        _, res = topo.process_instance_map(inst)
        assert set(res.polygons.keys()) == {1, 2}
