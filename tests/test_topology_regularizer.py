#!/usr/bin/env python3
"""Tests — régularisation topologique globale (partition partagée)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_batiments.core.topology_regularizer import TopologicalBuildingRegularizer, TopologyConfig


@pytest.fixture
def topo() -> TopologicalBuildingRegularizer:
    return TopologicalBuildingRegularizer(TopologyConfig(min_component_area=10))


def _two_adjacent_rects(h=80, w=80) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    inst[15:65, 10:38] = 1
    inst[15:65, 38:68] = 2
    return inst


def _row_of_buildings(n=4, h=80, w=120) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    step = w // n
    for i in range(n):
        inst[20:60, i * step + 2:(i + 1) * step - 2] = i + 1
    return inst


def _t_junction(h=80, w=80) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    inst[10:70, 10:35] = 1
    inst[35:55, 35:70] = 2
    return inst


def _l_adjacent(h=80, w=80) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    inst[10:50, 10:40] = 1
    inst[10:30, 40:70] = 2
    inst[30:50, 40:60] = 1
    return inst


def _small_gap(h=80, w=80) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    inst[20:60, 10:38] = 1
    inst[20:60, 42:70] = 2  # gap 4 px
    return inst


def _small_overlap(h=80, w=80) -> np.ndarray:
    """Raster sans overlap ; overlap simulé après vectorisation indépendante."""
    inst = np.zeros((h, w), np.int32)
    inst[20:60, 10:40] = 1
    inst[20:60, 38:70] = 2  # 2 px partagés au raster
    return inst


def _rotated_row(h=100, w=100) -> np.ndarray:
    import cv2
    inst = np.zeros((h, w), np.int32)
    for i, cx in enumerate([25, 50, 75]):
        rect = ((cx, 50), (18, 30), 25)
        box = cv2.boxPoints(rect).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [box], 1)
        inst[mask > 0] = i + 1
    return inst


def _border_cut(h=80, w=80) -> np.ndarray:
    inst = np.zeros((h, w), np.int32)
    inst[0:40, 20:60] = 1
    return inst


def _assert_no_overlaps(polys: dict[int, Polygon], tol: float = 0.5):
    ids = sorted(polys.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            inter = polys[a].intersection(polys[b])
            assert inter.area <= tol, f"overlap {a}/{b} area={inter.area}"


def _assert_valid(polys: dict[int, Polygon]):
    for lab, p in polys.items():
        assert p.is_valid, f"invalid poly {lab}"
        assert p.area > 0


def _assert_ids_preserved(polys: dict[int, Polygon], inst: np.ndarray, min_area=10):
    expected = {lab for lab in range(1, int(inst.max()) + 1) if (inst == lab).sum() >= min_area}
    assert expected.issubset(set(polys.keys())), f"missing ids {expected - set(polys.keys())}"


def _shared_boundary_coords(p1: Polygon, p2: Polygon) -> bool:
    """Les frontières communes ont des coordonnées identiques."""
    shared = p1.boundary.intersection(p2.boundary)
    if shared.is_empty:
        return False
    return shared.length > 0


class TestPartition:
    def test_two_adjacent_no_overlap(self, topo):
        inst = _two_adjacent_rects()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        _assert_valid(res.polygons)
        _assert_ids_preserved(res.polygons, inst)
        assert res.metrics.overlap_area <= 0.5

    def test_shared_boundary_exists(self, topo):
        inst = _two_adjacent_rects()
        _, res = topo.process_instance_map(inst)
        assert _shared_boundary_coords(res.polygons[1], res.polygons[2])

    def test_row_of_buildings(self, topo):
        inst = _row_of_buildings(5)
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        _assert_ids_preserved(res.polygons, inst)

    def test_t_junction(self, topo):
        inst = _t_junction()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        _assert_valid(res.polygons)

    def test_l_adjacent(self, topo):
        inst = _l_adjacent()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)

    def test_small_gap(self, topo):
        inst = _small_gap()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        assert res.metrics.n_instances_out >= 2

    def test_raster_touching_no_overlap(self, topo):
        inst = _small_overlap()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)

    def test_rotated_buildings(self, topo):
        inst = _rotated_row()
        _, res = topo.process_instance_map(inst)
        _assert_no_overlaps(res.polygons)
        _assert_valid(res.polygons)

    def test_border_cut(self, topo):
        inst = _border_cut()
        _, res = topo.process_instance_map(inst)
        assert 1 in res.polygons
        _assert_valid(res.polygons)


class TestSegments:
    def test_single_boundary_between_neighbors(self, topo):
        inst = _two_adjacent_rects()
        segs = topo.extract_boundary_segments(inst)
        shared = [(p0, p1, la, lb) for p0, p1, la, lb in segs if la > 0 and lb > 0]
        assert len(shared) >= 1
        # une frontière verticale commune
        vertical = [s for s in shared if abs(s[0][0] - s[1][0]) < 1e-6]
        assert vertical

    def test_snap_merges_close_vertices(self, topo):
        inst = _two_adjacent_rects()
        raw = topo.extract_boundary_segments(inst)
        snapped, nodes = topo.snap_vertices_global(raw)
        assert len(nodes) <= len(raw) * 2


class TestAssertions:
    def test_zero_overlap_metric(self, topo):
        inst = _row_of_buildings(6)
        _, res = topo.process_instance_map(inst)
        assert res.metrics.overlap_area <= topo.config.overlap_max_area

    def test_geometries_valid(self, topo):
        inst = _row_of_buildings(4)
        gdf, res = topo.process_instance_map(inst)
        for geom in gdf.geometry:
            assert geom.is_valid

    def test_iou_acceptable(self, topo):
        inst = _two_adjacent_rects()
        _, res = topo.process_instance_map(inst)
        assert res.metrics.mean_iou_vs_raster >= 0.75
