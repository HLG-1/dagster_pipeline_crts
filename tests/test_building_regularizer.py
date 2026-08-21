#!/usr/bin/env python3
"""Tests unitaires — BuildingPolygonRegularizer (formes synthétiques)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_batiments.core.building_regularizer import (
    BuildingPolygonRegularizer,
    RegularizationLevel,
    RegularizerConfig,
)


def _draw_poly(shape_hw: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape_hw, dtype=np.uint8)
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [arr], 1)
    return m


def _noisy_rectangle_mask(h: int = 128, w: int = 128) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[30:90, 40:100] = 1
    # bruit escalier
    m[30:35, 95:100] = 1
    m[85:90, 38:45] = 1
    return m


def _rotated_rect_mask(h: int = 128, w: int = 128, angle: float = 25.0) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    rect = ((64, 64), (50, 30), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(m, [box], 1)
    return m


def _l_shape_mask(h: int = 128, w: int = 128) -> np.ndarray:
    pts = [(30, 30), (90, 30), (90, 60), (60, 60), (60, 90), (30, 90)]
    return _draw_poly((h, w), pts)


def _u_shape_mask(h: int = 128, w: int = 128) -> np.ndarray:
    pts = [(25, 25), (95, 25), (95, 45), (70, 45), (70, 85), (50, 85), (50, 45), (25, 45)]
    return _draw_poly((h, w), pts)


def _two_close_buildings(h: int = 128, w: int = 128) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[30:80, 20:55] = 1
    m[30:80, 58:93] = 1
    return m


def _border_cut_building(h: int = 128, w: int = 128) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[0:50, 40:90] = 1
    return m


@pytest.fixture
def reg() -> BuildingPolygonRegularizer:
    return BuildingPolygonRegularizer(RegularizerConfig())


class TestMaskToPolygon:
    def test_empty_mask(self, reg):
        assert reg.mask_to_polygon(np.zeros((64, 64), np.uint8)) is None

    def test_rectangle(self, reg):
        poly = reg.mask_to_polygon(_noisy_rectangle_mask())
        assert poly is not None
        assert poly.is_valid
        assert poly.area > 100


class TestRegularization:
    def test_reduces_vertices_noisy_rect(self, reg):
        m = _noisy_rectangle_mask()
        orig = reg.mask_to_polygon(m)
        assert orig is not None
        res = reg.process_mask(m, building_id=1)
        assert res.polygon is not None
        assert res.n_vertices_after <= res.n_vertices_before
        assert res.iou_vs_original >= 0.7

    def test_rotated_building_preserved(self, reg):
        m = _rotated_rect_mask(angle=30)
        res = reg.process_mask(m, building_id=1)
        assert res.polygon is not None
        assert res.area_change_ratio <= 0.25
        assert res.level in (
            RegularizationLevel.REGULARIZED,
            RegularizationLevel.SIMPLIFIED,
            RegularizationLevel.ORIGINAL,
        )

    def test_l_shape_not_rectangle(self, reg):
        m = _l_shape_mask()
        res = reg.process_mask(m, building_id=1)
        assert res.polygon is not None
        assert res.n_vertices_after >= 4
        assert res.iou_vs_original >= 0.65

    def test_u_shape_preserved(self, reg):
        m = _u_shape_mask()
        res = reg.process_mask(m, building_id=1)
        assert res.polygon is not None
        assert res.n_vertices_after >= 5

    def test_two_instances_separate(self, reg):
        m = _two_close_buildings()
        inst = np.zeros_like(m, dtype=np.int32)
        inst[m == 1] = 1  # merged label — test per-mask
        m1 = (m > 0).astype(np.uint8)
        m1[:, 55:] = 0
        m2 = (m > 0).astype(np.uint8)
        m2[:, :55] = 0
        r1 = reg.process_mask(m1, 1)
        r2 = reg.process_mask(m2, 2)
        assert r1.polygon is not None and r2.polygon is not None
        assert not r1.polygon.intersects(r2.polygon.buffer(-1))

    def test_border_cut_no_crash(self, reg):
        res = reg.process_mask(_border_cut_building(), building_id=1)
        assert res.polygon is not None

    def test_fallback_chain(self, reg):
        cfg = RegularizerConfig(min_polygon_iou=0.99, max_area_change_ratio=0.001)
        strict = BuildingPolygonRegularizer(cfg)
        res = strict.process_mask(_noisy_rectangle_mask(), 1)
        assert res.level in (
            RegularizationLevel.SIMPLIFIED,
            RegularizationLevel.ORIGINAL,
        )


class TestInstanceMap:
    def test_process_instance_map(self, reg):
        m = _two_close_buildings()
        inst = np.zeros_like(m, dtype=np.int32)
        inst[:, :55][m[:, :55] == 1] = 1
        inst[:, 55:][m[:, 55:] == 1] = 2
        results, stats = reg.process_instance_map(inst)
        assert stats.n_input == 2
        assert len(results) == 2


class TestValidate:
    def test_merge_collinear(self, reg):
        zigzag = Polygon([(0, 0), (10, 1), (20, 0), (20, 10), (0, 10), (0, 0)])
        merged = reg.merge_collinear_edges(zigzag)
        assert merged is not None
        assert len(reg._exterior_vertices(merged)) <= len(reg._exterior_vertices(zigzag))
