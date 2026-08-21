#!/usr/bin/env python3
"""Tests — classification de forme et rectangles orientés."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_batiments.core.shape_refiner import (
    BuildingShapeRefiner,
    ShapeClass,
    ShapeRefinerConfig,
)


@pytest.fixture
def refiner():
    return BuildingShapeRefiner(ShapeRefinerConfig())


def _rect_mask(h=80, w=80, angle=0, cx=40, cy=40, rw=30, rh=20):
    m = np.zeros((h, w), np.uint8)
    rect = ((cx, cy), (rw, rh), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(m, [box], 1)
    return m


def _noisy_rect_mask(h=80, w=80):
    m = np.zeros((h, w), np.uint8)
    m[20:60, 25:65] = 1
    m[20:28, 60:68] = 1  # petit décrochement
    m[55:60, 22:28] = 1
    return m


def _l_mask(h=80, w=80):
    m = np.zeros((h, w), np.uint8)
    m[15:55, 15:45] = 1
    m[15:35, 45:70] = 1
    return m


def _mask_to_poly(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
    pts.append(pts[0])
    return Polygon(pts)


class TestRectangle:
    def test_classify_rotated_rect(self, refiner):
        m = _rect_mask(angle=30)
        p = _mask_to_poly(m)
        cls, _ = refiner.classify_building_shape(p, m)
        assert cls == ShapeClass.RECTANGLE

    def test_fit_4_vertices(self, refiner):
        m = _noisy_rect_mask()
        p = _mask_to_poly(m)
        res = refiner.refine_building(1, p, m)
        assert res.polygon is not None
        assert res.n_vertices_after == 4
        assert res.shape_class == ShapeClass.RECTANGLE

    def test_noisy_bump_removed(self, refiner):
        m = _noisy_rect_mask()
        p = _mask_to_poly(m)
        assert len(list(p.exterior.coords)) - 1 > 4
        res = refiner.refine_building(1, p, m)
        assert res.n_vertices_after == 4
        assert res.metrics.mask_iou_final >= 0.75


class TestLShape:
    def test_l_not_forced_to_4_if_significant(self, refiner):
        m = _l_mask()
        p = _mask_to_poly(m)
        res = refiner.refine_building(1, p, m)
        assert res.polygon is not None
        # L peut rester complexe OU rectangle si IoU rect suffisant
        if res.shape_class == ShapeClass.RECTANGLE:
            assert res.metrics.mask_iou_final >= 0.7
        else:
            assert res.n_vertices_after >= 4


class TestParsimony:
    def test_refine_all_pct_quad(self, refiner):
        inst = np.zeros((80, 80), np.int32)
        for i, (cx, cy) in enumerate([(20, 20), (60, 20), (20, 60), (60, 60)]):
            m = _rect_mask(cx=cx, cy=cy, rw=18, rh=14)
            inst[m > 0] = i + 1
        polys = {}
        for lab in range(1, 5):
            polys[lab] = _mask_to_poly((inst == lab).astype(np.uint8))
        out, stats = refiner.refine_all(polys, inst)
        assert stats.pct_4_vertices >= 75.0
