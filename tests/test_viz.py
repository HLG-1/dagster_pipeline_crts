"""
Tests pour les fonctions de visualisation (overlays transparents avec contours)
et de vectorisation avec régularisation.
"""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Polygon
import geopandas as gpd

from pipeline_batiments.core.viz import make_overlay, compose_layer_view, draw_contours
from pipeline_batiments.core.instances import (
    render_gdf_outlined,
    render_instances_outlined,
    instances_to_polygons,
    separate_building_instances,
)
from pipeline_batiments.core.polygons import stitch_polygons


def test_make_overlay():
    rgb = np.full((100, 100, 3), 100, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    overlay = make_overlay(rgb, mask, color_hex="#FF6600", alpha=0.5)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    # Les pixels hors masque restent inchangés
    assert np.all(overlay[0, 0] == 100)
    # Les pixels sous le masque sont modifiés par le blend
    assert not np.all(overlay[50, 50] == 100)


def test_render_gdf_outlined():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    poly = Polygon([(10, 10), (60, 10), (60, 50), (10, 50)])
    gdf = gpd.GeoDataFrame([{"geometry": poly, "building_id": 1}], crs="EPSG:4326")
    transform = Affine.identity()

    out = render_gdf_outlined(
        rgb,
        gdf,
        transform=transform,
        fill_hex="#22C55E",
        line_hex="#EF4444",
        alpha=0.5,
        line_thickness=2,
    )
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8
    # Intérieur rempli
    assert out[30, 30, 1] > 0  # Green channel active
    # Bordure présente
    assert np.any(out[:, :, 0] > 0)  # Red channel active for outline


def test_render_instances_outlined():
    inst = np.zeros((100, 100), dtype=np.int32)
    inst[10:50, 10:50] = 1
    inst[50:90, 50:90] = 2

    out = render_instances_outlined(inst, fill_hex="#F97316", line_hex="#FFFFFF")
    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8
    assert np.any(out[25, 25] > 0)


def test_instances_to_polygons_with_regularization():
    inst = np.zeros((100, 100), dtype=np.int32)
    inst[20:60, 20:60] = 1
    transform = Affine.identity()

    gdf = instances_to_polygons(inst, transform, "EPSG:4326", regularize=True, method="rectfit")
    assert len(gdf) == 1
    poly = gdf.iloc[0].geometry
    assert isinstance(poly, Polygon)
    # Vérifier que le rectangle régularisé a 4 sommets
    assert len(poly.exterior.coords) - 1 == 4


def test_stitch_polygons():
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2 = Polygon([(2, 0), (12, 0), (12, 10), (2, 10)])
    gdf = gpd.GeoDataFrame([{"geometry": p1}, {"geometry": p2}], crs="EPSG:4326")

    merged = stitch_polygons(gdf, "EPSG:4326", overlap_frac=0.3)
    assert len(merged) == 1
    assert merged.attrs["n_merged"] == 1
