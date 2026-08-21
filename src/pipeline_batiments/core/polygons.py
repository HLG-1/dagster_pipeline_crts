"""
Conversion masques d'instances -> polygones SIG, et fusion des polygones
chevauchants entre tuiles (bâtiments coupés par les bords).

Fonctions utilisées par assets/segmentation.py (étape 04) et assets/fusion.py
(étape 05).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .geo_io import safe_crs
from .instances import instances_to_polygons, instances_to_polygons_rectfit


def stitch_polygons(gdf: gpd.GeoDataFrame, crs_str: str, overlap_frac: float = 0.3) -> gpd.GeoDataFrame:
    """Fusionne les polygones qui se recouvrent fortement (dédoublonnage + coutures
    entre tuiles). Utilise un union-find sur un index spatial (STRtree) :
    deux polygones sont fusionnés si leur intersection couvre au moins
    `overlap_frac` de l'aire du plus petit des deux.

    Critère d'acceptation : un bâtiment à cheval sur 2 tuiles -> 1 seul
    polygone final. Ne plante pas si gdf est vide.
    """
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(columns=["geometry", "building_id", "area_m2"], crs=safe_crs(crs_str))

    geoms = list(gdf.geometry)
    n = len(geoms)
    if n <= 1:
        out = gdf.copy().reset_index(drop=True)
        out.attrs["n_merged"] = 0
        return out

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    tree = STRtree(geoms)
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        for j in tree.query(g):
            j = int(j)
            if j <= i:
                continue
            gj = geoms[j]
            if gj is None or gj.is_empty:
                continue
            inter = g.intersection(gj).area
            if inter <= 0:
                continue
            if inter / max(min(g.area, gj.area), 1e-8) >= overlap_frac:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    rows = []
    n_merged = 0
    for members in groups.values():
        valid_members = [geoms[i] for i in members if geoms[i] is not None and not geoms[i].is_empty]
        if not valid_members:
            continue
        if len(valid_members) > 1:
            n_merged += len(valid_members) - 1
        merged = unary_union(valid_members)
        if merged.is_empty:
            continue
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda p: p.area)
        rows.append({"geometry": merged, "area_m2": round(merged.area, 2)})

    crs = safe_crs(crs_str)
    out = gpd.GeoDataFrame(rows, crs=crs)
    out.attrs["n_merged"] = n_merged
    return out.reset_index(drop=True)
