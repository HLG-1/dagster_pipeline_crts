"""
Etape 05 - Fusion des polygones inter-tuiles.

Utilise core.polygons.stitch_polygons : un batiment a cheval sur 2 tuiles
(dans la zone d'overlap) devient 1 seul polygone final. Garantit des
building_id finaux coherents (1..N sans trous).
"""
from __future__ import annotations

import os
import yaml
import geopandas as gpd
from dagster import asset

from pipeline_batiments.core.geo_io import safe_crs
from pipeline_batiments.core.polygons import stitch_polygons


def _load_config() -> dict:
    cfg_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@asset(group_name="pipeline_batiments", description="Polygones bâtiments après fusion des chevauchements inter-tuiles.")
def polygones_fusionnes(
    context,
    orthophoto_meta: dict,
    polygones_par_tuile: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    crs_str = safe_crs(orthophoto_meta["crs"])

    if len(polygones_par_tuile) == 0:
        context.log.warning("Aucun polygone en entree - export d'un GeoDataFrame vide (pas de crash).")
        empty = gpd.GeoDataFrame(columns=["geometry", "building_id", "area_m2"], crs=crs_str)
        context.add_output_metadata({"n_batiments": 0, "n_fusions": 0})
        return empty

    cfg = _load_config()
    overlap_frac = float(cfg.get("fusion", {}).get("iou_merge_threshold", 0.5))

    merged = stitch_polygons(polygones_par_tuile, crs_str, overlap_frac=overlap_frac)
    merged = merged.reset_index(drop=True)
    merged["building_id"] = range(1, len(merged) + 1)

    n_merged = merged.attrs.get("n_merged", 0)
    context.log.info(f"{len(merged)} batiments apres fusion ({n_merged} paires fusionnees)")
    context.add_output_metadata({"n_batiments": len(merged), "n_fusions": n_merged})
    return merged
