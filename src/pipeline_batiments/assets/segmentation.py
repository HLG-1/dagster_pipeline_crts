"""
Etape 04 - Segmentation SAM 3 (masques) ou repli YOLO segment, par tuile.

Pour chaque tuile : point interieur par boite (multiprompt), appel SAM3
(health-check + retries geres par SAM3Resource), puis vectorisation
immediate en polygones (coords monde, via le transform de la fenetre).
La fusion inter-tuiles (etape 05) travaille sur ces polygones deja
georeferences.
"""
from __future__ import annotations

import os
import yaml
import geopandas as gpd
import pandas as pd
from dagster import asset
from rasterio.windows import Window

from pipeline_batiments.core.geo_io import read_window_rgb, safe_crs, window_transform
from pipeline_batiments.core.multiprompt import points_for_boxes
from pipeline_batiments.core.polygons import instances_to_polygons
from pipeline_batiments.resources.sam3_resource import SAM3Resource


def _load_config() -> dict:
    cfg_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@asset(group_name="pipeline_batiments", description="Polygones bâtiments par tuile (vectorisation SIG).")
def polygones_par_tuile(
    context,
    orthophoto_meta: dict,
    boites_yolo: list[dict],
    sam3: SAM3Resource,
) -> gpd.GeoDataFrame:
    cfg = _load_config()
    reg_cfg = cfg.get("regularization", {})
    regularize = bool(reg_cfg.get("enabled", False))
    reg_method = str(reg_cfg.get("method", "rectfit"))
    simplify_tol = float(reg_cfg.get("simplify_tolerance", 0.5))
    min_area_px = int(reg_cfg.get("min_area_px", 50))

    sam3_ok = sam3.health_check()
    if not sam3_ok:
        context.log.warning(
            f"Service SAM3 indisponible sur {sam3.url} — utilisation des masques natifs YOLO en repli."
        )

    path = orthophoto_meta["path"]
    crs_str = safe_crs(orthophoto_meta["crs"])
    parts: list[gpd.GeoDataFrame] = []

    for i, tile in enumerate(boites_yolo):
        boxes = tile.get("boxes_xyxy", [])
        native_inst = tile.get("instances")

        if not boxes and (native_inst is None or int(native_inst.max()) == 0):
            continue

        window = Window(tile["col"], tile["row"], tile["width"], tile["height"])
        label = None

        if sam3_ok and boxes:
            rgb = read_window_rgb(path, window)
            points = points_for_boxes(rgb, boxes)
            try:
                label, _meta = sam3.predict_instances(
                    rgb, boxes_xyxy=boxes, points_xy=points, sep="per_box",
                )
            except Exception as exc:
                context.log.warning(f"Tuile {i + 1} : echec SAM3 ({exc}), repli sur YOLO segment")
                label = None

        if (label is None or int(label.max()) == 0) and native_inst is not None and int(native_inst.max()) > 0:
            label = native_inst

        if label is None or int(label.max()) == 0:
            continue

        tile_transform = window_transform(path, window)
        gdf = instances_to_polygons(
            label,
            tile_transform,
            crs_str,
            simplify_tol=simplify_tol,
            min_area_px=min_area_px,
            regularize=regularize,
            method=reg_method,
        )
        if len(gdf):
            parts.append(gdf)

        context.log.info(f"Tuile {i + 1}/{len(boites_yolo)} : {len(gdf)} polygones")

    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "building_id", "area_m2", "area_px"], crs=crs_str)

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs_str)
    context.add_output_metadata({"n_polygones_avant_fusion": len(merged)})
    return merged
