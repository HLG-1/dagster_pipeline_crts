"""
Etape 01 - Lecture / validation de l'orthophoto.

N'ouvre pas toute l'image en RAM (contrairement a geo.py::load_geotiff
original) : seules les metadonnees sont lues via
core.geo_io.raster_meta. La lecture des pixels se fait plus tard, tuile
par tuile, dans assets/detection.py et assets/segmentation.py via
core.geo_io.read_window_rgb - ce qui repond a l'exigence du cahier des
charges pour les grandes orthophotos.
"""
from __future__ import annotations

import os

import yaml
from dagster import MetadataValue, asset, Field

from pipeline_batiments.core.geo_io import raster_meta


def _load_config() -> dict:
    cfg_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@asset(
    group_name="pipeline_batiments",
    description="Valide le GeoTIFF et extrait ses metadonnees (CRS, dimensions).",
    config_schema={"input_path": Field(str, is_required=False)}
)
def orthophoto_meta(context) -> dict:
    input_path = context.op_config.get("input_path")
    if not input_path:
        cfg = _load_config()
        input_path = os.environ.get("INPUT_PATH", cfg.get("input", {}).get("path", ""))
    if not input_path:
        raise ValueError("input_path must be set via config, INPUT_PATH env var, or pipeline.yaml")
    meta = raster_meta(input_path)

    resolution = _pixel_resolution(meta["transform"])
    context.log.info(
        f"Ortho '{meta['stem']}' : {meta['width']}x{meta['height']} px, "
        f"{meta['n_bands']} bandes, CRS={meta['crs']}, "
        f"resolution ~{resolution:.3f} u/px"
    )
    context.add_output_metadata({
        "stem": meta["stem"],
        "dimensions": f"{meta['width']}x{meta['height']}",
        "crs": meta["crs"],
        "n_bands": meta["n_bands"],
        "resolution_approx": MetadataValue.float(round(resolution, 4)),
    })
    return meta


def _pixel_resolution(transform) -> float:
    """Resolution sol approximative (taille de pixel), a partir du transform."""
    return abs(transform.a)
