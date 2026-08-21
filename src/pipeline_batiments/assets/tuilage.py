"""
Etape 02 - Decoupage en tuiles chevauchantes.

Calcule uniquement la grille de fenetres. Chaque fenetre est lue a la demande par les etapes
suivantes via core.geo_io.read_window_rgb.
"""
from __future__ import annotations

import os
import yaml
from dagster import asset

from pipeline_batiments.core.tiling import tile_positions


def _load_config() -> dict:
    cfg_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@asset(group_name="pipeline_batiments", description="Grille de tuiles chevauchantes couvrant l'orthophoto.")
def fenetres_tuiles(
    context,
    orthophoto_meta: dict,
) -> list[dict]:
    cfg = _load_config()
    tuilage_cfg = cfg.get("tuilage", {})
    tile_size = int(tuilage_cfg.get("tile_size", 512))
    overlap = int(tuilage_cfg.get("overlap", 64))

    positions = tile_positions(
        height=orthophoto_meta["height"],
        width=orthophoto_meta["width"],
        tile_size=tile_size,
        overlap=overlap,
    )
    windows = [
        {"row": r, "col": c, "height": h, "width": w}
        for (r, c, h, w) in positions
    ]

    context.log.info(
        f"{len(windows)} tuiles ({tile_size}px, overlap {overlap}px) "
        f"pour '{orthophoto_meta['stem']}'"
    )
    context.add_output_metadata({"n_tuiles": len(windows), "tile_size": tile_size, "overlap": overlap})
    return windows
