"""
Etape 06 - Export masque + livrables SIG.

Ecrit GeoJSON + GPKG + Shapefile (core.geo_io.export_all), un masque PNG
binaire, un overlay PNG transparent haute définition avec contours lissés
(approche render_gdf_outlined de finetunig_portable), et un fichier
`<stem>__<timestamp>_run.json` de métadonnées (traçabilité : params, n_batiments,
latence, versions).
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from dagster import asset
from PIL import Image

from pipeline_batiments.core.geo_io import _to_rgb_uint8, export_all
from pipeline_batiments.core.instances import render_gdf_outlined


@asset(group_name="pipeline_batiments", description="Export GeoJSON/GPKG/Shapefile + masque PNG + overlay transparent + métadonnées de run.")
def export_sig(
    context,
    orthophoto_meta: dict,
    polygones_fusionnes: gpd.GeoDataFrame,
) -> dict:
    t0 = time.perf_counter()
    stem = orthophoto_meta["stem"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_stem = f"{stem}__{timestamp}"

    out_dir = Path("results") / run_stem
    paths = export_all(polygones_fusionnes, out_dir, stem=stem)

    mask_path = out_dir / f"{stem}_mask.png"
    _write_mask_png(polygones_fusionnes, orthophoto_meta, mask_path)
    paths["mask_png"] = mask_path

    overlay_path = out_dir / f"{stem}_overlay.png"
    _write_overlay_png(polygones_fusionnes, orthophoto_meta, overlay_path)
    paths["overlay_png"] = overlay_path

    run_meta = {
        "ortho": orthophoto_meta["path"],
        "stem": stem,
        "crs": orthophoto_meta["crs"],
        "n_buildings": len(polygones_fusionnes),
        "generated_at": timestamp,
        "latency_ms_export": round((time.perf_counter() - t0) * 1000, 1),
        "python_version": platform.python_version(),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    run_json_path = out_dir / f"{run_stem}_run.json"
    run_json_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))

    context.log.info(f"Export termine : {len(polygones_fusionnes)} batiments -> {out_dir}")
    context.add_output_metadata({
        "n_buildings": len(polygones_fusionnes),
        "output_dir": str(out_dir),
        "geojson": str(paths["geojson"]),
        "gpkg": str(paths["gpkg"]),
        "mask_png": str(mask_path),
        "overlay_png": str(overlay_path),
    })
    return run_meta


def _write_mask_png(gdf: gpd.GeoDataFrame, orthophoto_meta: dict, out_path: Path) -> None:
    """Aperçu binaire (bâti/fond) rasterisé à la résolution de l'ortho d'origine."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = orthophoto_meta["height"], orthophoto_meta["width"]

    if len(gdf) == 0:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        shapes = [(geom, 255) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        mask = rasterio.features.rasterize(
            shapes,
            out_shape=(h, w),
            transform=orthophoto_meta["transform"],
            fill=0,
            dtype="uint8",
        )

    Image.fromarray(mask, mode="L").save(out_path)


def _write_overlay_png(
    gdf: gpd.GeoDataFrame,
    orthophoto_meta: dict,
    out_path: Path,
    fill_hex: str = "#F97316",
    line_hex: str = "#EF4444",
    alpha: float = 0.48,
    line_thickness: int = 2,
) -> None:
    """Overlay PNG haute qualité : orthophoto originale + masques transparents avec contours anti-crénelés (méthode finetunig_portable)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ortho_path = orthophoto_meta["path"]
    try:
        with rasterio.open(ortho_path) as src:
            data = src.read()
        rgb, _ = _to_rgb_uint8(data)
    except Exception:
        rgb = np.asarray(Image.open(ortho_path).convert("RGB"))

    transform = orthophoto_meta["transform"]

    if gdf is not None and len(gdf) > 0:
        overlay_rgb = render_gdf_outlined(
            rgb,
            gdf,
            transform=transform,
            fill_hex=fill_hex,
            line_hex=line_hex,
            alpha=alpha,
            line_thickness=line_thickness,
        )
    else:
        overlay_rgb = rgb

    Image.fromarray(overlay_rgb).save(out_path)
