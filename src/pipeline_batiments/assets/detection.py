"""
Etape 03 - Detection YOLO (boites), par tuile.

Lit chaque fenetre directement depuis le GeoTIFF (pas de pre-chargement
complet en RAM, cf ingestion.py) et appelle la resource YOLO (modele
charge une seule fois pour tout le run).

Mise a jour 2024-08 : Utilise detect_instances() au lieu de detect() pour
avoir les masques natifs YOLO disponibles en fallback pour SAM3.
"""
from __future__ import annotations

from rasterio.windows import Window
from dagster import asset

from pipeline_batiments.core.geo_io import read_window_rgb
from pipeline_batiments.resources.yolo_resource import YOLOResource


@asset(group_name="pipeline_batiments", description="Boites YOLO par tuile.")
def boites_yolo(
    context,
    orthophoto_meta: dict,
    fenetres_tuiles: list[dict],
    yolo: YOLOResource,
) -> list[dict]:
    path = orthophoto_meta["path"]
    results = []
    total_boxes = 0

    for i, tw in enumerate(fenetres_tuiles):
        window = Window(tw["col"], tw["row"], tw["width"], tw["height"])
        rgb = read_window_rgb(path, window)
        # IMPORTANT: utiliser detect_instances pour avoir les masques natifs YOLO disponibles en fallback
        out = yolo.detect_instances(rgb)
        boxes = out.get("boxes_xyxy", [])
        instances = out.get("instances")
        # Ne stocker le masque natif que s'il contient des instances réelles pour économiser la mémoire
        if instances is not None and int(instances.max()) == 0:
            instances = None
        total_boxes += len(boxes)
        results.append({**tw, "boxes_xyxy": boxes, "instances": instances})
        if len(boxes) > 0 or instances is not None:
            context.log.info(f"Tuile {i + 1}/{len(fenetres_tuiles)} : {len(boxes)} boites, {int(instances.max()) if instances is not None else 0} instances")

    context.add_output_metadata({
        "n_tuiles": len(fenetres_tuiles),
        "n_boites_total": total_boxes,
    })
    return results
