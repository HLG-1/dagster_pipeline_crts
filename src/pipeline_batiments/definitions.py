"""
Point d'entree Dagster : assemble les 6 assets (etapes 01 a 06), les
resources (SAM3, YOLO), et le job du pipeline complet.

Lancement :
    dagster dev -f src/pipeline_batiments/definitions.py

Puis dans l'UI : materialiser `pipeline_complet_job` avec la config
(input_path, weights_path, sam3 url...) - cf config/pipeline.yaml pour les
valeurs par defaut.
"""
from __future__ import annotations

import os

import yaml
from dagster import Definitions

from pipeline_batiments.assets import detection, export, fusion, ingestion, segmentation, tuilage  # noqa: F401 (charge les @asset)
from pipeline_batiments.jobs import _all_assets as all_assets
from pipeline_batiments.jobs import pipeline_complet_job
from pipeline_batiments.resources.sam3_resource import SAM3Resource
from pipeline_batiments.resources.yolo_resource import YOLOResource
from pipeline_batiments.sensors import watch_new_orthophotos


def _load_default_config() -> dict:
    cfg_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


_cfg = _load_default_config()

defs = Definitions(
    assets=all_assets,
    jobs=[pipeline_complet_job],
    sensors=[watch_new_orthophotos],
    resources={
        "sam3": SAM3Resource(
            url=os.environ.get("SAM3_URL", _cfg.get("sam3", {}).get("url", "http://127.0.0.1:8077")),
            timeout_s=_cfg.get("sam3", {}).get("timeout_s", 30),
            max_retries=_cfg.get("sam3", {}).get("max_retries", 3),
            backoff_s=_cfg.get("sam3", {}).get("backoff_s", 2),
        ),
        "yolo": YOLOResource(
            weights_path=os.environ.get(
                "YOLO_WEIGHTS_PATH", _cfg.get("yolo", {}).get("weights_path", "checkpoints/yolo11/best.pt")
            ),
            conf_threshold=_cfg.get("yolo", {}).get("conf_threshold", 0.15),
        ),
    },
)
