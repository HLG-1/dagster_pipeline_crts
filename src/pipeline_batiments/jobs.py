"""
Jobs Dagster : selection d'assets a materialiser ensemble.

pipeline_complet_job regroupe les 6 assets (etapes 01 a 06) en un seul job
executable depuis l'UI (Launchpad) ou declenche automatiquement par un
sensor (cf sensors.py::watch_new_orthophotos).

Ce fichier est intentionnellement independant de definitions.py (qui
l'importe, ainsi que sensors.py) pour eviter tout import circulaire :
definitions.py -> jobs.py
definitions.py -> sensors.py -> jobs.py
"""
from __future__ import annotations

from dagster import define_asset_job, load_assets_from_modules

from pipeline_batiments.assets import detection, export, fusion, ingestion, segmentation, tuilage

_all_assets = load_assets_from_modules([ingestion, tuilage, detection, segmentation, fusion, export])

pipeline_complet_job = define_asset_job(
    name="pipeline_complet",
    selection=_all_assets,
    description="Orthophoto -> lecture -> tuilage -> YOLO -> SAM3 -> fusion -> export SIG",
)

