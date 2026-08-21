"""
Sensor Dagster : surveille un dossier (config/pipeline.yaml -> sensor.watch_dir)
et declenche automatiquement pipeline_complet_job des qu'une nouvelle
orthophoto .tif/.tiff y apparait.

Realise le cas d'usage du cahier des charges : "je depose une orthophoto,
le systeme... me rend le masque + les polygones, sans que j'ouvre un
notebook."

Mecanique :
    - le dagster-daemon appelle `watch_new_orthophotos` toutes les
      `poll_interval_s` secondes (minimum_interval_seconds) ;
    - a chaque appel, on liste les fichiers du dossier surveille et on les
      compare au cursor (ensemble des fichiers deja traites, stocke en
      JSON dans context.cursor - c'est la seule "memoire" persistee entre
      deux appels) ;
    - pour chaque fichier nouveau, on emet un RunRequest qui lance
      pipeline_complet_job avec input_path pointant vers ce fichier.
"""
from __future__ import annotations

import json
from pathlib import Path

from dagster import (
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    sensor,
)

from pipeline_batiments.jobs import pipeline_complet_job

_VALID_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _load_watch_dir() -> Path:
    """Lit sensor.watch_dir depuis config/pipeline.yaml (repli : data/incoming)."""
    import os

    import yaml

    cfg_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    return Path(cfg.get("sensor", {}).get("watch_dir", "data/incoming"))


@sensor(
    job=pipeline_complet_job,
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.STOPPED,  # activation manuelle depuis l'UI, cf etape 5
    description="Declenche pipeline_complet_job pour chaque nouvelle orthophoto deposee dans le dossier surveille.",
)
def watch_new_orthophotos(context: SensorEvaluationContext):
    watch_dir = _load_watch_dir()
    watch_dir.mkdir(parents=True, exist_ok=True)

    already_seen: set[str] = set(json.loads(context.cursor)) if context.cursor else set()

    candidates = sorted(
        p for p in watch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VALID_EXT
    )
    new_files = [p for p in candidates if p.name not in already_seen]

    if not new_files:
        return SkipReason(f"Aucune nouvelle orthophoto dans {watch_dir}")

    run_requests = []
    for path in new_files:
        run_requests.append(
            RunRequest(
                run_key=path.name,  # deduplication cote Dagster (idempotence)
                run_config={
                    "ops": {
                        "orthophoto_meta": {
                            "config": {"input_path": str(path)}
                        }
                    }
                },
                tags={"source": "sensor", "ortho_file": path.name},
            )
        )
        already_seen.add(path.name)

    context.update_cursor(json.dumps(sorted(already_seen)))
    return run_requests
