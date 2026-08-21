"""
Resource Dagster pour parler au service SAM3 (process externe long-vivant,
GPU dedie, cf services/sam3_service/). Adapte du client HTTP
applib/core/instance_pipeline.py :: sam3_instances, sam3_health du
collegue, avec ajout du health-check obligatoire et des retries/backoff
exiges par le cahier des charges (etape 04).
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from dagster import ConfigurableResource, get_dagster_logger
from PIL import Image
from pydantic import Field


class SAM3ServiceError(Exception):
    """Le service SAM3 est injoignable ou a renvoye une erreur, apres retries."""


class SAM3Resource(ConfigurableResource):
    """Client HTTP du micro-service SAM3 (endpoint /predict_instances).

    Config (cf config/pipeline.yaml -> sam3.*) :
        url          : ex "http://127.0.0.1:8077"
        timeout_s    : timeout par requete
        max_retries  : nombre de tentatives avant echec
        backoff_s    : delai initial entre tentatives (doublé a chaque essai)
    """

    url: str = Field(description="URL du service SAM3, ex http://127.0.0.1:8077")
    timeout_s: int = 30
    max_retries: int = 3
    backoff_s: int = 2

    def health_check(self) -> bool:
        """GET /health. A appeler obligatoirement avant tout batch (exigence
        du cahier des charges : 'health-check obligatoire avant le batch').
        """
        log = get_dagster_logger()
        try:
            with urllib.request.urlopen(f"{self.url.rstrip('/')}/health", timeout=5) as r:
                out = json.loads(r.read())
                ready = bool(out.get("ready", False))
                if not ready:
                    log.warning(f"Service SAM3 repond mais n'est pas pret : {out}")
                return ready
        except Exception as e:
            log.error(f"Service SAM3 injoignable sur {self.url} : {e}")
            return False

    def predict_instances(
        self,
        rgb: np.ndarray,
        boxes_xyxy: list[list[float]] | None = None,
        points_xy: list[list[float]] | None = None,
        prompt: str = "building",
        conf: float = 0.0,
        sep: str = "per_box",
    ) -> tuple[np.ndarray, list[dict]]:
        """POST /predict_instances -> (carte d'instances int32 HxW, meta par batiment).

        Retries avec backoff exponentiel si le service est temporairement
        indisponible (GPU sature, redemarrage en cours) - cf critere
        d'acceptation : 'le pipeline survit a un redemarrage du service SAM3'.
        """
        log = get_dagster_logger()
        payload = self._build_payload(rgb, prompt, boxes_xyxy, points_xy, conf, sep)

        delay = self.backoff_s
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call(payload)
            except (urllib.error.URLError, OSError, TimeoutError, RuntimeError) as e:
                last_err = e
                log.warning(
                    f"SAM3 /predict_instances echec (tentative {attempt}/{self.max_retries}) : {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2

        raise SAM3ServiceError(
            f"Service SAM3 injoignable apres {self.max_retries} tentatives : {last_err}"
        )

    @staticmethod
    def _build_payload(rgb, prompt, boxes_xyxy, points_xy, conf, sep) -> dict:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        payload = {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "prompt": prompt,
            "conf": float(conf),
            "sep": sep,
        }
        if boxes_xyxy:
            payload["boxes_xyxy"] = [[float(v) for v in b] for b in boxes_xyxy]
        if points_xy:
            payload["points_xy"] = [
                [float(p[0]), float(p[1])] if p is not None else None for p in points_xy
            ]
        return payload

    def _call(self, payload: dict) -> tuple[np.ndarray, list[dict]]:
        req = urllib.request.Request(
            f"{self.url.rstrip('/')}/predict_instances",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            res = json.loads(r.read())

        if "label_b64" not in res:
            raise RuntimeError(f"Reponse SAM3 invalide : {res.get('error', res)}")

        h, w = res["shape"]
        label = (
            np.frombuffer(base64.b64decode(res["label_b64"]), dtype="<i4")
            .reshape(h, w)
            .astype(np.int32)
        )
        return label, res.get("instances", [])
