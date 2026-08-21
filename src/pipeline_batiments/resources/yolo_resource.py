"""
Resource Dagster pour le detecteur YOLO. Modele charge UNE SEULE FOIS par
run (Dagster instancie la resource une fois et la reutilise pour tous les
assets qui en dependent) - exigence explicite du cahier des charges.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from dagster import ConfigurableResource, get_dagster_logger
from pydantic import PrivateAttr


class YOLOResource(ConfigurableResource):
    """Config (cf config/pipeline.yaml -> yolo.*) :
        weights_path    : chemin vers best.pt
        conf_threshold  : seuil de confiance (defaut 0.20)
        iou_threshold   : seuil NMS interne Ultralytics (defaut 0.45)
    """

    weights_path: str
    conf_threshold: float = 0.20
    iou_threshold: float = 0.45

    _detector: object = PrivateAttr(default=None)

    def setup_for_execution(self, context) -> None:
        """Charge le modele une seule fois, au demarrage du run (pas a
        chaque tuile)."""
        log = get_dagster_logger()
        from pipeline_batiments.yolo11.config import YOLO11Config, is_segmentation_weights
        from pipeline_batiments.yolo11.detect import YOLODetector
        from pipeline_batiments.yolo11.env import resolved_device
        from pipeline_batiments.yolo11.model import YOLO11ModelRegistry

        weights = Path(self.weights_path)

        # Charger le modèle une fois pour détecter la vraie tâche
        # (best.pt peut être un SegmentationModel même si le nom ne contient pas "seg")
        model, _ = YOLO11ModelRegistry.get(weights)
        real_task = getattr(model, "task", None) or "detect"
        is_seg = (real_task == "segment") or is_segmentation_weights(weights)

        cfg = YOLO11Config(
            pretrained_weights=weights,
            best_weights=weights,
            model=self.weights_path,
            use_finetuned=True,          # utilise best_weights en priorité
            detection_only=not is_seg,   # False si le modèle est bien un modèle segment
            infer_imgsz=1024,            # correspond à la taille réelle des tuiles
        )
        cfg.device = resolved_device()
        self._detector = YOLODetector(cfg)
        log.info(
            f"Modele YOLO charge depuis {self.weights_path} "
            f"(device={cfg.device}, task={real_task}, "
            f"detection_only={cfg.detection_only}, infer_imgsz=1024)"
        )

    def detect(self, rgb: np.ndarray) -> dict:
        """Detecte les batiments sur une tuile RGB uint8.

        Renvoie {"boxes_xyxy": [[x1,y1,x2,y2], ...], "instances": <masques natifs>}.
        """
        if self._detector is None:
            raise RuntimeError("YOLOResource non initialisee - setup_for_execution n'a pas ete appele")
        return self._detector.detect_instances(
            rgb, conf=self.conf_threshold, iou=self.iou_threshold
        )
