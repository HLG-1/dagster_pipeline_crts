"""
Post-traitement unifié → instances bâtiment (tous modèles).

Un seul pipeline, activable via ``use_postprocess`` :

  **Désactivé** — masque sémantique (seuil) uniquement, overlay lisse sans
  traits de séparation artificiels (idéal RSBuilding / SegFormer / UNet).

  **Activé** — nettoyage morphologique + étiquetage instances robuste :
    1. Seuil sur la carte de probabilité
    2. Ouverture / fermeture morphologique + suppression des speckles
    3. Composantes connexes (défaut, fiable — pas de watershed)
    4. Optionnel : watershed guidé par la proba (séparation bâtiments collés)

Modèles natifs instances (YOLO→SAM) : filtre d'aire + comblement de trous
par instance, sans watershed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .geo_io import clean_mask, safe_crs
from .instances import (
    connected_instances,
    instances_to_polygons,
    separate_building_instances,
    separate_instances,
    split_large_instance_map,
)


class SeparationMode(str, Enum):
    NONE = "none"
    CONNECTED = "connected"
    BUILDING = "building"
    WATERSHED = "watershed"


SEPARATION_LABELS: dict[str, str] = {
    SeparationMode.NONE.value: "Aucune (masque binaire seulement)",
    SeparationMode.CONNECTED.value: "Composantes connexes (sans split collés)",
    SeparationMode.BUILDING.value: "Auto bâtiments collés (recommandé)",
    SeparationMode.WATERSHED.value: "Watershed legacy",
}


@dataclass
class PostProcessConfig:
    threshold: float = 0.40
    min_area: int = 50
    clean_morphology: bool = True
    separation: SeparationMode = SeparationMode.BUILDING
    min_distance: int = 0  # 0 = auto
    use_prob_boundary: bool = True
    opening_px: int = 3


@dataclass
class PostProcessResult:
    mask: np.ndarray
    instances: np.ndarray
    n_buildings: int
    steps_applied: list[str] = field(default_factory=list)


def apply_postprocess(
    prob: np.ndarray,
    cfg: PostProcessConfig,
    *,
    enabled: bool = True,
) -> PostProcessResult:
    """Proba (H,W) → masque binaire + carte d'instances (pipeline unifié)."""
    steps: list[str] = [f"seuil ≥ {cfg.threshold:.2f}"]

    binary = (prob > cfg.threshold).astype(np.uint8)
    if not enabled:
        return PostProcessResult(
            mask=binary,
            instances=np.zeros_like(binary, dtype=np.int32),
            n_buildings=0,
            steps_applied=steps + ["post-traitement désactivé (masque sémantique seul)"],
        )

    if cfg.clean_morphology:
        binary = clean_mask(binary, min_area=max(10, cfg.min_area // 2))
        steps.append(f"nettoyage morphologique (min {cfg.min_area} px)")

    if cfg.separation == SeparationMode.NONE:
        inst = np.zeros_like(binary, dtype=np.int32)
        steps.append("pas de séparation d'instances")
    elif cfg.separation == SeparationMode.CONNECTED:
        inst = connected_instances(binary, min_area=cfg.min_area)
        steps.append(f"composantes connexes (min {cfg.min_area} px)")
    elif cfg.separation == SeparationMode.BUILDING:
        md = cfg.min_distance if cfg.min_distance > 0 else None
        prob_guide = prob if cfg.use_prob_boundary else None
        inst = separate_building_instances(
            binary,
            prob=prob_guide,
            min_area=cfg.min_area,
            min_distance=md,
            opening_px=cfg.opening_px,
        )
        steps.append(
            f"séparation auto bâtiments (min {cfg.min_area} px, ouverture {cfg.opening_px})"
        )
    else:
        boundary = prob if cfg.use_prob_boundary else None
        inst = separate_instances(
            binary,
            min_area=cfg.min_area,
            min_distance=cfg.min_distance,
            boundary=boundary,
        )
        steps.append(
            f"watershed (min_distance={cfg.min_distance}, min_area={cfg.min_area} px)"
        )

    return PostProcessResult(
        mask=binary,
        instances=inst,
        n_buildings=int(inst.max()),
        steps_applied=steps,
    )


def refine_native_instances(
    instances: np.ndarray,
    min_area: int,
    prob: np.ndarray | None = None,
) -> np.ndarray:
    """Post-traitement modèles natifs instances (YOLO→SAM, etc.)."""
    if instances is None or instances.max() == 0:
        return instances

    out = np.zeros_like(instances, dtype=np.int32)
    nid = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for lab in range(1, int(instances.max()) + 1):
        comp = (instances == lab).astype(np.uint8)
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k, iterations=1)
        area = int(comp.sum())
        if area >= min_area:
            nid += 1
            out[comp > 0] = nid

    return split_large_instance_map(out, prob=prob, min_area=min_area)


def filter_instances_by_area(
    instances: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """Filtre les instances trop petites (modèles natifs YOLO→SAM3)."""
    out = np.zeros_like(instances, dtype=np.int32)
    nid = 0
    for lab in range(1, int(instances.max()) + 1):
        comp = instances == lab
        if comp.sum() >= min_area:
            nid += 1
            out[comp] = nid
    return out


def instances_to_gdf(instances: np.ndarray, transform, crs):
    return instances_to_polygons(instances, transform, safe_crs(crs))
