"""
YOLO11 instance segmentation — bâtiments satellite (production).

Modules:
    config / env     — .env + dataclass
    model            — chargement unique .pt
    preprocess       — RGB uint8, imgsz natif
    postprocess      — fusion masques + morphologie
    service          — pipeline métier
    infer            — façade CLI
"""
from .config import (
    BEST_WEIGHTS,
    YOLO11Config,
    default_config,
    resolve_weights_path,
)

# Les imports lourds (infer, service, train) sont paresseux : ils ne sont
# chargés que si on les demande explicitement. Cela évite que dataset.py
# (qui importe applib) soit exécuté au simple import de pipeline_batiments.yolo11.
def __getattr__(name):
    if name == "YOLO11Segmenter":
        from .infer import YOLO11Segmenter
        return YOLO11Segmenter
    if name in ("YOLO11BuildingService", "get_service"):
        from .service import YOLO11BuildingService, get_service
        return locals()[name]
    if name == "train_yolo11":
        from .train import train_yolo11
        return train_yolo11
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "YOLO11Config",
    "YOLO11BuildingService",
    "YOLO11Segmenter",
    "get_service",
    "default_config",
    "resolve_weights_path",
    "train_yolo11",
    "BEST_WEIGHTS",
]
