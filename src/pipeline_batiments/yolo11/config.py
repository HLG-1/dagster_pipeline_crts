"""
Configuration YOLO — test COCO sans fine-tuning (YOLO11-l/x par défaut).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .env import env_bool, env_float, env_int, env_str, load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
IMAGES_SRC = ROOT / "data" / "images"
MASKS_SRC = ROOT / "data" / "mask"
YOLO_DATA_ROOT = ROOT / "data" / "yolo11"
DATASET_YAML = ROOT / "dataset.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints" / "yolo11"
BEST_WEIGHTS = CHECKPOINT_DIR / "best.pt"
RUNS_DIR = ROOT / "runs" / "segment"
PRETRAINED_DIR = ROOT / "checkpoints" / "pretrained"

# Poids locaux optionnels (pas le défaut « sanity check COCO »)
LOCAL_BUILDING_V8 = PRETRAINED_DIR / "yolov8n_building.pt"

# Variantes Ultralytics recommandées (sans fine-tune)
YOLO11_VARIANTS: dict[str, str] = {
    "building": str(PRETRAINED_DIR / "yolov8n_building.pt"),  # défaut hybride YOLO→SAM
    "l-seg": "yolo11l-seg.pt",
    "x-seg": "yolo11x-seg.pt",
    "l": "yolo11l.pt",
    "x": "yolo11x.pt",
}
DEFAULT_VARIANT = "building"
DEFAULT_COCO_MODEL = YOLO11_VARIANTS["l-seg"]

# Classes COCO proches « structure » (optionnel, désactivé par défaut)
COCO_BUILDING_LIKE = frozenset({0, 56, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72})


def variant_to_model(variant: str) -> str:
    v = variant.strip().lower()
    return YOLO11_VARIANTS.get(v, variant)


def is_segmentation_weights(path: Path | str) -> bool:
    n = Path(path).name.lower()
    return "seg" in n or n.endswith("-seg.pt") or "building" in n


@dataclass
class YOLO11Config:
    """YOLO11 / Ultralytics — test générique COCO (pas de fine-tune par défaut)."""

    pretrained_weights: Path = field(default_factory=lambda: LOCAL_BUILDING_V8)
    best_weights: Path = field(default_factory=lambda: BEST_WEIGHTS)
    use_finetuned: bool = False
    use_local_building_v8: bool = False

    model: str = DEFAULT_COCO_MODEL
    variant: str = DEFAULT_VARIANT
    task: str = "segment"
    data_yaml: Path = field(default_factory=lambda: DATASET_YAML)
    imgsz: int = 1024
    epochs: int = 100
    batch: int | str = "auto"
    device: str = "auto"
    workers: int = 8
    patience: int = 30
    amp: bool = True
    fraction: float = 1.0
    val: bool = True
    plots: bool = True
    save: bool = True
    project: str = "runs/segment"
    name: str = "building_yolo11"

    train_ratio: float = 0.75
    val_ratio: float = 0.25
    seed: int = 42

    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    fliplr: float = 0.5
    flipud: float = 0.5
    degrees: float = 15.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    perspective: float = 0.0005
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    min_polygon_area_px: int = 25
    class_id: int = 0
    class_name: str = "building"

    conf: float = 0.15
    iou: float = 0.45
    infer_imgsz: int = 1024   # CORRIGÉ : correspond à la taille réelle des tuiles (1024px)
    tile_size: int = 1024     # CORRIGÉ : correspond à assets/tuilage.py tile_size=1024
    mask_threshold: float = 0.25
    min_mask_area_px: int = 0
    filter_coco_building_like: bool = False
    detection_only: bool = False  # CORRIGÉ : False par défaut (modèle segment)

    @classmethod
    def from_env(cls) -> YOLO11Config:
        variant = env_str("YOLO11_VARIANT", DEFAULT_VARIANT)
        model_name = env_str("YOLO11_MODEL", "") or variant_to_model(variant)

        weights = env_str("YOLO11_WEIGHTS", "")
        if weights:
            pretrained = Path(weights)
        elif env_bool("YOLO11_USE_LOCAL_BUILDING", False) and LOCAL_BUILDING_V8.is_file():
            pretrained = LOCAL_BUILDING_V8
            model_name = LOCAL_BUILDING_V8.name
        else:
            pretrained = Path(model_name)

        if not pretrained.is_absolute() and not pretrained.is_file():
            for cand in (ROOT / pretrained, PRETRAINED_DIR / pretrained.name):
                if cand.is_file():
                    pretrained = cand
                    break

        task = "segment" if is_segmentation_weights(pretrained) or is_segmentation_weights(model_name) else "detect"

        return cls(
            pretrained_weights=pretrained,
            best_weights=Path(env_str("YOLO11_BEST_WEIGHTS", str(BEST_WEIGHTS))),
            use_finetuned=env_bool("YOLO11_USE_FINETUNED", False),
            use_local_building_v8=env_bool("YOLO11_USE_LOCAL_BUILDING", False),
            model=model_name,
            variant=variant,
            task=task,
            device=env_str("YOLO11_DEVICE", "auto"),
            conf=env_float("YOLO11_CONF", 0.20),
            iou=env_float("YOLO11_IOU", 0.45),
            infer_imgsz=env_int("YOLO11_IMGSZ", 1024),       # CORRIGÉ : 1024 par défaut
            tile_size=env_int("YOLO11_TILE_SIZE", 1024),     # CORRIGÉ : 1024 par défaut
            mask_threshold=env_float("YOLO11_MASK_THRESHOLD", 0.25),
            min_mask_area_px=env_int("YOLO11_MIN_MASK_AREA", 0),
            filter_coco_building_like=env_bool("YOLO11_FILTER_BUILDING_LIKE", False),
            detection_only=env_bool("YOLO11_DETECTION_ONLY", False),  # CORRIGÉ : False par défaut
            class_id=env_int("YOLO11_CLASS_ID", 0),
        )


def resolve_weights_path(cfg: YOLO11Config | None = None) -> Path:
    """Fine-tune local → fichier explicite → nom modèle Ultralytics (téléchargement auto)."""
    cfg = cfg or YOLO11Config.from_env()
    if cfg.use_finetuned and cfg.best_weights.is_file() and cfg.best_weights.stat().st_size > 1024:
        return cfg.best_weights.resolve()
    w = cfg.pretrained_weights
    if w.is_file() and w.stat().st_size > 1024:
        return w.resolve()
    return Path(cfg.model)


def default_config(**overrides) -> YOLO11Config:
    cfg = YOLO11Config.from_env()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


SANITY_CHECK_NOTICE = (
    "Modèle COCO (images naturelles) — sanity check sur satellite, pas de prod sans fine-tune RS."
)
