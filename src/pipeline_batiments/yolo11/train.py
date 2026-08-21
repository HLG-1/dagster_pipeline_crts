"""
Entraînement YOLO11 segmentation — GPU, AMP, early stopping, mAP mask.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import BEST_WEIGHTS, CHECKPOINT_DIR, YOLO11Config, default_config
from .dataset import build_yolo_dataset, ensure_dataset_yaml


def train_yolo11(cfg: YOLO11Config | None = None, prepare_data: bool = True) -> dict:
    """
    Lance l'entraînement Ultralytics.

    Équivalent CLI :
        yolo segment train model=yolov8n-seg.pt data=dataset.yaml \\
            imgsz=1024 batch=auto epochs=100 device=0
    """
    from ultralytics import YOLO

    cfg = cfg or default_config()

    if prepare_data:
        build_yolo_dataset(
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            min_polygon_area_px=cfg.min_polygon_area_px,
            class_id=cfg.class_id,
            class_name=cfg.class_name,
            force=True,
        )
    else:
        ensure_dataset_yaml(class_name=cfg.class_name)

    data_yaml = Path(cfg.data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset.yaml introuvable : {data_yaml}")

    model = YOLO(cfg.model)
    batch = cfg.batch if cfg.batch != "auto" else -1

    results = model.train(
        task=cfg.task,
        data=str(data_yaml.resolve()),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=batch,
        device=cfg.device,
        workers=cfg.workers,
        patience=cfg.patience,
        amp=cfg.amp,
        fraction=cfg.fraction,
        val=cfg.val,
        plots=cfg.plots,
        save=cfg.save,
        project=cfg.project,
        name=cfg.name,
        exist_ok=True,
        pretrained=True,
        # Augmentations satellite
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        fliplr=cfg.fliplr,
        flipud=cfg.flipud,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
    )

    run_dir = Path(cfg.project) / cfg.name
    best_src = run_dir / "weights" / "best.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if best_src.is_file():
        shutil.copy2(best_src, BEST_WEIGHTS)
        print(f"✅ Meilleurs poids copiés → {BEST_WEIGHTS}")
    else:
        alt = Path(getattr(results, "save_dir", run_dir)) / "weights" / "best.pt"
        if alt.is_file():
            shutil.copy2(alt, BEST_WEIGHTS)

    metrics = {}
    if hasattr(results, "results_dict"):
        metrics = dict(results.results_dict)
    elif hasattr(results, "metrics"):
        metrics = getattr(results.metrics, "results_dict", {}) or {}

    return {
        "save_dir": str(run_dir),
        "best_weights": str(BEST_WEIGHTS) if BEST_WEIGHTS.is_file() else str(best_src),
        "metrics": metrics,
    }


def main() -> None:
    import argparse

    from .env import resolved_device

    p = argparse.ArgumentParser(description="Fine-tuning YOLO segment bâtiments (GPU)")
    p.add_argument("--prepare", action="store_true", help="Préparer le dataset YOLO avant entraînement")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="0, cuda, cpu, auto")
    p.add_argument("--model", default=None, help="ex. yolov8m-seg.pt")
    args = p.parse_args()

    cfg = default_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.model:
        cfg.model = args.model
    if args.device is not None:
        d = args.device.strip().lower()
        cfg.device = d if d in ("cpu", "cuda", "auto") or d.isdigit() else resolved_device()
    elif cfg.device == "auto":
        cfg.device = resolved_device()
        if cfg.device == "cuda":
            cfg.device = "0"

    print(f"Entraînement YOLO — device={cfg.device} epochs={cfg.epochs} imgsz={cfg.imgsz}")
    out = train_yolo11(cfg, prepare_data=args.prepare)
    print("Terminé :", out.get("best_weights"))


if __name__ == "__main__":
    main()
