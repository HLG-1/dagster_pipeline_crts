"""
Prétraitement images satellite → entrée Ultralytics YOLO11-seg.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessedImage:
    """Tuile prête pour predict()."""

    rgb: np.ndarray  # H×W×3 uint8 RGB
    source_path: Path | None
    original_size: tuple[int, int]  # (h, w)
    infer_imgsz: int


def read_image(path: str | Path) -> np.ndarray:
    """GeoTIFF ou image → RGB uint8."""
    path = Path(path)
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.stack([arr[0]] * 3, axis=-1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
    return rgb


def prepare_tile(
    source: np.ndarray | str | Path,
    *,
    tile_size: int | None = None,
) -> PreprocessedImage:
    """
    Normalise l'entrée : RGB uint8, imgsz = max(h, w) (évite 0 masque sur tuiles 512).
    """
    src_path = Path(source) if isinstance(source, (str, Path)) and Path(source).suffix else None
    if isinstance(source, np.ndarray):
        rgb = source
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.max() > 1 else (rgb * 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        # NOTE : on suppose que le tableau numpy entrant est déjà en RGB uint8
        # (garanti par read_window_rgb dans geo_io.py).
        # L'heuristique BGR/RGB a été supprimée car elle causait des inversions
        # erronées sur les images satellite (toits gris, végétation, etc.).

    else:
        rgb = read_image(Path(source))
        src_path = Path(source)

    h, w = rgb.shape[:2]
    if tile_size and (h != tile_size or w != tile_size):
        rgb = cv2.resize(rgb, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        h, w = tile_size, tile_size

    infer_imgsz = max(h, w)
    return PreprocessedImage(
        rgb=rgb,
        source_path=src_path,
        original_size=(h, w),
        infer_imgsz=infer_imgsz,
    )
