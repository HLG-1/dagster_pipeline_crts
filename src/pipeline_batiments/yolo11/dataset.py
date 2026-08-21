"""
Prépare le dataset Ultralytics YOLO11 segmentation (polygones normalisés).

Source (priorité) :
  - dataset_bati_vector/<split>/{images, masks_vec}  (GPKG, splits ETL)
  - data/images/imgN.tif + data/mask/maskN.tif         (legacy raster)

Cible : data/yolo11/images/{train,val}/ + labels/{train,val}/*.txt
"""
from __future__ import annotations

import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import yaml

from .config import IMAGES_SRC, MASKS_SRC, ROOT, YOLO_DATA_ROOT

ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from applib.core.dataset import (  # noqa: E402
    DATASET_ROOT,
    has_mask,
    images_dir,
    is_vector,
    list_images,
    mask_path_for,
)
from applib.core.benchmark_lot import benchmark_tile_names


def _read_rgb_bgr(path: Path) -> np.ndarray:
    """GeoTIFF ou image → BGR uint8 pour OpenCV / YOLO."""
    with rasterio.open(path) as src:
        data = src.read()
    if data.shape[0] >= 3:
        bands = [data[0], data[1], data[2]]
    else:
        bands = [data[0], data[0], data[0]]
    rgb = np.stack(
        [np.clip(b, 0, 255).astype(np.uint8) if b.dtype == np.uint8 else
         (np.clip(b.astype(np.float32) / max(float(b.max()), 1) * 255, 0, 255).astype(np.uint8))
         for b in bands],
        axis=-1,
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


_IMG_RE = re.compile(r"^img(\d+)$", re.I)


def _list_legacy_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(IMAGES_SRC.glob("img*.tif")):
        m = _IMG_RE.match(img.stem)
        if not m:
            continue
        mask = MASKS_SRC / f"mask{m.group(1)}.tif"
        if mask.is_file():
            pairs.append((img, mask))
    return pairs


def _list_split_pairs(split: str) -> list[tuple[Path, Path | None, str]]:
    """(image, masque|None, kind) — kind ∈ {vector, raster}."""
    out: list[tuple[Path, Path | None, str]] = []
    for img in list_images(split):
        if not has_mask(img):
            continue
        mp = mask_path_for(img)
        out.append((img, mp, "vector" if is_vector(split) else "raster"))
    return out


def _mask_to_polygons(
    mask: np.ndarray,
    class_id: int = 0,
    min_area: int = 25,
) -> list[str]:
    """Masque binaire → lignes label YOLO segment (polygones normalisés)."""
    if mask.max() > 1:
        binary = (mask > 127).astype(np.uint8)
    else:
        binary = (mask > 0).astype(np.uint8)

    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = pts
        coords: list[str] = []
        for x, y in approx:
            coords.append(f"{np.clip(x / w, 0, 1):.6f}")
            coords.append(f"{np.clip(y / h, 0, 1):.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines


def _gdf_to_polygons(gdf, transform, h: int, w: int, class_id: int, min_area: int) -> list[str]:
    """GeoDataFrame GPKG → lignes label YOLO (coords normalisées pixel)."""
    if gdf is None or len(gdf) == 0:
        return []

    inv = ~transform
    lines: list[str] = []

    def _add_ring(ring, min_px: int):
        coords: list[str] = []
        xs, ys = [], []
        for pt in ring:
            x, y = float(pt[0]), float(pt[1])
            col, row = inv * (float(x), float(y))
            xs.append(col)
            ys.append(row)
            coords.append(f"{np.clip(col / w, 0, 1):.6f}")
            coords.append(f"{np.clip(row / h, 0, 1):.6f}")
        if len(coords) < 6:
            return
        area = 0.5 * abs(
            sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(len(xs) - 1))
            + xs[-1] * ys[0] - xs[0] * ys[-1]
        )
        if area < min_area:
            return
        lines.append(f"{class_id} " + " ".join(coords))

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            _add_ring(list(geom.exterior.coords), min_area)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                _add_ring(list(poly.exterior.coords), min_area)

    return lines


def _write_split(
    items: list[tuple[Path, Path | None, str]],
    split: str,
    cfg_min_area: int,
    class_id: int,
) -> int:
    img_out = YOLO_DATA_ROOT / "images" / split
    lbl_out = YOLO_DATA_ROOT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path, mask_path, kind in items:
        stem = img_path.stem
        img = _read_rgb_bgr(img_path)

        out_img = img_out / f"{stem}.jpg"
        cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        if kind == "vector":
            import geopandas as gpd

            with rasterio.open(img_path) as src:
                transform = src.transform
                h, w = src.height, src.width
            gdf = gpd.read_file(mask_path)
            lines = _gdf_to_polygons(gdf, transform, h, w, class_id, cfg_min_area)
        else:
            mask = _read_mask(mask_path)
            lines = _mask_to_polygons(mask, class_id=class_id, min_area=cfg_min_area)

        lbl_file = lbl_out / f"{stem}.txt"
        lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1
    return n


def _list_lot_pairs(split: str, tile_names: list[str]) -> list[tuple[Path, Path | None, str]]:
    """Paires image/masque pour un lot figé (benchmark pilote)."""
    img_root = images_dir(split)
    out: list[tuple[Path, Path | None, str]] = []
    for name in tile_names:
        img = img_root / name
        if not img.is_file() or not has_mask(img):
            continue
        mp = mask_path_for(img)
        out.append((img, mp, "vector" if is_vector(split) else "raster"))
    return out


def build_yolo_dataset(
    train_ratio: float = 0.75,
    val_ratio: float = 0.25,
    seed: int = 42,
    min_polygon_area_px: int = 25,
    class_id: int = 0,
    class_name: str = "building",
    force: bool = False,
    use_etl_splits: bool = True,
    tile_list: Path | None = None,
) -> Path:
    """
    Construit data/yolo11/ et dataset.yaml à la racine du projet.
    Retourne le chemin du dataset.yaml.

    Si ``tile_list`` est fourni (ex. metadata/benchmark_10_test.json), n'utilise
    que ces tuiles avec un split train/val aléatoire reproductible.

    Si ``dataset_bati_vector`` est présent et ``use_etl_splits=True``, reprend
    les splits train/val ETL (pas de mélange aléatoire).
    """
    if force and YOLO_DATA_ROOT.exists():
        shutil.rmtree(YOLO_DATA_ROOT)
    YOLO_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    source_desc = ""
    lot_path = Path(tile_list) if tile_list else None

    if lot_path is not None and lot_path.is_file():
        from applib.core.benchmark_lot import benchmark_split

        split = benchmark_split(lot_path)
        names = benchmark_tile_names(lot_path)
        items = _list_lot_pairs(split, names)
        if not items:
            raise FileNotFoundError(f"Aucune tuile valide dans le lot {lot_path}")

        rng = random.Random(seed)
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(n * val_ratio)) if n > 1 else 0
        if n_val >= n:
            n_val = n - 1
        val_items = shuffled[:n_val]
        train_items = shuffled[n_val:]

        n_train = _write_split(train_items, "train", min_polygon_area_px, class_id)
        n_val = _write_split(val_items, "val", min_polygon_area_px, class_id)
        source_desc = f"{DATASET_ROOT} (lot {lot_path.name}, {n} tuiles)"
        yaml_path = ensure_dataset_yaml(
            class_name=class_name,
            n_train=n_train,
            n_val=n_val,
            source=source_desc,
        )
        return yaml_path

    if use_etl_splits and DATASET_ROOT is not None:
        train_items = _list_split_pairs("train")
        val_items = _list_split_pairs("val")
        if train_items:
            n_train = _write_split(train_items, "train", min_polygon_area_px, class_id)
            n_val = _write_split(val_items, "val", min_polygon_area_px, class_id) if val_items else 0
            source_desc = str(DATASET_ROOT.resolve())
            yaml_path = ensure_dataset_yaml(
                class_name=class_name,
                n_train=n_train,
                n_val=n_val,
                source=source_desc,
            )
            return yaml_path

    pairs = _list_legacy_pairs()
    if not pairs:
        raise FileNotFoundError(
            f"Aucune paire img/mask (dataset_bati_vector ou {IMAGES_SRC} / {MASKS_SRC})"
        )

    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    if n_val >= n:
        n_val = n - 1
    val_pairs = [(a, b, "raster") for a, b in shuffled[:n_val]]
    train_pairs = [(a, b, "raster") for a, b in shuffled[n_val:]]

    n_train = _write_split(train_pairs, "train", min_polygon_area_px, class_id)
    n_val = _write_split(val_pairs, "val", min_polygon_area_px, class_id) if val_pairs else 0

    yaml_path = ensure_dataset_yaml(
        class_name=class_name,
        n_train=n_train,
        n_val=n_val,
        source=str(IMAGES_SRC.resolve()),
    )
    return yaml_path


def ensure_dataset_yaml(
    class_name: str = "building",
    n_train: int = 0,
    n_val: int = 0,
    source: str = "",
) -> Path:
    """Génère / corrige dataset.yaml (format segmentation instance, polygones)."""
    root_abs = YOLO_DATA_ROOT.resolve()
    src = source or str(IMAGES_SRC.resolve())
    data = {
        "path": str(root_abs),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
        "nc": 1,
        "task": "segment",
        "description": "Bâtiments aériens — polygones depuis labels vectoriels GPKG",
        "source_dataset": src,
    }
    if n_train or n_val:
        data["counts"] = {"train": n_train, "val": n_val}

    yolo_yaml = YOLO_DATA_ROOT / "dataset.yaml"
    yolo_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(yolo_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    root_yaml = ROOT / "dataset.yaml"
    root_data = {
        "path": str(root_abs),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
        "nc": 1,
    }
    with open(root_yaml, "w", encoding="utf-8") as f:
        yaml.dump(root_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return root_yaml
