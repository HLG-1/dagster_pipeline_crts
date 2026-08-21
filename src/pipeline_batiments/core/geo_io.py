"""
I/O geospatiales : validation GeoTIFF, lecture par fenetres (pour les
grandes orthophotos, sans tout charger en RAM), conversion RGB uint8 pour
l'inference, export SIG (GeoJSON / GPKG / Shapefile).

Ce module ajoute `raster_meta` (metadonnees seules, aucune
lecture de pixels) et `read_window_rgb` (lecture d'une seule fenetre), pour
repondre a l'exigence du cahier des charges : "pour les tres grandes
orthophotos, ne plus charger toute l'image en RAM - lire par fenetres
(rasterio.windows)".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from shapely.geometry import shape
import geopandas as gpd


# Metadonnees / validation (etape 01) 

_SUPPORTED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def raster_meta(path: str | Path) -> dict:
    """Ouvre l'orthophoto (GeoTIFF, PNG, JPG) et renvoie ses metadonnees.
    Pour les grands GeoTIFF, src.read() n'est pas appele ici.
    Pour les PNG/JPG simples, un CRS (EPSG:4326) et un transform par defaut
    sont fournis automatiquement pour permettre le traitement direct de patchs.
    """
    from affine import Affine
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() not in _SUPPORTED_EXT:
        raise ValueError(f"Format non supporte (attendu {_SUPPORTED_EXT}) : {path}")

    # Essai 1 : lecture rasterio (GeoTIFF ou raster standard)
    try:
        with rasterio.open(path) as src:
            crs_str = safe_crs(src.crs)
            tf = src.transform if src.transform is not None else Affine.identity()
            bounds = src.bounds if src.bounds is not None else (0, 0, src.width, src.height)
            return {
                "path": str(path),
                "stem": path.stem,
                "crs": crs_str,
                "transform": tf,
                "bounds": bounds,
                "width": src.width,
                "height": src.height,
                "n_bands": src.count,
                "dtype": str(src.dtypes[0]) if src.dtypes else "uint8",
            }
    except Exception:
        pass

    # Essai 2 : repli PIL pour les images PNG / JPG sans driver GDAL
    with Image.open(path) as img:
        w, h = img.size
        mode = img.mode
        n_bands = len(img.getbands()) if hasattr(img, "getbands") else 3
        return {
            "path": str(path),
            "stem": path.stem,
            "crs": "EPSG:4326",
            "transform": Affine.identity(),
            "bounds": (0, 0, w, h),
            "width": w,
            "height": h,
            "n_bands": n_bands,
            "dtype": "uint8",
        }


# Conversion bandes -> RGB uint8 (pour YOLO / SAM3)
def _to_rgb_uint8(data: np.ndarray) -> tuple[np.ndarray, str]:
    if data.ndim == 3 and data.shape[2] in (1, 3, 4) and data.shape[0] > 4:
        # Format HxWxC
        if data.shape[2] == 1:
            data = np.repeat(data, 3, axis=-1)
        elif data.shape[2] == 4:
            data = data[:, :, :3]
        return data.astype(np.uint8), "h_w_c"

    n_bands = data.shape[0]
    if n_bands >= 3:
        bands = [data[0], data[1], data[2]]
    elif n_bands == 2:
        bands = [data[0], data[1], data[0]]
    else:
        bands = [data[0], data[0], data[0]]

    dtype = bands[0].dtype
    max_val = max(float(b.max()) for b in bands)

    if dtype == np.uint8 and max_val <= 255:
        rgb = np.stack([np.clip(b, 0, 255).astype(np.uint8) for b in bands], axis=-1)
        return rgb, "whu_uint8"

    if max_val > 255:
        rgb = np.stack([_stretch_band_uint8(b) for b in bands], axis=-1)
        return rgb, "uint16_stretched"

    channels = []
    for b in bands:
        b = b.astype(np.float32)
        lo, hi = float(b.min()), float(b.max())
        if hi > lo:
            b = (b - lo) / (hi - lo) * 255.0
        channels.append(np.clip(b, 0, 255).astype(np.uint8))
    return np.stack(channels, axis=-1), "float_scaled"


def _stretch_band_uint8(band: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    b = band.astype(np.float32)
    valid = b[b > 0]
    if valid.size > 100:
        lo, hi = np.percentile(valid, [p_lo, p_hi])
    else:
        lo, hi = float(b.min()), float(b.max())
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
    out = np.power(out, 0.92)
    return (out * 255.0).astype(np.uint8)


#Lecture fenetree (grandes orthophotos ou petits patchs) 

def read_window_rgb(path: str | Path, window: rasterio.windows.Window) -> np.ndarray:
    """Lit uniquement la fenetre demandee (pas toute l'image) et la convertit
    en RGB uint8 pour l'inference YOLO/SAM3. Fonctionne sur GeoTIFF, PNG, JPG.
    """
    path = Path(path)
    try:
        with rasterio.open(path) as src:
            data = src.read(window=window)
        rgb, _profile = _to_rgb_uint8(data)
        return rgb
    except Exception:
        from PIL import Image
        with Image.open(path) as pil_img:
            full = np.asarray(pil_img.convert("RGB"))
        r, c = int(round(window.row_off)), int(round(window.col_off))
        h, w = int(round(window.height)), int(round(window.width))
        return full[r : r + h, c : c + w]


def window_transform(path: str | Path, window: rasterio.windows.Window):
    """Transform (georeference) correspondant a une fenetre donnee, pour
    reprojeter les polygones locaux (coords tuile) vers les coords monde."""
    from affine import Affine
    path = Path(path)
    try:
        with rasterio.open(path) as src:
            if src.transform is not None:
                return rasterio.windows.transform(window, src.transform)
    except Exception:
        pass
    return Affine.translation(window.col_off, window.row_off)


# CRS 

def safe_crs(crs_value) -> str:
    """Toujours renvoyer un CRS valide pour GeoPandas (repli EPSG:4326)."""
    if crs_value is None:
        return "EPSG:4326"
    try:
        s = crs_value.to_string() if hasattr(crs_value, "to_string") else str(crs_value).strip()
    except Exception:
        s = ""
    if not s or s.lower() in ("none", "null", "unknown", "epsg:none", "none:none"):
        return "EPSG:4326"
    try:
        from pyproj import CRS
        CRS.from_user_input(s)
        return s
    except Exception:
        return "EPSG:4326"


# Exports SIG 

def export_geojson(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = gdf.to_crs("EPSG:4326") if gdf.crs and str(gdf.crs) != "EPSG:4326" else gdf
    out.to_file(path, driver="GeoJSON")


def export_gpkg(gdf: gpd.GeoDataFrame, path: str | Path, layer: str = "buildings") -> None:
    """GeoPackage (.gpkg) - CRS d'origine conserve (pas de reprojection)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG", layer=layer)


def export_shapefile(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="ESRI Shapefile")


def export_all(gdf: gpd.GeoDataFrame, out_dir: str | Path, stem: str = "buildings") -> dict:
    """Ecrit GeoJSON + GPKG + Shapefile. Renvoie {format: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    gj = out_dir / f"{stem}.geojson"
    export_geojson(gdf, gj)
    paths["geojson"] = gj

    gp = out_dir / f"{stem}.gpkg"
    export_gpkg(gdf, gp)
    paths["gpkg"] = gp

    shp_dir = out_dir / f"{stem}_shp"
    export_shapefile(gdf, shp_dir / f"{stem}.shp")
    paths["shapefile_dir"] = shp_dir

    return paths


def clean_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Nettoyage morphologique d'un masque binaire (retire les petits bruits et trous)."""
    import cv2
    mask_u8 = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask_u8)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lbl] = 1
    return clean

