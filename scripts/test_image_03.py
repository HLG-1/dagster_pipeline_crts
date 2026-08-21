"""
Test de segmentation sur l'orthophoto 03.tif.

Permet de tester :
  - soit un secteur/crop rapide (ex: 2048x2048 px) pour validation immediate en quelques secondes
  - soit l'image entiere
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.windows import Window

from pipeline_batiments.core.geo_io import export_all, raster_meta, read_window_rgb, window_transform
from pipeline_batiments.core.polygons import instances_to_polygons, stitch_polygons
from pipeline_batiments.core.tiling import tile_positions
from pipeline_batiments.core.instances import render_gdf_outlined
from pipeline_batiments.yolo11.config import YOLO11Config
from pipeline_batiments.yolo11.detect import YOLODetector


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test sur 03.tif")
    parser.add_argument("--image", type=str, default=r"C:\Users\HP\Downloads\03.tif", help="Chemin de 03.tif")
    parser.add_argument("--roi-size", type=int, default=2048, help="Taille du secteur en pixels (0 pour tout)")
    parser.add_argument("--row", type=int, default=12000, help="Ligne de depart du secteur")
    parser.add_argument("--col", type=int, default=14000, help="Colonne de depart du secteur")
    parser.add_argument("--conf", type=float, default=0.15, help="Seuil confiance YOLO")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        # Essayer a la racine de C:\Users\HP
        alt = Path(r"C:\Users\HP\03.tif")
        if alt.is_file():
            img_path = alt
        else:
            print(f"Erreur : Image introuvable a {img_path} ni {alt}")
            return

    print("==================================================")
    print(f"  Test de segmentation sur : {img_path.name}")
    print("==================================================")

    meta = raster_meta(img_path)
    w_full, h_full = meta["width"], meta["height"]
    crs_str = meta["crs"]
    print(f"Dimensions completes : {w_full} x {h_full} px ({img_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"CRS : {crs_str[:60]}...")

    # Chargement YOLO sur CPU
    weights = ROOT / "checkpoints" / "yolo11" / "best.pt"
    if not weights.is_file():
        weights = ROOT / "finetunig_portable" / "checkpoints" / "pretrained" / "yolov8n_building.pt"

    print(f"\nChargement YOLO CPU ({weights.name})...")
    cfg = YOLO11Config(
        pretrained_weights=weights,
        best_weights=weights,
        model=str(weights),
        device="cpu",
        infer_imgsz=512,
        conf=args.conf,
    )
    detector = YOLODetector(cfg)
    print("Modele pret !")

    if args.roi_size > 0:
        r_start = min(args.row, max(0, h_full - args.roi_size))
        c_start = min(args.col, max(0, w_full - args.roi_size))
        h_crop, w_crop = min(args.roi_size, h_full), min(args.roi_size, w_full)
        print(f"\nMode Secteur : {w_crop}x{h_crop} px (offset row={r_start}, col={c_start})")
    else:
        r_start, c_start = 0, 0
        h_crop, w_crop = h_full, w_full
        print(f"\nMode Image Entiere : {w_crop}x{h_crop} px")

    # Tuilage 512x512 avec overlap 64
    tile_size, overlap = 512, 64
    positions = tile_positions(h_crop, w_crop, tile_size=tile_size, overlap=overlap)
    print(f"Nombre de tuiles a traiter : {len(positions)}")

    parts = []
    total_dets = 0
    t0 = time.perf_counter()

    for i, (r_rel, c_rel, th, tw) in enumerate(positions):
        r_abs = r_start + r_rel
        c_abs = c_start + c_rel
        win = Window(c_abs, r_abs, tw, th)

        rgb_tile = read_window_rgb(img_path, win)
        out = detector.detect_instances(rgb_tile)
        n_d = out["n_detections"]
        total_dets += n_d

        inst = out.get("instances")
        if inst is not None and int(inst.max()) > 0:
            win_tf = window_transform(img_path, win)
            gdf = instances_to_polygons(inst, win_tf, crs_str, simplify_tol=0.5, regularize=False)
            if len(gdf):
                parts.append(gdf)

        if (i + 1) % 5 == 0 or (i + 1) == len(positions):
            print(f"  Avancement : {i+1}/{len(positions)} tuiles traites ({total_dets} batiments detectes)...")

    duration = time.perf_counter() - t0
    print(f"\nInference terminee en {duration:.1f}s (~{duration/len(positions):.2f}s par tuile)")

    out_dir = ROOT / "results" / f"test_03_roi_{w_crop}x{h_crop}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if parts:
        merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs_str)
        stitched = stitch_polygons(merged, crs_str, overlap_frac=0.5)
        stitched = stitched.reset_index(drop=True)
        stitched["building_id"] = range(1, len(stitched) + 1)
        print(f"Batiments finaux apres fusion inter-tuiles : {len(stitched)}")

        paths = export_all(stitched, out_dir, stem="03_buildings")
        print(f"\nSorties SIG generees :")
        print(f"  GeoJSON : {paths['geojson']}")
        print(f"  GPKG    : {paths['gpkg']}")
        print(f"  Shapefile : {paths['shapefile_dir']}")

        # Generation Overlay visuel du secteur
        if args.roi_size > 0:
            sector_win = Window(c_start, r_start, w_crop, h_crop)
            sector_rgb = read_window_rgb(img_path, sector_win)
            sector_tf = window_transform(img_path, sector_win)
            overlay = render_gdf_outlined(sector_rgb, stitched, transform=sector_tf, line_thickness=2)
            overlay_path = out_dir / "03_sector_overlay.jpg"
            Image.fromarray(overlay).save(overlay_path)
            print(f"  Apercu visuel : {overlay_path}")
    else:
        print("Aucun batiment detecte sur ce secteur.")

    print("\n[SUCCES] Test sur 03.tif valide !")


if __name__ == "__main__":
    main()
