"""
Test sur plusieurs patches de l'image pour valider YOLO+SAM3.
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
from rasterio.windows import Window

from pipeline_batiments.core.geo_io import read_window_rgb, window_transform
from pipeline_batiments.core.polygons import instances_to_polygons, stitch_polygons
from pipeline_batiments.core.tiling import tile_positions
from pipeline_batiments.core.multiprompt import points_for_boxes
from pipeline_batiments.core.instances import render_gdf_outlined
from pipeline_batiments.yolo11.config import YOLO11Config
from pipeline_batiments.yolo11.detect import YOLODetector
from pipeline_batiments.resources.sam3_resource import SAM3Resource
from pipeline_batiments.core.geo_io import raster_meta
from PIL import Image


def test_single_patch(img_path, detector, sam3, r_start, c_start, h_crop, w_crop, use_sam3, min_area, crs_str, tile_size=512, overlap=64):
    """Test sur un seul patch de l'image."""
    positions = tile_positions(h_crop, w_crop, tile_size=tile_size, overlap=overlap)
    print(f"  Nombre de tuiles : {len(positions)}")

    parts = []
    total_dets = 0
    t0 = time.perf_counter()

    for i, (r_rel, c_rel, th, tw) in enumerate(positions):
        r_abs = r_start + r_rel
        c_abs = c_start + c_rel
        win = Window(c_abs, r_abs, tw, th)

        rgb_tile = read_window_rgb(img_path, win)

        if use_sam3:
            # Utiliser YOLO + SAM3 avec fallback vers YOLO natif
            out = detector.detect_instances(rgb_tile)
            boxes = out.get("boxes_xyxy", [])
            yolo_native = out.get("instances")
            n_d = out["n_detections"]
            total_dets += n_d

            if boxes:
                points = points_for_boxes(rgb_tile, boxes)
                try:
                    label, _meta = sam3.predict_instances(
                        rgb_tile, boxes_xyxy=boxes, points_xy=points, sep="per_box",
                    )
                    
                    # Filtrer les petites surfaces
                    if int(label.max()) > 0:
                        out_label = np.zeros_like(label)
                        nid = 0
                        for lab in range(1, int(label.max()) + 1):
                            m = label == lab
                            if m.sum() >= min_area:
                                nid += 1
                                out_label[m] = nid
                        label = out_label
                    
                    if int(label.max()) == 0:
                        inst = yolo_native  # Fallback vers YOLO natif
                    else:
                        inst = label
                except Exception as e:
                    inst = yolo_native  # Fallback vers YOLO natif
            else:
                inst = yolo_native
        else:
            # Utiliser YOLO natif uniquement
            out = detector.detect_instances(rgb_tile)
            n_d = out["n_detections"]
            total_dets += n_d
            inst = out.get("instances")

        if inst is not None and int(inst.max()) > 0:
            win_tf = window_transform(img_path, win)
            gdf = instances_to_polygons(inst, win_tf, crs_str, simplify_tol=0.5, regularize=False)
            if len(gdf):
                parts.append(gdf)

    duration = time.perf_counter() - t0

    if parts:
        merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs_str)
        stitched = stitch_polygons(merged, crs_str, overlap_frac=0.5)
        stitched = stitched.reset_index(drop=True)
        return len(stitched), duration, stitched
    else:
        return 0, duration, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test sur plusieurs patches")
    parser.add_argument("--image", type=str, default=r"/home/hajar/bench_haut/crts2_bench_haut/data/zone3/03.tif", help="Chemin de l'image")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"], help="Device pour YOLO")
    parser.add_argument("--use-sam3", action="store_true", help="Utiliser SAM3")
    parser.add_argument("--sam3-url", type=str, default="http://127.0.0.1:8077", help="URL du service SAM3")
    parser.add_argument("--min-area", type=int, default=0, help="Surface minimale en pixels")
    parser.add_argument("--conf", type=float, default=0.15, help="Seuil confiance YOLO")
    parser.add_argument("--save-visuals", action="store_true", help="Sauvegarder les images input et overlay")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Erreur : Image introuvable a {img_path}")
        return

    print("="*60)
    print("  TEST MULTI-PATCHS YOLO+SAM3")
    print("="*60)

    meta = raster_meta(img_path)
    w_full, h_full = meta["width"], meta["height"]
    crs_str = meta["crs"]
    print(f"Image : {img_path.name}")
    print(f"Dimensions : {w_full} x {h_full} px")

    # Chargement YOLO
    weights = ROOT / "checkpoints" / "yolo11" / "best.pt"
    if not weights.is_file():
        weights = ROOT / "checkpoints" / "pretrained" / "yolov8n_building.pt"
    if not weights.is_file():
        weights = ROOT / "finetunig_portable" / "checkpoints" / "pretrained" / "yolov8n_building.pt"

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nChargement YOLO sur {device} ({weights.name})...")
    cfg = YOLO11Config(
        pretrained_weights=weights,
        best_weights=weights,
        model=str(weights),
        device=device,
        infer_imgsz=512,
        conf=args.conf,
    )
    detector = YOLODetector(cfg)
    print("YOLO pret !")

    # Initialisation SAM3 si demandé
    sam3 = None
    if args.use_sam3:
        print(f"\nInitialisation SAM3 ({args.sam3_url})...")
        sam3 = SAM3Resource(url=args.sam3_url, timeout_s=300, max_retries=3, backoff_s=5)
        if not sam3.health_check():
            print("ERREUR: Service SAM3 indisponible")
            return
        print("SAM3 pret !")

    # Définir plusieurs patches représentatifs
    patches = [
        {"name": "Centre", "row": h_full//2 - 512, "col": w_full//2 - 512, "size": 1024},
        {"name": "Coin_HG", "row": 0, "col": 0, "size": 1024},
        {"name": "Coin_HD", "row": 0, "col": w_full - 1024, "size": 1024},
        {"name": "Coin_BG", "row": h_full - 1024, "col": 0, "size": 1024},
        {"name": "Coin_BD", "row": h_full - 1024, "col": w_full - 1024, "size": 1024},
        {"name": "Zone_Mid_Haut", "row": h_full//4 - 512, "col": w_full//2 - 512, "size": 1024},
        {"name": "Zone_Mid_Bas", "row": 3*h_full//4 - 512, "col": w_full//2 - 512, "size": 1024},
    ]
    
    mode_str = "YOLO+SAM3" if args.use_sam3 else "YOLO natif"
    print(f"\n{'='*60}")
    print(f"  TEST MODE : {mode_str}")
    print(f"{'='*60}")
    
    # Créer le dossier de sortie si on sauvegarde les visuels
    import tempfile
    if args.save_visuals:
        out_dir = Path(tempfile.gettempdir()) / "multi_patches_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Résultats visuels sauvegardés dans : {out_dir}")
    
    all_results = []
    
    for patch in patches:
        print(f"\n--- Patch: {patch['name']} ---")
        
        # S'assurer que le patch est dans les limites
        r_start = max(0, min(patch["row"], h_full - patch["size"]))
        c_start = max(0, min(patch["col"], w_full - patch["size"]))
        h_crop = min(patch["size"], h_full - r_start)
        w_crop = min(patch["size"], w_full - c_start)
        
        print(f"Position: ({r_start}, {c_start}), taille: {w_crop}x{h_crop}")
        
        # Test sur ce patch
        n_buildings, duration, gdf = test_single_patch(
            img_path, detector, sam3, r_start, c_start, h_crop, w_crop,
            args.use_sam3, args.min_area, crs_str
        )
        
        print(f"Resultat: {n_buildings} bâtiments, {duration:.1f}s")
        
        # Sauvegarder les visuels si demandé
        if args.save_visuals and gdf is not None and len(gdf) > 0:
            patch_dir = out_dir / patch["name"]
            patch_dir.mkdir(parents=True, exist_ok=True)
            
            # Lire l'image RGB du patch
            sector_win = Window(c_start, r_start, w_crop, h_crop)
            sector_rgb = read_window_rgb(img_path, sector_win)
            sector_tf = window_transform(img_path, sector_win)
            
            # Sauvegarder l'image input
            input_path = patch_dir / "input.jpg"
            Image.fromarray(sector_rgb).save(input_path)
            
            # Créer et sauvegarder l'overlay
            overlay = render_gdf_outlined(sector_rgb, gdf, transform=sector_tf, line_thickness=2)
            overlay_path = patch_dir / "overlay.jpg"
            Image.fromarray(overlay).save(overlay_path)
            
            # Créer l'image combinée input + overlay
            input_bgr = cv2.cvtColor(sector_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            # Redimensionner si trop grand
            max_display = 800
            h, w = input_bgr.shape[:2]
            if max(h, w) > max_display:
                scale = max_display / max(h, w)
                input_bgr = cv2.resize(input_bgr, (int(w * scale), int(h * scale)))
                overlay_bgr = cv2.resize(overlay_bgr, (int(w * scale), int(h * scale)))
            
            combined = np.hstack([input_bgr, overlay_bgr])
            cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Overlay", (input_bgr.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            combined_path = patch_dir / "combined.jpg"
            cv2.imwrite(str(combined_path), combined)
            
            print(f"  Visuels sauvegardés dans : {patch_dir}")
        elif args.save_visuals:
            print(f"  Pas de visuels (aucun bâtiment détecté)")
        
        all_results.append({
            "name": patch["name"],
            "n_buildings": n_buildings,
            "duration": duration
        })
    
    # Résumé
    print(f"\n{'='*60}")
    print("  RÉSUMÉ")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"{result['name']:15s}: {result['n_buildings']:3d} bâtiments, {result['duration']:5.1f}s")
    
    total_buildings = sum(r["n_buildings"] for r in all_results)
    total_time = sum(r["duration"] for r in all_results)
    avg_time = total_time / len(all_results)
    
    print(f"\nTotal: {total_buildings} bâtiments sur {len(all_results)} patches")
    print(f"Temps total: {total_time:.1f}s, Moyenne: {avg_time:.1f}s/patch")
    print(f"\n[SUCCES] Test multi-patches terminé !")


if __name__ == "__main__":
    main()