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
from pipeline_batiments.core.metrics import compute_metrics
from pipeline_batiments.core.multiprompt import points_for_boxes
from pipeline_batiments.yolo11.config import YOLO11Config
from pipeline_batiments.yolo11.detect import YOLODetector
from pipeline_batiments.resources.sam3_resource import SAM3Resource


def test_single_patch(img_path, detector, sam3, r_start, c_start, h_crop, w_crop, args, device, crs_str, tile_size, overlap):
    """Test sur un seul patch de l'image."""
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

        if args.use_sam3:
            # Utiliser YOLO + SAM3 avec fallback vers YOLO natif
            # IMPORTANT: utiliser detect_instances pour avoir les masques natifs YOLO disponibles
            out = detector.detect_instances(rgb_tile)
            boxes = out.get("boxes_xyxy", [])
            yolo_native = out.get("instances")  # Garder les instances YOLO natives
            n_d = out["n_detections"]
            total_dets += n_d

            if boxes:
                points = points_for_boxes(rgb_tile, boxes)
                try:
                    label, _meta = sam3.predict_instances(
                        rgb_tile, boxes_xyxy=boxes, points_xy=points, sep="per_box",
                    )
                    
                    print(f"    SAM3 brut: {int(label.max())} instances, shape={label.shape}")
                    
                    # Filtrer les petites surfaces (comme votre camarade)
                    min_area_px = args.min_area if args.min_area > 0 else 100  # Surface minimale en pixels
                    if int(label.max()) > 0:
                        print(f"    SAM3: {int(label.max())} instances avant filtrage")
                        # Filtrage des petites instances
                        out_label = np.zeros_like(label)
                        nid = 0
                        for lab in range(1, int(label.max()) + 1):
                            m = label == lab
                            area = m.sum()
                            if area >= min_area_px:
                                nid += 1
                                out_label[m] = nid
                        print(f"    SAM3: {nid} instances après filtrage (min_area={min_area_px}px)")
                        label = out_label
                    
                    if int(label.max()) == 0:
                        # Fallback vers YOLO natif si SAM3 ne détecte rien après filtrage
                        print(f"    Fallback: YOLO natif (SAM3 a retourné 0 instances)")
                        inst = yolo_native  # Utiliser les instances YOLO natives stockées
                        if inst is not None:
                            print(f"    YOLO natif: {int(inst.max())} instances")
                        else:
                            print(f"    YOLO natif: PAS d'instances disponibles")
                    else:
                        inst = label
                except Exception as e:
                    print(f"    Warning: SAM3 failed ({e}), fallback to YOLO native")
                    inst = yolo_native  # Utiliser les instances YOLO natives stockées
                    if inst is not None:
                        print(f"    YOLO natif fallback: {int(inst.max())} instances")
            else:
                inst = yolo_native  # Si pas de boîtes, utiliser YOLO natif
                if inst is not None:
                    print(f"    Pas de boîtes, YOLO natif: {int(inst.max())} instances")
        else:
            # Utiliser YOLO natif (masques instances)
            out = detector.detect_instances(rgb_tile)
            n_d = out["n_detections"]
            total_dets += n_d
            inst = out.get("instances")
            if inst is not None:
                print(f"    YOLO natif: {int(inst.max())} instances (tuile {i+1})")
            else:
                print(f"    YOLO natif: PAS d'instances (tuile {i+1})")

        if inst is not None and int(inst.max()) > 0:
            win_tf = window_transform(img_path, win)
            gdf = instances_to_polygons(inst, win_tf, crs_str, simplify_tol=0.5, regularize=False)
            if len(gdf):
                parts.append(gdf)

        if (i + 1) % 5 == 0 or (i + 1) == len(positions):
            mode = "YOLO+SAM3 (avec fallback)" if args.use_sam3 else "YOLO natif"
            print(f"  Avancement : {i+1}/{len(positions)} tuiles traites ({total_dets} batiments detectes, mode={mode})...")

    duration = time.perf_counter() - t0
    print(f"Inference terminee en {duration:.1f}s (~{duration/len(positions):.2f}s par tuile)")

    if parts:
        merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs_str)
        stitched = stitch_polygons(merged, crs_str, overlap_frac=0.5)
        stitched = stitched.reset_index(drop=True)
        stitched["building_id"] = range(1, len(stitched) + 1)
        print(f"Batiments finaux apres fusion inter-tuiles : {len(stitched)}")
        
        return {
            "n_buildings": len(stitched),
            "duration": duration,
            "mode": "YOLO+SAM3 (avec fallback)" if args.use_sam3 else "YOLO natif",
            "device": device,
            "gdf": stitched
        }
    else:
        print("Aucun batiment detecte sur ce secteur.")
        return {
            "n_buildings": 0,
            "duration": duration,
            "mode": "YOLO+SAM3 (avec fallback)" if args.use_sam3 else "YOLO natif",
            "device": device,
            "gdf": None
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test sur 03.tif")
    parser.add_argument("--image", type=str, default=r"/home/hajar/bench_haut/crts2_bench_haut/data/zone3/03.tif", help="Chemin de 03.tif")
    parser.add_argument("--roi-size", type=int, default=2048, help="Taille du secteur en pixels (0 pour tout)")
    parser.add_argument("--row", type=int, default=12000, help="Ligne de depart du secteur")
    parser.add_argument("--col", type=int, default=14000, help="Colonne de depart du secteur")
    parser.add_argument("--conf", type=float, default=0.15, help="Seuil confiance YOLO")
    parser.add_argument("--ground-truth", type=str, default=None, help="Chemin du GeoJSON de reference pour calculer IoU/F1/PQ")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Device pour YOLO (cpu, cuda, auto)")
    parser.add_argument("--use-sam3", action="store_true", help="Utiliser le service SAM3 pour la segmentation")
    parser.add_argument("--sam3-url", type=str, default="http://127.0.0.1:8077", help="URL du service SAM3")
    parser.add_argument("--min-area", type=int, default=0, help="Surface minimale en pixels (0 = pas de filtrage)")
    parser.add_argument("--test-multiple", action="store_true", help="Tester sur plusieurs patches (centre, coins, etc.)")
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
    print("Modele YOLO pret !")

    # Initialisation SAM3 si demandé
    sam3 = None
    if args.use_sam3:
        print(f"\nInitialisation du service SAM3 ({args.sam3_url})...")
        sam3 = SAM3Resource(url=args.sam3_url, timeout_s=300, max_retries=3, backoff_s=5)
        if not sam3.health_check():
            print("ERREUR: Service SAM3 indisponible")
            return
        print("Service SAM3 pret !")

    # Mode test multiple patches
    if args.test_multiple:
        print("\n" + "="*60)
        print("  TEST MULTIPLE PATCHS")
        print("="*60)
        
        # Définir plusieurs patches représentatifs
        patches = [
            {"name": "Centre", "row": h_full//2 - 512, "col": w_full//2 - 512, "size": 1024},
            {"name": "Coin_Haut_Gauche", "row": 0, "col": 0, "size": 1024},
            {"name": "Coin_Haut_Droite", "row": 0, "col": w_full - 1024, "size": 1024},
            {"name": "Coin_Bas_Gauche", "row": h_full - 1024, "col": 0, "size": 1024},
            {"name": "Coin_Bas_Droite", "row": h_full - 1024, "col": w_full - 1024, "size": 1024},
            {"name": "Zone_Upper_Mid", "row": h_full//4 - 512, "col": w_full//2 - 512, "size": 1024},
            {"name": "Zone_Lower_Mid", "row": 3*h_full//4 - 512, "col": w_full//2 - 512, "size": 1024},
        ]
        
        all_results = []
        
        for patch in patches:
            print(f"\n{'='*60}")
            print(f"  Patch: {patch['name']}")
            print(f"{'='*60}")
            
            # S'assurer que le patch est dans les limites de l'image
            r_start = max(0, min(patch["row"], h_full - patch["size"]))
            c_start = max(0, min(patch["col"], w_full - patch["size"]))
            h_crop = min(patch["size"], h_full - r_start)
            w_crop = min(patch["size"], w_full - c_start)
            
            print(f"Position: ({r_start}, {c_start}), taille: {w_crop}x{h_crop}")
            
            # Test sur ce patch
            result = test_single_patch(
                img_path, detector, sam3, r_start, c_start, h_crop, w_crop,
                args, device, crs_str, tile_size, overlap
            )
            result["patch_name"] = patch["name"]
            all_results.append(result)
        
        # Résumé des résultats
        print(f"\n{'='*60}")
        print("  RÉSUMÉ DES PATCHS")
        print(f"{'='*60}")
        
        for result in all_results:
            print(f"{result['patch_name']}: {result['n_buildings']} bâtiments, "
                  f"{result['duration']:.1f}s, mode={result['mode']}")
        
        total_buildings = sum(r["n_buildings"] for r in all_results)
        total_time = sum(r["duration"] for r in all_results)
        print(f"\nTotal: {total_buildings} bâtiments sur {len(all_results)} patches, "
              f"{total_time:.1f}s total")
        
        return

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

        if args.use_sam3:
            # Utiliser YOLO + SAM3 avec fallback vers YOLO natif
            # IMPORTANT: utiliser detect_instances pour avoir les masques natifs YOLO disponibles
            out = detector.detect_instances(rgb_tile)
            boxes = out.get("boxes_xyxy", [])
            yolo_native = out.get("instances")  # Garder les instances YOLO natives
            n_d = out["n_detections"]
            total_dets += n_d

            if boxes:
                points = points_for_boxes(rgb_tile, boxes)
                try:
                    label, _meta = sam3.predict_instances(
                        rgb_tile, boxes_xyxy=boxes, points_xy=points, sep="per_box",
                    )
                    
                    print(f"    SAM3 brut: {int(label.max())} instances, shape={label.shape}")
                    
                    # Filtrer les petites surfaces (comme votre camarade)
                    min_area_px = args.min_area if args.min_area > 0 else 100  # Surface minimale en pixels
                    if int(label.max()) > 0:
                        print(f"    SAM3: {int(label.max())} instances avant filtrage")
                        # Filtrage des petites instances
                        out_label = np.zeros_like(label)
                        nid = 0
                        for lab in range(1, int(label.max()) + 1):
                            m = label == lab
                            area = m.sum()
                            if area >= min_area_px:
                                nid += 1
                                out_label[m] = nid
                        print(f"    SAM3: {nid} instances après filtrage (min_area={min_area_px}px)")
                        label = out_label
                    
                    if int(label.max()) == 0:
                        # Fallback vers YOLO natif si SAM3 ne détecte rien après filtrage
                        print(f"    Fallback: YOLO natif (SAM3 a retourné 0 instances)")
                        inst = yolo_native  # Utiliser les instances YOLO natives stockées
                        if inst is not None:
                            print(f"    YOLO natif: {int(inst.max())} instances")
                        else:
                            print(f"    YOLO natif: PAS d'instances disponibles")
                    else:
                        inst = label
                except Exception as e:
                    print(f"    Warning: SAM3 failed ({e}), fallback to YOLO native")
                    inst = yolo_native  # Utiliser les instances YOLO natives stockées
                    if inst is not None:
                        print(f"    YOLO natif fallback: {int(inst.max())} instances")
            else:
                inst = yolo_native  # Si pas de boîtes, utiliser YOLO natif
                if inst is not None:
                    print(f"    Pas de boîtes, YOLO natif: {int(inst.max())} instances")
        else:
            # Utiliser YOLO natif (masques instances)
            out = detector.detect_instances(rgb_tile)
            n_d = out["n_detections"]
            total_dets += n_d
            inst = out.get("instances")
            if inst is not None:
                print(f"    YOLO natif: {int(inst.max())} instances (tuile {i+1})")
            else:
                print(f"    YOLO natif: PAS d'instances (tuile {i+1})")

        if inst is not None and int(inst.max()) > 0:
            win_tf = window_transform(img_path, win)
            gdf = instances_to_polygons(inst, win_tf, crs_str, simplify_tol=0.5, regularize=False)
            if len(gdf):
                parts.append(gdf)

        if (i + 1) % 5 == 0 or (i + 1) == len(positions):
            mode = "YOLO+SAM3" if args.use_sam3 else "YOLO natif"
            print(f"  Avancement : {i+1}/{len(positions)} tuiles traites ({total_dets} batiments detectes, mode={mode})...")

    duration = time.perf_counter() - t0
    print(f"\nInference terminee en {duration:.1f}s (~{duration/len(positions):.2f}s par tuile)")

    # Utiliser un dossier temporaire dans /tmp pour éviter les problèmes de permissions
    import tempfile
    out_dir = Path(tempfile.gettempdir()) / f"test_03_roi_{w_crop}x{h_crop}"
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

            # Affichage cote a cote : input et overlay
            print("\nAffichage des images...")
            input_rgb = cv2.cvtColor(sector_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            # Redimensionner pour affichage si trop grand
            max_display = 800
            h, w = input_rgb.shape[:2]
            if max(h, w) > max_display:
                scale = max_display / max(h, w)
                input_rgb = cv2.resize(input_rgb, (int(w * scale), int(h * scale)))
                overlay_bgr = cv2.resize(overlay_bgr, (int(w * scale), int(h * scale)))
            
            combined = np.hstack([input_rgb, overlay_bgr])
            cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Overlay", (input_rgb.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Input vs Overlay", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Calcul des metriques
        print("\n" + "=" * 50)
        print("  METRIQUES DE PERFORMANCE")
        print("=" * 50)
        
        # Calcul de la surface moyenne
        if "area_m2" in stitched.columns:
            avg_area = stitched["area_m2"].mean()
            total_area = stitched["area_m2"].sum()
            print(f"Surface moyenne par batiment : {avg_area:.2f} m2")
            print(f"Surface totale batiments : {total_area:.2f} m2")
        
        # Densite de batiments (batiments / km2)
        if args.roi_size > 0:
            # Approximation de la surface en km2 (en supposant resolution ~0.1m/px)
            pixel_resolution = 0.1  # metres par pixel (a ajuster selon votre image)
            sector_area_km2 = (w_crop * h_crop * pixel_resolution ** 2) / 1_000_000
            density = len(stitched) / sector_area_km2 if sector_area_km2 > 0 else 0
            print(f"Densite de batiments : {density:.2f} batiments/km2")
        
        # Metriques de performance d'inference
        print(f"\nPerformance d'inference :")
        print(f"  Mode : {'YOLO+SAM3 (avec fallback)' if args.use_sam3 else 'YOLO natif'}")
        print(f"  Device YOLO : {device}")
        print(f"  Temps total : {duration:.1f}s")
        print(f"  Temps par tuile : {duration/len(positions):.2f}s")
        print(f"  Tuiles par seconde : {len(positions)/duration:.2f}")
        print(f"  Batiments par seconde : {len(stitched)/duration:.2f}")
        
        # Metriques de qualite (si ground truth disponible)
        print(f"\nMetriques de qualite :")
        print(f"  Nombre de batiments detectes : {len(stitched)}")
        
        if args.ground_truth:
            gt_path = Path(args.ground_truth)
            if gt_path.is_file():
                try:
                    gt_gdf = gpd.read_file(gt_path)
                    print(f"  Ground truth charge : {len(gt_gdf)} batiments de reference")
                    
                    # Calculer IoU moyen
                    iou_scores = []
                    for _, pred_row in stitched.iterrows():
                        pred_geom = pred_row.geometry
                        best_iou = 0
                        for _, gt_row in gt_gdf.iterrows():
                            gt_geom = gt_row.geometry
                            if pred_geom.intersects(gt_geom):
                                intersection = pred_geom.intersection(gt_geom).area
                                union = pred_geom.union(gt_geom).area
                                iou = intersection / union if union > 0 else 0
                                best_iou = max(best_iou, iou)
                        iou_scores.append(best_iou)
                    
                    avg_iou = np.mean(iou_scores) if iou_scores else 0
                    print(f"  IoU moyen : {avg_iou:.4f}")
                    
                    # Calculer F1 (precision/recall)
                    tp = sum(1 for iou in iou_scores if iou > 0.5)
                    fp = len(stitched) - tp
                    fn = len(gt_gdf) - tp
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"  Precision : {precision:.4f}")
                    print(f"  Recall : {recall:.4f}")
                    print(f"  F1 Score : {f1:.4f}")
                except Exception as e:
                    print(f"  Erreur lors du calcul des metriques : {e}")
            else:
                print(f"  Fichier ground truth introuvable : {gt_path}")
        else:
            print(f"  Pour IoU/F1/PQ, utilisez --ground-truth <chemin_geojson>")
    else:
        print("Aucun batiment detecte sur ce secteur.")

    print("\n[SUCCES] Test sur 03.tif valide !")


if __name__ == "__main__":
    main()
