#!/usr/bin/env python3
"""
Test pour comparer YOLO direct vs YOLO via pipeline (read_window_rgb).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline_batiments.core.geo_io import read_window_rgb, raster_meta
from rasterio.windows import Window


def test_yolo_direct(image_path):
    """Test YOLO en lecture directe (comme notre script de test)."""
    print(f"🧪 Test YOLO direct sur : {image_path}")
    
    model = YOLO("checkpoints/yolo11/best.pt")
    results = model(image_path, conf=0.20, iou=0.45, imgsz=512)
    
    boxes = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            boxes.append({
                'box': xyxy.tolist(),
                'conf': float(conf)
            })
    
    print(f"✅ YOLO direct : {len(boxes)} boîtes")
    return boxes


def test_yolo_via_pipeline(image_path):
    """Test YOLO via la méthode du pipeline (read_window_rgb)."""
    print(f"\n🧪 Test YOLO via pipeline sur : {image_path}")
    
    # Charger les métadonnées
    meta = raster_meta(image_path)
    print(f"   Métadonnées : {meta['width']}x{meta['height']}")
    
    # Créer une fenêtre couvrant toute l'image
    window = Window(0, 0, meta['width'], meta['height'])
    
    # Lire via read_window_rgb
    rgb = read_window_rgb(image_path, window)
    print(f"   Image lue : {rgb.shape}, dtype={rgb.dtype}")
    
    # Test YOLO sur cette image
    model = YOLO("checkpoints/yolo11/best.pt")
    results = model(rgb, conf=0.20, iou=0.45, imgsz=512)
    
    boxes = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            boxes.append({
                'box': xyxy.tolist(),
                'conf': float(conf)
            })
    
    print(f"✅ YOLO via pipeline : {len(boxes)} boîtes")
    return boxes


def test_yolo_opencv_vs_rasterio(image_path):
    """Compare lecture OpenCV vs rasterio.read_window_rgb."""
    print(f"\n🧪 Comparaison lecture OpenCV vs rasterio : {image_path}")
    
    # Lecture OpenCV
    img_cv = cv2.imread(image_path)
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    print(f"   OpenCV : {img_cv_rgb.shape}, dtype={img_cv_rgb.dtype}")
    
    # Lecture rasterio
    meta = raster_meta(image_path)
    window = Window(0, 0, meta['width'], meta['height'])
    img_rasterio = read_window_rgb(image_path, window)
    print(f"   Rasterio : {img_rasterio.shape}, dtype={img_rasterio.dtype}")
    
    # Comparer
    if img_cv_rgb.shape == img_rasterio.shape:
        diff = np.abs(img_cv_rgb.astype(float) - img_rasterio.astype(float))
        max_diff = diff.max()
        print(f"   Différence max : {max_diff}")
        if max_diff < 5:
            print("   ✅ Images identiques (différence < 5)")
        else:
            print(f"   ⚠️  Images différentes (différence = {max_diff})")
    else:
        print(f"   ❌ Shapes différents : {img_cv_rgb.shape} vs {img_rasterio.shape}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_yolo_pipeline_vs_direct.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"❌ Image introuvable : {image_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("COMPARAISON YOLO DIRECT vs PIPELINE")
    print("=" * 60)
    
    # Test direct
    boxes_direct = test_yolo_direct(image_path)
    
    # Test via pipeline
    boxes_pipeline = test_yolo_via_pipeline(image_path)
    
    # Comparaison lecture
    test_yolo_opencv_vs_rasterio(image_path)
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"YOLO direct : {len(boxes_direct)} boîtes")
    print(f"YOLO pipeline : {len(boxes_pipeline)} boîtes")
    
    if len(boxes_direct) != len(boxes_pipeline):
        print(f"⚠️  DIFFÉRENCE DÉTECTÉE : {abs(len(boxes_direct) - len(boxes_pipeline))} boîtes")
    else:
        print("✅ Résultats identiques")


if __name__ == "__main__":
    main()