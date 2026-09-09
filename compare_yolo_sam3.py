#!/usr/bin/env python3
"""
Script de comparaison YOLO vs SAM3 pour diagnostiquer le problème.
Teste :
1. YOLO seul (détection native)
2. SAM3 avec les boîtes YOLO
3. Génère des overlays comparatifs
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO


def encode_image_to_base64(image_path):
    """Encode une image en base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_yolo_detection(image_path, weights_path="checkpoints/yolo11/best.pt"):
    """Test la détection YOLO seule."""
    print(f"🧪 Test YOLO sur : {image_path}")
    print(f"   Poids : {weights_path}")
    
    # Charger le modèle YOLO
    model = YOLO(weights_path)
    
    # Inférence
    results = model(image_path, conf=0.20, iou=0.45, imgsz=512)
    
    # Extraire les boîtes
    boxes = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            conf = box.conf[0].cpu().numpy()
            boxes.append({
                'box': xyxy.tolist(),
                'conf': float(conf)
            })
    
    print(f"✅ YOLO : {len(boxes)} boîtes détectées")
    for i, box in enumerate(boxes):
        print(f"   Boîte {i+1}: {box['box']}, conf={box['conf']:.3f}")
    
    return boxes


def test_sam3_with_yolo_boxes(image_path, yolo_boxes, sam3_url="http://127.0.0.1:8077"):
    """Test SAM3 avec les boîtes YOLO."""
    print(f"\n🧪 Test SAM3 avec {len(yolo_boxes)} boîtes YOLO")
    
    # Charger l'image pour dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Encoder l'image
    image_b64 = encode_image_to_base64(image_path)
    
    # Préparer les boîtes et points pour SAM3
    boxes_xyxy = [box['box'] for box in yolo_boxes]
    
    # Calculer les points centroïdes pour chaque boîte
    points_xy = []
    for box in yolo_boxes:
        x1, y1, x2, y2 = box['box']
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        points_xy.append([cx, cy])
    
    print(f"   Boîtes : {len(boxes_xyxy)}")
    print(f"   Points : {len(points_xy)}")
    
    payload = {
        "image": image_b64,
        "prompt": "building",
        "conf": 0.25,
        "sep": "per_box",
        "boxes_xyxy": boxes_xyxy,
        "points_xy": points_xy
    }
    
    print("🚀 Envoi requête SAM3...")
    try:
        response = requests.post(
            f"{sam3_url}/predict_instances",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ SAM3 : {result['n_instances']} instances")
        print(f"   Latence : {result['latency_ms']:.1f} ms")
        
        # Décoder le masque
        label_b64 = result['label_b64']
        img_bytes = base64.b64decode(label_b64)
        instances_mask = np.frombuffer(img_bytes, dtype=np.int32)
        instances_mask = instances_mask.reshape(result['shape'])
        
        return instances_mask, result
        
    except Exception as e:
        print(f"❌ Erreur SAM3 : {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response : {e.response.text}")
        return None, None


def create_yolo_overlay(image_path, yolo_boxes, output_path):
    """Crée un overlay avec les boîtes YOLO."""
    print(f"🎨 Création overlay YOLO : {output_path}")
    
    original = cv2.imread(image_path)
    overlay = original.copy()
    
    # Dessiner chaque boîte
    for i, box_info in enumerate(yolo_boxes):
        box = box_info['box']
        conf = box_info['conf']
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Rectangle rouge
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (239, 68, 68), 2)
        
        # Label avec confiance
        label = f"#{i+1}: {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (239, 68, 68), 2)
    
    cv2.putText(overlay, f"YOLO: {len(yolo_boxes)} detections", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, overlay)
    print(f"✅ Overlay YOLO sauvegardé")


def create_sam3_overlay(image_path, instances_mask, output_path):
    """Crée un overlay avec les masques SAM3."""
    print(f"🎨 Création overlay SAM3 : {output_path}")
    
    original = cv2.imread(image_path)
    overlay = original.copy()
    
    # Compter les instances
    unique_labels = np.unique(instances_mask)
    n_instances = len([l for l in unique_labels if l > 0])
    print(f"   Instances : {n_instances}")
    
    if n_instances == 0:
        cv2.putText(overlay, "NO BUILDINGS DETECTED", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Masque coloré
        color_mask = np.zeros_like(original)
        
        for label in unique_labels:
            if label == 0:
                continue
                
            mask = (instances_mask == label).astype(np.uint8)
            color = [249, 115, 22]  # Orange
            color_mask[mask == 1] = color
        
        # Fusion
        alpha = 0.48
        beta = 1 - alpha
        overlay = cv2.addWeighted(original, alpha, color_mask, beta, 0)
        
        # Contours
        for label in unique_labels:
            if label == 0:
                continue
                
            mask = (instances_mask == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (239, 68, 68), 2)
        
        cv2.putText(overlay, f"SAM3: {n_instances} buildings", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, overlay)
    print(f"✅ Overlay SAM3 sauvegardé")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_yolo_sam3.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"❌ Image introuvable : {image_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("COMPARAISON YOLO vs SAM3")
    print("=" * 60)
    
    # Test YOLO
    yolo_boxes = test_yolo_detection(image_path)
    
    if yolo_boxes:
        # Overlay YOLO
        yolo_output = Path(image_path).stem + "_yolo_overlay.png"
        create_yolo_overlay(image_path, yolo_boxes, yolo_output)
    
    # Test SAM3 avec boîtes YOLO
    if yolo_boxes:
        sam3_mask, sam3_result = test_sam3_with_yolo_boxes(image_path, yolo_boxes)
        
        if sam3_mask is not None:
            sam3_output = Path(image_path).stem + "_sam3_with_yolo_overlay.png"
            create_sam3_overlay(image_path, sam3_mask, sam3_output)
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"Image : {image_path}")
    print(f"YOLO détecte : {len(yolo_boxes)} boîtes")
    if yolo_boxes and sam3_mask is not None:
        unique_labels = np.unique(sam3_mask)
        sam3_count = len([l for l in unique_labels if l > 0])
        print(f"SAM3 avec boîtes YOLO : {sam3_count} instances")
        print(f"Ratio : {sam3_count}/{len(yolo_boxes)} = {sam3_count/len(yolo_boxes):.1%}")
    
    print("\n📁 Fichiers générés :")
    if yolo_boxes:
        print(f"   - {Path(image_path).stem}_yolo_overlay.png")
    if yolo_boxes and sam3_mask is not None:
        print(f"   - {Path(image_path).stem}_sam3_with_yolo_overlay.png")


if __name__ == "__main__":
    main()