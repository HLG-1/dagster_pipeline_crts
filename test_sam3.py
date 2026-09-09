#!/usr/bin/env python3
"""
Script de test simple pour diagnostiquer le problème SAM3.
Compare l'entrée vs l'overlay pour voir la détection.
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests


def encode_image_to_base64(image_path):
    """Encode une image en base64 pour l'envoi à SAM3."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_image(b64_string):
    """Décode une image base64 en tableau numpy."""
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)


def test_sam3_detection(image_path, sam3_url="http://127.0.0.1:8077"):
    """Test la détection SAM3 sur une image."""
    print(f"🧪 Test SAM3 sur : {image_path}")
    print(f"🌐 URL SAM3 : {sam3_url}")
    
    # Vérifier que le service est disponible
    try:
        response = requests.get(f"{sam3_url}/health", timeout=5)
        health = response.json()
        print(f"✅ SAM3 Health : {health}")
    except Exception as e:
        print(f"❌ Erreur connexion SAM3 : {e}")
        return False
    
    # Encoder l'image
    print("📷 Encodage image...")
    image_b64 = encode_image_to_base64(image_path)
    
    # Préparer le payload
    payload = {
        "image": image_b64,
        "prompt": "building",
        "conf": 0.25,
        "sep": "grouped",  # Test sans boîtes pour voir la détection native
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
        print(f"✅ Réponse SAM3 reçue")
        print(f"   Shape : {result['shape']}")
        print(f"   Instances : {len(result.get('instances', []))}")
        print(f"   Latence : {result.get('latency_ms', 0):.1f} ms")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur requête SAM3 : {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response : {e.response.text}")
        return False


def test_sam3_with_boxes(image_path, sam3_url="http://127.0.0.1:8077"):
    """Test SAM3 avec des boîtes de détection simulées."""
    print(f"\n🧪 Test SAM3 avec boîtes sur : {image_path}")
    
    # Charger l'image pour obtenir ses dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"   Dimensions image : {w}x{h}")
    
    # Encoder l'image
    image_b64 = encode_image_to_base64(image_path)
    
    # Simuler une boîte couvrant tout l'image (pour tester)
    boxes = [[0, 0, w, h]]
    points = [[w//2, h//2]]  # Point central
    
    payload = {
        "image": image_b64,
        "prompt": "building",
        "conf": 0.25,
        "sep": "per_box",
        "boxes_xyxy": boxes,
        "points_xy": points
    }
    
    print("🚀 Envoi requête SAM3 avec boîte...")
    try:
        response = requests.post(
            f"{sam3_url}/predict_instances",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Réponse SAM3 reçue")
        print(f"   Shape : {result['shape']}")
        print(f"   Instances : {len(result.get('instances', []))}")
        print(f"   Latence : {result.get('latency_ms', 0):.1f} ms")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur requête SAM3 : {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response : {e.response.text}")
        return False


def create_overlay(image_path, sam3_result, output_path):
    """Crée un overlay visuel pour comparer input vs output."""
    print(f"🎨 Création overlay : {output_path}")
    
    # Charger l'image originale
    original = cv2.imread(image_path)
    overlay = original.copy()
    
    # Décoder le masque d'instances
    label_b64 = sam3_result.get('label_b64')
    if not label_b64:
        print("⚠️  Pas de masque dans la réponse SAM3")
        cv2.putText(overlay, "NO MASK DATA", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, overlay)
        return None
        
    instances_mask = decode_base64_to_image(label_b64)
    
    if instances_mask is None:
        print("⚠️  Impossible de décoder le masque")
        cv2.putText(overlay, "MASK DECODE ERROR", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, overlay)
        return None
        
    print(f"   Masque shape : {instances_mask.shape}")
    print(f"   Valeurs uniques : {np.unique(instances_mask)}")
    
    # Créer l'overlay
    overlay = original.copy()
    
    # Pour chaque instance (sauf 0 qui est le fond)
    unique_labels = np.unique(instances_mask)
    n_instances = len([l for l in unique_labels if l > 0])
    print(f"   Nombre d'instances détectées : {n_instances}")
    
    if n_instances == 0:
        print("⚠️  Aucune instance détectée par SAM3 !")
        # Sauvegarder l'image originale avec un texte d'erreur
        cv2.putText(overlay, "NO DETECTION", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, overlay)
        return None
    
    # Créer un masque coloré
    color_mask = np.zeros_like(original)
    
    # Appliquer une couleur semi-transparente pour chaque instance
    for label in unique_labels:
        if label == 0:  # Ignorer le fond
            continue
            
        mask = (instances_mask == label).astype(np.uint8)
        
        # Couleur orange (comme dans le pipeline)
        color = [249, 115, 22]  # RGB = #F97316
        
        # Appliquer la couleur au masque
        color_mask[mask == 1] = color
    
    # Fusionner avec transparence
    alpha = 0.48
    beta = 1 - alpha
    overlay = cv2.addWeighted(original, alpha, color_mask, beta, 0)
    
    # Ajouter les contours
    for label in unique_labels:
        if label == 0:
            continue
            
        mask = (instances_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Couleur rouge pour les contours
        cv2.drawContours(overlay, contours, -1, (239, 68, 68), 2)  # RGB = #EF4444
    
    # Ajouter le nombre de bâtiments détectés
    cv2.putText(overlay, f"SAM3: {n_instances} buildings", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, overlay)
    print(f"✅ Overlay sauvegardé : {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_sam3.py <image_path>")
        print("Exemple: python test_sam3.py data/incoming/zone1_00007_IMG.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    sam3_url = "http://127.0.0.1:8077"
    
    if not Path(image_path).exists():
        print(f"❌ Image introuvable : {image_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TEST 1 : Détection SAM3 native (sans boîtes)")
    print("=" * 60)
    result1 = test_sam3_detection(image_path, sam3_url)
    
    if result1:
        output1 = Path(image_path).stem + "_sam3_native_overlay.png"
        result_overlay1 = create_overlay(image_path, result1, output1)
    
    print("\n" + "=" * 60)
    print("TEST 2 : Détection SAM3 avec boîte (simulation)")
    print("=" * 60)
    result2 = test_sam3_with_boxes(image_path, sam3_url)
    
    if result2:
        output2 = Path(image_path).stem + "_sam3_box_overlay.png"
        result_overlay2 = create_overlay(image_path, result2, output2)
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"Image : {image_path}")
    if result1:
        print(f"Test 1 (native) : {len([i for i in result1.get('instances', [])])} instances")
    if result2:
        print(f"Test 2 (avec boîte) : {len([i for i in result2.get('instances', [])])} instances")
    
    print("\n📁 Fichiers générés :")
    if result1 and result_overlay1:
        print(f"   - {output1}")
    if result2 and result_overlay2:
        print(f"   - {output2}")


if __name__ == "__main__":
    main()