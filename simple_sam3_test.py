#!/usr/bin/env python3
"""
Script simple pour tester SAM3 et créer un overlay input vs output.
Se concentre sur le décodage correct du masque.
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import requests


def encode_image_to_base64(image_path):
    """Encode une image en base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_sam3_and_create_overlay(image_path, sam3_url="http://127.0.0.1:8077"):
    """Test SAM3 et crée un overlay simple."""
    print(f"🧪 Test SAM3 sur : {image_path}")
    
    # Charger l'image originale
    original = cv2.imread(image_path)
    h, w = original.shape[:2]
    print(f"   Image : {w}x{h}")
    
    # Encoder l'image
    image_b64 = encode_image_to_base64(image_path)
    
    # Test avec boîte couvrant toute l'image
    payload = {
        "image": image_b64,
        "prompt": "building",
        "conf": 0.25,
        "sep": "per_box",
        "boxes_xyxy": [[0, 0, w, h]],
        "points_xy": [[w//2, h//2]]
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
        
        print(f"✅ Réponse reçue")
        print(f"   Shape : {result['shape']}")
        print(f"   Instances : {result['n_instances']}")
        print(f"   Latence : {result['latency_ms']:.1f} ms")
        
        # Décoder le masque
        label_b64 = result['label_b64']
        print(f"📷 Décodage masque ({len(label_b64)} caractères)...")
        
        # Essayer différents décodages
        try:
            # Méthode 1: Décode base64 puis numpy frombuffer
            img_bytes = base64.b64decode(label_b64)
            print(f"   Bytes décodés : {len(img_bytes)} octets")
            
            # Essayer comme int32
            instances_mask = np.frombuffer(img_bytes, dtype=np.int32)
            print(f"   Shape int32 : {instances_mask.shape}")
            print(f"   Valeurs uniques : {np.unique(instances_mask)}")
            
            # Reshape selon la shape retournée
            instances_mask = instances_mask.reshape(result['shape'])
            print(f"   Shape après reshape : {instances_mask.shape}")
            
        except Exception as e:
            print(f"   ❌ Erreur décodage int32 : {e}")
            
            # Essayer comme uint8
            try:
                instances_mask = np.frombuffer(img_bytes, dtype=np.uint8)
                print(f"   Shape uint8 : {instances_mask.shape}")
                instances_mask = instances_mask.reshape(result['shape'])
                print(f"   Shape après reshape : {instances_mask.shape}")
                print(f"   Valeurs uniques : {np.unique(instances_mask)}")
            except Exception as e2:
                print(f"   ❌ Erreur décodage uint8 : {e2}")
                return None
        
        # Créer l'overlay
        print("🎨 Création overlay...")
        overlay = original.copy()
        
        # Compter les instances
        unique_labels = np.unique(instances_mask)
        n_instances = len([l for l in unique_labels if l > 0])
        print(f"   Instances détectées : {n_instances}")
        
        if n_instances == 0:
            print("⚠️  Aucune instance détectée (que des 0)")
            cv2.putText(overlay, "NO BUILDINGS DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Créer un masque coloré
            color_mask = np.zeros_like(original)
            
            # Pour chaque instance
            for label in unique_labels:
                if label == 0:  # Fond
                    continue
                    
                mask = (instances_mask == label).astype(np.uint8)
                
                # Couleur orange
                color = [249, 115, 22]  # RGB
                color_mask[mask == 1] = color
            
            # Fusionner avec transparence
            alpha = 0.48
            beta = 1 - alpha
            overlay = cv2.addWeighted(original, alpha, color_mask, beta, 0)
            
            # Ajouter contours
            for label in unique_labels:
                if label == 0:
                    continue
                    
                mask = (instances_mask == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (239, 68, 68), 2)  # Rouge
            
            cv2.putText(overlay, f"SAM3: {n_instances} buildings", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Sauvegarder
        output_path = Path(image_path).stem + "_sam3_simple_overlay.png"
        cv2.imwrite(output_path, overlay)
        print(f"✅ Overlay sauvegardé : {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response : {e.response.text}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_sam3_test.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"❌ Image introuvable : {image_path}")
        sys.exit(1)
    
    test_sam3_and_create_overlay(image_path)