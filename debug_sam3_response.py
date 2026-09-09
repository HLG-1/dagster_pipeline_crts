#!/usr/bin/env python3
"""
Script pour déboguer la structure de réponse SAM3.
"""
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image_to_base64(image_path):
    """Encode une image en base64 pour l'envoi à SAM3."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def debug_sam3_response(image_path, sam3_url="http://127.0.0.1:8077"):
    """Examine la structure complète de la réponse SAM3."""
    print(f"🔍 Débogage réponse SAM3 pour : {image_path}")
    
    # Encoder l'image
    image_b64 = encode_image_to_base64(image_path)
    
    # Test avec boîte
    payload = {
        "image": image_b64,
        "prompt": "building",
        "conf": 0.25,
        "sep": "per_box",
        "boxes_xyxy": [[0, 0, 512, 512]],
        "points_xy": [[256, 256]]
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
        
        print("\n📋 Structure complète de la réponse :")
        print(json.dumps(result, indent=2, default=str))
        
        print("\n🔑 Clés disponibles :")
        for key in result.keys():
            value = result[key]
            if isinstance(value, (str, int, float, bool)):
                print(f"   {key}: {value}")
            elif isinstance(value, list):
                print(f"   {key}: list de {len(value)} éléments")
                if key == 'instances' and len(value) > 0:
                    print(f"      Premier élément : {value[0]}")
            elif isinstance(value, dict):
                print(f"   {key}: dict avec {len(value)} clés")
            else:
                print(f"   {key}: {type(value)}")
        
        # Vérifier spécifiquement label_b64
        if 'label_b64' in result:
            label_b64 = result['label_b64']
            print(f"\n🔍 label_b64 présent : {label_b64 is not None}")
            if label_b64:
                print(f"   Longueur : {len(label_b64)} caractères")
                print(f"   Début : {label_b64[:50]}...")
            else:
                print("   ⚠️ label_b64 est None ou vide !")
        else:
            print("\n⚠️ Clé 'label_b64' absente de la réponse !")
            
    except Exception as e:
        print(f"❌ Erreur : {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response : {e.response.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_sam3_response.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"❌ Image introuvable : {image_path}")
        sys.exit(1)
    
    debug_sam3_response(image_path)