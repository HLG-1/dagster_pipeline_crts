#!/usr/bin/env python3
"""
Script final de comparaison : Input vs Output du pipeline complet.
Montre l'image originale et l'overlay final généré par le pipeline.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def create_final_comparison(input_path, overlay_path, output_path):
    """Crée une comparaison côte à côte : input vs output."""
    print(f"🎨 Création comparaison finale : {output_path}")
    
    # Charger les images
    input_img = cv2.imread(input_path)
    overlay_img = cv2.imread(overlay_path)
    
    if input_img is None:
        print(f"❌ Impossible de charger : {input_path}")
        return
    
    if overlay_img is None:
        print(f"❌ Impossible de charger : {overlay_path}")
        return
    
    # Vérifier les dimensions
    h1, w1 = input_img.shape[:2]
    h2, w2 = overlay_img.shape[:2]
    
    print(f"   Input : {w1}x{h1}")
    print(f"   Overlay : {w2}x{h2}")
    
    # Redimensionner si nécessaire
    if h1 != h2 or w1 != w2:
        print(f"   Redimensionnement overlay vers {w1}x{h1}")
        overlay_img = cv2.resize(overlay_img, (w1, h1))
    
    # Créer l'image côte à côte
    comparison = np.hstack([input_img, overlay_img])
    
    # Ajouter des labels
    cv2.putText(comparison, "INPUT", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "PIPELINE OUTPUT", (w1 + 20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Ligne de séparation
    cv2.line(comparison, (w1, 0), (w1, h1), (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"✅ Comparaison sauvegardée : {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python final_comparison.py <input_image> <overlay_image>")
        print("Exemple: python final_comparison.py data/incoming/zone1_00007_IMG.png results/zone1_00007_IMG__timestamp/zone1_00007_IMG_overlay.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    overlay_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"❌ Input introuvable : {input_path}")
        sys.exit(1)
    
    if not Path(overlay_path).exists():
        print(f"❌ Overlay introuvable : {overlay_path}")
        sys.exit(1)
    
    output_path = Path(input_path).stem + "_final_comparison.png"
    create_final_comparison(input_path, overlay_path, output_path)
    
    print(f"\n📁 Résultat : {output_path}")


if __name__ == "__main__":
    main()