"""
Test rapide d'inférence YOLO sur CPU (sans GPU CUDA ni service SAM3).

Usage :
    python scripts/test_yolo_cpu.py
    python scripts/test_yolo_cpu.py --image chemin/vers/image.tif
"""
import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image

from pipeline_batiments.yolo11.config import YOLO11Config
from pipeline_batiments.yolo11.detect import YOLODetector


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test YOLO sur CPU")
    parser.add_argument("--image", type=str, default=None, help="Chemin vers une image ou un GeoTIFF")
    parser.add_argument("--weights", type=str, default="checkpoints/yolo11/best.pt", help="Chemin des poids YOLO")
    parser.add_argument("--conf", type=float, default=0.15, help="Seuil de confiance (ex: 0.15)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        alt = ROOT / "finetunig_portable" / "checkpoints" / "pretrained" / "yolov8n_building.pt"
        if alt.is_file():
            weights_path = alt

    print(f"=== Test Inference YOLO (Mode CPU) ===")
    print(f"Poids : {weights_path}")
    print(f"Device: CPU")
    print(f"Conf  : {args.conf}")

    cfg = YOLO11Config(
        pretrained_weights=weights_path,
        best_weights=weights_path,
        model=str(weights_path),
        device="cpu",
        infer_imgsz=512,
        conf=args.conf,
    )

    print("\nChargement du modele...")
    detector = YOLODetector(cfg)
    print("Modele charge avec succes !")

    # Image de test
    if args.image and Path(args.image).is_file():
        img_path = Path(args.image)
        if img_path.suffix.lower() in [".tif", ".tiff"]:
            import rasterio
            with rasterio.open(img_path) as src:
                arr = src.read()
                if arr.shape[0] >= 3:
                    rgb = np.transpose(arr[:3], (1, 2, 0))
                else:
                    rgb = np.stack([arr[0]] * 3, axis=-1)
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb = np.asarray(Image.open(img_path).convert("RGB"))
    else:
        sample = ROOT / "finetunig_portable" / "third_party" / "sam3" / "assets" / "images" / "test_image.jpg"
        if sample.is_file():
            rgb = np.asarray(Image.open(sample).convert("RGB"))
        else:
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)

    h, w = rgb.shape[:2]
    print(f"Image  : {w}x{h} px")

    print("Lancement de l'inference...")
    res = detector.detect_instances(rgb)

    print("\n=== Resultats ===")
    print(f"Temps d'inference : {res['latency_ms']} ms (~{res['latency_ms']/1000:.2f}s)")
    print(f"Nombre de boites  : {len(res['boxes_xyxy'])}")
    print(f"Masques detectes  : {res['n_detections']}")

    out_dir = ROOT / "results" / "test_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "test_yolo_cpu_overlay.jpg"
    
    vis = rgb.copy()
    for box in res.get("boxes_xyxy", []):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 128), 2)
    Image.fromarray(vis).save(out_img)

    print(f"Apercu sauvegarde dans : {out_img}")
    print("\n[OK] YOLO fonctionne sur CPU sans GPU !")


if __name__ == "__main__":
    main()
