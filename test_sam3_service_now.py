#!/usr/bin/env python3
"""
Test direct du service SAM3 HTTP pour vérifier s'il fonctionne.
"""
import base64
import io
import json
import urllib.request
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration
SAM3_URL = "http://localhost:8077"
TEST_IMAGE = Path("/home/hajar/dagster_pipeline_crts/data/incoming/zone1_00007_IMG.png")

print(f"🔌 Test du service SAM3: {SAM3_URL}")

# 1. Health check
print("\n1️⃣ Health check...")
try:
    with urllib.request.urlopen(f"{SAM3_URL}/health", timeout=5) as r:
        health = json.loads(r.read())
        print(f"✓ Service prêt: {health.get('ready')}")
        print(f"✓ Backend: {health.get('backend')}")
        print(f"✓ Variant: {health.get('variant')}")
        print(f"✓ FT Checkpoint: {health.get('ft_checkpoint')}")
        print(f"✓ Device: {health.get('device')}")
except Exception as e:
    print(f"✗ Erreur health check: {e}")
    exit(1)

# 2. Charger une image de test
print(f"\n2️⃣ Chargement de l'image de test: {TEST_IMAGE}")
if not TEST_IMAGE.exists():
    print(f"✗ Image non trouvée, création d'une image synthétique")
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_pil = Image.fromarray(img_array)
else:
    print(f"✓ Image trouvée")
    test_pil = Image.open(TEST_IMAGE).convert("RGB")
    img_array = np.array(test_pil)

print(f"📷 Dimensions: {img_array.shape}")

# 3. Encoder l'image en base64
print("\n3️⃣ Encodage de l'image...")
buf = io.BytesIO()
test_pil.save(buf, format="PNG")
img_b64 = base64.b64encode(buf.getvalue()).decode()
print(f"✓ Image encodée ({len(img_b64)} caractères)")

# 4. Test de segmentation avec prompt texte
print("\n4️⃣ Test de segmentation (prompt texte)...")
payload = {
    "image": img_b64,
    "prompt": "building",
    "conf": 0.25,
    "sep": "grouped"
}

try:
    req = urllib.request.Request(
        f"{SAM3_URL}/predict_instances",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    
    print(f"✓ Segmentation réussie:")
    print(f"  - Forme: {result.get('shape')}")
    print(f"  - Instances: {result.get('n_instances')}")
    print(f"  - Méthode: {result.get('sep')}")
    print(f"  - Latence: {result.get('latency_ms')}ms")
    
    instances = result.get('instances', [])
    if instances:
        print(f"  - Première instance: {instances[0]}")
    
    # 5. Test avec des boîtes (simulation YOLO)
    print("\n5️⃣ Test de segmentation avec boîtes (simulation YOLO)...")
    # Créer quelques boîtes fictives
    h, w = img_array.shape[:2]
    boxes = [
        [50, 50, 150, 150],
        [200, 100, 300, 200],
        [100, 200, 250, 350],
    ]
    
    payload_boxes = {
        "image": img_b64,
        "prompt": "building",
        "conf": 0.25,
        "sep": "per_box",
        "boxes_xyxy": boxes
    }
    
    req_boxes = urllib.request.Request(
        f"{SAM3_URL}/predict_instances",
        data=json.dumps(payload_boxes).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_boxes, timeout=30) as r:
        result_boxes = json.loads(r.read())
    
    print(f"✓ Segmentation avec boîtes réussie:")
    print(f"  - Instances: {result_boxes.get('n_instances')}")
    print(f"  - Latence: {result_boxes.get('latency_ms')}ms")
    
    instances_boxes = result_boxes.get('instances', [])
    if instances_boxes:
        print(f"  - Détail instances:")
        for i, inst in enumerate(instances_boxes[:3]):
            print(f"    Instance {i+1}: score={inst.get('score'):.3f}, box={inst.get('box')}")
    
    print("\n✅ Tests réussis ! Le service SAM3 fonctionne.")
    print(f"⚠️  Note: Le checkpoint FT est {'CHARGÉ' if health.get('ft_checkpoint') else 'NON CHARGÉ'}")
    
except Exception as e:
    print(f"✗ Erreur segmentation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)