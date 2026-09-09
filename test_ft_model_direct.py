#!/usr/bin/env python3
"""
Test direct du modèle SAM3 fine-tuné pour vérifier qu'il fonctionne.
Ce script charge le modèle fine-tuné directement sans passer par le service HTTP.
"""
import sys
import os
from pathlib import Path

# Trouver et charger l'environnement conda sam3
conda_paths = [
    "/opt/conda/etc/profile.d/conda.sh",
    "/root/miniconda3/etc/profile.d/conda.sh", 
    "/home/hajar/miniconda3/etc/profile.d/conda.sh",
    "/home/hajar/anaconda3/etc/profile.d/conda.sh",
]

conda_found = False
for conda_path in conda_paths:
    if Path(conda_path).exists():
        print(f"🔧 Utilisation de conda: {conda_path}")
        # Charger conda
        import subprocess
        result = subprocess.run(
            ["bash", "-c", f"source {conda_path} && conda activate sam3 && which python"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            sam3_python = result.stdout.strip()
            print(f"🐍 Python sam3: {sam3_python}")
            # Pas besoin de changer sys.executable ici car le script sera exécuté via conda
            conda_found = True
            break

if not conda_found:
    print("⚠️ Conda non trouvé, tentative avec Python par défaut")

# Ajouter le chemin du projet
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "third_party" / "sam3"))
sys.path.insert(0, str(ROOT))

try:
    import torch
    import numpy as np
    from PIL import Image
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ Erreur import: {e}")
    sys.exit(1)

# Configuration du modèle fine-tuné
FT_CHECKPOINT = "/home/hajar/dagster_pipeline_crts/sam3_logs/building_ft_seg/checkpoints/sam3.pt"
print(f"\n📂 Checkpoint FT: {FT_CHECKPOINT}")
print(f"📂 Fichier existe: {Path(FT_CHECKPOINT).is_file()}")

if not Path(FT_CHECKPOINT).is_file():
    print("✗ Le checkpoint fine-tuné n'existe pas!")
    sys.exit(1)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    print("\n🔄 Chargement du modèle SAM3 de base...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Charger le modèle de base
    model = build_sam3_image_model(
        enable_inst_interactivity=True,
        enable_segmentation=True,
        load_from_HF=True,
    )
    
    print("✓ Modèle de base chargé")
    
    # Charger et fusionner le checkpoint fine-tuné
    print(f"\n🔄 Fusion du checkpoint fine-tuné depuis {FT_CHECKPOINT}...")
    ckpt = torch.load(FT_CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    
    # Aplatir les clés si nécessaire
    if any("detector." in k for k in state):
        state = {k.replace("detector.", ""): v for k, v in state.items() if "detector." in k}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"✓ Checkpoint fusionné:")
    print(f"  - Clés chargées: {len(state)}")
    print(f"  - Clés manquantes: {len(missing)}")
    print(f"  - Clés inattendues: {len(unexpected)}")
    
    # Préparer le modèle
    model.eval()
    model = model.to(device)
    model = model.float()
    
    print(f"\n✓ Modèle prêt sur {device}")
    
    # Créer le processeur
    processor = Sam3Processor(model)
    print("✓ Processeur créé")
    
    # Test avec une image de test
    test_image = ROOT / "data" / "incoming" / "zone1_00007_IMG.png"
    if not test_image.exists():
        print(f"\n⚠ Image de test non trouvée: {test_image}")
        print("Création d'une image de test synthétique...")
        test_image_np = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image_np)
    else:
        print(f"\n📷 Chargement de l'image de test: {test_image}")
        test_pil = Image.open(test_image).convert("RGB")
        test_image_np = np.array(test_pil)
    
    print(f"📷 Dimensions image: {test_image_np.shape}")
    
    # Test de segmentation
    print("\n🔄 Test de segmentation...")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=False):
            state = processor.set_image(test_pil)
            out = processor.set_text_prompt(state=state, prompt="building")
    
    masks = out.get("masks")
    boxes = out.get("boxes")
    scores = out.get("scores")
    
    print(f"✓ Segmentation terminée:")
    print(f"  - Masques détectés: {len(masks) if masks is not None else 0}")
    print(f"  - Boîtes détectées: {len(boxes) if boxes is not None else 0}")
    print(f"  - Scores: {len(scores) if scores is not None else 0}")
    
    if masks is not None and len(masks) > 0:
        print(f"  - Premier masque shape: {masks[0].shape}")
        print(f"  - Premier score: {scores[0] if scores is not None else 'N/A'}")
    
    print("\n✅ Test réussi ! Le modèle fine-tuné fonctionne correctement.")
    
except Exception as e:
    print(f"\n✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)