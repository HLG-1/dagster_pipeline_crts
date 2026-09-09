#!/bin/bash
# Script pour relancer SAM3 avec le modèle fine-tuné

echo "🔄 Arrêt du service SAM3 existant..."
sudo pkill -f "sam3_service.server"

echo "⏱️  Attente de l'arrêt..."
sleep 5

echo "🚀 Lancement SAM3 avec modèle fine-tuné en tant que root..."
cd /home/hajar/dagster_pipeline_crts

# Variables d'environnement pour le modèle fine-tuné
export SAM3_FT_CHECKPOINT=/home/hajar/dagster_pipeline_crts/sam3_logs/building_ft_seg/checkpoints/sam3.pt
export SAM3_DEVICE=cuda
export SAM3_DTYPE=bfloat16
export SAM3_PORT=8077
export SAM3_USE_FT=1
export SAM3_ENV=sam3

# Lancement avec sudo pour utiliser l'environnement root
sudo bash -c '
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate sam3 2>/dev/null || conda activate base
cd /home/hajar/dagster_pipeline_crts
export SAM3_FT_CHECKPOINT=/home/hajar/dagster_pipeline_crts/sam3_logs/building_ft_seg/checkpoints/sam3.pt
export SAM3_DEVICE=cuda
export SAM3_DTYPE=bfloat16
export SAM3_PORT=8077
export SAM3_USE_FT=1
export SAM3_ENV=sam3
nohup python -m services.sam3_service.server > /tmp/sam3_service.log 2>&1 &
'

echo "⏱️  Attente du démarrage..."
sleep 15

echo "🔍 Vérification du service..."
curl -s http://localhost:8077/health | python3 -m json.tool

echo "✅ Service SAM3 relancé avec modèle fine-tuné"