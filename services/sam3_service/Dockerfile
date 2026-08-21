FROM python:3.10-slim

# Dependances systeme pour OpenCV, Git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Mise a jour de pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Installation de PyTorch avec support CUDA 13.0 (L40s / PyTorch 2.11.0+cu130)
# Index officiel Pytorch pour cu130
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# Copie et installation du package sam3 (third_party local - pas de clone git)
COPY third_party/sam3 /app/third_party/sam3
RUN if [ -f "/app/third_party/sam3/pyproject.toml" ]; then \
        pip install --no-cache-dir -e /app/third_party/sam3 ; \
    else \
        echo "ERREUR : third_party/sam3/pyproject.toml introuvable" && exit 1 ; \
    fi

# Installation des dependances specifiques du micro-service SAM3
COPY services/sam3_service/requirements-sam3.txt /app/services/sam3_service/requirements-sam3.txt
RUN pip install --no-cache-dir -r /app/services/sam3_service/requirements-sam3.txt && \
    pip install --no-cache-dir einops pycocotools psutil && \
    (pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git" || true)

# Fixe le conflit numpy / opencv (numpy<2 exige par sam3, opencv<4.10 pour compat).
# Installe EN DERNIER pour avoir la priorite sur les sous-dependances.
RUN pip install --no-cache-dir "numpy<2" "opencv-python-headless<4.10"

# Verification legere : imports sans chargement de poids
# (on ne peut pas appeler build_sam3_image_model au build, les poids sont
#  montes en volume et absents ici — on verifie juste que les modules importent)
RUN python -c "\
import numpy, cv2, torch; \
import sam3; \
print('Build OK — numpy', numpy.__version__, '| torch', torch.__version__, '| CUDA disponible au runtime:', torch.cuda.is_available())"

# Code du service (PAS les poids : montes en volume dans docker-compose)
COPY services/ /app/services/

ENV SAM3_HOST=0.0.0.0 \
    SAM3_PORT=8077 \
    SAM3_DEVICE=cuda \
    SAM3_DTYPE=float16 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8077
CMD ["python", "-m", "services.sam3_service.server"]