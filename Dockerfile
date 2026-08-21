FROM python:3.10-slim

# Dependances systeme pour SIG (GDAL, Rasterio, GeoPandas) et OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Mise a jour de pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Installation de PyTorch avec support CUDA 13.0 (L40s / PyTorch 2.11.0+cu130)
# Doit etre fait AVANT pip install -e . pour que le resolver trouve torch
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# Copie des fichiers de configuration de projet et installation des dependances
# (torch est deja installe ci-dessus, pip install -e . ne le reinstallera pas)
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/

RUN pip install --no-cache-dir -e .

# Copie de la suite du code et des configurations
COPY config/ /app/config/
COPY workspace.yaml /app/workspace.yaml
COPY dagster.yaml /app/dagster.yaml

# Creation des dossiers de runtime
RUN mkdir -p /app/.dagster_home /app/checkpoints /app/data/incoming /app/data/processed /app/results

ENV DAGSTER_HOME=/app/.dagster_home \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 3000

# Commande par defaut pour dagster-webserver (surchargee dans docker-compose pour dagster-daemon)
CMD ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000", "-w", "workspace.yaml"]
