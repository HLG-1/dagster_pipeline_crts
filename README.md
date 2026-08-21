# Pipeline Bâtiments — YOLO 11 + SAM 3 + Dagster

Orchestration complète d'extraction et de régularisation d'empreintes de bâtiments à partir d'orthophotos GeoTIFF.
Le pipeline détecte les bâtiments avec YOLO 11 (fine-tuné), les segmente instance par instance avec SAM 3 (multiprompt), applique une régularisation géométrique vectorielle avancée (rectangles tournés, formes orthogonales L/T/U, topologie mitoyenne), puis exporte les empreintes en GeoJSON / GeoPackage / Shapefile ainsi que des masques transparents avec contours vectoriels lissés.

```
Orthophoto GeoTIFF
       │
       ▼
 01 · Ingestion     ─ lecture métadonnées (CRS, dimensions) sans charger les pixels en RAM
       │
       ▼
 02 · Tuilage       ─ découpage en fenêtres de tuiles (ex. 1024 × 1024 px, overlap 128 px)
       │
       ▼
 03 · Détection     ─ YOLO 11 best.pt → bounding boxes par tuile (GPU)
       │
       ▼
 04 · Segmentation  ─ Multiprompt (boîte + point) → SAM 3 (FastAPI GPU) → Régularisation géométrique
       │
       ▼
 05 · Fusion        ─ Raccordement inter-tuiles (STRtree stitch), dédoublonnage & contrôle topologique
       │
       ▼
 06 · Export SIG    ─ GeoJSON + GPKG + Shapefile + Masque binaire + Overlay transparent lissé + run.json
```

---

## Prérequis matériels et logiciels

| Élément | Minimum | Recommandé (config cible) |
|---|---|---|
| GPU | NVIDIA avec ≥ 12 GB VRAM | **NVIDIA L40s** (48 GB) |
| CUDA | 12.x | **13.0** |
| RAM système | 16 GB | 32 GB |
| Espace disque | 25 GB libres | 50 GB |
| OS | Linux (ou WSL2 sur Windows) | Ubuntu 22.04 / 24.04 |
| Docker Engine | ≥ 24 | 26+ |
| Docker Compose | ≥ 2.20 | 2.27+ |
| NVIDIA Container Toolkit | requis | requis |

---

## Structure du projet

```
dagster_pipeline_crts/
├── Dockerfile                          # Image Dagster (webserver + daemon)
├── docker-compose.yml                  # Orchestration des 3 conteneurs
├── pyproject.toml                      # Dépendances Python du pipeline
├── dagster.yaml                        # Configuration instance Dagster
├── workspace.yaml                      # Point d'entrée Dagster → definitions.py
│
├── config/
│   └── pipeline.yaml                   # Paramètres métier (tuile, seuils, régularisation, export)
│
├── checkpoints/
│   └── yolo11/
│       └── best.pt                     # Modèle YOLO 11 (~53 MB)
│
├── sam3_logs/
│   └── building_ft_seg/
│       └── checkpoints/
│           └── sam3.pt                 # Checkpoint SAM 3 (~9.4 GB)
│
├── third_party/
│   └── sam3/                           # Code source SAM 3 Meta officiel (local)
│
├── services/
│   └── sam3_service/                   # Micro-service FastAPI SAM 3 (port 8077)
│       ├── Dockerfile
│       ├── server.py
│       ├── config.py
│       └── requirements-sam3.txt
│
├── src/
│   └── pipeline_batiments/
│       ├── definitions.py              # Point d'entrée Dagster (assets + resources)
│       ├── jobs.py                     # Job pipeline_complet (6 assets)
│       ├── sensors.py                  # Sensor watch_new_orthophotos
│       ├── assets/                     # 6 étapes orchestrées
│       │   ├── ingestion.py
│       │   ├── tuilage.py
│       │   ├── detection.py
│       │   ├── segmentation.py         # Segmentation SAM3 + régularisation
│       │   ├── fusion.py               # Fusion spatiale STRtree
│       │   └── export.py               # Export SIG + Overlays transparents
│       ├── core/                       # Algorithmes géométriques et traitement d'image
│       │   ├── building_regularizer.py # Régularisation par équerrage 90° et simplification
│       │   ├── rect_regularizer.py     # Re-fit géométrique rectilinéaire et minAreaRect
│       │   ├── topology_regularizer.py # Alignement topologique des murs mitoyens
│       │   ├── shape_refiner.py        # Classification et affinage de forme (parcimonie 4 sommets)
│       │   ├── instances.py            # render_gdf_outlined, separate_instances, métriques
│       │   ├── viz.py                  # make_overlay, compose_layer_view, contours
│       │   ├── postprocess.py          # Nettoyage morphologique de masques
│       │   ├── metrics.py              # Calcul IoU, F1, PQ, latence
│       │   ├── polygons.py             # Vectorisation et fusion d'instances
│       │   ├── geo_io.py               # Lecture raster fenêtrée, reprojection CRS, export SIG
│       │   └── multiprompt.py          # Génération des points centraux par boîte
│       ├── resources/
│       │   ├── yolo_resource.py        # Resource YOLO (chargement unique)
│       │   └── sam3_resource.py        # Resource SAM3 (HTTP + retries + backoff)
│       └── yolo11/                     # Module YOLO 11 Ultralytics
│
├── data/
│   ├── incoming/                       # Dossier surveillé par le sensor
│   └── processed/
│
├── results/                            # Sorties du pipeline (GeoJSON, GPKG, PNG, JSON)
└── tests/                              # Tests unitaires automatisés pytest (60 tests)
```

---

## Fonctionnalités Avancées

### 1. Régularisation Géométrique & Topologique
- **Re-fit Rectilinéaire (`rect_regularizer`)** : Re-fit en rectangle minimum tourné (`minAreaRect`) pour les bâtiments réguliers (4 sommets garantis) ou en polygone orthogonal (angles à 90° pour formes en L/T/U).
- **Orientation partagée par îlot** : Détection des groupes de bâtiments voisins pour harmoniser les orientations dominantes et fusionner les murs mitoyens face-à-face.
- **Régularisation topologique (`topology_regularizer`)** : Résolution des chevauchements résiduels et suppression des micro-espaces entre bâtiments contigus.

### 2. Rendu des Masques Transparents (`render_gdf_outlined`)
- **Reprojection affine précise** : Correspondance sub-pixel entre les coordonnées géoréférencées (monde) et la grille image via `rasterio.transform`.
- **Transparence contrôlée** : Remplissage semi-transparent (`alpha=0.48` par défaut) permettant d'inspecter l'orthophoto sous-jacente.
- **Contours vectoriels anticrénelés (`cv2.LINE_AA`)** : Lignes de délimitation franches et nettes pour distinguer immédiatement les bâtiments accolés.

---

## Installation sur un nouveau PC

### Prérequis
- GPU NVIDIA avec ≥ 12 GB VRAM (recommandé)
- CUDA 12.x ou 13.0
- Docker Engine ≥ 24
- Docker Compose ≥ 2.20
- NVIDIA Container Toolkit
- Git

### Étape 1 · Cloner le dépôt
```bash
git clone https://github.com/HLG-1/dagster_pipeline_crts.git
cd dagster_pipeline_crts
```

### Étape 2 · Installer les dépendances système (Linux/Ubuntu)
```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Installer Docker et Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Installer NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Étape 3 · Télécharger les modèles de poids
```bash
# Créer les dossiers nécessaires
mkdir -p checkpoints/yolo11
mkdir -p sam3_logs/building_ft_seg/checkpoints

# Télécharger le modèle YOLO 11 (~53 MB)
# Placer le fichier best.pt dans checkpoints/yolo11/best.pt
# (Contactez l'équipe pour obtenir le modèle fine-tuné)

# Télécharger le modèle SAM 3 (~9.4 GB)
# Placer le fichier sam3.pt dans sam3_logs/building_ft_seg/checkpoints/sam3.pt
# (Contactez l'équipe pour obtenir le checkpoint)
```

### Étape 4 · Configurer l'environnement
```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env pour configurer les variables
nano .env
```

Variables importantes dans `.env` :
- `HF_TOKEN=your_hugging_face_token_here` (si nécessaire pour SAM3)
- `SAM3_DEVICE=cuda` (ou `cpu` si pas de GPU)
- `SAM3_DTYPE=float16` (ou `float32` pour plus de précision)
- `SAM3_URL=http://127.0.0.1:8077` (ou `http://sam3:8077` dans Docker)
- `YOLO_WEIGHTS_PATH=checkpoints/yolo11/best.pt`

### Étape 5 · Construire et démarrer les conteneurs
```bash
# Construction des images Docker
docker compose build

# Démarrage en arrière-plan
docker compose up -d

# Vérifier l'état
docker compose ps
```

### Étape 6 · Vérifier les services
```bash
# Vérifier le micro-service SAM3
curl http://localhost:8077/health

# Vérifier Dagster webserver
curl http://localhost:3000
```

### Étape 7 · Lancer le pipeline
1. Ouvrir le navigateur sur **http://localhost:3000**
2. Naviguer vers **Assets** → Groupe `pipeline_batiments`
3. Cliquer sur **Materialize all** pour lancer le pipeline complet
4. Placer une orthophoto GeoTIFF dans `data/incoming/`
5. Le sensor détectera automatiquement le fichier et lancera le pipeline

---

## Installation et Lancement (Docker Compose)

### 1 · Vérifier les fichiers de poids
```bash
# Modèle YOLO 11 (~53 MB)
ls -lh checkpoints/yolo11/best.pt

# Modèle SAM 3 (~9.4 GB)
ls -lh sam3_logs/building_ft_seg/checkpoints/sam3.pt
```

### 2 · Configurer l'environnement
```bash
cp .env.example .env
```

### 3 · Construire et démarrer les conteneurs
```bash
# Construction des 3 images
docker compose build

# Démarrage en arrière-plan
docker compose up -d
```

### 4 · Vérifier le bon fonctionnement
```bash
# État des conteneurs
docker compose ps

# Vérifier le micro-service SAM3
curl http://localhost:8077/health
```

### 5 · Accéder à l'interface Dagster
Ouvrir votre navigateur sur **http://localhost:3000** :
- Naviguer vers **Assets** → Groupe `pipeline_batiments`
- Cliquer sur **Materialize all** pour lancer le pipeline complet.

---

## Sorties du Pipeline (`results/`)

Chaque exécution génère un dossier horodaté `results/<stem>__<timestamp>/` contenant :

| Fichier | Format | Description |
| :--- | :--- | :--- |
| `<stem>.geojson` | GeoJSON (WGS84 / Source) | Empreintes vectorielles régularisées avec attributs (`building_id`, `area_m2`, `area_px`). |
| `<stem>.gpkg` | GeoPackage | Couche SIG standard pour QGIS / ArcGIS. |
| `<stem>_shp/` | ESRI Shapefile | Ensemble de fichiers `.shp`, `.dbf`, `.prj`, `.shx`. |
| `<stem>_mask.png` | PNG Grayscale (L) | Masque binaire plein format (255 = bâti, 0 = fond). |
| `<stem>_overlay.png` | PNG RGB (24-bit) | Orthophoto originale avec masques transparents et contours vectoriels lissés. |
| `<stem>__<timestamp>_run.json` | JSON | Métadonnées de traçabilité (latences, nombre de bâtiments, versions, chemins). |

---

## Configuration (`config/pipeline.yaml`)

```yaml
tuilage:
  tile_size: 1024
  overlap: 128

yolo:
  conf_threshold: 0.20
  weights_path: checkpoints/yolo11/best.pt

sam3:
  url: http://sam3:8077
  timeout_s: 300
  max_retries: 3
  backoff_s: 5

regularization:
  enabled: true
  method: rectfit             # rectfit | topology | none
  simplify_tolerance: 0.5
  min_area_px: 50

fusion:
  min_area_px: 50
  iou_merge_threshold: 0.3

export:
  formats: [geojson, gpkg, shapefile, mask_png, overlay_png]
  output_dir: results
  overlay:
    alpha: 0.48
    fill_color: "#F97316"
    line_color: "#EF4444"
    line_thickness: 2

sensor:
  watch_dir: data/incoming
  poll_interval_s: 30

input:
  path: data/incoming/01.tif
```

---

## Tests Automatisés

Le projet inclut une suite de tests unitaires couvrant l'ensemble des modules régularisateurs, la vectorisation et le rendu visuel :

```bash
python -m pytest tests/
```
*60 tests unitaires validés avec succès.*
