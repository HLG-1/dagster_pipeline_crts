"""
Micro-service SAM 3 (Meta officiel) — FastAPI, port 8077.

Endpoints :
  GET  /health
  POST /predict_instances

Contrat compatible avec applib/core/instance_pipeline.py :
  - image (base64 PNG)
  - prompt (texte, ex. "building")
  - boxes_xyxy, points_xy (optionnels)
  - sep : per_box | argmax | grouped
  - conf (seuil score, optionnel)

Environnement dédié conda `sam3` (torch>=2.7, Python 3.12).
Lancer : bash scripts/run_sam3_service.sh
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from .config import (
    SAM3_CHECKPOINT,
    SAM3_CONFIDENCE,
    SAM3_DEVICE,
    SAM3_DTYPE,
    SAM3_FT_CHECKPOINT,
    SAM3_HF_REPO,
    SAM3_LABEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sam3_service")

app = FastAPI(title="SAM3 Service (Meta)", version="1.0.0")

_MODEL = None
_PROCESSOR = None
_DEVICE: torch.device | None = None
_LOADED_FT: str | None = None


def _resolve_device() -> torch.device:
    want = (SAM3_DEVICE or "cuda").lower()
    if want in ("", "auto", "default", "cuda", "gpu", "0"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA demandé mais indisponible ; bascule sur CPU")
        return torch.device("cpu")
    if want == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def _resolve_dtype(dev: torch.device) -> torch.dtype:
    if dev.type != "cuda":
        return torch.float32
    m = (SAM3_DTYPE or "float32").lower()
    if m in ("bf16", "bfloat16"):
        return torch.bfloat16
    if m == "float32":
        return torch.float32
    return torch.float16


def _merge_finetune_checkpoint(model: torch.nn.Module, ckpt_path: str) -> dict[str, Any]:
    """Fusionne un checkpoint trainer (building_ft) sur un modèle HF complet.

    Le FT local n'entraîne que backbone / transformer / geometry / scoring
    (pas de tête interactive ni segmentation). On charge donc HF + overlay.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint FT introuvable : {path}")

    log.info("Fusion poids fine-tunés depuis %s …", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Format checkpoint inattendu : {type(state)}")

    # Format HF wrapé (detector.*) → aplatir
    if any("detector." in k for k in state):
        flat = {k.replace("detector.", ""): v for k, v in state.items() if "detector." in k}
    else:
        flat = state

    missing, unexpected = model.load_state_dict(flat, strict=False)
    info = {
        "path": str(path),
        "n_loaded": len(flat),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }
    log.info(
        "FT fusionné : %s clés · missing=%s · unexpected=%s · epoch=%s",
        info["n_loaded"], info["n_missing"], info["n_unexpected"], info["epoch"],
    )
    return info


def _fix_dtype_hooks(model: torch.nn.Module) -> int:
    """Aligne le dtype des activations sur celui des poids (évite bf16/fp32 mix)."""

    def _pre_hook(mod: torch.nn.Module, args: tuple):
        if not args:
            return args
        x = args[0]
        w = getattr(mod, "weight", None)
        if (
            torch.is_tensor(x)
            and torch.is_tensor(w)
            and x.is_floating_point()
            and w.is_floating_point()
            and x.dtype != w.dtype
        ):
            return (x.to(dtype=w.dtype),) + args[1:]
        return args

    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(_pre_hook)
            n += 1
    return n


def _exit_leaked_bf16_autocast(model: torch.nn.Module) -> bool:
    """Le tracker SAM3 fait bf16_context.__enter__() permanent → casse le mode concept."""
    exited = False
    try:
        pred = getattr(model, "inst_interactive_predictor", None)
        inner = getattr(pred, "model", None) if pred is not None else None
        ctx = getattr(inner, "bf16_context", None) if inner is not None else None
        if ctx is not None and torch.is_autocast_enabled():
            ctx.__exit__(None, None, None)
            exited = True
    except Exception as exc:  # noqa: BLE001
        log.warning("Impossible de sortir bf16_context: %s", exc)
    return exited


def _load_model() -> None:
    global _MODEL, _PROCESSOR, _DEVICE, _LOADED_FT
    if _MODEL is not None:
        return

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _DEVICE = _resolve_device()
    dtype = _resolve_dtype(_DEVICE)
    ft_path = (SAM3_FT_CHECKPOINT or "").strip()
    log.info(
        "Chargement SAM3 Meta — device=%s dtype=%s repo=%s label=%s ft=%s",
        _DEVICE, dtype, SAM3_HF_REPO, SAM3_LABEL, ft_path or "(aucun)",
    )

    kwargs: dict[str, Any] = {
        "enable_inst_interactivity": True,
        "enable_segmentation": True,
    }
    # Poids de base : fichier local si disponible, sinon init sans checkpoint.
    base_ckpt = (SAM3_CHECKPOINT or "").strip()
    if base_ckpt:
        kwargs["checkpoint_path"] = base_ckpt
        kwargs["load_from_HF"] = False
    else:
        kwargs["load_from_HF"] = False

    try:
        model = build_sam3_image_model(**kwargs)
    except FileNotFoundError as exc:
        log.warning("Checkpoint local introuvable, démarrage sans poids pré-entraînés : %s", exc)
        kwargs.pop("checkpoint_path", None)
        kwargs["load_from_HF"] = False
        model = build_sam3_image_model(**kwargs)
    if _exit_leaked_bf16_autocast(model):
        log.info("bf16_context tracker désactivé (mode concept float32 OK)")
    if ft_path:
        info = _merge_finetune_checkpoint(model, ft_path)
        _LOADED_FT = info.get("path") or ft_path
    else:
        _LOADED_FT = None

    model.eval()
    if _DEVICE.type == "cuda":
        model = model.to(_DEVICE)
    model = model.float()
    n_hooks = _fix_dtype_hooks(model)
    _MODEL = model
    _PROCESSOR = Sam3Processor(model)
    log.info(
        "SAM3 prêt sur %s (variant=%s, weights=float32, dtype_hooks=%s, autocast=%s)",
        _DEVICE, SAM3_LABEL, n_hooks, torch.is_autocast_enabled(),
    )


class PredictRequest(BaseModel):
    image: str
    prompt: str = "building"
    conf: float = Field(default=SAM3_CONFIDENCE, ge=0.0, le=1.0)
    sep: str = "per_box"
    boxes_xyxy: list[list[float]] | None = None
    points_xy: list[list[float] | None] | None = None


def _decode_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(pil)


def _encode_label(label: np.ndarray) -> str:
    return base64.b64encode(label.astype("<i4").tobytes()).decode("ascii")


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _mask_to_numpy(m) -> np.ndarray:
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[0]
    return (m > 0.5).astype(np.uint8)


def _predict_inst_box(
    inference_state,
    box: list[float],
    point: tuple[float, float] | None,
) -> tuple[np.ndarray, float]:
    """Un masque par boîte via predict_inst (API interactive SAM3)."""
    box_arr = np.array(box, dtype=np.float32).reshape(1, 4)
    pt = point or _box_center(box)
    pt_arr = np.array([[pt[0], pt[1]]], dtype=np.float32)
    pt_lab = np.array([1], dtype=np.int32)

    with torch.inference_mode():
        if _DEVICE and _DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=_resolve_dtype(_DEVICE)):
                out = _MODEL.predict_inst(
                    inference_state,
                    point_coords=pt_arr,
                    point_labels=pt_lab,
                    box=box_arr,
                    multimask_output=False,
                )
        else:
            out = _MODEL.predict_inst(
                inference_state,
                point_coords=pt_arr,
                point_labels=pt_lab,
                box=box_arr,
                multimask_output=False,
            )

    masks, scores = out[0], out[1]
    if masks is None or len(masks) == 0:
        return np.zeros((1, 1), dtype=np.uint8), 0.0
    m = _mask_to_numpy(masks)
    sc = float(scores[0]) if scores is not None and len(scores) else 1.0
    return m, sc


def _separate_per_box(
    rgb: np.ndarray,
    boxes: list[list[float]],
    points: list[list[float] | None] | None,
    conf_thr: float,
) -> tuple[np.ndarray, list[dict]]:
    h, w = rgb.shape[:2]
    label = np.zeros((h, w), dtype=np.int32)
    meta: list[dict] = []
    pil = Image.fromarray(rgb)
    state = _PROCESSOR.set_image(pil)

    nid = 0
    for i, box in enumerate(boxes):
        pt = None
        if points and i < len(points) and points[i] is not None:
            pt = (float(points[i][0]), float(points[i][1]))
        mask, score = _predict_inst_box(state, box, pt)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if score < conf_thr or mask.sum() == 0:
            continue
        nid += 1
        label[mask > 0] = nid
        meta.append({"id": nid, "box": box, "score": score, "sep": "per_box"})
    return label, meta


def _separate_argmax(
    rgb: np.ndarray,
    boxes: list[list[float]],
    points: list[list[float] | None] | None,
    conf_thr: float,
) -> tuple[np.ndarray, list[dict]]:
    h, w = rgb.shape[:2]
    label = np.zeros((h, w), dtype=np.int32)
    score_map = np.zeros((h, w), dtype=np.float32)
    meta: list[dict] = []
    pil = Image.fromarray(rgb)
    state = _PROCESSOR.set_image(pil)

    for i, box in enumerate(boxes):
        pt = None
        if points and i < len(points) and points[i] is not None:
            pt = (float(points[i][0]), float(points[i][1]))
        mask, score = _predict_inst_box(state, box, pt)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if score < conf_thr:
            continue
        m = mask.astype(np.float32) * score
        better = (mask > 0) & (m >= score_map)
        label[better] = i + 1
        score_map[better] = m[better]
        meta.append({"id": i + 1, "box": box, "score": score, "sep": "argmax"})

    if label.max() > 0:
        # Renuméroter 1..N
        out = np.zeros_like(label)
        nid = 0
        for lab in np.unique(label[label > 0]):
            nid += 1
            out[label == lab] = nid
        label = out
    return label, meta


def _separate_grouped(
    rgb: np.ndarray,
    prompt: str,
    conf_thr: float,
) -> tuple[np.ndarray, list[dict]]:
    """Prompt texte SAM3 → une instance par masque détecté."""
    h, w = rgb.shape[:2]
    pil = Image.fromarray(rgb)
    # Mode concept : float32 strict (autocast half/bf16 casse le VL backbone).
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=False):
            state = _PROCESSOR.set_image(pil)
            out = _PROCESSOR.set_text_prompt(state=state, prompt=prompt)

    masks = out.get("masks")
    boxes = out.get("boxes")
    scores = out.get("scores")
    if masks is None:
        return np.zeros((h, w), dtype=np.int32), []

    label = np.zeros((h, w), dtype=np.int32)
    meta: list[dict] = []
    nid = 0
    n = len(masks)
    for i in range(n):
        sc = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if sc < conf_thr:
            continue
        m = _mask_to_numpy(masks[i])
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        if m.sum() == 0:
            continue
        nid += 1
        label[m > 0] = nid
        box = boxes[i].tolist() if boxes is not None and i < len(boxes) else None
        meta.append({"id": nid, "box": box, "score": sc, "sep": "grouped"})
    return label, meta


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _load_model()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        vram = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / 1024**3, 1)
        return {
            "ready": _MODEL is not None,
            "backend": "sam3_meta",
            "variant": SAM3_LABEL,
            "ft_checkpoint": _LOADED_FT,
            "device": str(_DEVICE),
            "gpu": gpu_name,
            "vram_gb": vram,
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "hf_repo": SAM3_HF_REPO,
        }
    except Exception as e:
        return {"ready": False, "backend": "sam3_meta", "error": str(e)}


@app.post("/predict_instances")
def predict_instances(req: PredictRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        _load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"SAM3 non chargé : {e}") from e

    rgb = _decode_image(req.image)
    h, w = rgb.shape[:2]
    sep = (req.sep or "per_box").strip().lower()
    boxes = req.boxes_xyxy or []
    points = req.points_xy

    try:
        if sep == "grouped" or (not boxes and sep != "per_box"):
            label, meta = _separate_grouped(rgb, req.prompt, req.conf)
        elif sep == "argmax" and boxes:
            label, meta = _separate_argmax(rgb, boxes, points, req.conf)
        elif boxes:
            label, meta = _separate_per_box(rgb, boxes, points, req.conf)
        else:
            label, meta = _separate_grouped(rgb, req.prompt, req.conf)
    except Exception as e:
        log.exception("predict_instances failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "shape": [h, w],
        "label_b64": _encode_label(label),
        "instances": meta,
        "n_instances": int(label.max()),
        "sep": sep,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "backend": "sam3_meta",
        "variant": SAM3_LABEL,
    }


def main() -> None:
    import uvicorn

    from .config import SAM3_HOST, SAM3_PORT

    uvicorn.run(
        "services.sam3_service.server:app",
        host=SAM3_HOST,
        port=SAM3_PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
