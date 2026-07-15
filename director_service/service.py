#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""ARDY Director control service.

Runs on the ARDY host (inside the ardy venv) and wraps NVIDIA ARDY's generation
API behind a small HTTP surface so an external director (the MCP bridge, or an
agent brain) can generate and choreograph humanoid motion by text.

It reuses the already-running LLM2Vec text-encoder service (default port 9550)
so it only needs to hold the ~156M-param motion denoiser in VRAM, not a second
copy of Llama-3.

Endpoints:
  GET  /health                      -> liveness + loaded models + device
  GET  /models                      -> known model nicknames
  POST /generate    {prompt,...}    -> one clip from one prompt
  POST /choreograph {steps:[...]}   -> one clip stitched from a prompt sequence

Motion is written as ARDY-native .npz under OUTPUT_DIR and the path is returned,
so ARDY's own viewer (scripts/visualize.py) can load it.
"""
import os
import threading
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ardy.model.load_model import load_model, load_text_encoder
from ardy.model.registry import resolve_model_name
from ardy.postprocess import post_process_motion

OUTPUT_DIR = os.environ.get("DIRECTOR_OUTPUT_DIR", os.path.expanduser("~/ardy/outputs/director"))
ENCODER_URL = os.environ.get("DIRECTOR_ENCODER_URL", "http://localhost:9550")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="ARDY Director", version="0.1.0")

# --- lazily-loaded, cached model registry (shared text encoder) --------------
_lock = threading.Lock()
_text_encoder = None
_models = {}  # nickname -> loaded Ardy model


def _encoder():
    global _text_encoder
    if _text_encoder is None:
        # Reuse the running encoder service; fall back to local only if unreachable.
        _text_encoder = load_text_encoder(mode="auto", url=ENCODER_URL)
    return _text_encoder


def _get_model(nickname: str):
    resolved = resolve_model_name(nickname)
    with _lock:
        if resolved not in _models:
            _models[resolved] = load_model(resolved, device=DEVICE, text_encoder=_encoder())
        return resolved, _models[resolved]


def _generate_clip(model, resolved_name, prompt, num_frames, diffusion_steps,
                   cfg_weight, seed, first_heading_angle, history_frames):
    """One synchronous ARDY generation → numpy motion dict (single sample)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lengths = torch.tensor([num_frames], device=DEVICE)
    pad_mask = torch.ones((1, num_frames), dtype=torch.bool, device=DEVICE)
    heading = torch.tensor([first_heading_angle], dtype=torch.float, device=DEVICE)

    with torch.no_grad():
        motion = model(
            [prompt.strip()],
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=heading,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=cfg_weight,
            crop_history_length=history_frames,
        )
        out = model.motion_rep.inverse(motion, is_normalized=True)

    if "g1" not in resolved_name.lower():
        corrected = post_process_motion(
            out["local_rot_mats"], out["root_positions"], out["foot_contacts"],
            model.skeleton, constraint_lst=None,
        )
        out.update(corrected)

    # squeeze the leading sample/batch dim (num_samples == 1) so every array is (frames, ...)
    result = {}
    for k, v in out.items():
        a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == 1:
            a = a[0]
        result[k] = a
    return result


def _save_npz(motion, fps, text, tag):
    path = os.path.join(OUTPUT_DIR, f"{tag}.npz")
    np.savez(path, fps=np.int64(fps), text=np.array(text), **motion)
    return path


# --- request models ----------------------------------------------------------
class GenerateReq(BaseModel):
    prompt: str
    model: str = "core"
    duration: float = Field(4.0, gt=0.1, le=30.0)
    seed: int | None = None
    diffusion_steps: int | None = None
    cfg_weight: float = 4.0
    heading_deg: float = 0.0  # initial facing, degrees about +Y (0 = +Z)


class ChoreoStep(BaseModel):
    prompt: str
    duration: float = Field(3.0, gt=0.1, le=30.0)


class ChoreographReq(BaseModel):
    steps: list[ChoreoStep]
    model: str = "core"
    seed: int | None = None
    diffusion_steps: int | None = None
    cfg_weight: float = 4.0


# --- endpoints ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "encoder_url": ENCODER_URL,
        "encoder_loaded": _text_encoder is not None,
        "models_loaded": list(_models.keys()),
        "output_dir": OUTPUT_DIR,
    }


@app.get("/models")
def models():
    return {"models": ["core", "core8", "g1", "g152", "soma"],
            "note": "core=27-joint avatar (default), g1=Unitree G1 robot (MuJoCo qpos)."}


@app.post("/generate")
def generate(req: GenerateReq):
    try:
        resolved, model = _get_model(req.model)
        fps = model.motion_rep.fps
        num_frames = int(req.duration * fps)
        steps = req.diffusion_steps or int(model.diffusion.num_base_steps)
        patch = model.num_frames_per_token
        hist = (int(round(10 * fps)) // patch) * patch  # trained ~10s window
        motion = _generate_clip(model, resolved, req.prompt, num_frames, steps,
                                req.cfg_weight, req.seed, np.deg2rad(req.heading_deg), hist)
        tag = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, req.prompt, tag)
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": num_frames,
                "duration_s": req.duration, "npz": path, "prompt": req.prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/choreograph")
def choreograph(req: ChoreographReq):
    """Generate each step, then chain segments by carrying the root XZ offset and
    heading so the character continues from where the previous step ended.

    v0.1 stitches at segment boundaries; seams are position-continuous but not
    yet velocity-smoothed (that is the streaming-autoregressive upgrade).
    """
    if not req.steps:
        raise HTTPException(status_code=400, detail="steps must be non-empty")
    try:
        resolved, model = _get_model(req.model)
        fps = model.motion_rep.fps
        steps = req.diffusion_steps or int(model.diffusion.num_base_steps)
        patch = model.num_frames_per_token
        hist = (int(round(10 * fps)) // patch) * patch

        seg_keys = ["local_rot_mats", "global_rot_mats", "posed_joints",
                    "root_positions", "foot_contacts", "global_root_heading"]
        acc = {k: [] for k in seg_keys}
        root_offset = np.zeros(3, dtype=np.float32)
        heading_rad = 0.0
        labels = []
        for i, st in enumerate(req.steps):
            nf = int(st.duration * fps)
            seed_i = None if req.seed is None else req.seed + i
            m = _generate_clip(model, resolved, st.prompt, nf, steps, req.cfg_weight,
                               seed_i, heading_rad, hist)
            # carry root XZ so the next segment starts where this one ended
            # (arrays are squeezed to (frames, ...); index the last XYZ axis, not frames)
            rp = m["root_positions"].copy()          # (frames, 3)
            rp[:, [0, 2]] += root_offset[[0, 2]]
            m["root_positions"] = rp
            m["posed_joints"][..., [0, 2]] += root_offset[[0, 2]]  # (frames, joints, 3)
            root_offset = rp[-1].copy()
            # ...and carry the facing too, else a step that turns is undone at the
            # seam. global_root_heading is [cos(theta), sin(theta)] per frame.
            gh = m.get("global_root_heading")
            if gh is not None and len(gh):
                heading_rad = float(np.arctan2(gh[-1][1], gh[-1][0]))
            for k in seg_keys:
                if k in m:
                    acc[k].append(m[k])
            labels.append(st.prompt)

        motion = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}
        tag = f"choreo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, " | ".join(labels), tag)
        total = int(sum(int(s.duration * fps) for s in req.steps))
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": total,
                "segments": len(req.steps), "npz": path, "sequence": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DIRECTOR_PORT", "9600")))
