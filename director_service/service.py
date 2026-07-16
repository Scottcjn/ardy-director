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
  POST /choreograph {steps:[...]}   -> one clip streamed through a prompt sequence

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

from director_service.streaming import (
    default_history_frames,
    history_length_for,
    plan_stream,
    seam_report,
)

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
    return _decode_motion(model, resolved_name, motion)


def _decode_motion(model, resolved_name, motion):
    """Normalized ARDY motion tensor → numpy motion dict (single sample).

    Shared by /generate and /choreograph so both land in the same
    representation, post-processing and shape convention.
    """
    with torch.no_grad():
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


def _stream_motion(model, plan, diffusion_steps, cfg_weight, seed, first_heading_angle):
    """Run a StreamPlan through ARDY's autoregressive streaming.

    One `autoregressive_step` per chunk, each conditioned on the tail of what is
    already generated, so a prompt change is denoised *into* the motion in
    flight instead of being glued onto it. Returns the normalized motion tensor
    for the whole run, [1, frames, feats].

    Two details that matter and are easy to get wrong:

    * `autoregressive_step` returns history + new frames, so only
      `samples[:, history_length:]` is new. Appending the whole return value
      duplicates the history window every step (this mirrors how the interactive
      demo merges: `motion_tensor[:, :history_end+1]` + `samples[:, history:]`).
    * The returned frames are already in world coordinates -- ARDY recenters the
      history internally and translates the result back by the accumulated
      global translation. Carrying a root offset by hand, as the old stitch did,
      would double-count it.
    """
    if seed is not None:
        # Seed once: this is one continuous stream, not N independent clips.
        torch.manual_seed(seed)
        np.random.seed(seed)

    fps = model.motion_rep.fps
    patch = model.num_frames_per_token
    horizon = model.gen_horizon_len
    history_crop = default_history_frames(fps, horizon, patch)

    text_cache = {}
    motion_tensor = None
    with torch.no_grad():
        for chunk in plan.chunks:
            if chunk.prompt not in text_cache:
                # model._encode_text uses model.text_encoder -- the shared
                # encoder handed to load_model -- so streaming re-encodes text
                # without pulling in a second copy of Llama-3. Cached per prompt:
                # the text does not change between chunks of the same step.
                text_cache[chunk.prompt] = model._encode_text([chunk.prompt.strip()])
            text_feat, text_pad_mask = text_cache[chunk.prompt]

            have = 0 if motion_tensor is None else motion_tensor.shape[1]
            history_length = history_length_for(have, history_crop, patch)
            history = motion_tensor[:, have - history_length:] if history_length else None

            samples = model.autoregressive_step(
                num_frames=history_length + horizon,
                num_denoising_steps=diffusion_steps,
                motion_mask=None,
                observed_motion=None,
                cfg_weight=cfg_weight,
                texts=None,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                init_history_sequence=history,
                # Only the first step sets the world pose; afterwards the history
                # carries it and passing these again would reset the stream.
                init_global_translation=None if history_length else torch.zeros(
                    (1, model.motion_rep.nfeats_dict["root_pos"]), device=DEVICE),
                init_first_heading_angle=None if history_length else torch.tensor(
                    [first_heading_angle], dtype=torch.float, device=DEVICE),
            )
            new = samples[:, history_length:]
            motion_tensor = new if motion_tensor is None else torch.cat([motion_tensor, new], dim=1)

    return motion_tensor


def _save_npz(motion, fps, text, tag, seam_frames=None):
    """Write ARDY-native .npz. `seam_frames` rides along so scripts/seam_report.py
    can find the prompt changes without being told; ARDY's viewer looks keys up
    by name, so the extra entry is inert to it.
    """
    path = os.path.join(OUTPUT_DIR, f"{tag}.npz")
    extra = {} if seam_frames is None else {"seam_frames": np.asarray(seam_frames, dtype=np.int64)}
    np.savez(path, fps=np.int64(fps), text=np.array(text), **extra, **motion)
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
    heading_deg: float = 0.0  # initial facing, degrees about +Y (0 = +Z)
    # "stream" runs ARDY's autoregressive streaming (default: velocity-smooth
    # across prompt changes). "stitch" is v0.1's independent-clips-glued-by-root-
    # offset behaviour, kept only so the two can be compared on the same seed --
    # see scripts/seam_report.py. It pops at every seam; don't ship with it.
    mode: str = "stream"


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
        # History must leave room for the generation horizon inside the trained
        # window -- v0.1 spent the whole 10 s on history, so every step ran a
        # window longer than the model was trained on.
        hist = default_history_frames(fps, model.gen_horizon_len, patch)
        motion = _generate_clip(model, resolved, req.prompt, num_frames, steps,
                                req.cfg_weight, req.seed, np.deg2rad(req.heading_deg), hist)
        tag = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, req.prompt, tag)
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": num_frames,
                "duration_s": req.duration, "npz": path, "prompt": req.prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def _choreograph_stitch(req, model, resolved, fps, steps, hist):
    """v0.1: generate each step independently, glue by carrying the root XZ.

    Kept for A/B against streaming on the same seed. Position is continuous
    across a seam; velocity is not, because the next segment was denoised
    knowing nothing about the one before it.
    """
    seg_keys = ["local_rot_mats", "global_rot_mats", "posed_joints",
                "root_positions", "foot_contacts", "global_root_heading"]
    acc = {k: [] for k in seg_keys}
    root_offset = np.zeros(3, dtype=np.float32)
    labels, seams, frame = [], [], 0
    for i, st in enumerate(req.steps):
        nf = int(st.duration * fps)
        seed_i = None if req.seed is None else req.seed + i
        m = _generate_clip(model, resolved, st.prompt, nf, steps, req.cfg_weight,
                           seed_i, np.deg2rad(req.heading_deg), hist)
        # carry root XZ so the next segment starts where this one ended
        # (arrays are squeezed to (frames, ...); index the last XYZ axis, not frames)
        rp = m["root_positions"].copy()          # (frames, 3)
        rp[:, [0, 2]] += root_offset[[0, 2]]
        m["root_positions"] = rp
        m["posed_joints"][..., [0, 2]] += root_offset[[0, 2]]  # (frames, joints, 3)
        root_offset = rp[-1].copy()
        for k in seg_keys:
            if k in m:
                acc[k].append(m[k])
        labels.append(st.prompt)
        if i > 0:
            seams.append(frame)
        frame += len(rp)

    motion = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}
    segments = [{"prompt": p, "frames": int(a.shape[0])}
                for p, a in zip(labels, acc["root_positions"])]
    return motion, labels, seams, segments


@app.post("/choreograph")
def choreograph(req: ChoreographReq):
    """Turn a prompt sequence into one clip.

    mode="stream" (default): ARDY's autoregressive streaming. Each window is
    denoised on the tail of the motion already generated, so changing the prompt
    changes where the motion is *going* without cutting what it is doing --
    velocity carries through the change. Step durations are quantized to the
    model's generation horizon (see streaming.plan_stream); `segments` in the
    response reports the frames actually produced.

    mode="stitch": v0.1 behaviour, independent clips glued by root offset. Only
    useful for comparing seams against stream on the same seed.
    """
    if not req.steps:
        raise HTTPException(status_code=400, detail="steps must be non-empty")
    if req.mode not in ("stream", "stitch"):
        raise HTTPException(status_code=400, detail=f"mode must be 'stream' or 'stitch', got {req.mode!r}")
    try:
        resolved, model = _get_model(req.model)
        fps = model.motion_rep.fps
        steps = req.diffusion_steps or int(model.diffusion.num_base_steps)
        patch = model.num_frames_per_token
        hist = default_history_frames(fps, model.gen_horizon_len, patch)

        if req.mode == "stitch":
            motion, labels, seams, segments = _choreograph_stitch(
                req, model, resolved, fps, steps, hist)
        else:
            plan = plan_stream(req.steps, fps, model.gen_horizon_len)
            motion_tensor = _stream_motion(model, plan, steps, req.cfg_weight,
                                           req.seed, np.deg2rad(req.heading_deg))
            motion = _decode_motion(model, resolved, motion_tensor)
            labels, seams, segments = plan.prompts, plan.seam_frames, plan.segments()

        tag = f"choreo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, " | ".join(labels), tag, seam_frames=seams)
        frames = int(motion["root_positions"].shape[0])
        # Measured on the clip we just wrote, not asserted: the endpoint reports
        # how big its seams actually are so a bad one is visible without a viewer.
        report = seam_report(motion["root_positions"], seams, fps)
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": frames,
                "duration_s": round(frames / float(fps), 3), "mode": req.mode,
                "segments": segments, "seam_frames": seams,
                "seam_velocity": report, "npz": path, "sequence": labels}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DIRECTOR_PORT", "9600")))
