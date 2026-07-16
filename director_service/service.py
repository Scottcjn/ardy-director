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
  GET  /cameras                     -> camera modes + their parameters
  POST /generate    {prompt,...}    -> one clip from one prompt
  POST /choreograph {steps:[...]}   -> one clip stitched from a prompt sequence
  POST /stage/preview {stage,...}   -> the staged path + camera track, no GPU

Motion is written as ARDY-native .npz under OUTPUT_DIR and the path is returned,
so ARDY's own viewer (scripts/visualize.py) can load it.

A request may also stage the shot: `stage` places the character (start position
plus timed waypoints, fed to ARDY as root-path constraints so it actually walks
the path) and `camera` bakes a per-frame camera track into the .npz.
"""
import os
import threading
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ardy.constraints import Root2DConstraintSet
from ardy.model.load_model import load_model, load_text_encoder
from ardy.model.registry import resolve_model_name
from ardy.postprocess import post_process_motion

try:  # README launches this file as a script; tests import it as a package.
    from . import camera as camera_mod
    from . import staging
except ImportError:  # pragma: no cover - script mode puts this dir on sys.path
    import camera as camera_mod
    import staging

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
                   cfg_weight, seed, first_heading_angle, history_frames,
                   constraint_lst=None):
    """One synchronous ARDY generation → numpy motion dict (single sample).

    `constraint_lst` (e.g. a staged root path) is turned into ARDY's observed
    motion + mask, exactly as scripts/generate.py does, and is also handed to
    the post-processor so the foot fix-up respects the constraint instead of
    fighting it.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lengths = torch.tensor([num_frames], device=DEVICE)
    pad_mask = torch.ones((1, num_frames), dtype=torch.bool, device=DEVICE)
    heading = torch.tensor([first_heading_angle], dtype=torch.float, device=DEVICE)

    observed_motion, motion_mask = None, None
    if constraint_lst:
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraint_lst, lengths, to_normalize=True, device=DEVICE,
        )

    with torch.no_grad():
        motion = model(
            [prompt.strip()],
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=heading,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            crop_history_length=history_frames,
        )
        out = model.motion_rep.inverse(motion, is_normalized=True)

    if "g1" not in resolved_name.lower():
        corrected = post_process_motion(
            out["local_rot_mats"], out["root_positions"], out["foot_contacts"],
            model.skeleton, constraint_lst=constraint_lst or None,
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


# --- staging / camera bridge -------------------------------------------------
def _build_root_constraints(model, plan, face_path=False):
    """numpy staging plan → ARDY's Root2DConstraintSet on the model's skeleton.

    Position only by default, which is what ARDY's own interactive demo
    constrains (scripts/interactive_demo/gen_constraints.py): pinning the facing
    every frame as well would forbid the character from turning on the spot at a
    mark, so `face_path` is opt-in. When it is on, ARDY reads the facing as an
    angle (Root2DConstraintSet.update_constraints takes cos/sin of what it is
    given), so the plan's radians go through as-is.
    """
    device = DEVICE
    frame_indices = torch.tensor(plan["frame_indices"], dtype=torch.long)
    root_2d = torch.tensor(plan["root_2d"], dtype=torch.float, device=device)
    headings = None
    if face_path:
        headings = torch.tensor(plan["headings"], dtype=torch.float, device=device)
    return [Root2DConstraintSet(model.skeleton, frame_indices, root_2d, global_root_heading=headings)]


def _plan_stage(stage, fps, num_frames):
    """Validate + plan, turning a staging complaint into a 400 rather than a 500."""
    try:
        return staging.plan_root_path(
            [wp.model_dump() for wp in stage.waypoints],
            fps=fps, num_frames=num_frames,
            start=(stage.start.x, stage.start.z), dense=stage.dense_path,
        )
    except staging.StagingError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _solve_camera(cam, root_positions, fps):
    try:
        return camera_mod.solve_camera_track(
            cam.mode, root_positions, fps,
            distance=cam.distance, height=cam.height, side=cam.side,
            look_height=cam.look_height, look_ahead=cam.look_ahead,
            orbit_deg_per_s=cam.orbit_deg_per_s, orbit_start_deg=cam.orbit_start_deg,
            smoothing=cam.smoothing, position=cam.position, look_at=cam.look_at,
        )
    except camera_mod.CameraError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _attach_camera(motion, track):
    motion["camera_positions"] = track["positions"]
    motion["camera_targets"] = track["targets"]
    motion["camera_mode"] = np.array(track["mode"])


# --- request models ----------------------------------------------------------
class Waypoint(BaseModel):
    x: float
    z: float
    at: float = Field(..., gt=0.0, description="seconds from the clip start")


class StartPos(BaseModel):
    x: float = 0.0
    z: float = 0.0


class Stage(BaseModel):
    """Where the action happens: a start mark and timed waypoints."""
    waypoints: list[Waypoint] = Field(..., min_length=1)
    start: StartPos = StartPos()
    dense_path: bool = True  # constrain every frame (a walked path) vs. the marks only
    face_path: bool = False  # also pin the facing along the path (stops it turning on the spot)


class Camera(BaseModel):
    """How the shot is framed. Baked per frame into the output .npz."""
    mode: str = "follow"  # follow | orbit | fixed | over_the_shoulder
    distance: float | None = None
    height: float | None = None
    side: float | None = None
    look_height: float | None = None
    look_ahead: float | None = None
    orbit_deg_per_s: float | None = None
    orbit_start_deg: float | None = None
    smoothing: float | None = None
    position: list[float] | None = None  # fixed mode: the tripod, [x, y, z]
    look_at: list[float] | None = None   # fixed mode: aim point; omit to track the character


class GenerateReq(BaseModel):
    prompt: str
    model: str = "core"
    duration: float = Field(4.0, gt=0.1, le=30.0)
    seed: int | None = None
    diffusion_steps: int | None = None
    cfg_weight: float = 4.0
    heading_deg: float | None = None  # initial facing, degrees about +Y (0 = +Z)
    stage: Stage | None = None
    camera: Camera | None = None


class ChoreoStep(BaseModel):
    prompt: str
    duration: float = Field(3.0, gt=0.1, le=30.0)


class ChoreographReq(BaseModel):
    steps: list[ChoreoStep]
    model: str = "core"
    seed: int | None = None
    diffusion_steps: int | None = None
    cfg_weight: float = 4.0
    camera: Camera | None = None


class StagePreviewReq(BaseModel):
    """Dry-run staging: no prompt, no model, no GPU."""
    stage: Stage
    duration: float = Field(4.0, gt=0.1, le=30.0)
    camera: Camera | None = None
    fps: int = Field(30, gt=0, le=240, description="frame rate to plan against (the core model is 30)")


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


@app.get("/cameras")
def cameras():
    return {"modes": list(camera_mod.MODES), "defaults": camera_mod.DEFAULTS,
            "params": {m: list(p) for m, p in camera_mod.PARAMS_FOR.items()},
            "note": "Camera is baked per frame into the .npz as camera_positions/camera_targets."}


@app.post("/stage/preview")
def stage_preview(req: StagePreviewReq):
    """Plan a stage without generating: the root path ARDY would be given, and
    the camera track that path implies. No model load, no GPU — cheap enough to
    iterate on the blocking before spending a generation.

    The camera here is solved against the *planned* path (a straight walk
    between marks); the real shot is solved against ARDY's actual motion, so
    expect the preview to be the intent, not the frame-exact result.
    """
    fps = req.fps
    num_frames = int(req.duration * fps)
    plan = _plan_stage(req.stage, fps, num_frames)
    out = {"ok": True, "fps": fps, "frames": num_frames, "duration_s": req.duration,
           "dense_path": req.stage.dense_path, "stage": staging.summarize(plan, fps),
           "root_path": [[round(float(x), 4), round(float(z), 4)] for x, z in plan["root_2d"]],
           "frame_indices": [int(i) for i in plan["frame_indices"]]}
    if req.camera:
        # Planned path is XZ on the ground; give the camera solver a 3D root.
        root3 = np.zeros((len(plan["root_2d"]), 3), dtype=np.float64)
        root3[:, [0, 2]] = plan["root_2d"]
        track = _solve_camera(req.camera, root3, fps)
        out["camera"] = {"mode": track["mode"], "params": track["params"],
                         "first_position": [round(float(v), 4) for v in track["positions"][0]],
                         "last_position": [round(float(v), 4) for v in track["positions"][-1]],
                         "frames": int(len(track["positions"]))}
    return out


@app.post("/generate")
def generate(req: GenerateReq):
    try:
        resolved, model = _get_model(req.model)
        fps = model.motion_rep.fps
        num_frames = int(req.duration * fps)
        steps = req.diffusion_steps or int(model.diffusion.num_base_steps)
        patch = model.num_frames_per_token
        hist = (int(round(10 * fps)) // patch) * patch  # trained ~10s window

        constraint_lst, plan = None, None
        heading_deg = req.heading_deg
        if req.stage:
            plan = _plan_stage(req.stage, fps, num_frames)
            constraint_lst = _build_root_constraints(model, plan, face_path=req.stage.face_path)
            if heading_deg is None:
                # Face the way the staged path leaves the start mark, unless the
                # director overrode it.
                heading_deg = float(np.rad2deg(plan["headings"][0]))
        heading_deg = 0.0 if heading_deg is None else heading_deg

        motion = _generate_clip(model, resolved, req.prompt, num_frames, steps,
                                req.cfg_weight, req.seed, np.deg2rad(heading_deg), hist,
                                constraint_lst=constraint_lst)
        result = {"ok": True, "model": resolved, "fps": int(fps), "frames": num_frames,
                  "duration_s": req.duration, "prompt": req.prompt,
                  "heading_deg": round(float(heading_deg), 2)}
        if plan:
            result["stage"] = staging.summarize(plan, fps)
        if req.camera:
            track = _solve_camera(req.camera, motion["root_positions"], fps)
            _attach_camera(motion, track)
            result["camera"] = {"mode": track["mode"], "params": track["params"]}
        tag = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        result["npz"] = _save_npz(motion, fps, req.prompt, tag)
        return result
    except HTTPException:
        raise
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
        heading_deg = 0.0
        labels = []
        for i, st in enumerate(req.steps):
            nf = int(st.duration * fps)
            seed_i = None if req.seed is None else req.seed + i
            m = _generate_clip(model, resolved, st.prompt, nf, steps, req.cfg_weight,
                               seed_i, np.deg2rad(heading_deg), hist)
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

        motion = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}
        result = {"ok": True, "model": resolved, "fps": int(fps),
                  "segments": len(req.steps), "sequence": labels}
        if req.camera:
            # Solve over the whole stitched path so the shot carries across seams.
            track = _solve_camera(req.camera, motion["root_positions"], fps)
            _attach_camera(motion, track)
            result["camera"] = {"mode": track["mode"], "params": track["params"]}
        tag = f"choreo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        result["npz"] = _save_npz(motion, fps, " | ".join(labels), tag)
        result["frames"] = int(sum(int(s.duration * fps) for s in req.steps))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DIRECTOR_PORT", "9600")))
