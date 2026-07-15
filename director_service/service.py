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
  POST /set_camera {mode,...}      -> set camera mode + params (broadcast to viewers)
  GET  /camera                     -> current camera config
  POST /set_stage  {waypoints,...} -> set staging waypoints (broadcast to viewers)
  GET  /stage                      -> current stage config
  WS   /ws                         -> WebSocket: receive play commands, stream frames

Motion is written as ARDY-native .npz under OUTPUT_DIR (for backward compat) AND
buffered for live streaming to connected WebSocket viewers.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ardy.model.load_model import load_model, load_text_encoder
from ardy.model.registry import resolve_model_name
from ardy.postprocess import post_process_motion

OUTPUT_DIR = os.environ.get("DIRECTOR_OUTPUT_DIR", os.path.expanduser("~/ardy/outputs/director"))
ENCODER_URL = os.environ.get("DIRECTOR_ENCODER_URL", "http://localhost:9550")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="ARDY Director", version="0.2.0")

# --- lazily-loaded, cached model registry (shared text encoder) --------------
_lock = threading.Lock()
_text_encoder = None
_models = {}  # nickname -> loaded Ardy model


def _encoder():
    global _text_encoder
    if _text_encoder is None:
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
                   root_constraint=None):
    """One synchronous ARDY generation -> numpy motion dict (single sample).

    root_constraint: optional (num_frames, 2) array of (x, z) waypoints to
    guide the root path.  Passed as observed_motion if ARDY's API accepts it;
    otherwise used as a post-hoc blend target.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lengths = torch.tensor([num_frames], device=DEVICE)
    pad_mask = torch.ones((1, num_frames), dtype=torch.bool, device=DEVICE)
    heading = torch.tensor([first_heading_angle], dtype=torch.float, device=DEVICE)

    # Build observed_motion / motion_mask from root constraint if provided
    observed_motion = None
    motion_mask = None
    if root_constraint is not None and len(root_constraint) > 0:
        try:
            path_2d = _interpolate_root_path(root_constraint, num_frames)  # (N, 2)
            obs_root = torch.zeros((1, num_frames, 3), dtype=torch.float32, device=DEVICE)
            obs_root[0, :, 0] = torch.from_numpy(path_2d[:, 0]).to(DEVICE)
            obs_root[0, :, 2] = torch.from_numpy(path_2d[:, 1]).to(DEVICE)
            obs = model.motion_rep.forward(
                {"root_positions": obs_root}
            )
            observed_motion = obs
            motion_mask = torch.zeros(
                (1, *obs.shape[1:-1], 1), dtype=torch.bool, device=DEVICE
            )
            motion_mask[..., :] = True
        except Exception:
            observed_motion = None
            motion_mask = None

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

    # Post-hoc root path blend if ARDY's constraint didn't take (observed fallback)
    if root_constraint is not None and len(root_constraint) > 0:
        _blend_root_path(result, root_constraint, num_frames)

    return result


# --- root-path helpers -------------------------------------------------------


def _interpolate_root_path(waypoints: list, num_frames: int) -> np.ndarray:
    """Interpolate list of {x, z} waypoints into a smooth (N, 2) path."""
    pts = np.asarray([(wp["x"], wp["z"]) if isinstance(wp, dict) else (wp.x, wp.z)
                      for wp in waypoints], dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((num_frames, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.tile(pts[0], (num_frames, 1))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(seg_lens)])
    total = float(cum_dist[-1])
    if total < 1e-8:
        return np.tile(pts[0], (num_frames, 1))
    t = np.linspace(0, total, num_frames)
    out = np.zeros((num_frames, 2), dtype=np.float32)
    for i in range(num_frames):
        idx = int(np.searchsorted(cum_dist, t[i], side="right")) - 1
        idx = max(0, min(idx, len(pts) - 2))
        seg_t = (t[i] - cum_dist[idx]) / max(seg_lens[idx], 1e-8)
        seg_t = float(np.clip(seg_t, 0, 1))
        out[i] = pts[idx] + seg_t * diffs[idx]
    return out


def _blend_root_path(motion: dict, waypoints: list, num_frames: int):
    """Post-hoc blend: shift root positions toward interpolated waypoints."""
    target = _interpolate_root_path(waypoints, num_frames)  # (N, 2)
    rp = motion.get("root_positions")
    pj = motion.get("posed_joints")
    if rp is None or pj is None:
        return
    blend = np.linspace(0.0, 1.0, num_frames, dtype=np.float32) ** 0.5
    for i in range(num_frames):
        dx = target[i, 0] - rp[i, 0]
        dz = target[i, 1] - rp[i, 2]
        b = blend[i]
        rp[i, 0] += dx * b
        rp[i, 2] += dz * b
        pj[i, :, 0] += dx * b
        pj[i, :, 2] += dz * b


def _resolve_waypoints(req_waypoints, req_stage_id, start_x, start_z):
    """Resolve waypoints from request body or stored stage state."""
    waypoints = []
    sx, sz = start_x, start_z
    heading = None

    if req_waypoints is not None:
        waypoints = req_waypoints
    elif req_stage_id:
        with _stage_lock:
            stored = _stage_state.get(req_stage_id)
        if stored:
            waypoints = stored.get("waypoints", [])
            if sx is None:
                sx = stored.get("start_x", 0.0)
            if sz is None:
                sz = stored.get("start_z", 0.0)
            if heading is None:
                heading = stored.get("start_heading_deg")

    if sx is not None and sz is not None and len(waypoints) > 0:
        first = waypoints[0]
        if isinstance(first, dict):
            first["x"] = sx
            first["z"] = sz
        else:
            first.x = sx
            first.z = sz

    if heading is None and len(waypoints) >= 2:
        w0 = waypoints[0]
        w1 = waypoints[1]
        dx = (w1["x"] if isinstance(w1, dict) else w1.x) - (w0["x"] if isinstance(w0, dict) else w0.x)
        dz = (w1["z"] if isinstance(w1, dict) else w1.z) - (w0["z"] if isinstance(w0, dict) else w0.z)
        if abs(dx) > 1e-6 or abs(dz) > 1e-6:
            heading = float(np.rad2deg(np.arctan2(dx, dz)))

    return waypoints, heading


# --- WebSocket connection manager + motion buffer ----------------------------
class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self._connections.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        with self._lock:
            conns = list(self._connections)
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._connections)


_manager = ConnectionManager()

# Motion buffer: id -> {motion, fps, prompt, label}
_motion_buf: dict[str, dict] = {}
_buf_lock = threading.Lock()


def _buffer_motion(motion: dict, fps: int, prompt: str, label: str = "") -> str:
    cid = uuid.uuid4().hex[:12]
    with _buf_lock:
        _motion_buf[cid] = {"motion": motion, "fps": fps, "prompt": prompt, "label": label or prompt}
    return cid


async def _stream_motion(cid: str, loop_count: int = 1):
    """Broadcast a buffered motion to all connected viewers at its native FPS."""
    with _buf_lock:
        entry = _motion_buf.get(cid)
    if entry is None:
        return

    motion = entry["motion"]
    fps = entry["fps"]
    posed = motion.get("posed_joints", None)
    root_pos = motion.get("root_positions", None)
    rots = motion.get("local_rot_mats", None)
    if posed is None:
        return

    num_frames = posed.shape[0]
    delay = 1.0 / fps

    for loop in range(loop_count):
        await _manager.broadcast({
            "type": "start",
            "id": cid,
            "prompt": entry["prompt"],
            "label": entry["label"],
            "fps": fps,
            "total_frames": num_frames,
            "loop": loop,
            "total_loops": loop_count,
        })
        for i in range(num_frames):
            frame = {
                "type": "frame",
                "id": cid,
                "frame": i,
                "total": num_frames,
                "posed_joints": posed[i].tolist(),
                "root_position": root_pos[i].tolist() if root_pos is not None else None,
            }
            if rots is not None:
                frame["local_rot_mats"] = rots[i].tolist()
            await _manager.broadcast(frame)
            await asyncio.sleep(delay)

    await _manager.broadcast({"type": "done", "id": cid})


# --- camera state ------------------------------------------------------------
_CAMERA_DEFAULTS = {
    "mode": "follow",
    "distance": 3.0, "height": 1.5,
    "radius": 4.0, "orbit_speed": 0.3,
    "position": [0.0, 2.0, 5.0],
    "target": [0.0, 1.0, 0.0],
    "side_offset": 0.3, "forward_offset": 0.5,
}

_camera_state: dict = dict(_CAMERA_DEFAULTS)
_camera_lock = threading.Lock()


def _get_camera_state() -> dict:
    with _camera_lock:
        return dict(_camera_state)


def _set_camera_state(state: dict):
    merged = dict(_CAMERA_DEFAULTS)
    merged.update(state)
    with _camera_lock:
        _camera_state.clear()
        _camera_state.update(merged)


# --- stage state ------------------------------------------------------------
# Stores named stage setups: id -> {waypoints, start_x, start_z, start_heading_deg}
_stage_state: dict[str, dict] = {}
_stage_lock = threading.Lock()


def _save_stage(sid: str, data: dict):
    with _stage_lock:
        _stage_state[sid] = dict(data)


# --- request / response models -----------------------------------------------
class GenerateReq(BaseModel):
    prompt: str
    model: str = "core"
    duration: float = Field(4.0, gt=0.1, le=30.0)
    seed: int | None = None
    diffusion_steps: int | None = None
    cfg_weight: float = 4.0
    heading_deg: float = 0.0
    stage_id: str | None = None
    waypoints: list[dict] | None = None
    start_x: float | None = None
    start_z: float | None = None


class ChoreoStep(BaseModel):
    prompt: str
    duration: float = Field(3.0, gt=0.1, le=30.0)


class ChoreographReq(BaseModel):
    steps: list[ChoreoStep]
    model: str = "core"
    seed: int | None = None
    diffusion_steps: int | None = None
    cfg_weight: float = 4.0
    stage_id: str | None = None
    waypoints: list[dict] | None = None
    start_x: float | None = None
    start_z: float | None = None


class CameraReq(BaseModel):
    mode: str = "follow"
    distance: float | None = None
    height: float | None = None
    radius: float | None = None
    orbit_speed: float | None = None
    position: list[float] | None = None
    target: list[float] | None = None
    side_offset: float | None = None
    forward_offset: float | None = None


class StageReq(BaseModel):
    name: str = "default"
    waypoints: list[dict] = []
    start_x: float = 0.0
    start_z: float = 0.0
    start_heading_deg: float = 0.0


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
async def generate(req: GenerateReq):
    try:
        resolved, model = _get_model(req.model)
        fps = model.motion_rep.fps
        num_frames = int(req.duration * fps)
        steps = req.diffusion_steps or int(model.diffusion.num_base_steps)
        patch = model.num_frames_per_token
        hist = (int(round(10 * fps)) // patch) * patch

        waypoints, heading = _resolve_waypoints(
            req.waypoints, req.stage_id, req.start_x, req.start_z,
        )
        if heading is not None:
            req.heading_deg = float(heading)

        motion = _generate_clip(
            model, resolved, req.prompt, num_frames, steps,
            req.cfg_weight, req.seed, np.deg2rad(req.heading_deg), hist,
            root_constraint=waypoints if waypoints else None,
        )
        tag = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, req.prompt, tag)
        cid = _buffer_motion(motion, fps, req.prompt)
        if _manager.count > 0:
            asyncio.create_task(_stream_motion(cid))
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": num_frames,
                "duration_s": req.duration, "npz": path, "prompt": req.prompt,
                "stream_id": cid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/choreograph")
async def choreograph(req: ChoreographReq):
    if not req.steps:
        raise HTTPException(status_code=400, detail="steps must be non-empty")
    try:
        resolved, model = _get_model(req.model)
        fps = model.motion_rep.fps
        steps = req.diffusion_steps or int(model.diffusion.num_base_steps)
        patch = model.num_frames_per_token
        hist = (int(round(10 * fps)) // patch) * patch

        waypoints, heading = _resolve_waypoints(
            req.waypoints, req.stage_id, req.start_x, req.start_z,
        )

        seg_keys = ["local_rot_mats", "global_rot_mats", "posed_joints",
                    "root_positions", "foot_contacts", "global_root_heading"]
        acc = {k: [] for k in seg_keys}
        root_offset = np.zeros(3, dtype=np.float32)
        heading_deg = float(heading) if heading is not None else 0.0
        labels = []
        for i, st in enumerate(req.steps):
            nf = int(st.duration * fps)
            seed_i = None if req.seed is None else req.seed + i
            m = _generate_clip(model, resolved, st.prompt, nf, steps, req.cfg_weight,
                               seed_i, np.deg2rad(heading_deg), hist)
            rp = m["root_positions"].copy()
            rp[:, [0, 2]] += root_offset[[0, 2]]
            m["root_positions"] = rp
            m["posed_joints"][..., [0, 2]] += root_offset[[0, 2]]
            root_offset = rp[-1].copy()
            for k in seg_keys:
                if k in m:
                    acc[k].append(m[k])
            labels.append(st.prompt)

        motion = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}

        # Post-hoc blend entire choreography toward waypoints if provided
        if waypoints:
            total_frames = int(sum(int(s.duration * fps) for s in req.steps))
            _blend_root_path(motion, waypoints, total_frames)

        tag = f"choreo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, " | ".join(labels), tag)
        cid = _buffer_motion(motion, fps, " | ".join(labels), labels[0] if labels else "")
        if _manager.count > 0:
            asyncio.create_task(_stream_motion(cid))
        total = int(sum(int(s.duration * fps) for s in req.steps))
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": total,
                "segments": len(req.steps), "npz": path, "sequence": labels,
                "stream_id": cid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# --- camera endpoints --------------------------------------------------------
@app.post("/set_camera")
async def set_camera(req: CameraReq):
    merged = dict(_CAMERA_DEFAULTS)
    for k, v in req.model_dump(exclude_none=True).items():
        if v is not None:
            merged[k] = v
    _set_camera_state(merged)
    asyncio.create_task(_broadcast_camera())
    return {"ok": True, "camera": _get_camera_state()}


@app.get("/camera")
def get_camera():
    return _get_camera_state()


async def _broadcast_camera():
    await _manager.broadcast({
        "type": "camera",
        "camera": _get_camera_state(),
    })


# --- stage endpoints ---------------------------------------------------------
@app.post("/set_stage")
async def set_stage(req: StageReq):
    data = req.model_dump()
    _save_stage(req.name, data)
    asyncio.create_task(_broadcast_stage(req.name))
    return {"ok": True, "stage": data}


@app.get("/stage")
def get_stage(name: str = "default"):
    with _stage_lock:
        s = _stage_state.get(name)
    if s is None:
        raise HTTPException(status_code=404, detail=f"stage '{name}' not found")
    return s


@app.get("/stages")
def list_stages():
    with _stage_lock:
        return {"stages": list(_stage_state.keys())}


async def _broadcast_stage(name: str):
    with _stage_lock:
        data = _stage_state.get(name)
    if data:
        await _manager.broadcast({
            "type": "stage",
            "name": name,
            "stage": data,
        })


# --- WebSocket endpoint ------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await _manager.connect(ws)
    # Push current camera + stage state on connect
    asyncio.create_task(_push_state_on_connect(ws))
    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("command")

            if cmd == "play":
                cid = data.get("id", "")
                loops = data.get("loop", 1)
                with _buf_lock:
                    exists = cid in _motion_buf
                if exists:
                    asyncio.create_task(_stream_motion(cid, loops))
                else:
                    await ws.send_json({"type": "error", "message": f"clip '{cid}' not found"})

            elif cmd == "stop":
                pass

            elif cmd == "list":
                with _buf_lock:
                    clips = [{"id": k, "label": v["label"], "prompt": v["prompt"],
                              "fps": v["fps"],
                              "frames": int(v["motion"]["posed_joints"].shape[0])}
                             for k, v in _motion_buf.items()]
                await ws.send_json({"type": "clip_list", "clips": clips})

            elif cmd == "generate":
                prompt = data.get("prompt", "")
                duration = data.get("duration", 4.0)
                model_name = data.get("model", "core")
                loops = data.get("loop", 1)
                if not prompt:
                    await ws.send_json({"type": "error", "message": "prompt is required"})
                    continue
                try:
                    resolved, model = _get_model(model_name)
                    fps = model.motion_rep.fps
                    nf = int(duration * fps)
                    steps = data.get("diffusion_steps") or int(model.diffusion.num_base_steps)
                    patch = model.num_frames_per_token
                    hist = (int(round(10 * fps)) // patch) * patch
                    motion = _generate_clip(
                        model, resolved, prompt, nf, steps,
                        data.get("cfg_weight", 4.0),
                        data.get("seed"), np.deg2rad(data.get("heading_deg", 0.0)),
                        hist,
                    )
                    cid = _buffer_motion(motion, fps, prompt)
                    tag = f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                    _save_npz(motion, fps, prompt, tag)
                    await ws.send_json({
                        "type": "generated", "id": cid, "fps": int(fps),
                        "frames": nf, "duration_s": duration, "npz": tag + ".npz",
                    })
                    if loops > 0:
                        asyncio.create_task(_stream_motion(cid, loops))
                except Exception as e:
                    await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})

            elif cmd == "set_camera":
                merged = dict(_CAMERA_DEFAULTS)
                for k in merged:
                    if k in data:
                        merged[k] = data[k]
                _set_camera_state(merged)
                asyncio.create_task(_broadcast_camera())

            elif cmd == "set_stage":
                name = data.get("name", "default")
                stage_data = {
                    "waypoints": data.get("waypoints", []),
                    "start_x": data.get("start_x", 0.0),
                    "start_z": data.get("start_z", 0.0),
                    "start_heading_deg": data.get("start_heading_deg", 0.0),
                }
                _save_stage(name, stage_data)
                asyncio.create_task(_broadcast_stage(name))

    except WebSocketDisconnect:
        _manager.disconnect(ws)


async def _push_state_on_connect(ws: WebSocket):
    try:
        await ws.send_json({
            "type": "camera",
            "camera": _get_camera_state(),
        })
        with _stage_lock:
            for sid, sdata in _stage_state.items():
                await ws.send_json({
                    "type": "stage",
                    "name": sid,
                    "stage": sdata,
                })
    except Exception:
        pass


# --- helpers -----------------------------------------------------------------
def _save_npz(motion, fps, text, tag):
    path = os.path.join(OUTPUT_DIR, f"{tag}.npz")
    np.savez(path, fps=np.int64(fps), text=np.array(text), **motion)
    return path


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DIRECTOR_PORT", "9600")))
