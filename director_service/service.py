from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import threading
import time
import uuid

#... (existing imports and code)

app = FastAPI(title="ARDY Director", version="0.1.0")

# --- WebSocket setup ---------------------------------------------------------
connected_websockets = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep the connection alive
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

def send_motion_to_websockets(motion_data):
    for ws in connected_websockets:
        try:
            await ws.send_text(json.dumps(motion_data))
        except WebSocketDisconnect:
            connected_websockets.remove(ws)

# --- Generate and Choreograph functions -------------------------------------
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

    # squeeze the leading sample/batch dim (num_samples == 1) so every array is (frames,...)
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

@app.post("/generate")
async def generate(req: GenerateReq):
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
        motion_data = {
            "model": resolved,
            "fps": int(fps),
            "frames": num_frames,
            "duration_s": req.duration,
            "npz": path,
            "prompt": req.prompt,
            "motion": motion
        }
        await send_motion_to_websockets(motion_data)
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": num_frames,
                "duration_s": req.duration, "npz": path, "prompt": req.prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@app.post("/choreograph")
async def choreograph(req: ChoreographReq):
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
            # (arrays are squeezed to (frames,...); index the last XYZ axis, not frames)
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
        tag = f"choreo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, " | ".join(labels), tag)
        total = int(sum(int(s.duration * fps) for s in req.steps))
        motion_data = {
            "model": resolved,
            "fps": int(fps),
            "frames": total,
            "segments": len(req.steps),
            "npz": path,
            "sequence": labels,
            "motion": motion
        }
        await send_motion_to_websockets(motion_data)
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": total,
                "segments": len(req.steps), "npz": path, "sequence": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DIRECTOR_PORT", "9600")))