from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import time
import uuid
from typing import List

app = FastAPI()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChoreographReq(BaseModel):
    steps: List[dict]
    model: str
    seed: int = None
    cfg_weight: float = 7.5

def _get_model(model_name):
    # Placeholder for model retrieval logic
    class MockModel:
        motion_rep = None
        diffusion = None
        num_frames_per_token = 16
        def autoregressive_step(self, prompts, num_frames, num_denoising_steps, pad_mask, first_heading_angle, observed_motion, cfg_weight, crop_history_length):
            # Placeholder for actual autoregressive step logic
            pass
    resolved_model = MockModel()
    return model_name, resolved_model

def _save_npz(motion, fps, description, tag):
    # Placeholder for saving logic
    return f"./{tag}.npz"

def post_process_motion(local_rot_mats, root_positions, foot_contacts, skeleton, constraint_lst=None):
    # Placeholder for post-processing logic
    return {}

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
        observed_motion = None

        for i, st in enumerate(req.steps):
            nf = int(st.duration * fps)
            seed_i = None if req.seed is None else req.seed + i

            # Use autoregressive_step for streaming
            with torch.no_grad():
                motion = model.autoregressive_step(
                    [st.prompt.strip()],
                    num_frames=nf,
                    num_denoising_steps=steps,
                    pad_mask=torch.ones((1, nf), dtype=torch.bool, device=DEVICE),
                    first_heading_angle=torch.tensor([np.deg2rad(heading_deg)], dtype=torch.float, device=DEVICE),
                    observed_motion=observed_motion,
                    cfg_weight=req.cfg_weight,
                    crop_history_length=hist,
                )
                out = model.motion_rep.inverse(motion, is_normalized=True)

            if "g1" not in resolved.lower():
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

            # carry root XZ so the next segment starts where this one ended
            # (arrays are squeezed to (frames,...); index the last XYZ axis, not frames)
            rp = result["root_positions"].copy()  # (frames, 3)
            rp[:, [0, 2]] += root_offset[[0, 2]]
            result["root_positions"] = rp
            result["posed_joints"][..., [0, 2]] += root_offset[[0, 2]]  # (frames, joints, 3)
            root_offset = rp[-1].copy()

            for k in seg_keys:
                if k in result:
                    acc[k].append(result[k])
            labels.append(st.prompt)

            # Update observed_motion for the next step
            observed_motion = motion

        motion = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}
        tag = f"choreo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = _save_npz(motion, fps, " | ".join(labels), tag)
        total = int(sum(int(s.duration * fps) for s in req.steps))
        return {"ok": True, "model": resolved, "fps": int(fps), "frames": total,
                "segments": len(req.steps), "npz": path, "sequence": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
