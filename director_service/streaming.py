# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Streaming choreography: window budgets, step planning, seam metrics.

`/choreograph` used to generate each step as an independent clip and glue the
clips together by carrying the root offset. Position was continuous across the
seam, velocity was not, so the character popped at every prompt change.

ARDY streams natively: `Ardy.autoregressive_step` takes the tail of what has
already been generated as `init_history_sequence`, so the next window is
denoised *conditioned on the motion in flight*. Change the text between steps
and the model carries the velocity through the change itself -- there is no
seam to smooth, because there is no cut.

This module holds the arithmetic that decides *what* to generate: how much
history the model may see, how many steps a prompt is worth, and where the
prompt changes land. Plain python/numpy in, plain python/numpy out -- no ARDY,
no torch -- so the planning and the seam metrics are testable without the GPU
host. `service.py` owns the tensor work.
"""
import math
from dataclasses import dataclass

import numpy as np

# ARDY is trained on windows of at most 10 seconds.
TRAINED_WINDOW_SECONDS = 10.0


def default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    """Longest history that, *together with* the generation horizon, fits ARDY's
    trained window.

    Ported from ARDY's own ``scripts/generate.py::_default_history_frames`` (and
    the same budget the interactive demo uses). The subtraction is the point:
    the model sees ``history + horizon`` in one window, so the history has to
    leave room for the frames being generated. Feed it a full 10 s of history
    and every step runs a window longer than anything the model was trained on,
    which ARDY's own docstring warns "degrades into jitter".
    """
    max_window = int(TRAINED_WINDOW_SECONDS * fps) // num_frames_per_token * num_frames_per_token
    usable = (max_window - gen_horizon_len) // num_frames_per_token * num_frames_per_token
    return max(0, usable)


def history_length_for(accumulated_frames: int, history_crop_length: int,
                       num_frames_per_token: int) -> int:
    """Frames of history to hand the next step, given what exists so far.

    Mirrors the interactive demo's ``_get_history_motion``: take what there is,
    clamp to the crop budget, quantize down to a whole number of tokens. Returns
    0 when there is not yet a full token of motion -- the caller passes
    ``init_history_sequence=None`` for that first step.
    """
    if accumulated_frames <= 0 or num_frames_per_token <= 0:
        return 0
    usable = min(accumulated_frames, history_crop_length)
    return max(0, usable // num_frames_per_token * num_frames_per_token)


@dataclass(frozen=True)
class Chunk:
    """One `autoregressive_step`: ``gen_horizon_len`` frames under one prompt."""
    prompt: str
    step_index: int          # which requested step this chunk belongs to
    starts_prompt: bool      # first chunk of its step -- a prompt change (seam)
    start_frame: int         # absolute frame index in the finished clip
    end_frame: int           # exclusive


@dataclass(frozen=True)
class StreamPlan:
    """The full sequence of autoregressive steps for a choreography request."""
    chunks: list
    fps: float
    gen_horizon_len: int
    prompts: list            # one prompt per requested step, in order

    @property
    def total_frames(self) -> int:
        return self.chunks[-1].end_frame if self.chunks else 0

    @property
    def seam_frames(self) -> list:
        """Absolute frame index of each prompt change (excluding the start).

        These are the frames the old stitch popped at, and the frames
        `seam_report.py` measures.
        """
        return [c.start_frame for c in self.chunks if c.starts_prompt and c.start_frame > 0]

    def segments(self) -> list:
        """Per-requested-step spans of the finished clip.

        Durations are quantized to the generation horizon (see `plan_stream`),
        so these are the frames the client actually got, not the ones it asked
        for.
        """
        out = []
        for i, prompt in enumerate(self.prompts):
            mine = [c for c in self.chunks if c.step_index == i]
            if not mine:
                continue
            start, end = mine[0].start_frame, mine[-1].end_frame
            out.append({
                "prompt": prompt,
                "start_frame": start,
                "end_frame": end,
                "frames": end - start,
                "duration_s": round((end - start) / self.fps, 3),
            })
        return out


def plan_stream(steps, fps: float, gen_horizon_len: int) -> StreamPlan:
    """Turn (prompt, duration) steps into a list of autoregressive steps.

    `gen_horizon_len` is ARDY's atomic unit of generation -- one
    `autoregressive_step` emits exactly that many frames -- so a step's duration
    is inherently quantized to it. We round to the nearest whole horizon (never
    below one), and report the frames actually produced in `segments()` rather
    than silently returning a clip of a different length than the caller reads
    off its own request.

    Trimming the overshoot instead is not an option: those frames are what the
    model conditioned the next window on. Cutting them back out would reopen
    exactly the positional jump this endpoint exists to remove.

    Args:
        steps: iterable of objects with `.prompt` and `.duration` (seconds).
        fps: frames per second of the model's motion representation.
        gen_horizon_len: frames ARDY emits per autoregressive step.
    """
    if gen_horizon_len <= 0:
        raise ValueError("gen_horizon_len must be positive")

    chunks, prompts, frame = [], [], 0
    for i, st in enumerate(steps):
        prompts.append(st.prompt)
        want = float(st.duration) * fps
        # floor(x + 0.5), not round(): python rounds halves to even, so a 2.5-
        # horizon beat would round down to 2 and a 3.5 one up to 4. Ties go up --
        # a beat the director asked for should never come back shorter by parity.
        n_chunks = max(1, int(math.floor(want / gen_horizon_len + 0.5)))
        for j in range(n_chunks):
            chunks.append(Chunk(
                prompt=st.prompt,
                step_index=i,
                starts_prompt=(j == 0),
                start_frame=frame,
                end_frame=frame + gen_horizon_len,
            ))
            frame += gen_horizon_len
    return StreamPlan(chunks=chunks, fps=float(fps), gen_horizon_len=int(gen_horizon_len),
                      prompts=prompts)


# --- seam metrics ------------------------------------------------------------
# Root velocity by finite difference of the root's world position. The motion
# representation carries its own velocity channel, but the finite difference is
# what a viewer actually shows: if consecutive frames disagree about where the
# root is going, that is the pop, whatever the feature vector says.

def root_velocity(root_positions: np.ndarray, fps: float) -> np.ndarray:
    """(frames, 3) world positions -> (frames-1, 3) velocities in units/second.

    ``v[i]`` is the velocity carrying frame ``i`` into frame ``i+1``.
    """
    rp = np.asarray(root_positions, dtype=np.float64)
    if rp.ndim != 2 or rp.shape[1] != 3:
        raise ValueError(f"root_positions must be (frames, 3), got {rp.shape}")
    if rp.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float64)
    return np.diff(rp, axis=0) * float(fps)


def _velocity_deltas(root_positions: np.ndarray, fps: float) -> np.ndarray:
    """``dv[i]`` = how much the root's velocity changed at frame ``i+1``."""
    v = root_velocity(root_positions, fps)
    if len(v) < 2:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(v, axis=0), axis=1)


def _seam_indices(frame: int, n_deltas: int) -> list:
    """Which velocity-change samples touch the seam starting at ``frame``.

    Velocity is a finite difference, so the sample ``p[frame] - p[frame-1]``
    straddles the cut: it belongs to neither segment and already carries the new
    segment's displacement. That puts the discontinuity at ``frame-1`` or
    ``frame`` depending on where the cut lands, so both are the seam.
    """
    return [i for i in (frame - 1, frame) if 1 <= i <= n_deltas]


def velocity_jump_at(root_positions: np.ndarray, frame: int, fps: float) -> float:
    """Magnitude of the velocity discontinuity at ``frame``, in units/second.

    ``frame`` is the first frame of a new segment. Smooth motion changes
    velocity gradually (bounded by what one frame of acceleration can do); a
    stitch changes it in a single step, which is the pop.
    """
    dv = _velocity_deltas(root_positions, fps)
    idx = _seam_indices(int(frame), len(dv))
    return max((float(dv[i - 1]) for i in idx), default=0.0)


def seam_report(root_positions: np.ndarray, seam_frames, fps: float) -> dict:
    """Velocity jump at each seam vs. the clip's typical frame-to-frame change.

    ``ratio`` is the honest number to look at: a seam jump means little on its
    own (a fast run has big velocities), but a seam that changes velocity many
    times harder than a normal frame is a pop you can see. Baseline is the
    median over non-seam frames, so one bad seam cannot inflate it.
    """
    dv = _velocity_deltas(root_positions, fps)
    if len(dv) == 0:
        return {"seams": [], "baseline_jump": 0.0, "max_ratio": 0.0}

    seams = sorted({int(f) for f in seam_frames if _seam_indices(int(f), len(dv))})
    # Both samples touching a seam are excluded from the baseline: leaving the
    # straddling one in would let a pop raise the bar it is measured against.
    touched = {i for f in seams for i in _seam_indices(f, len(dv))}
    non_seam = [dv[i - 1] for i in range(1, len(dv) + 1) if i not in touched]
    baseline = float(np.median(non_seam)) if non_seam else 0.0

    rows = []
    for f in seams:
        jump = max(float(dv[i - 1]) for i in _seam_indices(f, len(dv)))
        rows.append({
            "frame": f,
            "time_s": round(f / float(fps), 3),
            "jump": jump,
            "ratio": (jump / baseline) if baseline > 1e-9 else float("inf") if jump > 1e-9 else 0.0,
        })
    return {
        "seams": rows,
        "baseline_jump": baseline,
        "max_ratio": max((r["ratio"] for r in rows), default=0.0),
    }
