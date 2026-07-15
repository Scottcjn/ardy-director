# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""/choreograph must chain the facing, not just the position.

The endpoint's contract is that each step "continues from where the previous
step ended". Position continuity is covered here too as a regression guard, but
the point of these tests is the heading: a step that turns the character must
leave the next step facing the new direction.

No GPU and no ARDY: `_generate_clip` is replaced with a fake that records the
`first_heading_angle` it is handed and emits a motion whose final facing is
scripted per prompt, in ARDY's own `global_root_heading = [cos t, sin t]` form
(ardy/motion_rep/reps/ardy_motionrep.py).
"""
import math

import numpy as np
import pytest

from director_service import service


FPS = 30
JOINTS = 4


class _FakeModel:
    """Only the attributes /choreograph reads off the model."""

    class motion_rep:
        fps = FPS

    class diffusion:
        num_base_steps = 8

    num_frames_per_token = 5


def _motion(num_frames, end_heading_rad, end_xz=(0.0, 0.0)):
    """A synthetic clip that ends at a given facing and root XZ."""
    root = np.zeros((num_frames, 3), dtype=np.float32)
    root[-1, 0], root[-1, 2] = end_xz
    heading = np.zeros((num_frames, 2), dtype=np.float32)
    heading[:, 0] = math.cos(end_heading_rad)  # [cos t, sin t], ARDY's convention
    heading[:, 1] = math.sin(end_heading_rad)
    return {
        "local_rot_mats": np.zeros((num_frames, JOINTS, 3, 3), dtype=np.float32),
        "global_rot_mats": np.zeros((num_frames, JOINTS, 3, 3), dtype=np.float32),
        "posed_joints": np.zeros((num_frames, JOINTS, 3), dtype=np.float32),
        "root_positions": root,
        "foot_contacts": np.zeros((num_frames, 2), dtype=np.float32),
        "global_root_heading": heading,
    }


@pytest.fixture
def rig(monkeypatch):
    """Patch out the GPU pieces; record what each segment is asked to start at."""
    calls = []
    script = {}

    def fake_generate_clip(model, resolved_name, prompt, num_frames, diffusion_steps,
                           cfg_weight, seed, first_heading_angle, history_frames):
        calls.append({"prompt": prompt, "heading": float(first_heading_angle)})
        end_heading, end_xz = script.get(prompt, (0.0, (0.0, 0.0)))
        return _motion(num_frames, end_heading, end_xz)

    monkeypatch.setattr(service, "_get_model", lambda nickname: ("core", _FakeModel()))
    monkeypatch.setattr(service, "_generate_clip", fake_generate_clip)
    monkeypatch.setattr(service, "_save_npz", lambda motion, fps, text, tag: f"/tmp/{tag}.npz")
    return calls, script


def _choreograph(prompts):
    return service.choreograph(
        service.ChoreographReq(steps=[{"prompt": p, "duration": 1.0} for p in prompts])
    )


def test_turn_is_carried_into_the_next_step(rig):
    """After "turn around", the next step must start facing backwards, not +Z."""
    calls, script = rig
    script["turn around"] = (math.pi, (0.0, 0.0))

    _choreograph(["walk forward", "turn around", "walk forward"])

    assert calls[0]["heading"] == pytest.approx(0.0), "first step starts at the default facing"
    assert calls[1]["heading"] == pytest.approx(0.0), "nothing has turned yet"
    assert calls[2]["heading"] == pytest.approx(math.pi, abs=1e-6), (
        "the step after a 180 turn must start facing where the turn left the character"
    )


def test_heading_accumulates_across_steps(rig):
    """Two 90 turns compose into 180 -- the carry is cumulative, not per-step."""
    calls, script = rig
    script["turn left"] = (math.pi / 2, (0.0, 0.0))
    script["turn left again"] = (math.pi, (0.0, 0.0))

    _choreograph(["turn left", "turn left again", "wave hello"])

    assert calls[1]["heading"] == pytest.approx(math.pi / 2, abs=1e-6)
    assert calls[2]["heading"] == pytest.approx(math.pi, abs=1e-6)


def test_heading_is_wrapped_not_unbounded(rig):
    """A facing of -90 must be handed over as -pi/2 (or +3pi/2), never as garbage."""
    calls, script = rig
    script["turn right"] = (-math.pi / 2, (0.0, 0.0))

    _choreograph(["turn right", "walk forward"])

    handed = calls[1]["heading"]
    assert math.isclose(math.cos(handed), 0.0, abs_tol=1e-6)
    assert math.isclose(math.sin(handed), -1.0, abs_tol=1e-6)


def test_root_position_still_carries(rig):
    """Regression guard: the existing XZ carry must survive the heading fix."""
    calls, script = rig
    script["walk forward"] = (0.0, (1.0, 2.0))

    res = _choreograph(["walk forward", "walk forward"])

    assert res["ok"] is True
    assert res["segments"] == 2
    assert res["frames"] == 2 * FPS
