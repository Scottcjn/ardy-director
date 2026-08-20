# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Service wiring: does a staged request reach ARDY as the right constraint?

ARDY itself is stubbed (conftest). The fake model records the tensors the
service hands it, so these tests assert the contract with ARDY's generation API
-- shapes, dtypes, and that the constraint is passed at all -- which is exactly
what silently degrades into "the character ignores the waypoints".
"""
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from director_service import service

FPS = 30


class FakeMotionRep:
    fps = FPS

    def __init__(self):
        self.conditions_calls = []

    def create_conditions_from_constraints_batched(self, constraint_lst, lengths, to_normalize, device):
        self.conditions_calls.append({"constraint_lst": constraint_lst, "lengths": lengths,
                                      "to_normalize": to_normalize, "device": device})
        n = int(lengths.max())
        return torch.zeros((1, n, 4)), torch.ones((1, n, 4), dtype=torch.bool)

    def inverse(self, motion, is_normalized):
        return motion


class FakeDiffusion:
    num_base_steps = 4


class FakeModel:
    """Returns a straight walk down +Z; records every call kwarg."""

    skeleton = "fake-skeleton"
    motion_rep = FakeMotionRep()
    diffusion = FakeDiffusion()
    num_frames_per_token = 4

    def __init__(self):
        self.calls = []
        self.motion_rep = FakeMotionRep()

    def __call__(self, texts, num_frames, **kw):
        self.calls.append({"texts": texts, "num_frames": num_frames, **kw})
        t = np.arange(num_frames) / FPS
        root = np.stack([np.zeros(num_frames), np.zeros(num_frames), t], axis=-1)
        return {
            "root_positions": torch.tensor(root[None], dtype=torch.float),
            "local_rot_mats": torch.zeros((1, num_frames, 27, 3, 3)),
            "global_rot_mats": torch.zeros((1, num_frames, 27, 3, 3)),
            "posed_joints": torch.zeros((1, num_frames, 27, 3)),
            "foot_contacts": torch.zeros((1, num_frames, 2)),
            "global_root_heading": torch.tensor(
                np.stack([np.ones(num_frames), np.zeros(num_frames)], axis=-1)[None], dtype=torch.float),
        }


@pytest.fixture
def model(monkeypatch, tmp_path):
    m = FakeModel()
    monkeypatch.setattr(service, "_get_model", lambda nick: ("core", m))
    monkeypatch.setattr(service, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(service, "post_process_motion", lambda *a, **kw: {})
    return m


@pytest.fixture
def client():
    return TestClient(service.app)


STAGE = {"waypoints": [{"x": 0.0, "z": 3.0, "at": 2.0}, {"x": 2.0, "z": 5.0, "at": 4.0}],
         "start": {"x": 0.0, "z": 0.0}, "dense_path": True}


# --- the point of the whole bounty: waypoints must become root constraints ----
def test_staged_generate_hands_ardy_a_root2d_constraint(client, model):
    r = client.post("/generate", json={"prompt": "walk", "duration": 6.0, "stage": STAGE})
    assert r.status_code == 200, r.text

    call = model.calls[-1]
    assert call["observed_motion"] is not None, "constraints built but never passed to ARDY"
    assert call["motion_mask"] is not None

    built = model.motion_rep.conditions_calls[-1]
    (constraint,) = built["constraint_lst"]
    assert constraint.skeleton == "fake-skeleton"
    assert len(constraint.frame_indices) == 121          # dense: frames 0..120
    assert constraint.root_2d.shape == (121, 2)          # ARDY wants XZ only
    assert built["to_normalize"] is True
    # The mark itself lands where and when it was called.
    assert constraint.frame_indices[60].item() == 60
    assert constraint.root_2d[60].tolist() == pytest.approx([0.0, 3.0], abs=1e-4)
    assert constraint.root_2d[-1].tolist() == pytest.approx([2.0, 5.0], abs=1e-4)


def test_constraint_tensors_are_the_dtypes_ardy_indexes_with(client, model):
    client.post("/generate", json={"prompt": "walk", "duration": 6.0, "stage": STAGE})
    (constraint,) = model.motion_rep.conditions_calls[-1]["constraint_lst"]
    assert constraint.frame_indices.dtype == torch.long   # used as an index
    assert constraint.root_2d.dtype == torch.float


def test_facing_is_not_pinned_by_default(client, model):
    """ARDY's own demo constrains position only; pinning the facing every frame
    would forbid turning on the spot at a mark, so it must be opt-in."""
    client.post("/generate", json={"prompt": "walk", "duration": 6.0, "stage": STAGE})
    (constraint,) = model.motion_rep.conditions_calls[-1]["constraint_lst"]
    assert constraint.global_root_heading is None


def test_face_path_pins_the_facing_along_the_path(client, model):
    staged = dict(STAGE, face_path=True)
    client.post("/generate", json={"prompt": "walk", "duration": 6.0, "stage": staged})
    (constraint,) = model.motion_rep.conditions_calls[-1]["constraint_lst"]
    assert constraint.global_root_heading is not None
    assert constraint.global_root_heading.shape == (121,)   # radians, one per frame
    assert constraint.global_root_heading.dtype == torch.float
    assert constraint.global_root_heading[30].item() == pytest.approx(0.0, abs=1e-4)  # walking +Z


def test_unstaged_generate_still_asks_ardy_for_nothing(client, model):
    r = client.post("/generate", json={"prompt": "walk", "duration": 2.0})
    assert r.status_code == 200
    call = model.calls[-1]
    assert call["observed_motion"] is None and call["motion_mask"] is None
    assert model.motion_rep.conditions_calls == []
    assert "stage" not in r.json()


def test_post_processing_is_told_about_the_constraint(client, model, monkeypatch):
    """Foot fix-up that ignores the constraint fights the staged path."""
    seen = {}
    monkeypatch.setattr(service, "post_process_motion",
                        lambda *a, **kw: seen.update(kw) or {})
    client.post("/generate", json={"prompt": "walk", "duration": 6.0, "stage": STAGE})
    assert seen["constraint_lst"] is not None
    client.post("/generate", json={"prompt": "walk", "duration": 2.0})
    assert seen["constraint_lst"] is None


def test_staged_generate_faces_the_way_the_path_leaves(client, model):
    stage = {"waypoints": [{"x": 3.0, "z": 0.0, "at": 2.0}]}  # walking +X
    r = client.post("/generate", json={"prompt": "walk", "duration": 4.0, "stage": stage})
    assert r.json()["heading_deg"] == pytest.approx(90.0, abs=1e-2)
    assert model.calls[-1]["first_heading_angle"].item() == pytest.approx(np.pi / 2, abs=1e-4)


def test_an_explicit_heading_still_wins(client, model):
    stage = {"waypoints": [{"x": 3.0, "z": 0.0, "at": 2.0}]}
    r = client.post("/generate", json={"prompt": "walk", "duration": 4.0,
                                       "stage": stage, "heading_deg": 0.0})
    assert r.json()["heading_deg"] == 0.0
    assert model.calls[-1]["first_heading_angle"].item() == pytest.approx(0.0, abs=1e-6)


# --- camera rides in the npz -------------------------------------------------
def test_camera_track_is_saved_alongside_the_motion(client, model):
    r = client.post("/generate", json={"prompt": "walk", "duration": 2.0,
                                       "camera": {"mode": "follow", "distance": 4.0}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["camera"]["mode"] == "follow"
    assert body["camera"]["params"]["distance"] == 4.0

    with np.load(body["npz"]) as npz:
        assert npz["camera_positions"].shape == (60, 3)
        assert npz["camera_targets"].shape == (60, 3)
        assert str(npz["camera_mode"]) == "follow"
        # Solved against the motion ARDY actually produced, not the plan.
        assert npz["camera_positions"][40][2] == pytest.approx(
            npz["root_positions"][40][2] - 4.0, abs=1e-3)


def test_no_camera_asked_no_camera_keys(client, model):
    r = client.post("/generate", json={"prompt": "walk", "duration": 2.0})
    with np.load(r.json()["npz"]) as npz:
        assert "camera_positions" not in npz.files
    assert "camera" not in r.json()


def test_choreograph_films_across_the_seams(client, model):
    r = client.post("/choreograph", json={
        "steps": [{"prompt": "walk", "duration": 1.0}, {"prompt": "wave", "duration": 1.0}],
        "camera": {"mode": "orbit"}})
    assert r.status_code == 200, r.text
    assert r.json()["camera"]["mode"] == "orbit"
    with np.load(r.json()["npz"]) as npz:
        assert len(npz["camera_positions"]) == len(npz["root_positions"])  # one camera, whole clip


# --- preview: no model, no GPU ----------------------------------------------
def test_preview_plans_without_touching_the_model(client, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("/stage/preview must not load a model")

    monkeypatch.setattr(service, "_get_model", boom)
    r = client.post("/stage/preview", json={"stage": STAGE, "duration": 6.0,
                                            "camera": {"mode": "follow"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage"]["last_frame"] == 120
    assert body["stage"]["path_length_m"] == pytest.approx(3.0 + np.hypot(2.0, 2.0), abs=1e-3)
    assert len(body["root_path"]) == 121
    assert body["root_path"][60] == pytest.approx([0.0, 3.0], abs=1e-3)
    assert body["camera"]["frames"] == 121


def test_preview_names_a_stage_that_cannot_fit(client):
    r = client.post("/stage/preview", json={
        "stage": {"waypoints": [{"x": 0.0, "z": 1.0, "at": 8.0}]}, "duration": 4.0})
    assert r.status_code == 400
    assert "lengthen the clip" in r.json()["detail"]


def test_bad_stage_is_a_400_not_a_500(client, model):
    """A director's mistake must not read as the service falling over."""
    r = client.post("/generate", json={
        "prompt": "walk", "duration": 4.0,
        "stage": {"waypoints": [{"x": 0.0, "z": 1.0, "at": 9.0}]}})
    assert r.status_code == 400, r.text
    assert "lengthen the clip" in r.json()["detail"]


def test_bad_camera_is_a_400_not_a_500(client, model):
    r = client.post("/generate", json={"prompt": "walk", "duration": 2.0,
                                       "camera": {"mode": "crane"}})
    assert r.status_code == 400, r.text
    assert "unknown camera mode" in r.json()["detail"]


def test_cameras_endpoint_lists_the_modes(client):
    body = client.get("/cameras").json()
    assert set(body["modes"]) == {"follow", "orbit", "fixed", "over_the_shoulder"}
    assert "distance" in body["defaults"]
