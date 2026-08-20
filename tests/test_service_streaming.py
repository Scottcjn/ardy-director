# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Service wiring: does /choreograph drive ARDY's streaming API correctly?

ARDY itself is stubbed (conftest). `FakeArdy` is not a simulation of the model
-- it is a recording of the contract `Ardy.autoregressive_step` actually
imposes, taken from ardy/model/ardy_model.py and the interactive demo it ships:

  * `num_frames` must be a whole number of tokens, and is history + horizon;
  * the return value is history + newly generated frames, not just the new ones;
  * history goes in as `init_history_sequence`; the world pose (translation and
    heading) is only set on the step that has no history.

Get any of those wrong and the failure is silent on a GPU -- duplicated frames,
a stream that resets its facing every beat, or a window longer than the model
was trained on -- so the fake asserts them the way the real model would.
"""
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from director_service import service
from director_service.streaming import default_history_frames

FPS = 30
HORIZON = 30       # frames per autoregressive step
PATCH = 10         # num_frames_per_token
FEATS = 8


class FakeMotionRep:
    fps = FPS
    nfeats_dict = {"root_pos": 3}

    def inverse(self, motion, is_normalized=True):
        # The service only reads shapes/values back out; carry the frame ids in
        # root_positions so tests can see exactly which frames survived.
        frames = motion.shape[1]
        ids = motion[..., 0]                      # [1, frames]
        root = torch.zeros((1, frames, 3))
        root[..., 2] = ids                        # walk +Z, one unit per frame id
        return {
            "root_positions": root,
            "posed_joints": torch.zeros((1, frames, 27, 3)),
            "local_rot_mats": torch.zeros((1, frames, 27, 3, 3)),
            "global_rot_mats": torch.zeros((1, frames, 27, 3, 3)),
            "foot_contacts": torch.zeros((1, frames, 2)),
            "global_root_heading": torch.zeros((1, frames)),
        }


class FakeDiffusion:
    num_base_steps = 8


class FakeArdy:
    """Records autoregressive_step calls and enforces ARDY's real contract."""

    gen_horizon_len = HORIZON
    num_frames_per_token = PATCH

    def __init__(self):
        self.motion_rep = FakeMotionRep()
        self.diffusion = FakeDiffusion()
        self.skeleton = object()
        self.calls = []
        self.encode_calls = []
        self.generate_calls = []
        self._next_frame = 0

    def __call__(self, texts, num_frames, **kw):
        """The single-clip path (`/generate`) -- Ardy.__call__."""
        self.generate_calls.append(dict(kw, texts=texts, num_frames=num_frames))
        return torch.zeros((1, num_frames, FEATS))

    def _encode_text(self, texts):
        self.encode_calls.append(list(texts))
        # Distinct, deterministic features per prompt so tests can tell which
        # prompt a step was conditioned on.
        val = float(sum(ord(c) for c in texts[0]) % 97)
        return torch.full((1, 4, 16), val), torch.ones((1, 4), dtype=torch.bool)

    def autoregressive_step(self, *, num_frames, num_denoising_steps, motion_mask,
                            observed_motion, cfg_weight, texts, text_feat, text_pad_mask,
                            init_history_sequence, init_global_translation,
                            init_first_heading_angle):
        assert num_frames % self.num_frames_per_token == 0, \
            "real ARDY asserts num_frames is a whole number of tokens"
        hist_len = 0 if init_history_sequence is None else init_history_sequence.shape[1]
        assert num_frames == hist_len + self.gen_horizon_len, \
            "the window is history + horizon"
        assert texts is None, "text is pre-encoded so the shared encoder is reused"

        self.calls.append({
            "num_frames": num_frames,
            "history_len": hist_len,
            "history": None if init_history_sequence is None else init_history_sequence.clone(),
            "text_feat": text_feat.clone(),
            "init_global_translation": init_global_translation,
            "init_first_heading_angle": init_first_heading_angle,
            "num_denoising_steps": num_denoising_steps,
            "cfg_weight": cfg_weight,
        })

        # New frames carry sequential ids; the real model likewise returns the
        # history followed by the generated window.
        ids = torch.arange(self._next_frame, self._next_frame + self.gen_horizon_len,
                           dtype=torch.float32)
        self._next_frame += self.gen_horizon_len
        new = ids.view(1, -1, 1).repeat(1, 1, FEATS)
        if init_history_sequence is None:
            return new
        return torch.cat([init_history_sequence, new], dim=1)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeArdy()
    monkeypatch.setattr(service, "_get_model", lambda nickname: ("core", model))
    # post_process_motion is stubbed to explode (conftest); the streaming path
    # runs it for non-g1 models, so record it instead of faking a correction.
    calls = []
    def _record(local_rot, root_pos, contacts, skeleton, constraint_lst=None, **kw):
        calls.append({"frames": root_pos.shape[1], "constraint_lst": constraint_lst})
        return {}
    monkeypatch.setattr(service, "post_process_motion", _record)
    model.postprocess_calls = calls
    return model


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "OUTPUT_DIR", str(tmp_path))
    return TestClient(service.app)


def _choreo(client, steps, **kw):
    body = {"steps": steps, **kw}
    r = client.post("/choreograph", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --- the streaming contract --------------------------------------------------

def test_history_is_fed_forward_across_a_prompt_change(client, fake_model):
    _choreo(client, [{"prompt": "walk forward", "duration": 2.0},
                     {"prompt": "wave hello", "duration": 1.0}])

    # 3 chunks: walk, walk, wave. The first has no history; every later step is
    # conditioned on the frames already generated -- including across the seam,
    # which is the whole point of the issue.
    assert len(fake_model.calls) == 3
    assert fake_model.calls[0]["history_len"] == 0
    assert fake_model.calls[1]["history_len"] == 30
    assert fake_model.calls[2]["history_len"] == 60

    # The step after the prompt change sees the *walk* frames it must flow out of.
    seam_history = fake_model.calls[2]["history"][0, :, 0]
    assert seam_history.tolist() == list(range(60))


def test_only_new_frames_are_appended(client, fake_model):
    # autoregressive_step returns history + new. Appending the whole return value
    # would duplicate the history window into the clip every single step.
    out = _choreo(client, [{"prompt": "walk", "duration": 3.0}])
    assert out["frames"] == 90
    npz = np.load(out["npz"])
    assert npz["root_positions"][:, 2].tolist() == [float(i) for i in range(90)]


def test_world_pose_is_set_once_and_then_carried_by_the_history(client, fake_model):
    _choreo(client, [{"prompt": "walk", "duration": 2.0}], heading_deg=90.0)

    first, second = fake_model.calls[0], fake_model.calls[1]
    assert first["init_global_translation"] is not None
    assert first["init_first_heading_angle"] is not None
    assert first["init_first_heading_angle"].item() == pytest.approx(np.pi / 2)
    # Re-sending the world pose once there is history would reset the stream.
    assert second["init_global_translation"] is None
    assert second["init_first_heading_angle"] is None


def test_history_never_exceeds_the_trained_window(client, fake_model):
    # 12 s of motion: long enough that the history budget has to bite.
    _choreo(client, [{"prompt": "walk", "duration": 12.0}])

    budget = default_history_frames(FPS, HORIZON, PATCH)
    max_window = int(10 * FPS) // PATCH * PATCH
    assert budget == 270
    for call in fake_model.calls:
        assert call["history_len"] <= budget
        assert call["num_frames"] <= max_window, "window outgrew what ARDY was trained on"
    # ...and it does saturate, rather than the test passing because it stayed small.
    assert fake_model.calls[-1]["history_len"] == budget


def test_history_is_the_tail_of_the_stream_not_the_head(client, fake_model):
    # Once the crop bites, the model must see the most recent frames.
    _choreo(client, [{"prompt": "walk", "duration": 12.0}])
    last = fake_model.calls[-1]
    ids = last["history"][0, :, 0]
    expected_end = sum(1 for _ in fake_model.calls[:-1]) * HORIZON
    assert ids[-1].item() == expected_end - 1
    assert ids[0].item() == expected_end - last["history_len"]


# --- prompts and the shared encoder ------------------------------------------

def test_prompt_change_reconditions_the_stream(client, fake_model):
    _choreo(client, [{"prompt": "walk forward", "duration": 1.0},
                     {"prompt": "wave hello", "duration": 1.0}])
    walk_feat, wave_feat = fake_model.calls[0]["text_feat"], fake_model.calls[1]["text_feat"]
    assert not torch.equal(walk_feat, wave_feat), "second beat must be conditioned on its own text"


def test_text_is_encoded_once_per_distinct_prompt(client, fake_model):
    # 6 chunks, 2 prompts: re-encoding per chunk would run Llama-3 six times.
    _choreo(client, [{"prompt": "walk forward", "duration": 3.0},
                     {"prompt": "wave hello", "duration": 3.0}])
    assert len(fake_model.calls) == 6
    assert fake_model.encode_calls == [["walk forward"], ["wave hello"]]


def test_streaming_does_not_load_a_second_text_encoder(client, fake_model, monkeypatch):
    # The service reuses the encoder service; a second copy would not fit in VRAM
    # next to the denoiser. (load_text_encoder is the conftest stub that raises.)
    loads = []
    monkeypatch.setattr(service, "load_text_encoder", lambda **kw: loads.append(kw))
    _choreo(client, [{"prompt": "walk", "duration": 2.0}, {"prompt": "wave", "duration": 2.0}])
    assert loads == []


# --- request handling --------------------------------------------------------

def test_segments_report_the_frames_actually_generated(client, fake_model):
    out = _choreo(client, [{"prompt": "walk", "duration": 2.4},   # -> 2 chunks
                           {"prompt": "wave", "duration": 1.0}])  # -> 1 chunk
    assert [s["prompt"] for s in out["segments"]] == ["walk", "wave"]
    assert [s["frames"] for s in out["segments"]] == [60, 30]
    assert out["seam_frames"] == [60]
    assert out["frames"] == 90


def test_seam_velocity_is_measured_on_the_clip(client, fake_model):
    # The fake walks one unit per frame with no discontinuity, so the reported
    # seam jump must be zero -- if the service measured the wrong frames, or
    # measured the plan instead of the motion, this would not hold.
    out = _choreo(client, [{"prompt": "walk", "duration": 1.0},
                           {"prompt": "wave", "duration": 1.0}])
    assert out["seam_velocity"]["seams"][0]["frame"] == 30
    assert out["seam_velocity"]["max_ratio"] == pytest.approx(0.0)


def test_npz_carries_its_seams_for_the_report_tool(client, fake_model):
    out = _choreo(client, [{"prompt": "walk", "duration": 1.0},
                           {"prompt": "wave", "duration": 1.0}])
    npz = np.load(out["npz"])
    assert npz["seam_frames"].tolist() == [30]
    # ARDY's viewer reads these by name; the extra key must not disturb them.
    for key in ("posed_joints", "global_rot_mats", "foot_contacts", "fps", "text"):
        assert key in npz


def test_single_prompt_clip_has_no_seams(client, fake_model):
    out = _choreo(client, [{"prompt": "walk", "duration": 2.0}])
    assert out["seam_frames"] == []
    assert out["seam_velocity"]["seams"] == []


def test_seed_seeds_the_stream_once(client, fake_model, monkeypatch):
    seeds = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: seeds.append(s))
    _choreo(client, [{"prompt": "walk", "duration": 2.0}, {"prompt": "wave", "duration": 2.0}],
            seed=42)
    # One continuous stream, one seed -- not seed+i per beat like the old stitch.
    assert seeds == [42]


def test_diffusion_steps_default_to_the_models_base_steps(client, fake_model):
    _choreo(client, [{"prompt": "walk", "duration": 1.0}])
    assert fake_model.calls[0]["num_denoising_steps"] == FakeDiffusion.num_base_steps


def test_postprocess_runs_once_over_the_whole_stream(client, fake_model):
    # Per-clip post-processing was part of the old stitch; the stream is one clip.
    _choreo(client, [{"prompt": "walk", "duration": 2.0}, {"prompt": "wave", "duration": 1.0}])
    assert len(fake_model.postprocess_calls) == 1
    assert fake_model.postprocess_calls[0]["frames"] == 90


def test_empty_steps_rejected(client, fake_model):
    assert client.post("/choreograph", json={"steps": []}).status_code == 400


def test_unknown_mode_rejected_as_bad_request_not_500(client, fake_model):
    r = client.post("/choreograph", json={"steps": [{"prompt": "walk", "duration": 1.0}],
                                          "mode": "blend"})
    assert r.status_code == 400
    assert "mode" in r.json()["detail"]


def test_model_failure_is_a_500_with_the_reason(client, fake_model, monkeypatch):
    def boom(**kw):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(fake_model, "autoregressive_step", boom)
    r = client.post("/choreograph", json={"steps": [{"prompt": "walk", "duration": 1.0}]})
    assert r.status_code == 500
    assert "CUDA out of memory" in r.json()["detail"]


# --- no regression to the single-prompt path ---------------------------------

def test_generate_history_budget_leaves_room_for_the_horizon(client, fake_model):
    # v0.1 passed the full 10 s window as crop_history_length, so /generate ran a
    # window longer than ARDY was trained on. It must now match ARDY's own script
    # (scripts/generate.py::_default_history_frames).
    r = client.post("/generate", json={"prompt": "walk forward", "duration": 2.0})
    assert r.status_code == 200, r.text

    seen = fake_model.generate_calls[0]
    assert seen["crop_history_length"] == default_history_frames(FPS, HORIZON, PATCH) == 270
    assert seen["crop_history_length"] + HORIZON <= int(10 * FPS)
    assert seen["texts"] == ["walk forward"]


def test_generate_still_returns_a_clip(client, fake_model):
    # The single-prompt path must be untouched apart from the window budget.
    r = client.post("/generate", json={"prompt": "walk forward", "duration": 2.0, "seed": 7})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["frames"] == 60 and out["prompt"] == "walk forward"
    assert np.load(out["npz"])["root_positions"].shape == (60, 3)


def test_stitch_mode_still_glues_clips(client, fake_model):
    # Kept for A/B only, but it must not rot: it is the "before" in before/after.
    out = _choreo(client, [{"prompt": "walk", "duration": 1.0},
                           {"prompt": "wave", "duration": 1.0}], mode="stitch")
    assert out["mode"] == "stitch"
    assert fake_model.calls == [], "stitch must not use the streaming API"
    assert len(fake_model.generate_calls) == 2, "one independent clip per beat"
    assert out["seam_frames"] == [30]
    assert [s["frames"] for s in out["segments"]] == [30, 30]
