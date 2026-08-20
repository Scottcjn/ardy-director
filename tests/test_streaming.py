# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Window budgets, step planning and seam metrics (no ARDY, no torch)."""
import numpy as np
import pytest

from director_service.streaming import (
    Chunk,
    StreamPlan,
    default_history_frames,
    history_length_for,
    plan_stream,
    root_velocity,
    seam_report,
    velocity_jump_at,
)


class Step:
    """Stands in for the pydantic ChoreoStep (only .prompt/.duration are used)."""

    def __init__(self, prompt, duration):
        self.prompt = prompt
        self.duration = duration


# --- window budget -----------------------------------------------------------

def test_history_leaves_room_for_the_horizon_inside_the_trained_window():
    # The bug this replaces: v0.1 used the whole 10 s window as history, so
    # history + horizon overran the window the model was trained on.
    fps, horizon, patch = 30.0, 30, 10
    hist = default_history_frames(fps, horizon, patch)
    max_window = int(10 * fps) // patch * patch
    assert hist + horizon <= max_window
    assert hist == 270  # 300-frame window minus the 30-frame horizon


def test_history_budget_matches_ardys_own_generate_script():
    # Ported formula must agree with ardy/scripts/generate.py::_default_history_frames
    # across the shapes real models use, not just the default one.
    for fps in (20.0, 30.0, 60.0):
        for horizon, patch in ((30, 10), (40, 8), (16, 4), (60, 30)):
            max_window = int(10 * fps) // patch * patch
            expected = max(0, (max_window - horizon) // patch * patch)
            assert default_history_frames(fps, horizon, patch) == expected


def test_history_budget_is_a_whole_number_of_tokens():
    # ARDY asserts num_frames % num_frames_per_token == 0; history feeds that sum.
    for patch in (4, 8, 10, 16):
        assert default_history_frames(30.0, 30, patch) % patch == 0


def test_history_budget_never_negative_when_horizon_exceeds_window():
    assert default_history_frames(1.0, 300, 10) == 0


def test_history_length_grows_then_saturates_at_the_crop():
    assert history_length_for(0, 270, 10) == 0        # nothing generated yet
    assert history_length_for(5, 270, 10) == 0        # less than one token
    assert history_length_for(30, 270, 10) == 30
    assert history_length_for(95, 270, 10) == 90      # quantized down
    assert history_length_for(1000, 270, 10) == 270   # clamped to the crop


# --- planning ----------------------------------------------------------------

def test_plan_covers_every_step_in_order_with_whole_horizons():
    plan = plan_stream([Step("walk", 3.0), Step("wave", 2.0)], fps=30.0, gen_horizon_len=30)
    assert [c.prompt for c in plan.chunks] == ["walk"] * 3 + ["wave"] * 2
    assert plan.total_frames == 150
    assert all(c.end_frame - c.start_frame == 30 for c in plan.chunks)
    # chunks tile the clip with no gap and no overlap
    assert [c.start_frame for c in plan.chunks] == [0, 30, 60, 90, 120]


def test_seam_frames_are_the_prompt_changes_not_the_chunk_boundaries():
    # Every chunk is a step boundary; only a *prompt change* is a seam. The old
    # stitch popped at these frames and nowhere else.
    plan = plan_stream([Step("walk", 3.0), Step("wave", 2.0), Step("sit", 1.0)],
                       fps=30.0, gen_horizon_len=30)
    assert plan.seam_frames == [90, 150]


def test_first_chunk_is_not_a_seam():
    plan = plan_stream([Step("walk", 3.0)], fps=30.0, gen_horizon_len=30)
    assert plan.chunks[0].starts_prompt is True
    assert plan.seam_frames == []  # frame 0 has nothing to pop against


def test_short_step_still_gets_one_chunk():
    # round(0.1*30/30) == 0; a beat the director asked for must not vanish.
    plan = plan_stream([Step("blink", 0.1), Step("wave", 2.0)], fps=30.0, gen_horizon_len=30)
    assert [c.prompt for c in plan.chunks] == ["blink", "wave", "wave"]


def test_duration_is_quantized_to_the_horizon_and_reported_honestly():
    # 2.5 s at horizon 1 s -> 3 chunks; the response must say 3 s, not 2.5.
    plan = plan_stream([Step("walk", 2.5)], fps=30.0, gen_horizon_len=30)
    seg = plan.segments()[0]
    assert seg["frames"] == 90
    assert seg["duration_s"] == 3.0


def test_segments_span_the_whole_clip_contiguously():
    plan = plan_stream([Step("a", 3.0), Step("b", 2.0), Step("c", 2.0)],
                       fps=30.0, gen_horizon_len=30)
    segs = plan.segments()
    assert [s["prompt"] for s in segs] == ["a", "b", "c"]
    assert segs[0]["start_frame"] == 0
    for prev, nxt in zip(segs, segs[1:]):
        assert prev["end_frame"] == nxt["start_frame"]
    assert segs[-1]["end_frame"] == plan.total_frames


def test_repeated_prompt_in_separate_steps_stays_two_segments():
    # Same text twice is still two beats -- segments must not collapse by prompt.
    plan = plan_stream([Step("walk", 1.0), Step("walk", 1.0)], fps=30.0, gen_horizon_len=30)
    assert len(plan.segments()) == 2
    assert plan.seam_frames == [30]


def test_empty_plan_has_no_frames():
    plan = plan_stream([], fps=30.0, gen_horizon_len=30)
    assert plan.total_frames == 0 and plan.segments() == [] and plan.seam_frames == []


def test_zero_horizon_rejected():
    with pytest.raises(ValueError):
        plan_stream([Step("walk", 1.0)], fps=30.0, gen_horizon_len=0)


# --- seam metrics ------------------------------------------------------------

def test_root_velocity_is_per_second_not_per_frame():
    rp = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.float32)  # 1 unit/frame
    v = root_velocity(rp, fps=30.0)
    assert v.shape == (2, 3)
    np.testing.assert_allclose(v[:, 2], [30.0, 30.0])  # 30 units/s


def test_root_velocity_rejects_wrong_shape():
    with pytest.raises(ValueError):
        root_velocity(np.zeros((10, 27, 3)), fps=30.0)


def test_root_velocity_of_a_single_frame_is_empty_not_a_crash():
    assert root_velocity(np.zeros((1, 3)), fps=30.0).shape == (0, 3)


def _constant_velocity(frames, per_frame):
    return np.cumsum(np.tile([0.0, 0.0, per_frame], (frames, 1)), axis=0)


def _ramp(frames, v0, accel_per_frame=0.0, start=0.0):
    """Root walking +Z, gently accelerating: a realistic non-zero baseline."""
    per_frame = v0 + accel_per_frame * np.arange(frames)
    z = start + np.cumsum(per_frame)
    return np.stack([np.zeros(frames), np.zeros(frames), z], axis=1)


def test_no_jump_when_velocity_is_continuous():
    assert velocity_jump_at(_constant_velocity(60, 0.1), frame=30, fps=30.0) == pytest.approx(0.0)


def test_jump_equals_the_velocity_delta_at_a_hard_cut():
    # Walk at 0.1 units/frame, then cut into 0.3 units/frame at frame 30:
    # a 0.2 units/frame change = 6 units/s at 30 fps.
    rp = np.concatenate([_constant_velocity(30, 0.1),
                         _constant_velocity(30, 0.3) + [0, 0, 3.0]])
    assert velocity_jump_at(rp, frame=30, fps=30.0) == pytest.approx(6.0, abs=1e-6)


def test_jump_is_found_wherever_the_cut_lands_in_the_finite_difference():
    # The p[30]-p[29] sample straddles the cut, so the discontinuity can show up
    # at either sample touching the seam. Both cuts below are 6 units/s.
    lands_early = np.concatenate([_constant_velocity(30, 0.1),
                                  _constant_velocity(30, 0.3) + [0, 0, 3.0]])
    lands_late = np.concatenate([_constant_velocity(30, 0.1),
                                 _constant_velocity(30, 0.3) + [0, 0, 2.8]])
    assert velocity_jump_at(lands_early, frame=30, fps=30.0) == pytest.approx(6.0, abs=1e-6)
    assert velocity_jump_at(lands_late, frame=30, fps=30.0) == pytest.approx(6.0, abs=1e-6)


def test_seam_report_flags_the_stitch_and_clears_the_smooth_clip():
    # Same gentle acceleration in both; only the stitched clip cuts at frame 30.
    smooth = _ramp(60, 0.1, 0.001)
    stitched = np.concatenate([_ramp(30, 0.1, 0.001),
                               _ramp(30, 0.3, 0.001, start=_ramp(30, 0.1, 0.001)[-1, 2])])

    smooth_report = seam_report(smooth, [30], fps=30.0)
    stitched_report = seam_report(stitched, [30], fps=30.0)

    # A smooth clip's seam is indistinguishable from any other frame: ratio ~1.
    assert smooth_report["baseline_jump"] == pytest.approx(0.03, abs=1e-6)
    assert smooth_report["max_ratio"] == pytest.approx(1.0, abs=0.05)
    # The stitch changes velocity ~170x harder than a normal frame does.
    assert stitched_report["max_ratio"] > 50


def test_baseline_excludes_seams_so_one_pop_cannot_hide_another():
    # Two hard cuts. If the baseline counted the seam samples, the median would
    # rise and the ratios would understate the pops.
    rp = np.concatenate([_constant_velocity(30, 0.1),
                         _constant_velocity(30, 0.3) + [0, 0, 3.0],
                         _constant_velocity(30, 0.05) + [0, 0, 12.0]])
    report = seam_report(rp, [30, 60], fps=30.0)
    assert [r["frame"] for r in report["seams"]] == [30, 60]
    assert report["baseline_jump"] == pytest.approx(0.0, abs=1e-9)
    assert all(r["jump"] > 1.0 for r in report["seams"])


def test_seam_report_reports_seam_times_in_seconds():
    rp = _constant_velocity(90, 0.1)
    report = seam_report(rp, [45], fps=30.0)
    assert report["seams"][0]["time_s"] == pytest.approx(1.5)


def test_seam_report_ignores_out_of_range_seams():
    rp = _constant_velocity(30, 0.1)
    assert seam_report(rp, [0, 999], fps=30.0)["seams"] == []


def test_seam_report_on_a_clip_too_short_to_have_velocity():
    assert seam_report(np.zeros((1, 3)), [0], fps=30.0)["seams"] == []
