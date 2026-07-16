# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Staging geometry: waypoints -> the root path ARDY is constrained to.

No ARDY here: staging.py is deliberately plain numpy, so this exercises the
real planning code rather than a stand-in.
"""
import numpy as np
import pytest

from director_service import staging

FPS = 30


def plan(waypoints, duration=6.0, **kw):
    return staging.plan_root_path(waypoints, fps=FPS, num_frames=int(duration * FPS), **kw)


def test_waypoint_lands_on_its_frame_and_position():
    p = plan([{"x": 0.0, "z": 3.0, "at": 2.0}])
    assert p["frame_indices"][-1] == 60
    assert p["root_2d"][-1] == pytest.approx([0.0, 3.0], abs=1e-5)
    assert p["root_2d"][0] == pytest.approx([0.0, 0.0], abs=1e-5)


def test_dense_path_walks_the_line_at_constant_speed():
    """Halfway in time is halfway along the leg -- that is what makes it a walk."""
    p = plan([{"x": 4.0, "z": 0.0, "at": 2.0}])
    assert len(p["frame_indices"]) == 61  # every frame 0..60 constrained
    mid = p["root_2d"][30]
    assert mid == pytest.approx([2.0, 0.0], abs=1e-4)


def test_sparse_path_constrains_only_the_marks():
    p = plan([{"x": 4.0, "z": 0.0, "at": 2.0}, {"x": 4.0, "z": 4.0, "at": 4.0}], dense=False)
    assert p["frame_indices"].tolist() == [0, 60, 120]


def test_start_position_places_frame_zero():
    p = plan([{"x": 5.0, "z": 5.0, "at": 2.0}], start=(1.0, -2.0))
    assert p["root_2d"][0] == pytest.approx([1.0, -2.0], abs=1e-5)


def test_heading_follows_the_path_in_ardys_convention():
    """ARDY: heading 0 faces +Z, so walking +X must read as +90 deg."""
    p = plan([{"x": 3.0, "z": 0.0, "at": 2.0}])
    assert np.rad2deg(p["headings"][0]) == pytest.approx(90.0, abs=1e-3)

    p_z = plan([{"x": 0.0, "z": 3.0, "at": 2.0}])
    assert np.rad2deg(p_z["headings"][0]) == pytest.approx(0.0, abs=1e-3)

    p_back = plan([{"x": 0.0, "z": -3.0, "at": 2.0}])
    assert abs(np.rad2deg(p_back["headings"][0])) == pytest.approx(180.0, abs=1e-3)


def test_heading_turns_at_the_corner():
    p = plan([{"x": 0.0, "z": 3.0, "at": 2.0}, {"x": 3.0, "z": 3.0, "at": 4.0}])
    assert np.rad2deg(p["headings"][10]) == pytest.approx(0.0, abs=1e-3)    # walking +Z
    assert np.rad2deg(p["headings"][90]) == pytest.approx(90.0, abs=1e-3)   # now walking +X


def test_standing_still_holds_the_last_real_heading():
    """A pause must not snap the facing back to +Z mid-clip."""
    p = plan([{"x": 3.0, "z": 0.0, "at": 2.0}, {"x": 3.0, "z": 0.0, "at": 4.0}])
    assert np.rad2deg(p["headings"][30]) == pytest.approx(90.0, abs=1e-3)
    assert np.rad2deg(p["headings"][90]) == pytest.approx(90.0, abs=1e-3)  # held through the pause
    assert np.rad2deg(p["headings"][-1]) == pytest.approx(90.0, abs=1e-3)


def test_a_stage_that_never_moves_faces_forward():
    p = plan([{"x": 0.0, "z": 0.0, "at": 2.0}])
    assert np.all(p["headings"] == 0.0)
    assert p["path_length_m"] == pytest.approx(0.0)


def test_path_length_and_speed_are_reported():
    p = plan([{"x": 3.0, "z": 4.0, "at": 2.0}])  # 3-4-5 triangle
    assert p["path_length_m"] == pytest.approx(5.0)
    s = staging.summarize(p, FPS)
    assert s["path_length_m"] == pytest.approx(5.0)
    assert s["mean_speed_mps"] == pytest.approx(2.5)
    assert s["last_frame"] == 60


def test_waypoint_past_the_clip_is_refused_not_rounded():
    """generate.py raises on out-of-range constraint frames; catch it earlier."""
    with pytest.raises(staging.StagingError, match="shorten the stage or lengthen the clip"):
        plan([{"x": 0.0, "z": 1.0, "at": 8.0}], duration=6.0)


def test_times_must_strictly_increase():
    with pytest.raises(staging.StagingError, match="strictly increase"):
        plan([{"x": 0.0, "z": 1.0, "at": 3.0}, {"x": 1.0, "z": 1.0, "at": 2.0}])


def test_two_marks_on_one_frame_are_refused():
    with pytest.raises(staging.StagingError, match="round to frame"):
        plan([{"x": 0.0, "z": 1.0, "at": 2.0}, {"x": 1.0, "z": 1.0, "at": 2.001}])


def test_a_waypoint_on_frame_zero_points_at_start_instead():
    with pytest.raises(staging.StagingError, match="use `start` for that"):
        plan([{"x": 1.0, "z": 1.0, "at": 0.001}])


def test_empty_and_non_finite_are_refused():
    with pytest.raises(staging.StagingError, match="non-empty"):
        plan([])
    with pytest.raises(staging.StagingError, match="finite"):
        plan([{"x": float("nan"), "z": 1.0, "at": 2.0}])
