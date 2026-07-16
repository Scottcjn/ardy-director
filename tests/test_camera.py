# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Camera solving: a root path -> per-frame eye/target arrays.

Like staging, camera.py is plain numpy, so these assert the real framing math.
"""
import numpy as np
import pytest

from director_service import camera

FPS = 30


def walk_along_z(frames=60, speed=1.0):
    """Root walking straight down +Z at `speed` m/s."""
    t = np.arange(frames) / FPS
    return np.stack([np.zeros(frames), np.zeros(frames), t * speed], axis=-1)


def test_follow_sits_behind_the_character_at_the_asked_distance():
    root = walk_along_z()
    track = camera.solve_camera_track("follow", root, FPS, distance=5.0, height=2.5, smoothing=0.0)
    eye, tgt = track["positions"], track["targets"]
    assert eye.shape == root.shape and tgt.shape == root.shape

    # Walking +Z: the camera trails in -Z, at the requested height.
    late = 40
    assert eye[late][2] == pytest.approx(root[late][2] - 5.0, abs=1e-4)
    assert eye[late][1] == pytest.approx(2.5, abs=1e-4)
    # Ground distance from the character is exactly `distance`.
    ground = np.linalg.norm((eye[late] - root[late])[[0, 2]])
    assert ground == pytest.approx(5.0, abs=1e-4)


def test_follow_aims_at_chest_height_not_the_floor():
    root = walk_along_z()
    track = camera.solve_camera_track("follow", root, FPS, look_height=1.2, smoothing=0.0)
    assert track["targets"][10][1] == pytest.approx(1.2, abs=1e-5)
    assert track["targets"][10][[0, 2]] == pytest.approx(root[10][[0, 2]], abs=1e-5)


def test_follow_swings_round_when_the_character_turns():
    """Walk +Z, then +X: the camera must end up trailing in -X, not stuck in -Z."""
    first = walk_along_z(30)
    turn = np.stack([np.arange(1, 31) / FPS, np.zeros(30), np.full(30, first[-1][2])], axis=-1)
    root = np.concatenate([first, turn])
    track = camera.solve_camera_track("follow", root, FPS, distance=5.0, smoothing=0.0)
    eye = track["positions"]
    assert eye[10][2] < root[10][2] - 4.0     # trailing along -Z early
    assert eye[-1][0] == pytest.approx(root[-1][0] - 5.0, abs=1e-3)  # trailing along -X late


def test_smoothing_makes_the_camera_lag_the_turn():
    """A hard turn should not teleport the camera; that is what smoothing buys."""
    first = walk_along_z(30)
    turn = np.stack([np.arange(1, 31) / FPS, np.zeros(30), np.full(30, first[-1][2])], axis=-1)
    root = np.concatenate([first, turn])

    snap = camera.solve_camera_track("follow", root, FPS, smoothing=0.0)["positions"]
    lazy = camera.solve_camera_track("follow", root, FPS, smoothing=0.9)["positions"]
    jump_snap = np.abs(np.diff(snap, axis=0)).max()
    jump_lazy = np.abs(np.diff(lazy, axis=0)).max()
    assert jump_lazy < jump_snap


def test_standing_still_films_from_a_stable_side():
    """No velocity to read: the camera must not spin on rounding noise."""
    root = np.zeros((60, 3))
    track = camera.solve_camera_track("follow", root, FPS, distance=5.0)
    eye = track["positions"]
    assert np.allclose(eye, eye[0], atol=1e-6)
    assert eye[0][2] == pytest.approx(-5.0, abs=1e-4)  # default facing +Z -> camera at -Z


def test_over_the_shoulder_is_offset_to_the_side_and_looks_ahead():
    root = walk_along_z()
    track = camera.solve_camera_track("over_the_shoulder", root, FPS, distance=5.0,
                                      side=1.2, look_ahead=1.0, smoothing=0.0)
    eye, tgt = track["positions"], track["targets"]
    i = 40
    assert eye[i][2] == pytest.approx(root[i][2] - 5.0, abs=1e-4)
    assert abs(eye[i][0]) == pytest.approx(1.2, abs=1e-4)              # off to one side
    assert tgt[i][2] == pytest.approx(root[i][2] + 1.0, abs=1e-4)      # aiming ahead of them
    follow = camera.solve_camera_track("follow", root, FPS, smoothing=0.0)["positions"]
    assert not np.allclose(eye[i], follow[i])


def test_orbit_circles_at_the_asked_radius_and_rate():
    root = np.zeros((90, 3))
    track = camera.solve_camera_track("orbit", root, FPS, distance=6.0, height=2.0,
                                      orbit_deg_per_s=90.0, orbit_start_deg=0.0)
    eye = track["positions"]
    radii = np.linalg.norm(eye[:, [0, 2]] - root[:, [0, 2]], axis=-1)
    assert np.allclose(radii, 6.0, atol=1e-4)
    assert np.allclose(eye[:, 1], 2.0, atol=1e-6)
    assert eye[0] == pytest.approx([0.0, 2.0, 6.0], abs=1e-4)   # starts on +Z
    assert eye[30] == pytest.approx([6.0, 2.0, 0.0], abs=1e-4)  # 90 deg/s -> +X after 1s


def test_orbit_rides_along_with_a_moving_character():
    root = walk_along_z()
    track = camera.solve_camera_track("orbit", root, FPS, distance=6.0)
    radii = np.linalg.norm((track["positions"] - root)[:, [0, 2]], axis=-1)
    assert np.allclose(radii, 6.0, atol=1e-4)


def test_fixed_camera_holds_still_and_pans_to_track():
    root = walk_along_z()
    track = camera.solve_camera_track("fixed", root, FPS, position=[3.0, 2.0, -4.0])
    eye, tgt = track["positions"], track["targets"]
    assert np.allclose(eye, np.array([3.0, 2.0, -4.0]), atol=1e-6)  # locked off
    assert tgt[0][2] != tgt[-1][2]                                  # but keeps them in frame
    assert tgt[-1][[0, 2]] == pytest.approx(root[-1][[0, 2]], abs=1e-5)


def test_fixed_with_look_at_stares_at_one_spot():
    root = walk_along_z()
    track = camera.solve_camera_track("fixed", root, FPS, position=[3.0, 2.0, -4.0],
                                      look_at=[0.0, 1.0, 0.0])
    assert np.allclose(track["targets"], np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_fixed_without_a_position_parks_behind_the_start_mark():
    root = walk_along_z()
    track = camera.solve_camera_track("fixed", root, FPS, distance=5.0, height=2.0)
    assert track["positions"][0] == pytest.approx([0.0, 2.0, -5.0], abs=1e-5)


def test_resolved_params_are_reported_per_mode():
    root = walk_along_z()
    track = camera.solve_camera_track("orbit", root, FPS, orbit_deg_per_s=45.0)
    assert track["mode"] == "orbit"
    assert track["params"]["orbit_deg_per_s"] == 45.0
    assert "side" not in track["params"]  # over-the-shoulder's knob, not orbit's


def test_every_frame_is_finite_and_accounted_for():
    root = walk_along_z(120)
    for mode in camera.MODES:
        track = camera.solve_camera_track(mode, root, FPS)
        assert len(track["positions"]) == 120, mode
        assert np.all(np.isfinite(track["positions"])), mode
        assert np.all(np.isfinite(track["targets"])), mode


def test_bad_asks_are_named():
    root = walk_along_z()
    with pytest.raises(camera.CameraError, match="unknown camera mode"):
        camera.solve_camera_track("dolly_zoom", root, FPS)
    with pytest.raises(camera.CameraError, match="nothing to film"):
        camera.solve_camera_track("follow", np.zeros((0, 3)), FPS)
    with pytest.raises(camera.CameraError, match=r"\[frames, 3\]"):
        camera.solve_camera_track("follow", np.zeros((10, 2)), FPS)
    with pytest.raises(camera.CameraError, match="smoothing"):
        camera.solve_camera_track("follow", root, FPS, smoothing=1.0)
    with pytest.raises(camera.CameraError, match=r"\[x, y, z\]"):
        camera.solve_camera_track("fixed", root, FPS, position=[1.0, 2.0])
