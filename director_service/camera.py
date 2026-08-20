# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Camera: bake a per-frame camera track over a generated root path.

ARDY gives motion; a director also has to say where the shot is looking from.
This module turns a camera mode plus the character's root path into two arrays
— eye position and look-at target, one per frame — which ride along in the
output .npz so any renderer (ARDY's viser viewer, Blender, three.js) can frame
the shot the way it was directed.

Pure numpy: no ARDY, no torch, no GPU. The framing is the same idea as the
interactive demo's live follow camera (`scripts/interactive_demo/camera.py`),
baked per frame instead of driven by a viewer's redraw.

World is Y-up; heading 0 faces +Z (ARDY's convention).
"""
import numpy as np

MODES = ("follow", "orbit", "fixed", "over_the_shoulder")

# Defaults are the demo's framing, rounded: far enough back for a full body.
DEFAULTS = {
    "distance": 5.0,      # metres behind the character (follow / over_the_shoulder)
    "height": 2.5,        # metres above the root
    "side": 1.2,          # lateral offset (over_the_shoulder only)
    "look_height": 1.2,   # aim at chest height, not the ankles
    "look_ahead": 1.0,    # metres ahead of the character (over_the_shoulder only)
    "orbit_deg_per_s": 20.0,
    "orbit_start_deg": 0.0,
    "smoothing": 0.1,     # 0 = snap to the character's facing, ->1 = very lazy camera
}

_UP = np.array([0.0, 1.0, 0.0])
_MOVING_MPS = 0.05  # below this the character is standing: don't read a direction from noise


class CameraError(ValueError):
    """A camera the director asked for that cannot be built."""


def solve_camera_track(mode, root_positions, fps, **params):
    """Bake eye/target arrays for `mode` over `root_positions`.

    Args:
        mode: one of MODES.
        root_positions: [F, 3] world root positions (the generated motion's
            `root_positions`).
        fps: frame rate, needed to read speed and to time the orbit.
        **params: overrides of DEFAULTS, plus `position` / `look_at`
            ([x, y, z]) for `fixed`.

    Returns:
        dict with `positions` [F, 3], `targets` [F, 3], `mode` and the resolved
        `params` (so the .npz records how the shot was framed).
    """
    if mode not in MODES:
        raise CameraError(f"unknown camera mode {mode!r}; expected one of {', '.join(MODES)}")
    root = np.asarray(root_positions, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise CameraError(f"root_positions must be [frames, 3], got {list(root.shape)}")
    if len(root) == 0:
        raise CameraError("root_positions is empty: nothing to film")
    if fps <= 0:
        raise CameraError(f"fps must be positive, got {fps}")

    cfg = dict(DEFAULTS)
    for k, v in params.items():
        if v is not None:
            cfg[k] = v
    if not 0.0 <= float(cfg["smoothing"]) < 1.0:
        raise CameraError(f"smoothing must be in [0, 1), got {cfg['smoothing']}")

    forward = _forward_directions(root, fps, float(cfg["smoothing"]))

    if mode == "fixed":
        positions, targets = _fixed(root, cfg)
    elif mode == "orbit":
        positions, targets = _orbit(root, fps, cfg)
    elif mode == "follow":
        positions, targets = _follow(root, forward, cfg)
    else:
        positions, targets = _over_the_shoulder(root, forward, cfg)

    resolved = {k: v for k, v in cfg.items() if k in PARAMS_FOR[mode]}
    return {
        "positions": positions.astype(np.float32),
        "targets": targets.astype(np.float32),
        "mode": mode,
        "params": resolved,
    }


PARAMS_FOR = {
    "follow": ("distance", "height", "look_height", "smoothing"),
    "over_the_shoulder": ("distance", "height", "side", "look_height", "look_ahead", "smoothing"),
    "orbit": ("distance", "height", "look_height", "orbit_deg_per_s", "orbit_start_deg"),
    "fixed": ("position", "look_at", "look_height"),
}


def _forward_directions(root, fps, smoothing):
    """Per-frame facing, read from where the root is heading, then smoothed.

    A camera that snaps to instantaneous velocity jitters on every footfall, so
    the direction is low-passed the way the demo's live camera does it.
    """
    n = len(root)
    vel = np.zeros((n, 3))
    if n > 1:
        vel[:-1] = np.diff(root, axis=0) * fps
        vel[-1] = vel[-2]
    vel[:, 1] = 0.0  # film the ground track; vertical bob is not a facing

    speed = np.linalg.norm(vel, axis=-1)
    forward = np.zeros((n, 3))
    forward[:] = np.array([0.0, 0.0, 1.0])  # standing still: face +Z
    moving = speed > _MOVING_MPS
    forward[moving] = vel[moving] / speed[moving, None]

    if smoothing <= 0.0:
        return forward

    alpha = 1.0 - smoothing
    out = np.empty_like(forward)
    acc = forward[0].copy()
    for i in range(n):
        acc = alpha * forward[i] + smoothing * acc
        norm = np.linalg.norm(acc)
        acc = forward[i].copy() if norm < 1e-8 else acc / norm
        out[i] = acc
    return out


def _right_of(forward):
    """Right-hand side of the character, in the ground plane."""
    return np.stack([forward[:, 2], np.zeros(len(forward)), -forward[:, 0]], axis=-1)


def _follow(root, forward, cfg):
    positions = root - forward * float(cfg["distance"]) + _UP * float(cfg["height"])
    targets = root + _UP * float(cfg["look_height"])
    return positions, targets


def _over_the_shoulder(root, forward, cfg):
    right = _right_of(forward)
    positions = (
        root
        - forward * float(cfg["distance"])
        - right * float(cfg["side"])
        + _UP * float(cfg["height"])
    )
    targets = root + forward * float(cfg["look_ahead"]) + _UP * float(cfg["look_height"])
    return positions, targets


def _orbit(root, fps, cfg):
    n = len(root)
    t = np.arange(n) / float(fps)
    angle = np.deg2rad(float(cfg["orbit_start_deg"]) + float(cfg["orbit_deg_per_s"]) * t)
    radius = float(cfg["distance"])
    offset = np.stack([np.sin(angle) * radius, np.full(n, float(cfg["height"])), np.cos(angle) * radius], axis=-1)
    positions = root + offset
    targets = root + _UP * float(cfg["look_height"])
    return positions, targets


def _fixed(root, cfg):
    n = len(root)
    position = cfg.get("position")
    if position is None:
        # No tripod given: park where the character starts, backed off and up,
        # so `fixed` is usable without the director having to survey the stage.
        position = root[0] + np.array([0.0, float(cfg["height"]), -float(cfg["distance"])])
    position = _as_xyz("position", position)
    positions = np.repeat(position[None, :], n, axis=0)

    look_at = cfg.get("look_at")
    if look_at is None:
        targets = root + _UP * float(cfg["look_height"])  # locked-off camera, panning to track
    else:
        targets = np.repeat(_as_xyz("look_at", look_at)[None, :], n, axis=0)
    cfg["position"] = position.tolist()
    cfg["look_at"] = None if look_at is None else _as_xyz("look_at", look_at).tolist()
    return positions, targets


def _as_xyz(name, value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3,):
        raise CameraError(f"{name} must be [x, y, z], got {list(np.shape(value))}")
    if not np.all(np.isfinite(a)):
        raise CameraError(f"{name} must be finite, got {a.tolist()}")
    return a
