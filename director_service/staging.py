# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Staging: turn a director's waypoints into an ARDY root-path plan.

The director says "start at the door, be at the desk by 3s"; ARDY wants a
`Root2DConstraintSet` — frame indices plus the root's XZ position (and,
optionally, the facing) at those frames. This module does that conversion and
nothing else: plain numpy in, plain numpy out, no ARDY and no torch, so the
geometry is testable without the GPU host.

`service.py` owns the ARDY-side wrapping (numpy plan -> Root2DConstraintSet on
the model's skeleton/device).

Heading convention follows ARDY's own: `compute_heading_angle` in
`ardy/motion_rep/tools.py` is `atan2(hip_diff_z, -hip_diff_x)`, which makes
heading 0 face +Z and heading = atan2(dx, dz) for a path travelling (dx, dz).
"""
import numpy as np

# The root path is a 2D (XZ) ground path; Y is ARDY's business, not the stage's.
_XZ = 2


class StagingError(ValueError):
    """A staging plan that ARDY could not be asked to honour."""


def _as_xz(name, value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (_XZ,):
        raise StagingError(f"{name} must be [x, z], got {list(np.shape(value))}")
    if not np.all(np.isfinite(a)):
        raise StagingError(f"{name} must be finite, got {a.tolist()}")
    return a


def plan_root_path(waypoints, fps, num_frames, start=(0.0, 0.0), dense=True):
    """Plan the root's ground path through `waypoints`.

    Args:
        waypoints: ordered list of {"x", "z", "at"} — where the root should be
            (metres, ARDY world XZ) and when (seconds from the clip start).
            Times must strictly increase; the character walks between them at
            whatever constant speed the spacing implies.
        fps: model frame rate.
        num_frames: length of the clip being generated. Every constrained frame
            must land inside it (ARDY drops nothing silently, it raises).
        start: root XZ at frame 0. This is the "start position" of the stage.
        dense: True constrains every frame from 0 to the last waypoint (a
            walked path); False constrains only the waypoint frames themselves
            and lets ARDY choose the route between them.

    Returns:
        dict with `frame_indices` (int64 [N]), `root_2d` (float32 [N, 2]),
        `headings` (float32 [N], radians, 0 = facing +Z) and `path_length_m`.
    """
    if fps <= 0:
        raise StagingError(f"fps must be positive, got {fps}")
    if num_frames <= 0:
        raise StagingError(f"num_frames must be positive, got {num_frames}")
    if not waypoints:
        raise StagingError("waypoints must be non-empty")

    start_xz = _as_xz("start", start)

    # --- validate the beat sheet before touching geometry --------------------
    times, points = [], []
    prev_t = 0.0
    for i, wp in enumerate(waypoints):
        t = float(wp["at"])
        if not np.isfinite(t):
            raise StagingError(f"waypoint {i}: 'at' must be finite, got {t}")
        if t <= prev_t:
            where = "the clip start (0s)" if i == 0 else f"waypoint {i - 1} ({prev_t}s)"
            raise StagingError(
                f"waypoint {i}: 'at' must come after {where}, got {t}s — "
                "waypoints are a timeline, so their times must strictly increase"
            )
        times.append(t)
        points.append(_as_xz(f"waypoint {i}", (wp["x"], wp["z"])))
        prev_t = t

    # ARDY indexes constraints by frame; a waypoint past the clip is a mistake
    # worth naming, not rounding away (generate.py raises on the same case).
    frames = [int(round(t * fps)) for t in times]
    last_frame = frames[-1]
    if last_frame >= num_frames:
        raise StagingError(
            f"waypoint at {times[-1]}s is frame {last_frame}, but the clip is only "
            f"{num_frames} frames ({num_frames / fps:.2f}s at {fps} fps); "
            "shorten the stage or lengthen the clip"
        )
    # Two waypoints can round onto one frame at low fps; that is an unhonourable
    # ask (two positions, one frame), so say so rather than let numpy pick.
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1]:
            raise StagingError(
                f"waypoints {i - 1} and {i} both round to frame {frames[i]} at {fps} fps; "
                "space them at least one frame apart"
            )
    if frames[0] == 0:
        raise StagingError(
            "the first waypoint lands on frame 0, which is the start position; "
            "use `start` for that and give the first waypoint a later time"
        )

    knot_frames = np.array([0] + frames, dtype=np.float64)
    knot_points = np.stack([start_xz] + points, axis=0)  # [K, 2]

    if dense:
        out_frames = np.arange(0, last_frame + 1, dtype=np.int64)
    else:
        out_frames = np.array([0] + frames, dtype=np.int64)

    # Piecewise-linear in each of X and Z: constant speed along each leg.
    root_2d = np.stack(
        [np.interp(out_frames.astype(np.float64), knot_frames, knot_points[:, d]) for d in range(_XZ)],
        axis=-1,
    )

    headings = _headings_along(root_2d)
    path_length = float(np.sum(np.linalg.norm(np.diff(knot_points, axis=0), axis=-1)))

    return {
        "frame_indices": out_frames,
        "root_2d": root_2d.astype(np.float32),
        "headings": headings.astype(np.float32),
        "path_length_m": path_length,
    }


def _headings_along(path_2d):
    """Facing angle (radians, 0 = +Z) at each point of an XZ path.

    Face the way you are about to walk. Where the path stands still there is no
    direction to read, so hold the last real one (and, before the first step,
    borrow the first) rather than snapping to +Z mid-stride.
    """
    n = len(path_2d)
    headings = np.zeros(n, dtype=np.float64)
    if n < 2:
        return headings

    steps = np.diff(path_2d, axis=0)  # [n-1, 2] — leg leaving each point
    moving = np.linalg.norm(steps, axis=-1) > 1e-6
    # heading = atan2(dx, dz): ARDY's 0 faces +Z, +90 deg faces +X.
    angles = np.arctan2(steps[:, 0], steps[:, 1])

    last = None
    for i in range(n - 1):
        if moving[i]:
            last = angles[i]
        headings[i] = last if last is not None else np.nan
    headings[n - 1] = last if last is not None else np.nan  # arrive facing the way you came in

    if np.isnan(headings).all():
        return np.zeros(n, dtype=np.float64)  # the stage never moves: face +Z
    # Frames before the first movement: face the first real heading.
    first_real = headings[~np.isnan(headings)][0]
    headings[np.isnan(headings)] = first_real
    return headings


def summarize(plan, fps):
    """Human/agent-readable digest of a plan (what the preview endpoint returns)."""
    frames = plan["frame_indices"]
    return {
        "constrained_frames": int(len(frames)),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "last_time_s": round(float(frames[-1]) / fps, 3),
        "path_length_m": round(plan["path_length_m"], 3),
        "start_heading_deg": round(float(np.rad2deg(plan["headings"][0])), 2),
        "mean_speed_mps": round(
            plan["path_length_m"] / (float(frames[-1]) / fps), 3
        ) if frames[-1] > 0 else 0.0,
    }
