#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Web front-end support for the ARDY Director service.

This module is deliberately free of torch / ardy imports so the browser-facing
plumbing (npz -> JSON, tag sanitising, static dir) can be unit-tested on a
machine with no GPU and no ARDY install. The service wires these into FastAPI
routes; everything here is pure numpy + stdlib.

The 3D viewport in web/index.html plays `posed_joints` (world-space joint
positions, one row per frame) as a stick figure, using the skeleton's
parent table to draw the bones. Both the joints and the parent table are read
back out of the same .npz the generator already writes, so serving a clip
never has to touch the model again.
"""
import os
import re

import numpy as np

# Directory of the single-page UI, served as static files by the service.
WEB_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "web"))

# npz tags are generated as f"{kind}_{int(time)}_{hex6}"; only ever these chars.
# Reject anything else up front so /motion/{tag} can never walk out of OUTPUT_DIR.
_TAG_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def safe_tag(tag):
    """Return the tag if it is a plain filename token, else None.

    Guards the /motion route against path traversal ("../", absolute paths,
    NUL) before the tag is ever joined onto OUTPUT_DIR.
    """
    if not isinstance(tag, str):
        return None
    if tag in (".", "..") or "/" in tag or "\\" in tag or "\x00" in tag:
        return None
    return tag if _TAG_RE.match(tag) else None


def npz_path_for(output_dir, tag):
    """Sanitise `tag` and return the .npz path under `output_dir`, or None."""
    t = safe_tag(tag)
    if t is None:
        return None
    return os.path.join(output_dir, f"{t}.npz")


def motion_payload(npz_path, stride=1, decimals=4):
    """Read an ARDY-native clip and return a JSON-serialisable playback payload.

    Returns a dict with:
      ok, fps, frames (before striding), rendered_frames, stride,
      joint_names, parents (parent index per joint, -1 = root),
      positions ([frame][joint][xyz]), text, seam_frames

    Raises FileNotFoundError if the clip is gone and ValueError if the clip has
    no `posed_joints` (e.g. a robot-qpos model that this viewport can't draw).
    The service turns those into 404 / 422 respectively.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    stride = max(1, int(stride))
    with np.load(npz_path, allow_pickle=False) as data:
        if "posed_joints" not in data.files:
            raise ValueError(
                "clip has no 'posed_joints'; this viewport renders avatar "
                "skeletons (core/soma), not robot qpos"
            )
        joints = np.asarray(data["posed_joints"], dtype=np.float64)  # (frames, J, 3)
        fps = int(data["fps"]) if "fps" in data.files else 30
        text = str(data["text"]) if "text" in data.files else ""
        if "joint_names" in data.files:
            names = [str(x) for x in data["joint_names"]]
        else:
            names = [str(i) for i in range(joints.shape[1])]
        if "joint_parents" in data.files:
            parents = [int(x) for x in data["joint_parents"]]
        else:
            parents = [-1] * joints.shape[1]
        seams = ([int(x) for x in data["seam_frames"]]
                 if "seam_frames" in data.files else [])

    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"unexpected posed_joints shape {joints.shape}")

    total = int(joints.shape[0])
    sampled = joints[::stride]
    positions = np.round(sampled, decimals).tolist()

    return {
        "ok": True,
        "fps": fps,
        "frames": total,
        "rendered_frames": int(sampled.shape[0]),
        "stride": stride,
        "joint_names": names,
        "parents": parents,
        "positions": positions,
        "text": text,
        "seam_frames": seams,
    }
