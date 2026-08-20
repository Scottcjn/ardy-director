#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Generate a labeled, Blender-oriented blocky humanoid OBJ.

This is the *worked example* for scripts/rig_avatar.py: a second avatar besides
ARDY's bundled one. It builds one box per bone, positioned along the core
skeleton, tags each box with a Blender/Rigify-style group name (``upper_arm.L``,
``forearm.R``, ``thigh.L`` ...) and an intentionally *Blender-native Z-up*
orientation and a *different scale* (authored ~40 units tall) so running the
rigger exercises the automatic up-axis fix and height fit.

    python make_blocky_humanoid.py            # writes blocky_humanoid.obj

The output is deliberately committed so the example is reproducible without
running Blender.
"""
import os
import numpy as np

# Core skeleton bone -> (parent joint pos, child joint pos) in ARDY Y-up metres,
# taken from ardy/assets/skeletons/cskel27 (the canonical bind pose). Each box
# spans parent->child. Names use Blender Rigify conventions to prove the mapper
# handles them (not ARDY's own names).
JOINTS = {
    "Hips": (0.000, 0.970, 0.000), "Spine": (0.000, 1.041, -0.047),
    "Spine1": (0.000, 1.134, -0.064), "Spine2": (0.000, 1.228, -0.072),
    "Spine3": (0.000, 1.323, -0.072), "Neck": (0.000, 1.571, -0.037),
    "Head": (0.000, 1.699, -0.014),
    "RightShoulder": (-0.032, 1.496, -0.019), "RightArm": (-0.191, 1.496, -0.019),
    "RightForeArm": (-0.486, 1.496, -0.019), "RightHand": (-0.719, 1.496, -0.019),
    "LeftShoulder": (0.032, 1.496, -0.019), "LeftArm": (0.191, 1.496, -0.019),
    "LeftForeArm": (0.486, 1.496, -0.019), "LeftHand": (0.719, 1.496, -0.019),
    "RightUpLeg": (-0.095, 0.942, 0.000), "RightLeg": (-0.095, 0.530, 0.000),
    "RightFoot": (-0.095, 0.074, 0.000), "RightToeBase": (-0.095, 0.015, 0.161),
    "LeftUpLeg": (0.095, 0.942, 0.000), "LeftLeg": (0.095, 0.530, 0.000),
    "LeftFoot": (0.095, 0.074, 0.000), "LeftToeBase": (0.095, 0.015, 0.161),
}

# (Blender-style group name, parent joint, child joint, box half-thickness m)
BONES = [
    ("spine.001", "Hips", "Spine", 0.09),
    ("spine.002", "Spine", "Spine1", 0.09),
    ("spine.003", "Spine1", "Spine2", 0.10),
    ("chest", "Spine2", "Spine3", 0.11),
    ("neck", "Spine3", "Neck", 0.04),
    ("head", "Neck", "Head", 0.09),
    ("shoulder.R", "Spine3", "RightShoulder", 0.05),
    ("upper_arm.R", "RightShoulder", "RightArm", 0.05),
    ("forearm.R", "RightArm", "RightForeArm", 0.045),
    ("hand.R", "RightForeArm", "RightHand", 0.04),
    ("shoulder.L", "Spine3", "LeftShoulder", 0.05),
    ("upper_arm.L", "LeftShoulder", "LeftArm", 0.05),
    ("forearm.L", "LeftArm", "LeftForeArm", 0.045),
    ("hand.L", "LeftForeArm", "LeftHand", 0.04),
    ("thigh.R", "Hips", "RightUpLeg", 0.06),
    ("shin.R", "RightUpLeg", "RightLeg", 0.055),
    ("foot.R", "RightLeg", "RightFoot", 0.05),
    ("toe.R", "RightFoot", "RightToeBase", 0.04),
    ("thigh.L", "Hips", "LeftUpLeg", 0.06),
    ("shin.L", "LeftUpLeg", "LeftLeg", 0.055),
    ("foot.L", "LeftLeg", "LeftFoot", 0.05),
    ("toe.L", "LeftFoot", "LeftToeBase", 0.04),
]

AUTHOR_SCALE = 40.0  # author ~40 units tall (not metres) -> exercises height fit


def ardy_to_blender(p):
    """ARDY Y-up (x, y, z) -> Blender Z-up (x, -z, y)."""
    x, y, z = p
    return (x, -z, y)


def box_verts(a, b, r):
    """8 corners of a box spanning a->b with square cross-section (half-size r)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    axis = b - a
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-9 else np.array([0, 1.0, 0])
    # two perpendicular vectors
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    corners = []
    for end in (a, b):
        for su in (-r, r):
            for sw in (-r, r):
                corners.append(end + su * u + sw * w)
    return corners  # 8 points, ordered a(--,-+,+-,++) then b(...)


# faces of the box given the 8-corner ordering above (1-based within the box)
_BOX_FACES = [
    (1, 2, 4), (1, 4, 3), (5, 8, 6), (5, 7, 8),   # end caps
    (1, 5, 6), (1, 6, 2), (3, 4, 8), (3, 8, 7),
    (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4),
]


def main():
    lines = ["# blocky humanoid (Blender Z-up, ~40 units) -- worked example for rig_avatar.py"]
    vcount = 0
    for name, pa, pb, r in BONES:
        a = ardy_to_blender(JOINTS[pa])
        b = ardy_to_blender(JOINTS[pb])
        a = tuple(c * AUTHOR_SCALE for c in a)
        b = tuple(c * AUTHOR_SCALE for c in b)
        verts = box_verts(a, b, r * AUTHOR_SCALE)
        lines.append(f"o {name}")
        for v in verts:
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
        for f in _BOX_FACES:
            lines.append(f"f {f[0]+vcount} {f[1]+vcount} {f[2]+vcount}")
        vcount += 8
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocky_humanoid.obj")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({vcount} verts, {len(BONES)} labeled parts)")


if __name__ == "__main__":
    main()
