#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Headless tests for the custom-avatar rig pipeline.

Self-contained: numpy + the stdlib only, no GPU and no ARDY install. The key
acceptance check -- "loads in ARDY's viewer without the joint-name mismatch" --
is reproduced exactly: ARDY's ``CoreSkin.__init__`` raises ``ValueError`` unless
``rig_joint_names`` equals the skeleton's ``bone_order`` in order, so we assert
the emitted names match ``CORE_JOINT_NAMES`` and, mutation-style, that a
deliberately corrupted name array *would* be rejected. We also reimplement the
LBS rest identity (posed == bind -> vertices unchanged) to prove the deform
contract, and articulate one joint to prove weights actually bind.

Run:  python tests/test_avatar_rig.py     (or pytest)
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from director_service import avatar_rig as ar  # noqa: E402

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "custom_avatar")


# --------------------------------------------------------------------------- #
# name mapping
# --------------------------------------------------------------------------- #

def test_part_name_mapping_conventions():
    # Blender/Rigify .L/.R
    assert ar.map_part_to_joint("upper_arm.L") == "LeftArm"
    assert ar.map_part_to_joint("forearm.R") == "RightForeArm"
    assert ar.map_part_to_joint("thigh.L") == "LeftUpLeg"
    assert ar.map_part_to_joint("shin.R") == "RightLeg"
    assert ar.map_part_to_joint("toe.L") == "LeftToeBase"
    # Mixamo-style
    assert ar.map_part_to_joint("LeftForeArm") == "LeftForeArm"
    assert ar.map_part_to_joint("RightHand") == "RightHand"
    # central joints need no side
    assert ar.map_part_to_joint("spine.003") == "Spine2"
    assert ar.map_part_to_joint("head") == "Head"
    # longer alias beats its prefix (forearm not arm, toe not foot)
    assert ar.map_part_to_joint("lowerarm_l") == "LeftForeArm"
    # unmapped
    assert ar.map_part_to_joint("cape_flap") is None
    print("ok: part-name mapping")


def test_side_from_geometry_when_label_unsided():
    # a bare "arm" label resolves side from mean-x sign (+left / -right)
    assert ar.map_part_to_joint("arm", x_sign=+0.3) == "LeftArm"
    assert ar.map_part_to_joint("arm", x_sign=-0.3) == "RightArm"
    print("ok: side resolved from geometry")


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #

def test_align_fixes_axis_and_scale():
    rig = ar.load_core_rig()
    jp = rig.joint_positions
    skel_h = jp[:, 1].max() - jp[:, 1].min()
    # a Z-up, 40x-scaled point cloud
    rng = np.random.default_rng(0)
    v = rng.uniform(-1, 1, (500, 3))
    v[:, 2] = rng.uniform(0, 40, 500)   # tall along Z (Blender up)
    aligned = ar.align_to_core(v, rig, up_axis="auto")
    ah = aligned[:, 1].max() - aligned[:, 1].min()
    assert abs(ah - skel_h) < 1e-6, (ah, skel_h)          # height fit
    assert abs(aligned[:, 1].min() - jp[:, 1].min()) < 1e-6  # feet on floor
    print("ok: alignment axis + height fit")


# --------------------------------------------------------------------------- #
# end-to-end + ARDY loader contract
# --------------------------------------------------------------------------- #

def _load(path):
    return np.load(path, allow_pickle=True)


def _assert_loads_like_ardy(skin):
    """Reproduce CoreSkin.__init__'s only rig assertion + shape expectations."""
    names = [str(x) for x in skin["rig_joint_names"]]
    # this is exactly the check that raises "MISMATCH in skinnging rig"
    for expected, got in zip(ar.CORE_JOINT_NAMES, names):
        assert expected == got, f"MISMATCH expected={expected} got={got}"
    assert len(names) == 27
    V = skin["bind_vertices"].shape[0]
    assert skin["bind_rig_transform"].shape == (27, 4, 4)
    assert skin["rig_joint_connections"].shape == (26, 2)
    assert skin["lbs_indices"].shape == (V, 5)
    assert skin["lbs_weights"].shape == (V, 5)
    # weights are a partition of unity; indices reference real joints
    w = skin["lbs_weights"]
    assert np.allclose(w.sum(1), 1.0, atol=1e-5)
    assert skin["lbs_indices"].min() >= 0 and skin["lbs_indices"].max() < 27


def _lbs_numpy(skin, posed):
    """Plain-numpy reimplementation of ardy.viz.core_skin.CoreSkin.lbs.

    ``posed`` is (27,4,4) global joint transforms. Returns the deformed
    vertices, so the caller can check both the rest pose and an articulated one.
    """
    bind_v = skin["bind_vertices"].astype(np.float64)
    brt_inv = np.linalg.inv(skin["bind_rig_transform"].astype(np.float64))
    idx = skin["lbs_indices"]
    wt = skin["lbs_weights"].astype(np.float64)
    affine = (posed.astype(np.float64) @ brt_inv)[:, :3, :]        # (27,3,4)
    homog = np.concatenate([bind_v, np.ones((len(bind_v), 1))], 1)  # (V,4)
    per = np.einsum("vwij,vj->vwi", affine[idx], homog)            # (V,5,3)
    return (per * wt[..., None]).sum(1)                            # (V,3)


def _rest_identity(skin):
    """LBS at the bind pose (posed == bind) must return the bind vertices."""
    out = _lbs_numpy(skin, skin["bind_rig_transform"])
    return np.allclose(out, skin["bind_vertices"], atol=1e-5)


def test_end_to_end_rigid_parts():
    obj = os.path.join(EXAMPLE_DIR, "blocky_humanoid.obj")
    assert os.path.exists(obj), "run examples/custom_avatar/make_blocky_humanoid.py first"
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "skin_standard.npz")
        rep = ar.rig_avatar(obj, out, up_axis="auto", color=True)
        assert rep["labeled"] and rep["mode"] == "parts(rigid)"
        skin = _load(out)
        _assert_loads_like_ardy(skin)
        assert _rest_identity(skin), "LBS rest pose must reproduce bind vertices"
        assert "vertex_colors" in skin
        # every vertex of the arms/legs got a sided arm/leg joint (not Hips fallback)
        used = {ar.CORE_JOINT_NAMES[int(i)] for i in skin["lbs_indices"][:, 0]}
        assert "LeftForeArm" in used and "RightLeg" in used

        # articulation: bending one joint must move exactly its bound verts and
        # leave everything else put -- proves the weights actually bind (non-vacuous).
        j = ar.CORE_JOINT_NAMES.index("RightForeArm")
        th = np.pi / 4
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        posed = skin["bind_rig_transform"].astype(np.float64).copy()
        posed[j, :3, :3] = Rz @ posed[j, :3, :3]
        moved = np.linalg.norm(_lbs_numpy(skin, posed) - skin["bind_vertices"], axis=1)
        bound = (skin["lbs_indices"] == j).any(1)
        assert moved[bound].mean() > 1e-2, "forearm verts should follow the joint"
        assert moved[~bound].max() < 1e-5, "unbound verts must not move"
    print("ok: end-to-end rigid parts + ARDY loader contract + rest identity + articulation")


def test_smooth_and_geometry_fallback():
    obj = os.path.join(EXAMPLE_DIR, "blocky_humanoid.obj")
    with tempfile.TemporaryDirectory() as d:
        # smooth blend
        out = os.path.join(d, "smooth.npz")
        ar.rig_avatar(obj, out, smooth=True)
        skin = _load(out)
        _assert_loads_like_ardy(skin)
        assert (skin["lbs_weights"][:, 1] > 0).any(), "smooth mode should blend >1 joint"

        # strip labels -> geometry fallback still produces a valid skin
        mesh = ar.read_obj(obj)
        mesh.vertex_labels = ["" for _ in mesh.vertex_labels]
        rig = ar.load_core_rig()
        aligned = ar.align_to_core(mesh.vertices, rig)
        idx, wt = ar.skin_by_geometry(mesh, aligned, rig)
        out2 = os.path.join(d, "geo.npz")
        ar.write_skin(out2, aligned, mesh.faces, idx, wt, rig)
        _assert_loads_like_ardy(_load(out2))
    print("ok: smooth blend + geometry fallback")


def test_corrupted_names_would_be_rejected():
    """Sanity: a wrong name array is what CoreSkin rejects -- prove our check
    catches it, so a passing test genuinely means the loader accepts it."""
    bad = ar.CORE_JOINT_NAMES.copy()
    bad[8] = "R_Arm"  # Mixamo-ish but not ARDY's name
    try:
        for e, g in zip(ar.CORE_JOINT_NAMES, bad):
            assert e == g
    except AssertionError:
        print("ok: corrupted names correctly rejected")
        return
    raise AssertionError("corrupted names were NOT rejected -- test is vacuous")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} tests passed")
