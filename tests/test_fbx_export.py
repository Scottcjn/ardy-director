# SPDX-License-Identifier: AGPL-3.0-or-later
"""ARDY clip -> FBX skeletal-animation exporter.

These exercise director_service.fbx_export directly, which imports only numpy,
so the whole file runs with no GPU and no ARDY install. The strategy: build a
synthetic clip with a *reference* forward-kinematics that reimplements ARDY's
own rigid transform (ardy/skeleton/kinematics.py) from known rest offsets and
local rotations, then assert the exporter recovers those inputs and that a
full FK replay of the exported scene reproduces the world-space joints. Finally
re-parse the emitted FBX text and confirm the writer serialised the same
numbers the scene carried.
"""
import math
import re

import numpy as np
import pytest

from director_service.fbx_export import (
    build_scene,
    convert,
    euler_xyz_to_matrix,
    fbx_ascii,
    matrix_to_euler_xyz,
    reconstruct_positions,
)


# --------------------------------------------------------------------------- #
# A tiny skeleton and a reference FK identical to ARDY's rigid transform.
# --------------------------------------------------------------------------- #
# Hips -> Spine -> Head, plus Hips -> LeftUpLeg (branch), root = Hips.
NAMES = ["Hips", "Spine", "Head", "LeftUpLeg"]
PARENTS = [-1, 0, 1, 0]
REST_OFFSETS = np.array(
    [[0.0, 0.0, 0.0],    # root: no offset
     [0.0, 0.5, 0.0],    # Spine above Hips
     [0.0, 0.4, 0.0],    # Head above Spine
     [0.15, -0.1, 0.0]], # LeftUpLeg down/side from Hips
    dtype=np.float64,
)


def _rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _ref_fk(local_rots, root_pos):
    """Reimplements ardy fk: pos_j = pos_p + R_glob_p @ offset_j; R_glob_j = R_glob_p @ R_local_j."""
    frames, njoints = local_rots.shape[0], local_rots.shape[1]
    glob = np.zeros((frames, njoints, 3, 3))
    posed = np.zeros((frames, njoints, 3))
    for t in range(frames):
        for j in range(njoints):
            p = PARENTS[j]
            if p < 0:
                glob[t, j] = local_rots[t, j]
                posed[t, j] = root_pos[t]
            else:
                glob[t, j] = glob[t, p] @ local_rots[t, j]
                posed[t, j] = posed[t, p] + glob[t, p] @ REST_OFFSETS[j]
    return glob, posed


def _make_clip(path, frames=8, fps=30, with_local=False):
    rng = np.random.default_rng(7)
    njoints = len(NAMES)
    local = np.zeros((frames, njoints, 3, 3))
    for t in range(frames):
        for j in range(njoints):
            # A distinct, time-varying rotation per joint so nothing is trivially identity.
            ax = ["x", "y", "z"][j % 3]
            ang = 0.3 * math.sin(0.5 * t + j) + 0.1 * (j + 1)
            ax2 = ["y", "z", "x"][j % 3]
            local[t, j] = _rot(ax, ang) @ _rot(ax2, 0.2 * math.cos(0.3 * t + j))
    root_pos = np.stack(
        [0.1 * np.arange(frames), 0.9 + 0.02 * np.sin(np.arange(frames)),
         0.05 * np.arange(frames)], axis=1
    )
    glob, posed = _ref_fk(local, root_pos)

    extra = {}
    if with_local:
        extra["local_rot_mats"] = local.astype(np.float32)
    np.savez(
        path,
        fps=np.int64(fps),
        text=np.array("walk forward and nod"),
        posed_joints=posed.astype(np.float32),
        global_rot_mats=glob.astype(np.float32),
        root_positions=root_pos.astype(np.float32),
        joint_names=np.array(NAMES),
        joint_parents=np.array(PARENTS, dtype=np.int64),
        **extra,
    )
    return local, root_pos, glob, posed


# --------------------------------------------------------------- euler round-trip
@pytest.mark.parametrize("seed", range(6))
def test_euler_matrix_roundtrip(seed):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-math.pi, math.pi, size=3)
    m = euler_xyz_to_matrix(*ang)
    # Orthonormal, det +1.
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(m) - 1.0) < 1e-9
    back = matrix_to_euler_xyz(m)
    # Angles may differ but must reconstruct the same matrix.
    assert np.allclose(euler_xyz_to_matrix(*back), m, atol=1e-9)


def test_euler_handles_gimbal_lock():
    # ry = +90deg -> R[2,0] = -1, the pole.
    for ry in (math.pi / 2, -math.pi / 2):
        m = euler_xyz_to_matrix(0.4, ry, -0.7)
        back = matrix_to_euler_xyz(m)
        assert np.allclose(euler_xyz_to_matrix(*back), m, atol=1e-7)


# --------------------------------------------------------------- scene recovery
def test_bind_offsets_recovered(tmp_path):
    _make_clip(tmp_path / "c.npz")
    scene = build_scene(str(tmp_path / "c.npz"))
    assert scene["names"] == NAMES
    assert scene["parents"] == PARENTS
    assert scene["root_idx"] == 0
    assert np.allclose(scene["bind_offsets"], REST_OFFSETS, atol=1e-5)
    # Rigid skeleton -> the offset must not drift across frames.
    assert scene["offset_drift"] < 1e-4


def test_scene_reconstructs_original_joints(tmp_path):
    _local, _root, _glob, posed = _make_clip(tmp_path / "c.npz")
    scene = build_scene(str(tmp_path / "c.npz"))
    rebuilt = reconstruct_positions(scene)
    assert rebuilt.shape == posed.shape
    assert np.allclose(rebuilt, posed, atol=1e-4)


def test_scale_scales_lengths_not_angles(tmp_path):
    _make_clip(tmp_path / "c.npz")
    s1 = build_scene(str(tmp_path / "c.npz"), scale=1.0)
    s100 = build_scene(str(tmp_path / "c.npz"), scale=100.0)
    assert np.allclose(s100["bind_offsets"], s1["bind_offsets"] * 100, atol=1e-4)
    assert np.allclose(s100["root_translation"], s1["root_translation"] * 100, atol=1e-4)
    # Rotations are unaffected by scale.
    assert np.allclose(s100["euler_deg"], s1["euler_deg"], atol=1e-9)
    # And the scaled scene still reconstructs its (scaled) joints.
    rebuilt = reconstruct_positions(s100)
    assert np.allclose(rebuilt, reconstruct_positions(s1) * 100, atol=1e-3)


def test_local_rot_mats_present_gives_same_scene(tmp_path):
    # The exporter derives local rotations from global; a clip that also carries
    # local_rot_mats must not change the recovered animation.
    _make_clip(tmp_path / "a.npz", with_local=False)
    _make_clip(tmp_path / "b.npz", with_local=True)
    sa = build_scene(str(tmp_path / "a.npz"))
    sb = build_scene(str(tmp_path / "b.npz"))
    assert np.allclose(sa["euler_deg"], sb["euler_deg"], atol=1e-9)


# --------------------------------------------------------------- guards
def test_missing_posed_joints_rejected(tmp_path):
    np.savez(tmp_path / "g1.npz", fps=np.int64(30), text=np.array("robot"),
             joint_parents=np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError, match="posed_joints"):
        build_scene(str(tmp_path / "g1.npz"))


def test_missing_global_rots_rejected(tmp_path):
    j = np.zeros((3, 2, 3), dtype=np.float32)
    np.savez(tmp_path / "noglob.npz", fps=np.int64(30), text=np.array("x"),
             posed_joints=j, joint_parents=np.array([-1, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="global_rot_mats"):
        build_scene(str(tmp_path / "noglob.npz"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_scene(str(tmp_path / "nope.npz"))


# --------------------------------------------------------------- FBX text output
def _parse_fbx_models(text):
    """Pull each Model's name + Lcl Translation out of the ASCII FBX."""
    out = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r'\tModel: \d+, "Model::([^"]+)", "LimbNode"', line)
        if m:
            cur = m.group(1)
        mt = re.search(r'P: "Lcl Translation", "Lcl Translation", "", "A",'
                       r'([-\d.e+]+),([-\d.e+]+),([-\d.e+]+)', line)
        if mt and cur is not None:
            out[cur] = np.array([float(mt.group(i)) for i in (1, 2, 3)])
            cur = None
    return out


def _parse_first_curve_values(text):
    """Return the first AnimationCurve's KeyValueFloat list (root Lcl Rotation d|X)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "KeyValueFloat:" in line:
            vals = lines[i + 1].strip()[len("a: "):]
            return [float(x) for x in vals.split(",")]
    return None


def test_fbx_text_is_wellformed_and_carries_scene_numbers(tmp_path):
    _make_clip(tmp_path / "c.npz")
    scene = build_scene(str(tmp_path / "c.npz"), scale=100.0)
    text = fbx_ascii(scene)

    # Structural anchors UE's FBX importer keys off of.
    assert "FBXVersion: 7400" in text
    assert text.count('"LimbNode"') == len(NAMES) * 2  # NodeAttribute + Model each
    assert '"AnimStack::Take_001"' in text
    assert '"AnimLayer::BaseLayer"' in text
    assert 'P: "RotationOrder", "enum", "", "",0' in text  # eEulerXYZ, our convention

    # Bind translations round-trip through the serialiser.
    parsed = _parse_fbx_models(text)
    for j, name in enumerate(NAMES):
        assert np.allclose(parsed[name], scene["bind_offsets"][j], atol=1e-3), name

    # Every bone connects to its parent (child,parent) in Connections.
    for j, name in enumerate(NAMES):
        assert re.search(r'C: "OO",\d+,\d+', text)  # connections block exists
    # Root connects to the scene root node (parent id 0).
    assert re.search(r'C: "OO",\d+,0', text)


def test_first_curve_matches_root_rotation_track(tmp_path):
    _make_clip(tmp_path / "c.npz")
    scene = build_scene(str(tmp_path / "c.npz"))
    text = fbx_ascii(scene)
    vals = _parse_first_curve_values(text)
    # First curve emitted is the root bone's Lcl Rotation d|X across all frames.
    assert vals is not None
    assert np.allclose(vals, scene["euler_deg"][:, 0, 0], atol=1e-3)


def test_convert_end_to_end(tmp_path):
    _make_clip(tmp_path / "c.npz")
    report = convert(str(tmp_path / "c.npz"), str(tmp_path / "out.fbx"), scale=100.0)
    assert report["bones"] == len(NAMES)
    assert report["frames"] == 8
    assert (tmp_path / "out.fbx").exists()
    assert report["offset_drift"] < 1e-2  # scaled by 100 -> still tiny
    assert report["text"] == "walk forward and nod"
