# SPDX-License-Identifier: AGPL-3.0-or-later
"""Web viewport plumbing: tag sanitising + npz -> playback JSON.

These exercise director_service.web directly, which imports only numpy, so the
whole file runs with no GPU and no ARDY install. The FastAPI wiring in
service.py is a thin wrapper over motion_payload / npz_path_for tested here.
"""
import numpy as np
import pytest

from director_service.web import motion_payload, npz_path_for, safe_tag


def _write_clip(path, frames=6, joints=4, fps=30, seams=None, with_topology=True):
    j = np.arange(frames * joints * 3, dtype=np.float32).reshape(frames, joints, 3)
    extra = {}
    if with_topology:
        extra["joint_names"] = np.array(["Hips", "Spine", "Head", "LeftHand"][:joints])
        extra["joint_parents"] = np.array([-1, 0, 1, 1][:joints], dtype=np.int64)
    if seams is not None:
        extra["seam_frames"] = np.asarray(seams, dtype=np.int64)
    np.savez(path, fps=np.int64(fps), text=np.array("walk then wave"),
             posed_joints=j, root_positions=j[:, 0, :], **extra)


# ---------------------------------------------------------------- safe_tag
@pytest.mark.parametrize("bad", [
    "../secret", "/etc/passwd", "a/b", "..", ".", "x\\y", "n\x00ull", "", "a" * 200, 5,
])
def test_safe_tag_rejects_traversal_and_junk(bad):
    assert safe_tag(bad) is None


@pytest.mark.parametrize("ok", ["gen_1720000000_abc123", "choreo_1.2-3", "clip.npz_tag"])
def test_safe_tag_accepts_plain_tokens(ok):
    assert safe_tag(ok) == ok


def test_npz_path_for_blocks_bad_tag(tmp_path):
    assert npz_path_for(str(tmp_path), "../../etc/passwd") is None
    good = npz_path_for(str(tmp_path), "gen_1_a")
    assert good == str(tmp_path / "gen_1_a.npz")


# ------------------------------------------------------------- motion_payload
def test_payload_shape_and_topology(tmp_path):
    p = tmp_path / "gen_1_a.npz"
    _write_clip(p, frames=6, joints=4)
    out = motion_payload(str(p))
    assert out["ok"] and out["fps"] == 30
    assert out["frames"] == 6 and out["rendered_frames"] == 6 and out["stride"] == 1
    assert out["joint_names"][0] == "Hips"
    assert out["parents"] == [-1, 0, 1, 1]
    assert len(out["positions"]) == 6
    assert len(out["positions"][0]) == 4 and len(out["positions"][0][0]) == 3
    assert out["text"] == "walk then wave"


def test_payload_stride_downsamples_frames(tmp_path):
    p = tmp_path / "gen_2_b.npz"
    _write_clip(p, frames=10)
    out = motion_payload(str(p), stride=3)
    assert out["frames"] == 10                     # original count preserved
    assert out["rendered_frames"] == 4             # frames 0,3,6,9
    assert len(out["positions"]) == 4
    # first sampled frame equals original frame 0
    assert out["positions"][0][0][0] == 0.0


def test_payload_carries_seam_frames(tmp_path):
    p = tmp_path / "choreo_3_c.npz"
    _write_clip(p, frames=8, seams=[3, 5])
    assert motion_payload(str(p))["seam_frames"] == [3, 5]


def test_payload_defaults_topology_when_absent(tmp_path):
    p = tmp_path / "gen_4_d.npz"
    _write_clip(p, frames=4, joints=3, with_topology=False)
    out = motion_payload(str(p))
    assert out["parents"] == [-1, -1, -1]
    assert out["joint_names"] == ["0", "1", "2"]


def test_missing_clip_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        motion_payload(str(tmp_path / "nope.npz"))


def test_clip_without_posed_joints_is_unrenderable(tmp_path):
    p = tmp_path / "gen_5_e.npz"
    np.savez(p, fps=np.int64(30), text=np.array("robot"),
             qpos=np.zeros((5, 30), dtype=np.float32))
    with pytest.raises(ValueError, match="posed_joints"):
        motion_payload(str(p))
