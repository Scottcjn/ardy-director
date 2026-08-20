#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Convert an ARDY director `.npz` clip to an FBX skeletal animation.

The output imports into Unreal Engine 5 (and Maya / Blender) as a joint
hierarchy with a per-frame animation, ready to drive a humanoid rig through
UE's IK Retargeter. ARDY's core skeleton already uses Mixamo-style bone names
(``Hips``, ``Spine``, ``LeftArm``, ``LeftForeArm`` ...), which is exactly what
the retargeter expects, so no custom bone glue is needed.

Why this module is GPU-free and torch-free (like ``director_service.web``):
the generator already bakes everything an exporter needs straight into the
``.npz`` -- world-space joint positions (``posed_joints``), global rotation
matrices (``global_rot_mats``), the root track (``root_positions``) and the
skeleton topology (``joint_names`` / ``joint_parents``). We never have to
reload the model, so this runs on any machine and is fully unit-testable.

The math, in one line
---------------------
ARDY poses joints with the standard SMPL rigid transform
(``ardy/skeleton/kinematics.py``)::

    pos_j       = pos_parent + R_global_parent @ rest_offset_j
    R_global_j  = R_global_parent @ R_local_j

FBX composes a node's world transform identically::

    world_j = world_parent @ Translate(Lcl Translation) @ Rotate(Lcl Rotation)

so the mapping is exact:

* ``Lcl Translation`` of a bone == its rest offset ``rest_offset_j`` (constant),
* ``Lcl Rotation``    of a bone == its per-frame local rotation ``R_local_j``,
* the root's ``Lcl Translation`` is animated with ``root_positions``.

The rest offset is frame-invariant for a rigid skeleton, so we recover it
directly from the clip -- no model asset (``joints.p``) required::

    rest_offset_j = R_global_parent(t)^T @ (pos_j(t) - pos_parent(t))

and the local rotation is recovered the same way::

    R_local_j = R_global_parent(t)^T @ R_global_j(t)   (root: R_local = R_global)

Everything below is pure numpy + stdlib.
"""
import os

import numpy as np

# FBX stores key times in "ktime" ticks; this is the count per second for the
# default time mode. round(frame / fps * FBX_KTIME_PER_SEC) is the key time.
FBX_KTIME_PER_SEC = 46186158000

# Rotations are written as eEulerXYZ (RotationOrder value 0). For column
# vectors this order composes as R = Rz @ Ry @ Rx (intrinsic X, then Y, then Z),
# which is what the Autodesk FBX SDK -- and therefore Unreal -- reconstructs.
FBX_ROTATION_ORDER_XYZ = 0


# --------------------------------------------------------------------------- #
# Euler <-> matrix, in the exact convention we tag the FBX nodes with.
# --------------------------------------------------------------------------- #
def euler_xyz_to_matrix(rx, ry, rz):
    """Build a rotation matrix from eEulerXYZ angles in radians (R = Rz@Ry@Rx)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def matrix_to_euler_xyz(mat):
    """Decompose a rotation matrix into eEulerXYZ angles in radians.

    Inverse of :func:`euler_xyz_to_matrix` (R = Rz@Ry@Rx). Handles the
    gimbal-lock pole (|R[2, 0]| ~= 1) without blowing up.
    """
    m = np.asarray(mat, dtype=np.float64)
    # R[2, 0] == -sin(ry)
    sy = -m[2, 0]
    sy = min(1.0, max(-1.0, sy))
    ry = np.arcsin(sy)
    cy = np.cos(ry)
    if abs(cy) > 1e-7:
        rx = np.arctan2(m[2, 1], m[2, 2])
        rz = np.arctan2(m[1, 0], m[0, 0])
    else:
        # Gimbal lock: pin rz, fold the remaining freedom into rx.
        rz = 0.0
        rx = np.arctan2(-m[1, 2], m[1, 1])
    return np.array([rx, ry, rz], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Load an ARDY clip and turn it into an FBX-ready scene (bind pose + curves).
# --------------------------------------------------------------------------- #
def _load_arrays(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        files = set(data.files)
        if "posed_joints" not in files:
            raise ValueError(
                "clip has no 'posed_joints'; FBX export needs an avatar "
                "skeleton (core/soma), not a robot-qpos model (g1)"
            )
        posed = np.asarray(data["posed_joints"], dtype=np.float64)  # (F, J, 3)
        glob = (np.asarray(data["global_rot_mats"], dtype=np.float64)
                if "global_rot_mats" in files else None)          # (F, J, 3, 3)
        local = (np.asarray(data["local_rot_mats"], dtype=np.float64)
                 if "local_rot_mats" in files else None)
        root = (np.asarray(data["root_positions"], dtype=np.float64)
                if "root_positions" in files else None)           # (F, 3)
        fps = int(data["fps"]) if "fps" in files else 30
        text = str(data["text"]) if "text" in files else ""
        if "joint_names" in files:
            names = [str(x) for x in data["joint_names"]]
        else:
            names = [str(i) for i in range(posed.shape[1])]
        if "joint_parents" in files:
            parents = [int(x) for x in data["joint_parents"]]
        else:
            raise ValueError(
                "clip has no 'joint_parents'; cannot build a bone hierarchy"
            )
    if posed.ndim != 3 or posed.shape[2] != 3:
        raise ValueError(f"unexpected posed_joints shape {posed.shape}")
    if glob is None:
        raise ValueError(
            "clip has no 'global_rot_mats'; FBX export needs joint rotations, "
            "not just positions"
        )
    return posed, glob, local, root, fps, text, names, parents


def build_scene(npz_path, scale=1.0):
    """Read a director clip and return an FBX-ready scene description.

    Returns a dict with:
      ``names``            bone names (Mixamo-style for the core skeleton)
      ``parents``          parent index per bone (-1 = root)
      ``root_idx``         index of the root bone
      ``fps``              frames per second
      ``frames``           frame count
      ``bind_offsets``     (J, 3) rest bone offset in the parent's frame
                           (the node's static Lcl Translation), already scaled
      ``euler_deg``        (F, J, 3) per-frame Lcl Rotation in degrees
      ``root_translation`` (F, 3) per-frame Lcl Translation of the root, scaled
      ``text``             the prompt that produced the clip
      ``offset_drift``     max frame-to-frame disagreement in the recovered
                           rest offsets (a rigid skeleton -> ~0; a sanity meter)

    ``scale`` multiplies every length (ARDY is metric; pass 100 for cm if you
    want Unreal-native units, though UE's import dialog can also rescale).
    """
    posed, glob, local, root, fps, text, names, parents = _load_arrays(npz_path)
    frames, njoints = posed.shape[0], posed.shape[1]
    parents = list(parents)
    root_idx = parents.index(-1)

    # Root world track: prefer the saved root_positions, else read it off the
    # posed joints (they are identical -- posed_joints[:, root] == root_positions).
    if root is None:
        root = posed[:, root_idx, :].copy()

    # Rest offset of each bone, in its parent's local frame. Rigid skeleton ->
    # frame-invariant, so we take frame 0 and measure the drift as a check.
    bind_offsets = np.zeros((njoints, 3), dtype=np.float64)
    offset_drift = 0.0
    for j in range(njoints):
        p = parents[j]
        if p < 0:
            continue  # root offset is 0; its world motion lives in the T-curve
        # o_j(t) = R_global_parent(t)^T @ (pos_j(t) - pos_parent(t))
        rel = posed[:, j, :] - posed[:, p, :]                    # (F, 3)
        off = np.einsum("fki,fk->fi", glob[:, p], rel)           # R_p^T @ rel
        bind_offsets[j] = off[0]
        if frames > 1:
            offset_drift = max(offset_drift, float(np.abs(off - off[0]).max()))

    # Per-frame local rotation of each bone. Derive from the global rotations so
    # the result is self-consistent with the positions we measured the offsets
    # from; fall back is unnecessary because global_rot_mats is required.
    euler_deg = np.zeros((frames, njoints, 3), dtype=np.float64)
    for j in range(njoints):
        p = parents[j]
        for t in range(frames):
            if p < 0:
                r_local = glob[t, j]
            else:
                r_local = glob[t, p].T @ glob[t, j]
            euler_deg[t, j] = np.degrees(matrix_to_euler_xyz(r_local))

    return {
        "names": names,
        "parents": parents,
        "root_idx": root_idx,
        "fps": int(fps),
        "frames": int(frames),
        "bind_offsets": bind_offsets * float(scale),
        "euler_deg": euler_deg,
        "root_translation": root * float(scale),
        "text": text,
        "offset_drift": offset_drift * float(scale),
    }


# --------------------------------------------------------------------------- #
# FK reconstruction -- used by the tests (and available to callers) to prove a
# scene reproduces the original world-space joints it was built from.
# --------------------------------------------------------------------------- #
def reconstruct_positions(scene):
    """Replay a scene's bind pose + Euler curves back into world positions.

    Returns ``(frames, J, 3)`` world-space joints, computed with the same rigid
    transform ARDY uses. If a scene is correct, this equals the clip's original
    ``posed_joints`` (up to ``scale``). This is how the exporter is verified with
    no GPU and no Unreal in the loop.
    """
    parents = scene["parents"]
    offsets = scene["bind_offsets"]
    euler = np.radians(scene["euler_deg"])
    root_tr = scene["root_translation"]
    root_idx = scene["root_idx"]
    frames, njoints = euler.shape[0], euler.shape[1]

    out = np.zeros((frames, njoints, 3), dtype=np.float64)
    for t in range(frames):
        gpos = [None] * njoints
        grot = [None] * njoints
        # parents always precede children in ARDY's bone order, so one pass works.
        for j in range(njoints):
            r_local = euler_xyz_to_matrix(*euler[t, j])
            p = parents[j]
            if p < 0:
                grot[j] = r_local
                gpos[j] = root_tr[t].copy()
            else:
                grot[j] = grot[p] @ r_local
                gpos[j] = gpos[p] + grot[p] @ offsets[j]
            out[t, j] = gpos[j]
    return out


# --------------------------------------------------------------------------- #
# FBX ASCII writer.
# --------------------------------------------------------------------------- #
class _Ids:
    """Hands out the unique int64 object ids FBX connects everything by."""

    def __init__(self, start=1000000):
        self._n = start

    def next(self):
        self._n += 1
        return self._n


def _fmt(x):
    """Compact but loss-free float formatting for FBX property values."""
    return repr(float(x))


def _prop(name, typ, label, flags, *values):
    vals = "".join("," + _fmt(v) if isinstance(v, float) else f",{v}" for v in values)
    return f'\t\t\tP: "{name}", "{typ}", "{label}", "{flags}"{vals}'


def write_fbx(scene, out_path):
    """Serialise a scene from :func:`build_scene` to an ASCII FBX 7.4 file."""
    text = fbx_ascii(scene)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path


def fbx_ascii(scene):
    """Return the ASCII FBX 7.4 document for a scene as a string."""
    names = scene["names"]
    parents = scene["parents"]
    offsets = scene["bind_offsets"]
    euler = scene["euler_deg"]
    root_tr = scene["root_translation"]
    root_idx = scene["root_idx"]
    fps = scene["fps"]
    frames = scene["frames"]
    njoints = len(names)

    ids = _Ids()
    model_id = [ids.next() for _ in range(njoints)]
    attr_id = [ids.next() for _ in range(njoints)]

    ktimes = [int(round(t / float(fps) * FBX_KTIME_PER_SEC)) for t in range(frames)]
    ktime_start, ktime_stop = (ktimes[0], ktimes[-1]) if frames else (0, 0)

    L = []
    a = L.append

    # ---- header ------------------------------------------------------------
    a("; FBX 7.4.0 project file")
    a("; Generated by ARDY director -> FBX exporter (director_service/fbx_export.py)")
    a("; ----------------------------------------------------")
    a("")
    a("FBXHeaderExtension:  {")
    a("\tFBXHeaderVersion: 1003")
    a("\tFBXVersion: 7400")
    a("\tCreator: \"ARDY director FBX exporter\"")
    a("}")
    a("GlobalSettings:  {")
    a("\tVersion: 1000")
    a("\tProperties70:  {")
    a("\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",1")
    a("\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1")
    a("\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",2")
    a("\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1")
    a("\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0")
    a("\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1")
    a("\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1")
    a(f"\t\tP: \"TimeMode\", \"enum\", \"\", \"\",{_default_time_mode(fps)}")
    a("\t}")
    a("}")

    # ---- definitions -------------------------------------------------------
    total_curvenodes = njoints + 1        # one R per bone + one T for the root
    total_curves = njoints * 3 + 3        # xyz per rotation curvenode + root T
    a("Definitions:  {")
    a("\tVersion: 100")
    a(f"\tCount: {njoints * 2 + total_curvenodes + total_curves + 3}")
    for otype, count in [
        ("GlobalSettings", 1),
        ("Model", njoints),
        ("NodeAttribute", njoints),
        ("AnimationStack", 1),
        ("AnimationLayer", 1),
        ("AnimationCurveNode", total_curvenodes),
        ("AnimationCurve", total_curves),
    ]:
        a(f"\tObjectType: \"{otype}\" {{")
        a(f"\t\tCount: {count}")
        a("\t}")
    a("}")

    # ---- objects -----------------------------------------------------------
    a("Objects:  {")

    for j in range(njoints):
        # NodeAttribute (LimbNode) -- makes the Model a bone/joint.
        a(f'\t;NodeAttribute::{names[j]}')
        a(f'\tNodeAttribute: {attr_id[j]}, "NodeAttribute::{names[j]}", "LimbNode" {{')
        a("\t\tProperties70:  {")
        a("\t\t\tP: \"Size\", \"double\", \"Number\", \"\",1")
        a("\t\t}")
        a("\t\tTypeFlags: \"Skeleton\"")
        a("\t}")

        # Model (LimbNode) -- the bone node itself, with its rest transform.
        off = offsets[j]
        a(f'\tModel: {model_id[j]}, "Model::{names[j]}", "LimbNode" {{')
        a("\t\tVersion: 232")
        a("\t\tProperties70:  {")
        a(_prop("RotationActive", "bool", "", "", 1))
        a(_prop("InheritType", "enum", "", "", 1))  # RSrs
        a(_prop("ScalingMax", "Vector3D", "Vector", "", 0.0, 0.0, 0.0))
        a(_prop("DefaultAttributeIndex", "int", "Integer", "", 0))
        a(_prop("RotationOrder", "enum", "", "", FBX_ROTATION_ORDER_XYZ))
        a(_prop("Lcl Translation", "Lcl Translation", "", "A",
                float(off[0]), float(off[1]), float(off[2])))
        a(_prop("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0))
        a(_prop("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0))
        a("\t\t}")
        a("\t\tShading: T")
        a("\t\tCulling: \"CullingOff\"")
        a("\t}")

    # AnimationStack + AnimationLayer
    stack_id = ids.next()
    layer_id = ids.next()
    a(f'\tAnimationStack: {stack_id}, "AnimStack::Take_001", "" {{')
    a("\t\tProperties70:  {")
    a(f"\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",{ktime_start}")
    a(f"\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",{ktime_stop}")
    a(f"\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",{ktime_start}")
    a(f"\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",{ktime_stop}")
    a("\t\t}")
    a("\t}")
    a(f'\tAnimationLayer: {layer_id}, "AnimLayer::BaseLayer", "" {{')
    a("\t}")

    # Curve nodes + curves. Collect connections as we go.
    conns = []  # (kind, child, parent, prop_or_none)

    def _curve(values):
        cid = ids.next()
        a(f'\tAnimationCurve: {cid}, "AnimCurve::", "" {{')
        a("\t\tDefault: 0")
        a("\t\tKeyVer: 4009")
        a("\t\tKeyTime: *%d {" % len(ktimes))
        a("\t\t\ta: " + ",".join(str(k) for k in ktimes))
        a("\t\t}")
        a("\t\tKeyValueFloat: *%d {" % len(values))
        a("\t\t\ta: " + ",".join(_fmt(v) for v in values))
        a("\t\t}")
        # Constant-in / constant-out attr flags per key (24840 = auto).
        a("\t\tKeyAttrFlags: *1 {")
        a("\t\t\ta: 24840")
        a("\t\t}")
        a("\t\tKeyAttrDataFloat: *4 {")
        a("\t\t\ta: 0,0,0,0")
        a("\t\t}")
        a("\t\tKeyAttrRefCount: *1 {")
        a("\t\t\ta: %d" % len(ktimes))
        a("\t\t}")
        a("\t}")
        return cid

    for j in range(njoints):
        # Rotation curve node (all bones animate rotation).
        r0 = euler[0, j] if frames else np.zeros(3)
        rcn = ids.next()
        a(f'\tAnimationCurveNode: {rcn}, "AnimCurveNode::R", "" {{')
        a("\t\tProperties70:  {")
        a(_prop("d|X", "Number", "", "A", float(r0[0])))
        a(_prop("d|Y", "Number", "", "A", float(r0[1])))
        a(_prop("d|Z", "Number", "", "A", float(r0[2])))
        a("\t\t}")
        a("\t}")
        cx = _curve(euler[:, j, 0])
        cy = _curve(euler[:, j, 1])
        cz = _curve(euler[:, j, 2])
        conns.append(("OP", rcn, model_id[j], "Lcl Rotation"))
        conns.append(("OP", cx, rcn, "d|X"))
        conns.append(("OP", cy, rcn, "d|Y"))
        conns.append(("OP", cz, rcn, "d|Z"))
        conns.append(("OO", rcn, layer_id, None))

        if j == root_idx:
            t0 = root_tr[0] if frames else np.zeros(3)
            tcn = ids.next()
            a(f'\tAnimationCurveNode: {tcn}, "AnimCurveNode::T", "" {{')
            a("\t\tProperties70:  {")
            a(_prop("d|X", "Number", "", "A", float(t0[0])))
            a(_prop("d|Y", "Number", "", "A", float(t0[1])))
            a(_prop("d|Z", "Number", "", "A", float(t0[2])))
            a("\t\t}")
            a("\t}")
            tx = _curve(root_tr[:, 0])
            ty = _curve(root_tr[:, 1])
            tz = _curve(root_tr[:, 2])
            conns.append(("OP", tcn, model_id[j], "Lcl Translation"))
            conns.append(("OP", tx, tcn, "d|X"))
            conns.append(("OP", ty, tcn, "d|Y"))
            conns.append(("OP", tz, tcn, "d|Z"))
            conns.append(("OO", tcn, layer_id, None))

    a("}")

    # ---- connections -------------------------------------------------------
    a("Connections:  {")
    # Layer -> Stack
    a(f"\tC: \"OO\",{layer_id},{stack_id}")
    # Bone hierarchy + attributes.
    for j in range(njoints):
        p = parents[j]
        parent_model = 0 if p < 0 else model_id[p]  # 0 == scene root node
        a(f"\tC: \"OO\",{model_id[j]},{parent_model}")
        a(f"\tC: \"OO\",{attr_id[j]},{model_id[j]}")
    # Animation graph.
    for kind, child, parent, prop in conns:
        if kind == "OP":
            a(f"\tC: \"OP\",{child},{parent}, \"{prop}\"")
        else:
            a(f"\tC: \"OO\",{child},{parent}")
    a("}")
    a("")
    return "\n".join(L)


def _default_time_mode(fps):
    """Map common fps to an FBX TimeMode enum; 0 (default) otherwise.

    The KeyTime ticks we write are absolute, so playback is correct regardless;
    this just makes DCC tools show a sensible frame rate.
    """
    return {24: 10, 25: 12, 30: 6, 48: 15, 50: 16, 60: 17}.get(int(fps), 0)


def fbx_document(npz_path, scale=1.0):
    """Read a clip and return (fbx_text, report) without touching disk.

    Used by the HTTP route to stream an FBX straight to the client.
    """
    scene = build_scene(npz_path, scale=scale)
    report = {
        "bones": len(scene["names"]),
        "frames": scene["frames"],
        "fps": scene["fps"],
        "offset_drift": scene["offset_drift"],
        "text": scene["text"],
    }
    return fbx_ascii(scene), report


def convert(npz_path, out_path, scale=1.0):
    """One-call: read a clip, build the scene, write the FBX. Returns a report."""
    scene = build_scene(npz_path, scale=scale)
    write_fbx(scene, out_path)
    return {
        "out": out_path,
        "bones": len(scene["names"]),
        "frames": scene["frames"],
        "fps": scene["fps"],
        "offset_drift": scene["offset_drift"],
        "text": scene["text"],
    }
