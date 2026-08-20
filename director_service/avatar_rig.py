#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Rig any Blender humanoid onto ARDY's 27-joint core skeleton.

We rigged the bundled low-poly avatar to ARDY by hand -- axis fix, scale fit,
per-part rigid skinning, flat-color groups. This module turns that one-off into
a reusable tool: give it a humanoid mesh in a neutral T-pose and it emits a
valid ``skin_standard.npz`` that ARDY's viewer loads with no joint-name
mismatch, plus optional per-part flat colors.

Why this is GPU-free, torch-free and ARDY-install-free (like
``director_service.fbx_export``): the skin file is self-contained data. ARDY's
``CoreSkin`` only reads seven arrays back out of it -- ``bind_vertices``,
``faces``, ``bind_rig_transform``, ``rig_joint_names``, ``lbs_indices``,
``lbs_weights``, ``rig_joint_connections`` -- and the *only* thing it asserts
about the rig is that ``rig_joint_names`` equals the skeleton's ``bone_order``
in order (else it raises ``ValueError: MISMATCH in skinnging rig``). So we ship
the canonical core rig (bind pose + names + bone edges) as a tiny bundled asset
and never touch a GPU.

The math, in one line
---------------------
ARDY skins a vertex ``v`` at frame ``t`` with linear blend skinning::

    v(t) = sum_j  w_vj * (T_posed_j(t) @ T_bind_j^-1) @ v_bind

At the bind pose ``T_posed == T_bind`` every affine collapses to identity, so
``v(t) == v_bind``. That is the whole contract the deform has to honour: the
custom mesh vertices must be authored *in the core rig's bind pose*, and each
vertex must be bound (``lbs_indices`` / ``lbs_weights``) to the joints whose
bind position it sits on. Get those two right and every clip ARDY generates
deforms the custom mesh correctly, because ``T_posed_j`` is exactly the same
transform the model already drives ``joint j`` with.

So the pipeline is:

1.  **Align** the incoming mesh into the core bind pose -- fix the up axis
    (Blender is Z-up, ARDY is Y-up), uniform-scale to the skeleton's height,
    and translate so feet rest on the floor and the body is centred on the
    spine. (:func:`align_to_core`)
2.  **Bind** every vertex to core joints. With part labels we bind rigidly
    (each part's verts -> its joint, weight 1); a ``--smooth`` option instead
    blends the nearest bones by inverse distance for bend-friendly seams. With
    no labels we fall back to pure geometry -- nearest bone segment.
    (:func:`skin_by_parts`, :func:`skin_by_geometry`)
3.  **Write** the seven arrays, reusing the canonical ``bind_rig_transform`` /
    ``rig_joint_names`` / ``rig_joint_connections`` verbatim so the name check
    can never fail. (:func:`write_skin`)

Part-name mapping (:data:`PART_ALIASES`) understands the common humanoid
conventions -- Mixamo (``LeftArm``), Blender Rigify / ``.L`` suffixes
(``upper_arm.L``), MakeHuman, plain ``arm_left`` -- and resolves the
left/right side from the label when present, otherwise from vertex geometry
(sign of x). No dependency beyond numpy + the stdlib OBJ reader below.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Canonical core rig (27 joints). Shipped as a ~4.5 KB asset next to this file
# so the tool needs no ARDY checkout. It carries only what a skin file reuses
# verbatim: the bind pose, the joint names, and the bone edges.
# ---------------------------------------------------------------------------

CORE_RIG_ASSET = os.path.join(os.path.dirname(__file__), "assets", "core_rig_cskel27.npz")

# The 27 core joints, in ARDY's ``bone_order``. Duplicated here as a literal so
# the module is self-documenting and testable even if the asset is missing.
CORE_JOINT_NAMES = [
    "Hips",
    "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandEnd", "RightHandThumb1",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd", "LeftHandThumb1",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]

MAX_INFLUENCES = 5  # ARDY's skin uses W=5 columns in lbs_indices / lbs_weights


@dataclass
class CoreRig:
    """The canonical bind pose + topology every emitted skin reuses verbatim."""

    joint_names: list
    bind_rig_transform: np.ndarray            # (27, 4, 4) float32
    connections: np.ndarray                   # (26, 2)   int64

    @property
    def joint_positions(self) -> np.ndarray:  # (27, 3) bind-pose world positions
        return self.bind_rig_transform[:, :3, 3].astype(np.float64)

    def index(self, name: str) -> int:
        return self.joint_names.index(name)


def load_core_rig(path: str = CORE_RIG_ASSET) -> CoreRig:
    """Load the bundled canonical core rig."""
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["rig_joint_names"]]
    if names != CORE_JOINT_NAMES:
        raise ValueError(
            "bundled core rig asset joint names drifted from CORE_JOINT_NAMES; "
            "regenerate assets/core_rig_cskel27.npz from ARDY's cskel27 skin"
        )
    return CoreRig(
        joint_names=names,
        bind_rig_transform=d["bind_rig_transform"].astype(np.float32),
        connections=d["rig_joint_connections"].astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Part-name -> core-joint mapping.
#
# Keys are core joint names; values are lowercase substrings that a rig/part is
# commonly named after across Blender, Mixamo, Rigify, MakeHuman. Longer / more
# specific aliases must win over their prefixes (``forearm`` before ``arm``,
# ``toe`` before ``foot``), so we match by longest alias first.
# ---------------------------------------------------------------------------

PART_ALIASES = {
    "Hips": ["hips", "pelvis", "hip", "root"],
    "Spine": ["spine", "spine_01", "spine01", "abdomen", "lowerback", "spine.001"],
    "Spine1": ["spine1", "spine_02", "spine02", "chest_lower", "spine.002"],
    "Spine2": ["spine2", "spine_03", "spine03", "chest", "spine.003"],
    "Spine3": ["spine3", "spine_04", "spine04", "chest_upper", "upperchest", "spine.004"],
    "Neck": ["neck"],
    "Head": ["head", "skull"],
    "Shoulder": ["shoulder", "clavicle", "collar"],
    "Arm": ["upperarm", "upper_arm", "arm"],
    "ForeArm": ["forearm", "lowerarm", "lower_arm", "elbow"],
    "Hand": ["hand", "wrist", "palm"],
    "HandEnd": ["handend", "hand_end", "fingers", "middle1", "hand_tip"],
    "HandThumb1": ["thumb"],
    "UpLeg": ["upleg", "upperleg", "upper_leg", "thigh", "hip_"],
    "Leg": ["lowerleg", "lower_leg", "shin", "calf", "knee", "leg"],
    "Foot": ["foot", "ankle"],
    "ToeBase": ["toe", "ball"],
}

# Which core joints are sided (need a Left/Right prefix) vs central.
_SIDED = {"Shoulder", "Arm", "ForeArm", "Hand", "HandEnd", "HandThumb1", "UpLeg", "Leg", "Foot", "ToeBase"}
_CENTRAL = {"Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"}

# Longest-first alias table for greedy matching: (alias, base_joint).
_ALIAS_TABLE = sorted(
    ((alias, base) for base, aliases in PART_ALIASES.items() for alias in aliases),
    key=lambda kv: -len(kv[0]),
)

# Side markers, covering the common humanoid naming styles:
#   * separated words / suffixes: "arm_l", "forearm.R", "hand left", "leg_right"
#   * short single-letter tags:   "arm.l", "j_r"
#   * concatenated CamelCase:     "LeftForeArm", "RightHand" (Mixamo)
_LEFT_SEP = re.compile(r"(?:^|[._\- ])(l|left|lft)(?:$|[._\- 0-9])")
_RIGHT_SEP = re.compile(r"(?:^|[._\- ])(r|right|rgt)(?:$|[._\- 0-9])")


def _detect_side(label: str) -> str | None:
    """Return 'Left'/'Right'/None from a part label."""
    low = label.lower()
    # concatenated leading word (Mixamo): LeftForeArm / RightHand
    if low.startswith("left"):
        return "Left"
    if low.startswith("right"):
        return "Right"
    left = _LEFT_SEP.search(low)
    right = _RIGHT_SEP.search(low)
    if left and not right:
        return "Left"
    if right and not left:
        return "Right"
    if left and right:
        # both present -> take the later one (e.g. "arm_ik.L" the .L wins)
        return "Left" if left.start() > right.start() else "Right"
    return None


def map_part_to_joint(label: str, x_sign: float | None = None) -> str | None:
    """Map a free-form part / bone label to a core joint name.

    ``x_sign`` (mean x of the part's vertices in aligned core space, +left /
    -right) breaks the side when the label itself carries no L/R marker.
    Returns ``None`` if nothing matches (caller can fall back to geometry).
    """
    low = label.lower()
    base = None
    for alias, cand in _ALIAS_TABLE:
        if alias in low:
            base = cand
            break
    if base is None:
        return None
    if base in _CENTRAL:
        return base
    if base in _SIDED:
        side = _detect_side(label)
        if side is None:
            if x_sign is None:
                return None
            side = "Left" if x_sign >= 0 else "Right"
        return side + base
    return base  # (unreachable; every base is central or sided)


# ---------------------------------------------------------------------------
# OBJ reader (stdlib only). Parses vertices, triangulated faces, and per-vertex
# part labels taken from ``o <name>`` / ``g <name>`` sections.
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    vertices: np.ndarray                       # (V, 3) float64
    faces: np.ndarray                          # (F, 3) int64
    vertex_labels: list = field(default_factory=list)  # len V, str or "" if none

    @property
    def has_labels(self) -> bool:
        return any(lbl for lbl in self.vertex_labels)


def read_obj(path: str) -> Mesh:
    """Read a Wavefront OBJ. Faces are fan-triangulated; groups/objects become
    per-vertex labels (a vertex takes the label of the first group it is used in).
    """
    verts: list = []
    faces: list = []
    labels: dict = {}       # vertex index -> label
    current = ""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith(("o ", "g ")):
                current = line[2:].strip()
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    vi = tok.split("/")[0]
                    if not vi:
                        continue
                    vi = int(vi)
                    vi = vi - 1 if vi > 0 else len(verts) + vi   # 1-based / negative
                    idx.append(vi)
                    labels.setdefault(vi, current)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), np.int64)
    vlabels = [labels.get(i, "") for i in range(len(V))]
    return Mesh(vertices=V, faces=F, vertex_labels=vlabels)


# ---------------------------------------------------------------------------
# Alignment: Blender Z-up -> ARDY Y-up, uniform scale to skeleton height,
# translate so feet touch the floor and the body centres on the spine.
# ---------------------------------------------------------------------------

def align_to_core(vertices: np.ndarray, rig: CoreRig, up_axis: str = "auto") -> np.ndarray:
    """Return a copy of ``vertices`` placed into the core bind pose frame.

    ``up_axis``: 'y' (already ARDY-oriented), 'z' (Blender), or 'auto' (pick the
    axis with the largest extent as up, matching a standing humanoid).
    """
    v = np.asarray(vertices, dtype=np.float64).copy()
    if v.shape[0] == 0:
        return v

    if up_axis == "auto":
        extent = v.max(axis=0) - v.min(axis=0)
        up_axis = "z" if extent[2] >= extent[1] else "y"
    if up_axis == "z":
        # Z-up (Blender) -> Y-up (ARDY): (x, y, z) -> (x, z, -y)
        v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)

    # Uniform scale so the mesh height matches the skeleton's foot->head span.
    jp = rig.joint_positions
    skel_h = jp[:, 1].max() - jp[:, 1].min()
    mesh_h = v[:, 1].max() - v[:, 1].min()
    if mesh_h > 1e-9:
        v *= skel_h / mesh_h

    # Drop feet to the skeleton floor, centre x/z on the skeleton root column.
    foot_y = jp[:, 1].min()
    v[:, 1] += foot_y - v[:, 1].min()
    centre = v[:, [0, 2]].mean(axis=0)
    v[:, 0] += jp[0, 0] - centre[0]
    v[:, 2] += jp[0, 2] - centre[1]
    return v


# ---------------------------------------------------------------------------
# Skinning.
# ---------------------------------------------------------------------------

def _pack_influences(idx_list, w_list) -> tuple:
    """Pad/truncate per-vertex (indices, weights) to MAX_INFLUENCES, weights
    renormalised to sum to 1. Returns (int32 [V,5], float32 [V,5])."""
    V = len(idx_list)
    idx = np.zeros((V, MAX_INFLUENCES), dtype=np.int32)
    wgt = np.zeros((V, MAX_INFLUENCES), dtype=np.float32)
    for i, (ids, ws) in enumerate(zip(idx_list, w_list)):
        order = np.argsort(ws)[::-1][:MAX_INFLUENCES]
        ids = np.asarray(ids)[order]
        ws = np.asarray(ws, dtype=np.float64)[order]
        s = ws.sum()
        if s <= 0:
            ids, ws = np.array([0]), np.array([1.0])
            s = 1.0
        n = len(ids)
        idx[i, :n] = ids
        wgt[i, :n] = ws / s
    return idx, wgt


def _label_joint_indices(mesh: Mesh, vertices: np.ndarray, rig: CoreRig) -> np.ndarray:
    """Best-effort core-joint index per vertex from part labels; -1 if unmapped.
    Side is resolved per *part* using the mean x of that part's vertices so an
    unsided label (e.g. ``arm``) still lands on the correct arm."""
    out = np.full(len(vertices), -1, dtype=np.int64)
    # group vertices by label to compute a stable per-part x sign
    by_label: dict = {}
    for i, lbl in enumerate(mesh.vertex_labels):
        by_label.setdefault(lbl, []).append(i)
    for lbl, ids in by_label.items():
        if not lbl:
            continue
        x_sign = float(np.mean(vertices[ids, 0]))
        jname = map_part_to_joint(lbl, x_sign=x_sign)
        if jname is None:
            continue
        ji = rig.index(jname)
        for i in ids:
            out[i] = ji
    return out


def skin_by_parts(mesh: Mesh, vertices: np.ndarray, rig: CoreRig, smooth: bool = False):
    """Rigid (or optionally smooth) skinning driven by part labels.

    Unlabeled or unmapped vertices fall back to nearest-bone geometry so a
    partially-labeled mesh still rigs completely.
    """
    label_ji = _label_joint_indices(mesh, vertices, rig)
    geo_idx, geo_w = skin_by_geometry(mesh, vertices, rig, smooth=smooth, _raw=True)

    idx_list, w_list = [], []
    for i in range(len(vertices)):
        if label_ji[i] >= 0 and not smooth:
            idx_list.append([label_ji[i]])
            w_list.append([1.0])
        elif label_ji[i] >= 0 and smooth:
            # anchor to the labeled joint but let neighbours blend for seams
            ids = list(geo_idx[i])
            ws = list(geo_w[i])
            if label_ji[i] in ids:
                ws[ids.index(label_ji[i])] += 1.0
            else:
                ids.append(label_ji[i]); ws.append(1.0)
            idx_list.append(ids); w_list.append(ws)
        else:
            idx_list.append(list(geo_idx[i]))
            w_list.append(list(geo_w[i]))
    return _pack_influences(idx_list, w_list)


def _bone_segments(rig: CoreRig):
    """Return (child_idx, parent_pos, child_pos) for each of the 26 bones."""
    jp = rig.joint_positions
    segs = []
    for a, b in rig.connections:
        segs.append((int(b), jp[int(a)], jp[int(b)]))
    return segs


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def skin_by_geometry(mesh: Mesh, vertices: np.ndarray, rig: CoreRig,
                     smooth: bool = False, k: int = 4, _raw: bool = False):
    """Pure-geometry skinning: bind each vertex to the nearest bone segment.

    ``smooth`` blends the ``k`` nearest bones by inverse distance (soft seams);
    otherwise every vertex gets a single hard bind (weight 1). Returns packed
    (int32, float32) arrays, or, with ``_raw=True``, python lists (used as the
    fallback inside :func:`skin_by_parts`).
    """
    segs = _bone_segments(rig)
    idx_list, w_list = [], []
    for p in vertices:
        dists = np.array([_point_segment_dist(p, a, b) for (_c, a, b) in segs])
        child_ids = np.array([c for (c, _a, _b) in segs])
        if smooth:
            order = np.argsort(dists)[:k]
            near_d = dists[order]
            w = 1.0 / (near_d + 1e-4)
            idx_list.append([int(child_ids[o]) for o in order])
            w_list.append([float(x) for x in w])
        else:
            j = int(np.argmin(dists))
            idx_list.append([int(child_ids[j])])
            w_list.append([1.0])
    if _raw:
        return idx_list, w_list
    return _pack_influences(idx_list, w_list)


# Default limb colors for the optional per-part flat coloring.
_DEFAULT_PART_COLOR = {
    "Hips": (120, 120, 140), "Spine": (120, 120, 140), "Spine1": (120, 120, 140),
    "Spine2": (110, 110, 130), "Spine3": (110, 110, 130),
    "Neck": (210, 180, 160), "Head": (230, 200, 175),
    "RightArm": (200, 90, 90), "RightForeArm": (200, 90, 90), "RightHand": (180, 70, 70),
    "LeftArm": (90, 120, 200), "LeftForeArm": (90, 120, 200), "LeftHand": (70, 100, 180),
    "RightUpLeg": (90, 170, 110), "RightLeg": (90, 170, 110), "RightFoot": (70, 150, 90),
    "LeftUpLeg": (170, 150, 90), "LeftLeg": (170, 150, 90), "LeftFoot": (150, 130, 70),
}


def per_vertex_colors(lbs_indices: np.ndarray, rig: CoreRig) -> np.ndarray:
    """Flat per-vertex RGB (uint8) from each vertex's dominant joint."""
    cols = np.full((len(lbs_indices), 3), 180, dtype=np.uint8)
    for i, row in enumerate(lbs_indices):
        jname = rig.joint_names[int(row[0])]
        cols[i] = _DEFAULT_PART_COLOR.get(jname, (180, 180, 180))
    return cols


# ---------------------------------------------------------------------------
# Writer + top-level pipeline.
# ---------------------------------------------------------------------------

def write_skin(path: str, vertices: np.ndarray, faces: np.ndarray,
               lbs_indices: np.ndarray, lbs_weights: np.ndarray, rig: CoreRig,
               colors: np.ndarray | None = None) -> None:
    """Write an ARDY ``skin_standard.npz``. ``rig_joint_names`` /
    ``bind_rig_transform`` / ``rig_joint_connections`` are taken verbatim from
    the canonical core rig so the loader's name check can never fail."""
    arrays = dict(
        bind_vertices=vertices.astype(np.float64),
        faces=faces.astype(np.int64),
        bind_rig_transform=rig.bind_rig_transform.astype(np.float32),
        rig_joint_names=np.array(rig.joint_names, dtype="<U15"),
        lbs_indices=lbs_indices.astype(np.int32),
        lbs_weights=lbs_weights.astype(np.float32),
        rig_joint_connections=rig.connections.astype(np.int64),
    )
    if colors is not None:
        arrays["vertex_colors"] = colors.astype(np.uint8)
    np.savez(path, **arrays)


def rig_avatar(obj_path: str, out_path: str, *, up_axis: str = "auto",
               smooth: bool = False, color: bool = False,
               rig_path: str = CORE_RIG_ASSET) -> dict:
    """End-to-end: OBJ humanoid -> ARDY ``skin_standard.npz``.

    Returns a small summary dict (joint coverage, influence stats) for the CLI.
    """
    rig = load_core_rig(rig_path)
    mesh = read_obj(obj_path)
    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"{obj_path}: no vertices parsed")

    aligned = align_to_core(mesh.vertices, rig, up_axis=up_axis)

    if mesh.has_labels:
        lbs_idx, lbs_w = skin_by_parts(mesh, aligned, rig, smooth=smooth)
        mode = "parts+smooth" if smooth else "parts(rigid)"
    else:
        lbs_idx, lbs_w = skin_by_geometry(mesh, aligned, rig, smooth=smooth)
        mode = "geometry+smooth" if smooth else "geometry(rigid)"

    cols = per_vertex_colors(lbs_idx, rig) if color else None
    write_skin(out_path, aligned, mesh.faces, lbs_idx, lbs_w, rig, colors=cols)

    # count only joints that actually carry weight (ignore the padded zero slots)
    used = {int(j) for j, w in zip(lbs_idx.ravel(), lbs_w.ravel()) if w > 0}
    return {
        "vertices": int(len(aligned)),
        "faces": int(len(mesh.faces)),
        "mode": mode,
        "joints_used": len(used),
        "labeled": bool(mesh.has_labels),
        "colors": bool(color),
        "out": out_path,
    }
