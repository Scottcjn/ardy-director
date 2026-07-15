#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Real-time viser viewer for ARDY Director.

Receives motion frames over WebSocket from the director_service and renders
them as a 27-joint humanoid skeleton in a viser 3D scene.  Updates in
real-time at the motion's native frame rate, so the character appears to
move as the model generates.

Usage:
    # On the ARDY host (or any machine that can reach the director):
    python director_service/viewer.py

    # With custom addresses:
    python director_service/viewer.py --director ws://192.168.0.136:9600 --port 9602

Open http://localhost:<port> in a browser to see the live view.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("viewer")

# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------
# ARDY's core model uses a 27-joint skeleton (approximate topology below).
# If your ARDY build uses a different joint order, load a custom skeleton
# via --skeleton <file.json> containing {"joint_names":[...], "bones":[[i,j],...]}.

DEFAULT_JOINT_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine_lower",   # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine_mid",     # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine_upper",   # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_elbow",    # 16
    "right_elbow",   # 17
    "left_wrist",    # 18
    "right_wrist",   # 19
    "left_hand",     # 20
    "right_hand",    # 21
    "head_end",      # 22
    "left_eye",      # 23
    "right_eye",     # 24
    "nose",          # 25
    "jaw",           # 26
]

DEFAULT_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5),
    (3, 6),
    (4, 7), (5, 8),
    (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (22, 15), (23, 15), (24, 15), (25, 15), (26, 15),
]

# Color per joint (RGB normalized 0-1)
JOINT_COLORS = np.array([
    [0.27, 0.53, 1.00],  # pelvis
    [1.00, 0.80, 0.27],  # left_hip
    [0.27, 1.00, 0.80],  # right_hip
    [0.27, 0.53, 1.00],  # spine_lower
    [1.00, 0.80, 0.27],  # left_knee
    [0.27, 1.00, 0.80],  # right_knee
    [0.27, 0.53, 1.00],  # spine_mid
    [1.00, 0.80, 0.27],  # left_ankle
    [0.27, 1.00, 0.80],  # right_ankle
    [0.27, 0.53, 1.00],  # spine_upper
    [1.00, 0.80, 0.27],  # left_foot
    [0.27, 1.00, 0.80],  # right_foot
    [0.27, 0.53, 1.00],  # neck
    [0.27, 1.00, 0.27],  # left_collar
    [1.00, 0.27, 1.00],  # right_collar
    [1.00, 0.40, 0.27],  # head
    [0.27, 1.00, 0.27],  # left_elbow
    [1.00, 0.27, 1.00],  # right_elbow
    [0.27, 1.00, 0.27],  # left_wrist
    [1.00, 0.27, 1.00],  # right_wrist
    [0.27, 1.00, 0.27],  # left_hand
    [1.00, 0.27, 1.00],  # right_hand
    [1.00, 0.40, 0.27],  # head_end
    [1.00, 0.60, 0.53],  # left_eye
    [1.00, 0.60, 0.53],  # right_eye
    [1.00, 0.60, 0.53],  # nose
    [1.00, 0.60, 0.53],  # jaw
])

BONE_COLOR = (0.55, 0.55, 0.55)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class ArdyViewer:
    """Viser-based viewer that streams motion frames from the director."""

    def __init__(
        self,
        director_url: str = "ws://localhost:9600",
        viser_port: int = 9601,
        skeleton_path: str | None = None,
    ):
        self.director_ws_url = director_url.rstrip("/") + "/ws"
        self.viser_port = viser_port

        if skeleton_path:
            with open(skeleton_path) as f:
                skel = json.load(f)
            self.joint_names: list[str] = skel["joint_names"]
            self.bones: list[tuple[int, int]] = [tuple(b) for b in skel["bones"]]
        else:
            self.joint_names = list(DEFAULT_JOINT_NAMES)
            self.bones = list(DEFAULT_BONES)

        self.num_joints = len(self.joint_names)
        self._server = None
        self._joint_handles: dict[int, object] = {}
        self._bone_handles: dict[int, object] = {}
        self._status_handle: object = None
        self._ready = asyncio.Event()

    # ---- viser scene -------------------------------------------------------

    def _build_skeleton(self):
        import viser

        self._server = viser.ViserServer(port=self.viser_port)

        for i in range(self.num_joints):
            color = JOINT_COLORS[i] if i < len(JOINT_COLORS) else (0.5, 0.5, 0.5)
            sphere = self._server.add_sphere(
                f"/joint/{i}",
                radius=0.05,
                color=tuple(color),
            )
            self._server.add_label(
                f"/joint/{i}/label",
                text=self.joint_names[i] if i < len(self.joint_names) else f"j{i}",
            )
            self._joint_handles[i] = sphere

        for bi, (p, c) in enumerate(self.bones):
            mesh = self._server.add_mesh_simple(
                f"/bone/{bi}",
                vertices=self._bone_verts(p, c, np.zeros((self.num_joints, 3))),
                faces=self._bone_faces(),
                color=BONE_COLOR,
            )
            self._bone_handles[bi] = mesh

        self._set_status("Connected — waiting for motion...")
        logger.info("Viser server ready on port %d", self.viser_port)
        logger.info("Open http://localhost:%d in a browser", self.viser_port)

    @staticmethod
    def _bone_verts(pi: int, ci: int, pos: np.ndarray) -> list[tuple[float, float, float]]:
        half = 0.015
        return [
            (-half, -half, -half),
            (half, -half, -half),
            (half, half, -half),
            (-half, half, -half),
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        ]

    @staticmethod
    def _bone_faces() -> list[tuple[int, int, int]]:
        return [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (2, 3, 7), (2, 7, 6),
            (0, 3, 7), (0, 7, 4),
            (1, 2, 6), (1, 6, 5),
        ]

    def _set_status(self, text: str):
        if self._server is not None:
            self._server.add_label("/status", text=text)

    # ---- per-frame update --------------------------------------------------

    def _update_pose(self, posed_joints: list) -> None:
        pos = np.asarray(posed_joints, dtype=np.float32)
        if pos.shape[0] < self.num_joints:
            return

        for i, handle in self._joint_handles.items():
            if i < pos.shape[0]:
                handle.position = (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))

        for bi, (p, c) in enumerate(self.bones):
            if p >= pos.shape[0] or c >= pos.shape[0]:
                continue
            handle = self._bone_handles.get(bi)
            if handle is None:
                continue
            p_pos = pos[p]
            c_pos = pos[c]
            mid = (p_pos + c_pos) / 2.0
            direction = c_pos - p_pos
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            direction /= length

            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            dot = float(np.dot(up, direction))
            cross = np.cross(up, direction)
            qw = dot + 1.0
            q_norm = np.sqrt(qw * qw + cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
            if q_norm < 1e-8:
                continue
            qw /= q_norm
            qx, qy, qz = cross[0] / q_norm, cross[1] / q_norm, cross[2] / q_norm

            handle.position = (float(mid[0]), float(mid[1]), float(mid[2]))
            handle.wxyz = (qw, qx, qy, qz)
            handle.scale = (1.0, length / 0.03, 1.0)

    # ---- WebSocket loop ----------------------------------------------------

    async def _ws_loop(self):
        import websockets

        while True:
            try:
                logger.info("Connecting to %s ...", self.director_ws_url)
                async with websockets.connect(self.director_ws_url) as ws:
                    logger.info("Connected to director")
                    self._set_status("Connected — streaming")
                    self._ready.set()

                    async for raw in ws:
                        msg = json.loads(raw)
                        mtype = msg.get("type")

                        if mtype == "start":
                            total = msg.get("total_frames", 0)
                            label = msg.get("label", "") or msg.get("prompt", "")
                            self._set_status(f"Playing: {label} ({total} frames)")

                        elif mtype == "frame":
                            posed = msg.get("posed_joints")
                            if posed is not None:
                                self._update_pose(posed)

                        elif mtype == "done":
                            self._set_status("Done — idle")

                        elif mtype == "error":
                            logger.error("Server error: %s", msg.get("message", ""))

            except ImportError:
                raise
            except Exception as exc:
                logger.warning("Disconnected (%s), reconnecting in 3s ...", exc)
                self._set_status("Disconnected — reconnecting...")
                self._ready.clear()
                await asyncio.sleep(3)

    async def run(self):
        self._build_skeleton()
        await self._ws_loop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="ARDY Director — Live Viewer")
    p.add_argument(
        "--director",
        default="ws://localhost:9600",
        help="Director service URL (default: ws://localhost:9600)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=9601,
        help="Viser viewer port (default: 9601)",
    )
    p.add_argument(
        "--skeleton",
        default=None,
        help="Path to custom skeleton JSON (joint_names + bones)",
    )
    args = p.parse_args()

    viewer = ArdyViewer(
        director_url=args.director,
        viser_port=args.port,
        skeleton_path=args.skeleton,
    )
    try:
        asyncio.run(viewer.run())
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error("Install deps: pip install viser websockets numpy")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()
