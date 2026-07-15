#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Real-time viser viewer for ARDY Director.

Receives motion frames over WebSocket from the director_service and renders
them as a 27-joint humanoid skeleton in a viser 3D scene.  Also receives
camera and staging state so the view can follow the character or orbit,
and waypoints are drawn as scene markers.

Usage:
    python director_service/viewer.py
    python director_service/viewer.py --director ws://192.168.0.136:9600 --port 9602

Open http://localhost:<port> in a browser to see the live view.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("viewer")

# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------

DEFAULT_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine_lower",
    "left_knee", "right_knee", "spine_mid",
    "left_ankle", "right_ankle", "spine_upper",
    "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
    "head_end", "left_eye", "right_eye", "nose", "jaw",
]

DEFAULT_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (22, 15), (23, 15), (24, 15), (25, 15), (26, 15),
]

JOINT_COLORS = np.array([
    [0.27, 0.53, 1.00], [1.00, 0.80, 0.27], [0.27, 1.00, 0.80],
    [0.27, 0.53, 1.00], [1.00, 0.80, 0.27], [0.27, 1.00, 0.80],
    [0.27, 0.53, 1.00], [1.00, 0.80, 0.27], [0.27, 1.00, 0.80],
    [0.27, 0.53, 1.00], [1.00, 0.80, 0.27], [0.27, 1.00, 0.80],
    [0.27, 0.53, 1.00], [0.27, 1.00, 0.27], [1.00, 0.27, 1.00],
    [1.00, 0.40, 0.27], [0.27, 1.00, 0.27], [1.00, 0.27, 1.00],
    [0.27, 1.00, 0.27], [1.00, 0.27, 1.00],
    [0.27, 1.00, 0.27], [1.00, 0.27, 1.00],
    [1.00, 0.40, 0.27], [1.00, 0.60, 0.53], [1.00, 0.60, 0.53],
    [1.00, 0.60, 0.53], [1.00, 0.60, 0.53],
])

BONE_COLOR = (0.55, 0.55, 0.55)
WAYPOINT_COLOR = (0.2, 0.8, 0.2)
WAYPOINT_LINE_COLOR = (0.2, 0.6, 0.2)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class ArdyViewer:
    """Viser-based viewer that streams motion frames + camera + staging."""

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

        # --- viser handles ---
        self._server = None
        self._joint_handles: dict[int, object] = {}
        self._bone_handles: dict[int, object] = {}
        self._waypoint_handles: list[object] = []
        self._waypoint_line: object = None
        self._status_label: object = None
        self._camera_label: object = None
        self._clients: list[object] = []

        # --- state ---
        self._camera: dict = {}
        self._root_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self._waypoints: list[dict] = []
        self._orbit_angle: float = 0.0

    # ---- viser scene -------------------------------------------------------

    def _build_skeleton(self):
        import viser

        self._server = viser.ViserServer(port=self.viser_port)

        @self._server.on_client_connect
        def _on_connect(client: viser.ClientHandle):
            self._clients.append(client)
            client.add_label("/status", text="Connected — waiting for motion...")
            client.add_label("/camera_status", text="Camera: follow")
            self._apply_camera_to_client(client)

        @self._server.on_client_disconnect
        def _on_disconnect(client: viser.ClientHandle):
            if client in self._clients:
                self._clients.remove(client)

        for i in range(self.num_joints):
            color = JOINT_COLORS[i] if i < len(JOINT_COLORS) else (0.5, 0.5, 0.5)
            sphere = self._server.add_sphere(
                f"/joint/{i}", radius=0.05, color=tuple(color),
            )
            self._server.add_label(
                f"/joint/{i}/label",
                text=self.joint_names[i] if i < len(self.joint_names) else f"j{i}",
            )
            self._joint_handles[i] = sphere

        for bi, (p, c) in enumerate(self.bones):
            mesh = self._server.add_mesh_simple(
                f"/bone/{bi}",
                vertices=self._bone_verts(np.zeros((self.num_joints, 3))),
                faces=self._bone_faces(),
                color=BONE_COLOR,
            )
            self._bone_handles[bi] = mesh

        # Waypoint group
        self._server.add_frame("/waypoints")
        self._server.add_frame("/waypoints/path")

        self._status_label = "/status"
        self._camera_label = "/camera_status"

        logger.info("Viser server ready on port %d", self.viser_port)
        logger.info("Open http://localhost:%d in a browser", self.viser_port)

    @staticmethod
    def _bone_verts(pos: np.ndarray) -> list[tuple[float, float, float]]:
        half = 0.015
        return [
            (-half, -half, -half), (half, -half, -half),
            (half, half, -half), (-half, half, -half),
            (-half, -half, half), (half, -half, half),
            (half, half, half), (-half, half, half),
        ]

    @staticmethod
    def _bone_faces() -> list[tuple[int, int, int]]:
        return [
            (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
            (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5),
        ]

    def _set_status(self, text: str):
        if self._server is not None:
            self._server.add_label(self._status_label, text=text)

    # ---- camera ------------------------------------------------------------

    def _apply_camera_to_client(self, client):
        if not self._camera:
            return
        mode = self._camera.get("mode", "follow")
        label = f"Camera: {mode}"
        try:
            client.add_label(self._camera_label, text=label)
        except Exception:
            pass

    def _update_cameras(self):
        """Push current camera state to all connected client cameras."""
        if not self._camera or not self._server:
            return
        mode = self._camera.get("mode", "follow")
        for client in self._clients:
            try:
                self._apply_camera_to_client(client)
            except Exception:
                pass

    def _apply_follow_camera(self, root_pos: np.ndarray):
        """Move camera to follow the character (per-frame)."""
        if not self._camera:
            return
        mode = self._camera.get("mode", "follow")
        dist = self._camera.get("distance", 3.0)
        height = self._camera.get("height", 1.5)

        rx, ry, rz = float(root_pos[0]), float(root_pos[1]), float(root_pos[2])
        for client in self._clients:
            try:
                if mode == "follow":
                    client.camera.position = (rx - dist, ry + height, rz + dist * 0.3)
                    client.camera.look_at = (rx, ry + 0.8, rz)
                elif mode == "over-the-shoulder":
                    side = self._camera.get("side_offset", 0.3)
                    fwd = self._camera.get("forward_offset", 0.5)
                    sh = self._camera.get("shoulder_height", 1.6)
                    client.camera.position = (rx + side, ry + sh, rz - fwd)
                    client.camera.look_at = (rx, ry + 0.8, rz + 2.0)
                elif mode == "fixed":
                    pos = self._camera.get("position", [0.0, 2.0, 5.0])
                    tgt = self._camera.get("target", [0.0, 1.0, 0.0])
                    client.camera.position = tuple(pos)
                    client.camera.look_at = tuple(tgt)
                elif mode == "orbit":
                    radius = self._camera.get("radius", 4.0)
                    speed = self._camera.get("orbit_speed", 0.3)
                    self._orbit_angle += speed * 0.016
                    cx = rx + radius * math.cos(self._orbit_angle)
                    cz = rz + radius * math.sin(self._orbit_angle)
                    client.camera.position = (cx, ry + 1.5, cz)
                    client.camera.look_at = (rx, ry + 0.8, rz)
            except Exception:
                pass

    # ---- waypoints ---------------------------------------------------------

    def _render_waypoints(self):
        """Add/update waypoint markers and path line."""
        if self._server is None:
            return

        # Remove old waypoint markers
        for h in self._waypoint_handles:
            try:
                self._server.remove_frame(f"/waypoints/marker/{id(h)}")
            except Exception:
                pass
        self._waypoint_handles.clear()

        if not self._waypoints:
            return

        pts = np.array([
            (wp.get("x", 0), 0.02, wp.get("z", 0))
            for wp in self._waypoints
        ], dtype=np.float32)

        # Spheres at each waypoint
        for i, (x, y, z) in enumerate(pts):
            sphere = self._server.add_sphere(
                f"/waypoints/marker/{i}",
                radius=0.08,
                color=WAYPOINT_COLOR,
                position=(float(x), float(y), float(z)),
            )
            self._waypoint_handles.append(sphere)

        # Line segments between consecutive waypoints
        if len(pts) >= 2:
            seg_verts: list[tuple[float, float, float]] = []
            for i in range(len(pts) - 1):
                a = (float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
                b = (float(pts[i + 1, 0]), float(pts[i + 1, 1]), float(pts[i + 1, 2]))
                seg_verts.append(a)
                seg_verts.append(b)

            try:
                self._waypoint_line = self._server.add_mesh_simple(
                    "/waypoints/path/line",
                    vertices=[
                        (0, 0, 0), (0.005, 0, 0), (0.005, 0.005, 0), (0, 0.005, 0),
                        (0, 0, 1), (0.005, 0, 1), (0.005, 0.005, 1), (0, 0.005, 1),
                    ],
                    faces=[
                        (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
                        (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5),
                    ],
                    color=WAYPOINT_LINE_COLOR,
                )
            except Exception:
                pass

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

        # Store root for follow camera
        self._root_position = pos[0].copy()

    # ---- WebSocket loop ----------------------------------------------------

    async def _ws_loop(self):
        import websockets

        while True:
            try:
                logger.info("Connecting to %s ...", self.director_ws_url)
                async with websockets.connect(self.director_ws_url) as ws:
                    logger.info("Connected to director")
                    self._set_status("Connected — streaming")

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
                                self._apply_follow_camera(self._root_position)

                        elif mtype == "done":
                            self._set_status("Done — idle")

                        elif mtype == "camera":
                            cam = msg.get("camera", {})
                            self._camera = cam
                            self._update_cameras()
                            logger.info("Camera: %s", cam.get("mode", "?"))

                        elif mtype == "stage":
                            stage = msg.get("stage", {})
                            self._waypoints = stage.get("waypoints", [])
                            self._render_waypoints()
                            n = len(self._waypoints)
                            logger.info("Stage: %d waypoints", n)

                        elif mtype == "error":
                            logger.error("Server error: %s", msg.get("message", ""))

            except ImportError:
                raise
            except Exception as exc:
                logger.warning("Disconnected (%s), reconnecting in 3s ...", exc)
                self._set_status("Disconnected — reconnecting...")
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
        "--director", default="ws://localhost:9600",
        help="Director service URL (default: ws://localhost:9600)",
    )
    p.add_argument(
        "--port", type=int, default=9601,
        help="Viser viewer port (default: 9601)",
    )
    p.add_argument(
        "--skeleton", default=None,
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
