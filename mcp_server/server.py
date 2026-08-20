#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""ARDY Director MCP server.

A stdio MCP server (spawned by Claude Code / any MCP client) that lets an agent
direct NVIDIA ARDY: generate humanoid motion from text, and choreograph a
sequence of prompts into one clip. It forwards to the ARDY Director control
service (default http://192.168.0.136:9600), which owns the GPU and the model.

Config via env:
  DIRECTOR_URL   base URL of the control service (default http://192.168.0.136:9600)
  DIRECTOR_TIMEOUT  per-request timeout seconds (default 180)
"""
import json
import os

import requests
try:  # mcp >= 2.0.0 removed mcp.server.fastmcp; FastMCP was renamed
    # MCPServer and moved to mcp.server.mcpserver. Same .tool()/.run() API.
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

DIRECTOR_URL = os.environ.get("DIRECTOR_URL", "http://192.168.0.136:9600").rstrip("/")
TIMEOUT = float(os.environ.get("DIRECTOR_TIMEOUT", "180"))

mcp = FastMCP("ardy-director")


def _post(path, payload):
    r = requests.post(f"{DIRECTOR_URL}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        return {"ok": False, "error": r.text, "status": r.status_code}
    return r.json()


@mcp.tool()
def ardy_generate(prompt: str, model: str = "core", duration: float = 4.0,
                  seed: int | None = None, cfg_weight: float = 4.0,
                  heading_deg: float = 0.0) -> str:
    """Generate one humanoid motion clip from a text prompt using NVIDIA ARDY.

    ARDY only knows motion within its training distribution (human/humanoid
    ground motion: walk, run, turn, sit, gesture, jump). Out-of-distribution
    prompts (e.g. "fly") collapse to the nearest learned motion.

    Args:
        prompt: What the character should do, e.g. "walk forward then wave".
        model: "core" (27-joint avatar, default) or "g1" (Unitree G1 robot, MuJoCo qpos out).
        duration: Seconds of motion (0.1-30).
        seed: Optional int for reproducibility.
        cfg_weight: Text guidance strength (higher = follows prompt harder).
        heading_deg: Initial facing in degrees about vertical (0 = +Z).

    Returns: JSON with the saved .npz path (loadable in ARDY's viewer) and metadata.
    """
    return json.dumps(_post("/generate", {
        "prompt": prompt, "model": model, "duration": duration,
        "seed": seed, "cfg_weight": cfg_weight, "heading_deg": heading_deg,
    }), indent=2)


@mcp.tool()
def ardy_choreograph(steps: list[dict], model: str = "core",
                     seed: int | None = None, cfg_weight: float = 4.0,
                     camera_mode: str | None = None) -> str:
    """Choreograph a sequence of prompts into one continuous motion clip.

    Each step runs ARDY and the segments are chained so the character continues
    from where the previous step ended. Use this to direct multi-beat action.

    Args:
        steps: list of {"prompt": str, "duration": float} beats, in order,
               e.g. [{"prompt":"walk to the desk","duration":3},
                     {"prompt":"sit down","duration":2},
                     {"prompt":"wave","duration":2}].
        model: "core" or "g1".
        seed: Optional base seed (each step uses seed+i).
        cfg_weight: Text guidance strength.
        camera_mode: Optional "follow" | "orbit" | "fixed" | "over_the_shoulder";
                     the camera is solved across the whole stitched clip.

    Returns: JSON with the stitched .npz path and the sequence of beats.
    """
    payload = {"steps": steps, "model": model, "seed": seed, "cfg_weight": cfg_weight}
    if camera_mode is not None:
        payload["camera"] = {"mode": camera_mode}
    return json.dumps(_post("/choreograph", payload), indent=2)


def _stage_payload(waypoints, start_x, start_z, dense_path, face_path=False):
    return {"waypoints": waypoints, "start": {"x": start_x, "z": start_z},
            "dense_path": dense_path, "face_path": face_path}


def _camera_payload(mode, distance, height, orbit_deg_per_s, position, look_at):
    cam = {"mode": mode}
    for key, value in (("distance", distance), ("height", height),
                       ("orbit_deg_per_s", orbit_deg_per_s),
                       ("position", position), ("look_at", look_at)):
        if value is not None:
            cam[key] = value
    return cam


@mcp.tool()
def ardy_stage(prompt: str, waypoints: list[dict], camera_mode: str = "follow",
               duration: float = 6.0, model: str = "core", seed: int | None = None,
               cfg_weight: float = 4.0, start_x: float = 0.0, start_z: float = 0.0,
               dense_path: bool = True, face_path: bool = False,
               camera_distance: float | None = None,
               camera_height: float | None = None, orbit_deg_per_s: float | None = None,
               camera_position: list[float] | None = None,
               camera_look_at: list[float] | None = None) -> str:
    """Direct a staged shot: walk the character a path and frame it with a camera.

    Unlike `ardy_generate` (which lets ARDY wander wherever the prompt takes
    it), the waypoints become ARDY root-path constraints, so the character
    actually arrives where and when you said. The camera track is baked into
    the output .npz per frame.

    Args:
        prompt: What the character is doing, e.g. "walk briskly, then stop".
        waypoints: ordered marks [{"x": float, "z": float, "at": seconds}, ...],
                   e.g. [{"x":0,"z":3,"at":2},{"x":2,"z":5,"at":4}].
                   Times strictly increase and must fit inside `duration`.
        camera_mode: "follow" (behind the character), "orbit" (circles it),
                     "fixed" (locked-off tripod) or "over_the_shoulder".
        duration: Seconds of motion (0.1-30).
        model: "core" or "g1".
        seed: Optional int for reproducibility.
        cfg_weight: Text guidance strength.
        start_x, start_z: The start mark (root XZ at frame 0).
        dense_path: True walks the straight line between marks; False constrains
                    only the marks and lets ARDY pick the route.
        face_path: Pin the facing along the path too. Off by default, which
                   leaves ARDY free to turn on the spot at a mark.
        camera_distance, camera_height: Framing overrides in metres.
        orbit_deg_per_s: Orbit speed (orbit mode).
        camera_position, camera_look_at: [x, y, z] for fixed mode; omit look_at
                    to keep the character in frame as it moves.

    Returns: JSON with the .npz path, the stage digest (path length, speed) and
    the resolved camera.
    """
    return json.dumps(_post("/generate", {
        "prompt": prompt, "model": model, "duration": duration, "seed": seed,
        "cfg_weight": cfg_weight,
        "stage": _stage_payload(waypoints, start_x, start_z, dense_path, face_path),
        "camera": _camera_payload(camera_mode, camera_distance, camera_height,
                                  orbit_deg_per_s, camera_position, camera_look_at),
    }), indent=2)


@mcp.tool()
def ardy_preview_stage(waypoints: list[dict], duration: float = 6.0,
                       camera_mode: str | None = None, start_x: float = 0.0,
                       start_z: float = 0.0, dense_path: bool = True,
                       fps: int = 30) -> str:
    """Check a stage before spending a generation: no model, no GPU, instant.

    Returns the root path ARDY would be constrained to (and, with a camera mode,
    where the camera would sit), or an error naming what is wrong with the
    blocking — a waypoint past the end of the clip, times out of order, two
    marks on one frame. Use it to iterate on staging, then call `ardy_stage`.

    Args:
        waypoints: [{"x": float, "z": float, "at": seconds}, ...].
        duration: Seconds the real clip will be.
        camera_mode: Optional "follow" | "orbit" | "fixed" | "over_the_shoulder".
        start_x, start_z: The start mark.
        dense_path: Constrain every frame vs. only the marks.
        fps: Frame rate to plan against (the core model runs at 30).

    Returns: JSON with the planned path, its length and mean walking speed.
    """
    payload = {"stage": _stage_payload(waypoints, start_x, start_z, dense_path),
               "duration": duration, "fps": fps}
    if camera_mode is not None:
        payload["camera"] = {"mode": camera_mode}
    return json.dumps(_post("/stage/preview", payload), indent=2)


@mcp.tool()
def ardy_list_cameras() -> str:
    """List camera modes for `ardy_stage` and their tunable parameters."""
    try:
        r = requests.get(f"{DIRECTOR_URL}/cameras", timeout=15)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


@mcp.tool()
def ardy_list_models() -> str:
    """List the ARDY motion models available (core avatar vs G1 robot)."""
    try:
        r = requests.get(f"{DIRECTOR_URL}/models", timeout=15)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


@mcp.tool()
def ardy_status() -> str:
    """Health of the ARDY Director service: device, loaded models, encoder link."""
    try:
        r = requests.get(f"{DIRECTOR_URL}/health", timeout=15)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}",
                           "hint": f"Is the control service up at {DIRECTOR_URL}?"})


if __name__ == "__main__":
    mcp.run()
