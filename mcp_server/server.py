#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""ARDY Director MCP server.

A stdio MCP server (spawned by Claude Code / any MCP client) that lets an agent
direct NVIDIA ARDY: generate humanoid motion from text, choreograph sequences,
control the camera, and set staging waypoints.

Config via env:
  DIRECTOR_URL   base URL of the control service (default http://192.168.0.136:9600)
  DIRECTOR_TIMEOUT  per-request timeout seconds (default 180)
"""
from __future__ import annotations

import json
import os

import requests
from mcp.server.fastmcp import FastMCP

DIRECTOR_URL = os.environ.get("DIRECTOR_URL", "http://192.168.0.136:9600").rstrip("/")
TIMEOUT = float(os.environ.get("DIRECTOR_TIMEOUT", "180"))

mcp = FastMCP("ardy-director")


def _post(path, payload):
    r = requests.post(f"{DIRECTOR_URL}{path}", json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        return {"ok": False, "error": r.text, "status": r.status_code}
    return r.json()


def _get(path):
    r = requests.get(f"{DIRECTOR_URL}{path}", timeout=TIMEOUT)
    if r.status_code != 200:
        return {"ok": False, "error": r.text, "status": r.status_code}
    return r.json()


# ---------------------------------------------------------------------------
# Motion generation
# ---------------------------------------------------------------------------


@mcp.tool()
def ardy_generate(
    prompt: str,
    model: str = "core",
    duration: float = 4.0,
    seed: int | None = None,
    cfg_weight: float = 4.0,
    heading_deg: float = 0.0,
    stage_id: str | None = None,
    waypoints: list[dict] | None = None,
    start_x: float | None = None,
    start_z: float | None = None,
) -> str:
    """Generate one humanoid motion clip from a text prompt using NVIDIA ARDY.

    Optionally guide the root path with waypoints (staged path the character
    walks along).  Waypoints can be provided inline or referenced from a
    previously saved stage setup via stage_id.

    Args:
        prompt: What the character should do, e.g. "walk forward then wave".
        model: "core" (27-joint avatar, default) or "g1" (Unitree G1 robot).
        duration: Seconds of motion (0.1-30).
        seed: Optional int for reproducibility.
        cfg_weight: Text guidance strength (higher = follows prompt harder).
        heading_deg: Initial facing in degrees about vertical (0 = +Z).
        stage_id: Named stage setup to use for waypoints (from ardy_set_stage).
        waypoints: Inline list of {"x": float, "z": float, "heading_deg"?: float}
                   — overrides stage_id if both provided.
        start_x: Override start X position.
        start_z: Override start Z position.

    Returns: JSON with the saved .npz path, stream_id, and metadata.
    """
    return json.dumps(_post("/generate", {
        "prompt": prompt, "model": model, "duration": duration,
        "seed": seed, "cfg_weight": cfg_weight, "heading_deg": heading_deg,
        "stage_id": stage_id, "waypoints": waypoints,
        "start_x": start_x, "start_z": start_z,
    }), indent=2)


@mcp.tool()
def ardy_choreograph(
    steps: list[dict],
    model: str = "core",
    seed: int | None = None,
    cfg_weight: float = 4.0,
    stage_id: str | None = None,
    waypoints: list[dict] | None = None,
    start_x: float | None = None,
    start_z: float | None = None,
) -> str:
    """Choreograph a sequence of prompts into one continuous motion clip.

    Each step runs ARDY and the segments are chained so the character continues
    from where the previous step ended.  Optionally guide the overall path with
    waypoints.

    Args:
        steps: list of {"prompt": str, "duration": float} beats.
        model: "core" or "g1".
        seed: Optional base seed (each step uses seed+i).
        cfg_weight: Text guidance strength.
        stage_id: Named stage setup for waypoints (from ardy_set_stage).
        waypoints: Inline waypoints — overrides stage_id.
        start_x: Override start X position.
        start_z: Override start Z position.

    Returns: JSON with the stitched .npz path and the sequence of beats.
    """
    return json.dumps(_post("/choreograph", {
        "steps": steps, "model": model, "seed": seed, "cfg_weight": cfg_weight,
        "stage_id": stage_id, "waypoints": waypoints,
        "start_x": start_x, "start_z": start_z,
    }), indent=2)


# ---------------------------------------------------------------------------
# Camera control
# ---------------------------------------------------------------------------


@mcp.tool()
def ardy_set_camera(
    mode: str = "follow",
    distance: float | None = None,
    height: float | None = None,
    radius: float | None = None,
    orbit_speed: float | None = None,
    position: list[float] | None = None,
    target: list[float] | None = None,
    side_offset: float | None = None,
    forward_offset: float | None = None,
) -> str:
    """Set the camera mode and parameters for the live viewer.

    The camera state is broadcast to all connected viewers in real-time.

    Modes:
      follow (default): camera trails behind the character.
        - distance: how far back (default 3.0)
        - height: how high up (default 1.5)

      orbit: camera circles the character.
        - radius: orbit radius (default 4.0)
        - orbit_speed: radians/sec (default 0.3)

      fixed: static camera position.
        - position: [x, y, z] camera position (default [0, 2, 5])
        - target: [x, y, z] look-at point (default [0, 1, 0])

      over-the-shoulder: behind the character's shoulder.
        - side_offset: lateral offset (default 0.3)
        - forward_offset: how far behind (default 0.5)
        - height: shoulder height (default 1.6)

    Args:
        mode: Camera mode — "follow", "orbit", "fixed", or "over-the-shoulder".
        distance: Follow distance.
        height: Follow height / over-the-shoulder height.
        radius: Orbit radius.
        orbit_speed: Orbit rotation speed (rad/s).
        position: Fixed camera position [x, y, z].
        target: Fixed camera look-at [x, y, z].
        side_offset: Over-the-shoulder side offset.
        forward_offset: Over-the-shoulder forward offset.

    Returns: JSON with the applied camera config.
    """
    payload = {"mode": mode}
    if distance is not None:
        payload["distance"] = distance
    if height is not None:
        payload["height"] = height
    if radius is not None:
        payload["radius"] = radius
    if orbit_speed is not None:
        payload["orbit_speed"] = orbit_speed
    if position is not None:
        payload["position"] = position
    if target is not None:
        payload["target"] = target
    if side_offset is not None:
        payload["side_offset"] = side_offset
    if forward_offset is not None:
        payload["forward_offset"] = forward_offset
    return json.dumps(_post("/set_camera", payload), indent=2)


@mcp.tool()
def ardy_get_camera() -> str:
    """Get the current camera configuration."""
    return json.dumps(_get("/camera"), indent=2)


# ---------------------------------------------------------------------------
# Staging / waypoints
# ---------------------------------------------------------------------------


@mcp.tool()
def ardy_set_stage(
    name: str = "default",
    waypoints: list[dict] | None = None,
    start_x: float = 0.0,
    start_z: float = 0.0,
    start_heading_deg: float = 0.0,
) -> str:
    """Define a named stage setup with waypoints the character walks along.

    Save a stage once, then reference it by name in ardy_generate or
    ardy_choreograph via stage_id.  Waypoints are interpolated into a
    smooth root path that guides where the character walks.

    Args:
        name: A label for this stage setup (default "default").
        waypoints: Ordered list of {"x": float, "z": float} positions.
                   The first waypoint sets the start position; subsequent
                   waypoints define the path.
        start_x: Starting X position (overrides first waypoint X).
        start_z: Starting Z position (overrides first waypoint Z).
        start_heading_deg: Initial facing direction in degrees.

    Returns: JSON confirming the saved stage.
    """
    return json.dumps(_post("/set_stage", {
        "name": name,
        "waypoints": waypoints or [],
        "start_x": start_x,
        "start_z": start_z,
        "start_heading_deg": start_heading_deg,
    }), indent=2)


@mcp.tool()
def ardy_get_stage(name: str = "default") -> str:
    """Retrieve a saved stage setup by name."""
    return json.dumps(_get(f"/stage?name={name}"), indent=2)


@mcp.tool()
def ardy_list_stages() -> str:
    """List all saved stage setups."""
    return json.dumps(_get("/stages"), indent=2)


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------


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
