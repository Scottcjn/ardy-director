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
                     seed: int | None = None, cfg_weight: float = 4.0) -> str:
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

    Returns: JSON with the stitched .npz path and the sequence of beats.
    """
    return json.dumps(_post("/choreograph", {
        "steps": steps, "model": model, "seed": seed, "cfg_weight": cfg_weight,
    }), indent=2)


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
