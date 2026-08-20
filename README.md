# ARDY Director

An agent-drivable bridge for [NVIDIA ARDY](https://github.com/nv-tlabs/ardy) — the
real-time, text-to-motion humanoid model (SIGGRAPH 2026). ARDY Director lets an
LLM act as a **motion director**: it turns text ("walk to the desk, sit, then
wave") into humanoid motion, either as single clips or stitched choreography.

It ships as two pieces:

| Piece | Runs where | What it does |
|-------|-----------|--------------|
| **`director_service`** | On the ARDY host, inside the `ardy` venv (GPU) | FastAPI wrapper around ARDY's generation API. Owns the model. |
| **`mcp_server`** | Anywhere an MCP client lives (e.g. your laptop) | stdio MCP server exposing `ardy_generate`, `ardy_choreograph`, etc. Forwards to the service. |

The split matters: the control service holds the ~156M-param motion denoiser on
the GPU and **reuses the already-running LLM2Vec text-encoder** (so it does not
load a second copy of Llama-3). The MCP server is a thin, dependency-light
forwarder your agent talks to.

## What ARDY can and can't do

ARDY is a *motion-completion* model trained on the Bones Rigplay mocap dataset —
human/humanoid **ground** motion: walking, running, turning, sitting, jumping,
gestures. Text steers **within** that manifold. Out-of-distribution prompts
("fly like superman") collapse to the nearest learned motion (a skip/leap). To
add motions it never saw, you fine-tune ARDY — the Director does not invent
motion, it directs what ARDY knows.

## Quick start

### 1. Start the control service (on the ARDY host)

The LLM2Vec text encoder must already be running (ARDY's
`scripts/run_text_encoder_server.py`, default port 9550).

```bash
cd ~/ardy && source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python /path/to/ardy-director/director_service/service.py     # serves :9600
```

Or use `deploy_136.sh` to copy + launch it on the default host.

### 2. Register the MCP server (on the client)

Add to your MCP config (`~/.mcp.json`):

```json
"ardy-director": {
  "type": "stdio",
  "command": "python3",
  "args": ["/home/scott/ardy-director/mcp_server/server.py"],
  "env": { "DIRECTOR_URL": "http://192.168.0.136:9600" }
}
```

Restart the MCP client to pick it up.

### 3. Direct some motion

```
ardy_generate(prompt="walk forward and wave", duration=5)
ardy_choreograph(steps=[
  {"prompt": "walk to the desk", "duration": 3},
  {"prompt": "sit down",         "duration": 2},
  {"prompt": "wave hello",       "duration": 2},
])
```

Each call returns the path to an ARDY-native `.npz`, loadable in ARDY's own
viewer (`scripts/visualize.py`) or renderable with your own skin (e.g. the
N64-Sophia flat-color rig).

## Stage a shot

A prompt says *what* the character does; it does not say **where** it happens or
**where the shot is looking from**. Staging adds both:

- **Waypoints** — marks on the floor with times (`{"x": 0, "z": 4, "at": 3}`).
  They become ARDY root-path constraints (`Root2DConstraintSet`), so the
  character actually walks the path instead of wandering wherever the prompt
  takes it. `start` places it at frame 0, and it starts out facing the way the
  path leaves that mark. The post-processor is told about the constraint too, so
  the foot fix-up works with the path rather than against it.
- **A camera** — `follow`, `orbit`, `fixed` or `over_the_shoulder`, baked per
  frame into the `.npz` (`camera_positions`, `camera_targets`, `camera_mode`),
  solved against the motion ARDY actually produced.

### Worked example: walk to the desk, then to the window, on a follow camera

First **preview** the blocking. This costs nothing — no model, no GPU — so you
can fix the staging before you pay for a generation:

```bash
curl -s localhost:9600/stage/preview -H 'content-type: application/json' -d '{
  "duration": 6.0,
  "stage": {"start": {"x": 0, "z": 0},
            "waypoints": [{"x": 0, "z": 4, "at": 3}, {"x": 3, "z": 6, "at": 5}]},
  "camera": {"mode": "follow"}
}'
```

```json
{"ok": true, "fps": 30, "frames": 180,
 "stage": {"constrained_frames": 151, "last_frame": 150, "last_time_s": 5.0,
           "path_length_m": 7.606, "start_heading_deg": 0.0, "mean_speed_mps": 1.521},
 "camera": {"mode": "follow", "first_position": [0.0, 2.5, -5.0], ...}}
```

`mean_speed_mps: 1.521` is a brisk walk — the marks are reachable. Ask for the
same path in half the time and ARDY would be told to sprint; the preview is
where you notice. A stage that cannot work (a waypoint past the end of the clip,
times out of order) comes back as a `400` naming the problem, not a shrug.

Happy with it? Send the same `stage` to `/generate` (see
[`examples/staged_shot.json`](examples/staged_shot.json)):

```bash
curl -s localhost:9600/generate -H 'content-type: application/json' \
     -d @examples/staged_shot.json
```

From an agent, that whole loop is two tools:

```
ardy_preview_stage(waypoints=[{"x":0,"z":4,"at":3},{"x":3,"z":6,"at":5}], duration=6, camera_mode="follow")
ardy_stage(prompt="walk to the desk, pause, then walk to the window",
           waypoints=[{"x":0,"z":4,"at":3},{"x":3,"z":6,"at":5}],
           camera_mode="follow", duration=6, seed=42)
```

Two knobs worth knowing:

- `dense_path: false` constrains only the marks and lets ARDY choose the route
  between them; the default (`true`) walks the straight line.
- `face_path: true` also pins the facing along the path. It is **off** by
  default, matching ARDY's own interactive demo, which constrains position only
  — pinning the facing every frame forbids the character from turning on the
  spot at a mark ("walk to the desk, then turn and wave"). Turn it on when you
  want the facing nailed down and the prompt is a pure travel beat.

## MCP tools

| Tool | Purpose |
|------|---------|
| `ardy_generate` | One clip from one prompt (`core` avatar or `g1` robot). |
| `ardy_choreograph` | A prompt sequence stitched into one continuous clip (optional `camera_mode`). |
| `ardy_stage` | A staged shot: walk a waypoint path, framed by a camera. |
| `ardy_preview_stage` | Dry-run a stage (path, speed, camera) with no model and no GPU. |
| `ardy_list_cameras` | Camera modes and their tunable parameters. |
| `ardy_list_models` | List motion models (core vs G1). |
| `ardy_status` | Service health: device, loaded models, encoder link. |

## Tests

The staging geometry and camera solving are plain numpy — no ARDY, no GPU — so
they run anywhere:

```bash
pip install pytest httpx      # plus the service requirements
python -m pytest tests/
```

ARDY itself is stubbed at import time (`tests/conftest.py`); nothing here fakes
ARDY's behaviour, so anything the tests assert about generation is a contract
with its API (constraint shapes and dtypes, what gets passed to the model), not
a claim about the motion it returns.

## Roadmap

- **v0.1 (now):** generate + position-chained choreography, model caching, encoder reuse.
- **Staging and camera (now):** waypoint root-path constraints, four camera modes
  baked per frame, GPU-free `/stage/preview`.
- **Streaming choreography:** velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **Live viewer injection:** push directed motion straight into the running viser demo.
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
