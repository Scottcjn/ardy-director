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

Each call returns the path to an ARDY-native `.npz` (loadable in ARDY's own
viewer `scripts/visualize.py`) **plus** a `stream_id` you can `play` on the
WebSocket to push the motion live into a running viser viewer.  See
[Live viewer](#live-viewer) below.

## MCP tools

| Tool | Purpose |
|------|---------|
| `ardy_generate` | One clip from one prompt (`core` avatar or `g1` robot). |
| `ardy_choreograph` | A prompt sequence stitched into one continuous clip. |
| `ardy_set_camera` | Set camera mode + params (`follow` / `orbit` / `fixed` / `over-the-shoulder`). |
| `ardy_get_camera` | Read back the current camera config. |
| `ardy_set_stage` | Save a named stage setup with waypoints for root-path guidance. |
| `ardy_get_stage` | Retrieve a saved stage by name. |
| `ardy_list_stages` | List all saved stage names. |
| `ardy_list_models` | List motion models (core vs G1). |
| `ardy_status` | Service health: device, loaded models, encoder link. |

## Live viewer

When you generate or choreograph motion, the service **also** buffers it in memory
and streams it at the motion's native frame rate to every connected WebSocket
viewer in real-time.  The character moves in the browser window as the frames
arrive — no file reloads, no manual playback.

### Start the viewer

```bash
# On the ARDY host (or any machine that can reach it):
cd ~/ardy && source venv/bin/activate
pip install -r /path/to/ardy-director/requirements-viewer.txt
python /path/to/ardy-director/director_service/viewer.py
```

Open **http://localhost:9601** in a browser — you'll see a 27-joint humanoid
skeleton.  Every `/generate` and `/choreograph` call immediately pushes the
motion into the view.

The viewer is a standalone [viser](https://viser.ai) server; you can run
multiple instances (`--port 9602`) to watch from different angles, or adjust
the skeleton topology with a custom JSON file:

```bash
python director_service/viewer.py --skeleton my_skeleton.json
```

### WebSocket protocol

Viewers connect to `ws://<host>:9600/ws`.  The service broadcasts these messages:

| Type | Direction | Payload |
|------|-----------|---------|
| `start` | service → viewer | `{id, prompt, fps, total_frames, num_joints}` |
| `frame` | service → viewer | `{id, frame, total, posed_joints: [[x,y,z],...], root_position, local_rot_mats}` |
| `done` | service → viewer | `{id}` |

Viewers can send commands:

```json
{"command": "list"}                              // list buffered clips
{"command": "play", "id": "<stream_id>", "loop": 2}  // replay a clip
{"command": "generate", "prompt": "wave", "duration": 3}  // generate + auto-stream
```

The `stream_id` returned by `/generate` and `/choreograph` can be passed to
`play` to replay a clip any time.

## Camera control

The director can control how viewers frame the shot.  Camera state is stored in
the service and broadcast to every connected viewer over WebSocket in real-time.

### Set the camera

```
ardy_set_camera(mode="follow", distance=3.0, height=1.5)
ardy_set_camera(mode="orbit", radius=5.0, orbit_speed=0.5)
ardy_set_camera(mode="fixed", position=[2.0, 3.0, 8.0], target=[0, 1, 0])
ardy_set_camera(mode="over-the-shoulder", side_offset=0.4, forward_offset=0.6)
```

| Mode | Params | Behaviour |
|------|--------|-----------|
| `follow` | `distance`, `height` | Camera trails behind the character at a fixed offset. |
| `orbit` | `radius`, `orbit_speed` | Camera circles the character continuously. |
| `fixed` | `position[x,y,z]`, `target[x,y,z]` | Static camera — useful for establishing shots. |
| `over-the-shoulder` | `side_offset`, `forward_offset`, `height` | Behind the character's shoulder, POV style. |

The viewer applies the camera mode every frame: in `follow` mode it tracks the
character's root position; in `orbit` mode it rotates around the character;
`fixed` ignores the character entirely.  Switch modes mid-scene — the viewer
transitions immediately.

## Staging / waypoints

Stage a path for the character to walk along.  Waypoints are interpolated into
a smooth root trajectory that guides the motion generation.

### Define a stage

```
ardy_set_stage(
    name="walkway",
    waypoints=[
        {"x": 0, "z": 0},
        {"x": 3, "z": 2},
        {"x": 6, "z": 0},
    ],
    start_heading_deg=0
)
```

### Use it in generation

```
ardy_generate(prompt="walk forward", duration=4, stage_id="walkway")
```

Or pass waypoints inline (no save needed):

```
ardy_generate(
    prompt="walk along the path",
    duration=5,
    waypoints=[{"x":0,"z":0}, {"x":4,"z":3}, {"x":8,"z":0}],
)
```

Waypoints affect the motion in two ways:
1. **At generation time** — passed as `observed_motion` to ARDY when the model
   supports root-path constraints (speculative — falls back gracefully).
2. **Post-hoc blend** — the generated root trajectory is blended toward the
   interpolated waypoints so the character walks the staged path even without
   native constraint support.

The viewer renders waypoints as green sphere markers connected by a path line.

## Worked example: walk, turn, and sit — staged

```python
# 1. Save a stage: walk in from the right, stop center-stage
ardy_set_stage(
    name="center_stage",
    waypoints=[{"x": -4, "z": 0}, {"x": 0, "z": 0}],
    start_heading_deg=90,
)

# 2. Set an orbit camera so the viewer gets a full view
ardy_set_camera(mode="orbit", radius=5.0, orbit_speed=0.25)

# 3. Walk to center
ardy_generate(prompt="walk forward confidently", duration=3, stage_id="center_stage")

# 4. Switch to follow camera for close-up
ardy_set_camera(mode="over-the-shoulder", side_offset=0.3)

# 5. Sit and wave at the end position
ardy_choreograph(steps=[
    {"prompt": "turn around", "duration": 1.5},
    {"prompt": "sit down on the floor", "duration": 2},
    {"prompt": "wave hello", "duration": 2},
])
```

## WebSocket protocol (extended)

In addition to `start`, `frame`, and `done`, the service broadcasts these
messages to connected viewers:

| Type | Direction | Payload |
|------|-----------|---------|
| `camera` | service → viewer | `{camera: {mode, distance, height, ...}}` |
| `stage` | service → viewer | `{name, stage: {waypoints, start_x, start_z, ...}}` |

On WebSocket connect, the service pushes the current camera and all saved stages
so new viewers synchronise immediately.

Viewers can also send camera and stage commands:

```json
{"command": "set_camera", "mode": "orbit", "radius": 5.0}
{"command": "set_stage", "waypoints": [{"x": 0, "z": 0}, {"x": 5, "z": 0}]}
```

## Roadmap

- **v0.1:** generate + position-chained choreography, model caching, encoder reuse.
- **v0.2 (now):** live viewer injection + camera control (4 modes) + staging/waypoints for root-path guidance.
- **v0.3 (next):** streaming choreography — velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **v0.4:** scene generation — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
