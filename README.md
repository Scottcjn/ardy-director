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

## Roadmap

- **v0.1 (now):** generate + position-chained choreography, model caching, encoder reuse.
- **Live viewer injection (now):** motion streams in real-time to a viser viewer over WebSocket.
- **Streaming choreography:** velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
