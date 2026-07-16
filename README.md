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

### Web prompting engine (no file juggling)

The service also ships a single-page front end — type a prompt (or a sequence of
beats), hit go, and watch the character move in the browser. Open it once the
service is up:

```
http://<ARDY-host>:9600/ui/          #   / redirects here
```

- **Single prompt** or **Choreograph** mode (add beats, reorder with ↑/↓, run as
  one continuous clip). Controls for duration, model (`core`/`soma`/`g1`), seed
  and initial facing.
- A three.js viewport plays the returned motion as a stick figure with
  play/pause and a frame scrubber; choreograph seams (prompt changes) are marked.
- If you serve the page from somewhere other than the ARDY host, point it at the
  service with `/ui/?api=http://<host>:9600`.

It is glue over the existing API, so it needs no extra service state:

| Route | Purpose |
| --- | --- |
| `GET /ui/` | the static SPA |
| `GET /motion/{tag}` | a generated clip as JSON (`posed_joints` + skeleton parents) for the viewport; `?stride=N` thins frames for slow links |

`/generate` and `/choreograph` now also return a `tag` the UI fetches motion by;
the joint names and parent table are stored in the `.npz` at generation time, so
replaying a clip never reloads the model.

## MCP tools

| Tool | Purpose |
|------|---------|
| `ardy_generate` | One clip from one prompt (`core` avatar or `g1` robot). |
| `ardy_choreograph` | A prompt sequence stitched into one continuous clip. |
| `ardy_list_models` | List motion models (core vs G1). |
| `ardy_status` | Service health: device, loaded models, encoder link. |

## Roadmap

- **v0.1 (now):** generate + position-chained choreography, model caching, encoder reuse.
- **Streaming choreography:** velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **Live viewer injection:** push directed motion straight into the running viser demo.
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
