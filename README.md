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

`ardy_choreograph` **streams**: each beat is generated on top of the motion
already in flight (ARDY's `autoregressive_step`), so the character carries its
momentum through a prompt change instead of popping at the boundary. Beat
durations are rounded to the model's generation horizon — the response reports
the frames each beat actually got, plus the measured root-velocity jump at every
prompt change:

```json
{"frames": 210, "duration_s": 7.0, "mode": "stream",
 "segments":   [{"prompt": "walk to the desk", "frames": 90, "duration_s": 3.0}, ...],
 "seam_frames": [90, 150],
 "seam_velocity": {"baseline_jump": 0.031, "max_ratio": 1.2,
                   "seams": [{"frame": 90, "time_s": 3.0, "jump": 0.037, "ratio": 1.2}]}}
```

`ratio` is how many times harder the root's velocity changes at the seam than on
a typical frame of the same clip: ~1 means the seam is indistinguishable from
ordinary motion, large means a visible pop.

### Checking the seams yourself

`mode: "stitch"` reproduces v0.1's behaviour (independent clips glued by root
offset) so the two can be compared on the same seed:

```bash
for m in stitch stream; do
  curl -s localhost:9600/choreograph -H 'content-type: application/json' \
    -d "{\"seed\":42,\"mode\":\"$m\",\"steps\":[
          {\"prompt\":\"walk forward confidently\",\"duration\":3},
          {\"prompt\":\"stop and turn around\",\"duration\":2}]}" | jq -r .npz
done
python scripts/seam_report.py <stream.npz> --before <stitch.npz> --plot seams.png
```

The `.npz` carries its own seam frames, so the report needs nothing but the
files. matplotlib is optional — without it you still get the numbers.

## MCP tools

| Tool | Purpose |
|------|---------|
| `ardy_generate` | One clip from one prompt (`core` avatar or `g1` robot). |
| `ardy_choreograph` | A prompt sequence streamed into one continuous clip, velocity-smooth across beats. |
| `ardy_list_models` | List motion models (core vs G1). |
| `ardy_status` | Service health: device, loaded models, encoder link. |

## Roadmap

- **v0.1:** generate + position-chained choreography, model caching, encoder reuse.
- **Streaming choreography (now):** velocity-smooth seams via ARDY's `autoregressive_step` — the prompt changes mid-stream instead of segments being stitched.
- **Live viewer injection:** push directed motion straight into the running viser demo.
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
