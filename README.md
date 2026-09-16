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

## MCP tools

| Tool | Purpose |
|------|---------|
| `ardy_generate` | One clip from one prompt (`core` avatar or `g1` robot). |
| `ardy_choreograph` | A prompt sequence stitched into one continuous clip. |
| `ardy_list_models` | List motion models (core vs G1). |
| `ardy_status` | Service health: device, loaded models, encoder link. |

## Prompt library

`examples/prompts.json` is a starting palette so the engine ships with something
better than a blank box — 36 usable prompts plus 5 documented failures.

| Category | Prompts | What it covers |
|----------|:-------:|----------------|
| `locomotion` | 8 | Walk, run, side-step; direction and style variations. |
| `turning` | 4 | Changing facing, in place or while moving. |
| `ground_transitions` | 5 | Sit, stand, crouch, bow. |
| `jumps` | 4 | Both feet leaving the ground. |
| `gestures` | 5 | Upper-body actions performed while standing. |
| `kicks_strikes` | 3 | Limb strikes; in-distribution but high-energy. |
| `dance` | 4 | Performance motion, often looping. |
| `idle` | 3 | Low-energy standing — filler between beats. |
| `out_of_distribution` | 5 | **Known failures.** Not recommendations — the model's edges, documented so they don't surprise you. |

Each entry carries a suggested `duration`, a one-line `note`, and a `provenance`:
`upstream_preset` (verbatim from ARDY's own demo presets — author-vetted phrasing,
don't "tidy" them), `upstream_doc` (an upstream worked example), or `derived`
(written here against the documented training distribution).

**Phrasing matters.** ARDY's text encoder was trained on Bones Rigplay captions,
which are third-person declarative sentences — `"A person is walking."` The
imperative fragments used elsewhere in this README (`"walk to the desk"`) are
off-distribution *phrasing* even when the motion itself is in-distribution, and
are likelier to drift. The library follows the caption form throughout.

### Verifying the library

⚠️ **The library ships unverified** — every prompt is written against ARDY's
documented distribution, but none has been confirmed on hardware. `"verified":
false` means nobody has watched it produce motion; read the notes as
expectations, not observations.

Confirming them is one command on a host that can reach the Director service:

```bash
python scripts/validate_prompts.py --url http://192.168.0.136:9600 --seed 42
python scripts/validate_prompts.py --category locomotion --diffusion-steps 8  # quick sweep
python scripts/validate_prompts.py --write-back    # fold results into prompts.json
```

It generates every prompt and reports mechanical signals — did it generate, how
far the root travelled, how much of the clip had a foot down — flagging clips
that contradict the library's own expectations (a locomotion prompt that never
moves; an out-of-distribution prompt that sailed through). Those flags say where
to point your eyes first. **They are not a verdict:** whether the motion matches
the prompt needs a human. `--write-back` records what was observed per prompt.

## Roadmap

- **v0.1 (now):** generate + position-chained choreography, model caching, encoder reuse.
- **Streaming choreography:** velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **Live viewer injection:** push directed motion straight into the running viser demo.
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
