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

## Custom avatars (bring your own Blender humanoid)

ARDY ships one hand-rigged avatar. `scripts/rig_avatar.py` turns any Blender
humanoid into an ARDY-compatible skin automatically — no GPU, no ARDY install,
numpy only.

```bash
# Export your humanoid from Blender in a neutral T-pose as OBJ, with
# "Objects/Groups as OBJ groups" on (so each body part carries a label).
python scripts/rig_avatar.py my_avatar.obj skin_standard.npz --color

# View it: the stock viewer loads the core skin from the cskel27 asset folder,
# so back up the bundled skin and swap yours in.
cd /path/to/ardy
cp assets/skeletons/cskel27/skin_standard.npz assets/skeletons/cskel27/skin_standard.bak.npz
cp /path/to/skin_standard.npz assets/skeletons/cskel27/skin_standard.npz
python scripts/visualize.py CLIP.npz            # character now wears your mesh

# (or, in code, point the skeleton at any folder holding joints.p + your skin:)
#   CoreSkeleton27(folder="my_skel")   # my_skel/ = joints.p (copied) + skin_standard.npz
```

What it does automatically:

- **Up-axis + scale fit** — Blender Z-up → ARDY Y-up, uniform-scaled to the
  core skeleton's height, feet dropped to the floor.
- **Part → joint mapping** onto the 27-joint core skeleton, understanding
  Mixamo (`LeftForeArm`), Rigify/`.L`-`.R` (`upper_arm.L`), MakeHuman and plain
  `arm_left` names. Left/right comes from the label, or from vertex geometry
  when the label is unsided.
- **Skinning** — rigid per-part by default, or `--smooth` for inverse-distance
  blended seams. An unlabeled mesh falls back to nearest-bone geometry.
- **Optional flat limb colors** with `--color`.

The emitted `skin_standard.npz` reuses the canonical `bind_rig_transform` /
`rig_joint_names` / bone edges verbatim, so ARDY's viewer loads it with no
joint-name mismatch and every generated clip deforms it correctly.

Worked example (a second avatar besides the bundled one):

```bash
python examples/custom_avatar/make_blocky_humanoid.py      # writes the OBJ
python scripts/rig_avatar.py examples/custom_avatar/blocky_humanoid.obj \
       examples/custom_avatar/skin_standard.npz --color
python tests/test_avatar_rig.py                            # headless checks
```

## Roadmap

- **v0.1 (now):** generate + position-chained choreography, model caching, encoder reuse.
- **Streaming choreography:** velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **Live viewer injection:** push directed motion straight into the running viser demo.
- **Custom avatars (done):** `scripts/rig_avatar.py` rigs any Blender humanoid onto the core skeleton (see above).
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
