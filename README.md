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

## ARDY → Unreal Engine (FBX export + retarget)

ARDY makes motion but renders nothing. Unreal Engine is the render layer —
environments, lighting, cameras, photoreal / MetaHuman characters. Because the
`core` skeleton uses Mixamo-style bone names (`Hips`, `Spine`, `LeftArm`,
`LeftForeArm`, …), which is exactly what UE's **IK Retargeter** expects, an ARDY
clip drops onto the UE5 Mannequin or a MetaHuman with no custom bone glue.

### 1. Export a clip to FBX

Every generated clip already bakes the joint positions, global rotations, the
root track and the skeleton topology into its `.npz`, so the exporter is pure
numpy — **no GPU, no model reload, no ARDY install**.

```bash
# straight off an .npz on disk:
python scripts/export_fbx.py ~/ardy/outputs/director/<tag>.npz walk.fbx --scale 100

# or pull it from a running service by tag:
curl -o walk.fbx "http://<ARDY-host>:9600/export/<tag>.fbx?scale=100"
```

It writes an ASCII **FBX 7.4** file: one `LimbNode` per bone in the ARDY
hierarchy, each bone's rest offset as its `Lcl Translation`, and a per-frame
`Lcl Rotation` curve (plus a root `Lcl Translation` curve for locomotion). The
math is the plain SMPL rigid transform ARDY already uses, so an FK replay of the
FBX reproduces the clip's world-space joints exactly (the exporter tests assert
this round-trip, and the CLI prints a `rigid check` drift that should read ~0).

`--scale` / `?scale=` multiplies lengths. ARDY is metric; pass `100` for
Unreal-native centimetres, or leave it at `1.0` and rescale in UE's import
dialog. Works for the `core` (27-joint) and `soma` avatar skeletons; the `g1`
robot model has no `posed_joints` avatar rig, so it is rejected with a clear
error rather than a broken file.

### 2. Import + IK Retarget onto the UE5 Mannequin

1. **Import** `walk.fbx` (Content Browser → *Import*). Tick *Skeletal Mesh* and
   *Import Animations*; leave *Convert Scene* on so UE maps our Y-up/Z-front
   axes. You get a Skeletal Mesh, a Skeleton asset and an Animation Sequence.
2. Create an **IK Rig** for the imported skeleton and one for the UE5 Mannequin
   (`SK_Mannequin`) — right-click → *Create IK Rig*. In each, set the *Retarget
   Root* to `Hips`/`pelvis` and add a *Retarget Chain* per limb (Spine, Head,
   Arm L/R, Leg L/R). Because both skeletons use Mixamo-ish names, the chains
   auto-name cleanly.
3. Create an **IK Retargeter** with the ARDY IK Rig as *Source* and the
   Mannequin IK Rig as *Target*. Confirm the chain mapping (spine→spine,
   leftarm→leftarm, …).
4. Right-click the ARDY Animation Sequence → *Retarget Animations* → pick the
   retargeter → export. The result plays on the Mannequin.

### 3. MetaHuman (bonus)

A MetaHuman ships with its own IK Rig (`IK_metahuman`). Build a second
Retargeter with the ARDY IK Rig as *Source* and the MetaHuman IK Rig as
*Target*, map the same chains, and retarget the same sequence — no re-export
needed. (Set the MetaHuman body's *Post-Process* to the retargeted anim, or bake
to an Animation Sequence for Sequencer.)

> Tooling note: FBX generation and the numeric correctness of the export are
> covered by `tests/test_fbx_export.py` (round-trip FK + re-parse), which runs
> headless. The in-engine screen capture of a clip on the Mannequin needs a GPU
> host (to generate a clip) plus a UE5 install; run the two commands in step 1
> on such a host and the FBX drops straight into the recipe above.

## MCP tools

| Tool | Purpose |
|------|---------|
| `ardy_generate` | One clip from one prompt (`core` avatar or `g1` robot). |
| `ardy_choreograph` | A prompt sequence stitched into one continuous clip. |
| `ardy_export_fbx` | Export a clip tag to an FBX for Unreal / Maya / Blender. |
| `ardy_list_models` | List motion models (core vs G1). |
| `ardy_status` | Service health: device, loaded models, encoder link. |

## Roadmap

- **v0.1 (now):** generate + position-chained choreography, model caching, encoder reuse, FBX export to Unreal / Maya.
- **Streaming choreography:** velocity-smooth seams via ARDY's `autoregressive_step` (change the prompt mid-stream instead of stitching segments).
- **Live viewer injection:** push directed motion straight into the running viser demo.
- **Scene generation:** the larger goal — compose full visual scenes (camera, staging, multiple characters, background) around ARDY's demo, with the LLM as director.

## License

AGPL-3.0-or-later © 2026 Elyan Labs LLC. Wraps NVIDIA ARDY (Apache-2.0 code /
NVIDIA Open Model weights) — those licenses govern ARDY itself.
