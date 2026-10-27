# Policies for the local Gen 2 runner

These adapters were derived from the supplied `aria_eval_scripts_refactored`
folder, not another GitHub branch. They separate inference from camera capture,
robot control, recording, preview and keyboard handling. The source experiment
scripts remain historical references. `source_manifest.json` records a source
hash and destination for every top-level source script.

## Which policy to use

| Directory | Source family | Model input |
| --- | --- | --- |
| `textvla` | `eval_openpi_textvla*` | Robot views, YAM state, text |
| `streamvla` | `eval_openpi_streamvla*` | Same plus current gaze-cropped Aria RGB |
| `policy_edith` | `eval_openpi_ours*`, keyframe/egoview ablations | Same plus a queued visual keyframe and subtask text |
| `vlm_textvla` | `eval_openpi_vlm_textvla*`, `0915_finetext` | Robot views/state and planner-generated subtask text; top camera for planning |
| `fam_hri` | `eval_fam_hri*`, `eval_famhri_before` | Visual-keyframe VLA with optional local FAM-HRI grounding/planning |
| `static_hold` | Generic hardware smoke policy | Hold the startup joint state |
| `zero_action_policy` | Generic observation diagnostic | Zero vector of any state dimension; motor execution forbidden |

All five migrated policies default to **manual/preset language**, a local
OpenPI server at `ws://127.0.0.1:8000`, and 15 Hz. Merely loading a default policy
opens no inference connection, camera, CAN interface or cloud session. Gemini
planning and the external FAM-HRI pipeline are explicit options.

The canonical Edith directory is `policy_edith`; `edith` is a compatibility
symlink to it. Run it with `python -m edith.scripts.policy_edith`.
The old `eval_openpi_ours_with_voice_260503.py` filename now launches this same
local runtime. It no longer imports the remote Aria camera or Piper environment.

The families consolidate duplicated experiment files; they are not claims of
bit-for-bit reproduction of every dated experiment. Task/language variants are
configuration, not separate camera/robot loops. Korean text works unchanged in
`instruction`, transcript input and custom prompt files. Old publication-video,
Whisper-caption and plotting code is not part of `act()`.

## Install and check without hardware

From the repository root:

```bash
uv pip install --python .venv/bin/python -e '.[policies]'
.venv/bin/python -m edith.scripts.eval_policy_local \
  --policy-dir policies/streamvla --check-policy-only
.venv/bin/python -m unittest discover -s tests -v
```

The OpenPI adapter uses the MessagePack wire protocol through a finite-timeout
WebSocket client, compatible with NumPy 2. Core capture and diagnostics require
no robot SDK. Select a separate robot repository through the station JSON
`robot.adapter` factory; see [the main README](../README.md).

Each folder contains `policy.json`, `policy.py`, presets and policy-specific
prompt assets. Shared runner-side code under `edith/policy_utils` supplies
transport, RGB preprocessing, chunk execution and queue validation. Install
Edith on the destination host; copy the policy folder and its own assets there.
Relative imports inside `policy.py` are isolated per policy, so two directories
can both contain a `helper.py` without sharing its global state.

## Checkpoint requirements

The supplied files are **evaluation clients**, not model checkpoints. Start
an OpenPI-compatible server with a YAM-trained checkpoint and the observation
keys required by your selected family. Changing robot drivers cannot convert
Piper weights or their normalization statistics into YAM weights.

The server's initial metadata message must include:

```json
{
  "robot_type": "yam",
  "action_space": "joint_position",
  "arm_order": ["left", "right"]
}
```

Only publish that metadata for a checkpoint trained with this action convention.
The client rejects absent/mismatched metadata, malformed action chunks, NaN/Inf,
grippers outside `[0,1]`, expired results, and command jumps above configured
per-step limits. These checks validate the interface; they do not certify a
checkpoint's behavior. A returned chunk must have shape `(T,14)`:

`left joint1..joint6 (rad), left gripper, right joint1..joint6 (rad), right gripper`.

The generic policy interface returns one `(D,)` action per `act()`, matching
the selected robot. These five learned clients specifically return **14 YAM
values**, not an action chunk. Replacing a robot adapter does not convert their
checkpoint contract. The adapter buffers up to `chunk_size` server actions and consumes one
per synchronized observation. Pending inference holds the last target. New
instructions, reset, preemption or replacement invalidate earlier actions.
An exhausted task queue holds; it never homes or resets a robot.

Images are RGB uint8 from the canonical observation. The old scripts mixed BGR
and RGB; the adapters never swap canonical RGB before inference. `image_size`
(default 224) letterboxes all model images; set it to the dimensions used during
training or `null` to preserve native sizes. Gaze crops use actual frame size
and calibrated pixels. Invalid gaze falls back to the full image by default;
`invalid_gaze: "error"` makes missing gaze fatal instead.

## Run with task configuration

```bash
# Live camera/inference evaluation; no CAN unless --execute-actions is passed.
.venv/bin/python -m edith.scripts.eval_policy_local \
  --station configs/station.local.json \
  --policy-dir policies/streamvla \
  --policy-config policies/streamvla/presets/passing_tools.json \
  --headless --duration 60
```

USB discovery is the default. If needed, use `--aria-ip <GLASSES_USB_NCM_IP>`
for the directly connected glasses' USB interface, not a Wi-Fi or external
publisher address. Aria capture always selects `StreamingInterface.USB_NCM`.
`--policy-config` loads a JSON object that overrides the manifest's `config`
fields; repeat it to apply files in order (last value wins, shallow merge).
It does not modify policy files. `--instruction` takes precedence over the
configured instruction; a nonempty transcript takes precedence over both.

`--output-fps` defaults to the policy's `control_hz` and must match it. Changing
cadence changes the meaning of an action chunk, so use the model's trained rate.
A slow model can cause hold frames between chunks; inference is asynchronous
so synchronized observations and planner buffering keep running.

## Voice and operator updates

Voice variants now consume a transcript through the policy observation:

```bash
.venv/bin/python -m edith.scripts.eval_policy_local \
  --station configs/station.local.json \
  --policy-dir policies/textvla --transcript-file /tmp/edith-transcript.txt \
  --duration 60 --headless
```

Create the UTF-8 file first. A local STT/UI process may atomically replace it
with the latest complete utterance. These adapters **do not transcribe
speech themselves**. Live microphone PCM is available as `obs["audio"]`, but
the learned clients do not transcribe or forward it to the VLA server.
`audio.samples` holds int32 `(frames, channels)` PCM with the actual
`sample_rate_hz`, per-frame host/device timestamps, mute flags and amplitude
scale. The [zero-action diagnostic](zero_action_policy/README.md) records this
input alongside the camera/perception/state observations and outputs. Preset language requires neither STT nor an
online speech service. Repeated unchanged text is not treated as a new spoken
intervention in successive planner updates.

For manual keyframes, `policy_edith` captures the current synchronized Aria view when
a new instruction arrives. Or configure explicit `subtasks` with `instruction`
and a `keyframe` image path relative to the policy folder. A missing image path
means capture at reset. Assets may not escape the policy directory.

Queue policies also accept `--policy-events-file /tmp/edith-event.json`. Write
one JSON object atomically, using a different `id` for each event:

```json
{"id":"edit-001","command":"Append","new_elements":[{"subtask_instruction":"pick up the cup","keyframe_interval":{"start":0,"end":0}}]}
```

The event captures the **current synchronized frame**; its interval must be
`[0,0]`. `{"id":"next-002","command":"next"}` advances the current task.
`Preempt`, `Replace` (`target_index`) and `Remove` (`target_indices`, future items
only) are supported. Default `min_subtask_steps: 40` prevents early completion
spikes, and `completion_threshold: 0.8` advances only at the end of a chunk.

## Gemini planning and ablations

`policy_edith/presets/gemini_*.json` and `vlm_textvla/presets/gemini_*.json` select the
source task prompts. Set `gemini_model` to a model available to your account
and export `GEMINI_API_KEY`. The blank model in these preset templates is
intentional: loading one without selecting a model fails clearly.

```json
{"gemini_model":"YOUR_AVAILABLE_MODEL_ID"}
```

Save the object above to a local model settings file, then layer it after the
preset:

```bash
.venv/bin/python -m edith.scripts.eval_policy_local \
  --station configs/station.local.json \
  --policy-dir policies/policy_edith \
  --policy-config policies/policy_edith/presets/gemini_passing_tools.json \
  --policy-config /path/to/model-settings.json \
  --duration 120 --headless
```

This explicitly sends selected RGB frames/transcripts to Gemini. Camera-to-host
transport and robot control remain local USB/CAN. Inference services can require
network access even though glasses no longer require Wi-Fi.

The planner receives bounded, timestamp-labelled sampled RGB frames instead of
an MP4 with inferred frame timing. Original initial/update prompt sections,
visual queue context, completion gating, and Append/Replace/Preempt/Remove
semantics are retained. Intervals select actual captured timestamps. Repeated
camera samples are deduplicated and clips do not overlap. Model/network failure
raises an error rather than silently executing an old plan. Index-dependent
planner edits are dropped if the queue changed while the request was running.

Edith presets cover `without_keyframe`, `without_egoview`, `black_keyframe`,
`full_egoview`, and `manual_keyframes`. Layer them after the task preset.
Task-specific phrases from the source remain editable JSON/prompt assets.
Use `vlm_textvla` for top-view planning with text-only VLA input.

## FAM-HRI

`fam_hri` works as a manually queued visual policy by default. To enable the
original detector/planner path, use `fam_hri/presets/local_pipeline.json` and
supply `fam_hri_root` (containing `fam_hri/server/bundle_pipeline.py`), SAM3
weights/dependencies, and `llm_model`. These external files were not provided.
The adapter uses the source `KFInput`, `PipelineOpts`, `build_labeler`,
`build_planner`, and `process_utterance` API. It selects local calibrated-gaze
dwell keyframes and maps returned subtasks to captured keyframes. Task-specific
label heuristics from the old bottle/tool demos are not silently applied;
configure the object vocabulary/context for the actual scene.

FAM-HRI requests run off the control thread; exceeding `planner_timeout_s`
aborts evaluation. Python cannot forcibly cancel a native detector call, so
shutdown discards its results; the daemon worker cannot send robot commands.

## Validation limits

Tests use synthetic observations, fake planners, and a real loopback WebSocket
server. They cover the protocol, RGB/crop/state contracts, action chunk handling,
queue completion/edits, stale data rejection and policy loading. No model was
queried in the cloud, no real checkpoint inference was validated, and no robot
was moved during this migration. Real-model validation needs the YAM checkpoint;
FAM-HRI additionally needs its external package and weights.
