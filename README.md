# Edith: synchronized Aria Gen 2 + YAM evaluation

Edith collects and evaluates one capture-time-aligned observation containing
Project Aria Gen 2 RGB, eye gaze, both-hand pose, three YAM RealSense views, and
robot state. The default 15 FPS timeline is anchored to every second frame of
the 30 FPS top robot camera. Aria RGB and hand tracking run at 10 Hz in the
`mp_streaming_demo` profile, so their nearest sample may be reused while its
original timestamp and sequence number remain unchanged.

The safe default is a static, no-CAN policy check. Motor execution is only
enabled by the explicit `--execute-actions` flag.

Glasses capture uses **USB-NCM on this computer**. No Wi-Fi or separate Aria
streaming computer is required. USB-NCM is an IP link over the USB cable;
its local HTTP receiver is part of the Gen 2 SDK.
See the [official USB streaming guide](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/streaming).

## Structure

```text
policies/<policy>/                 one deployable policy directory
piper_teleop/aria_utils/           Aria Gen 2 capture, gaze/hand schema, projection
piper_teleop/robot_utils/          robot interface and ABC/YAM implementation
piper_teleop/teleop_utils/         optional GELLO validation
piper_teleop/eval_utils/           sync, observation, recording, action lease
piper_teleop/envs/                 sensor backends
piper_teleop/policy_utils/         policy-directory loader
piper_teleop/scripts/              collection, dashboard, and evaluation CLIs
tests/                             hardware-free contract tests
```

A new policy requires only a directory with `policy.json`, a Python entrypoint,
and any checkpoints or normalization assets it uses. A new robot implements
`RobotAdapter` and is registered in `robot_utils/factory.py`; policy and Aria
code do not change.

The `aria_eval_scripts*` folders contain historical experiment entry points.
`eval_openpi_ours_with_voice_260503.py` now redirects to the local `policy_edith`
launcher and uses its current CLI flags.
Some still include station-specific paths and online-service assumptions; the
supported portable path is `eval_policy_local` plus `RobotEvaluationEnv`.

## Install

Python 3.11 is the tested version. The ABC/i2rt deployment is a separate local
checkout and must match the station configuration.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-local-yam.txt
uv pip install --python .venv/bin/python -e /path/to/abc-robot-setting/i2rt
uv pip install --python .venv/bin/python -e .
```

Pair and authenticate the glasses once through the Aria Companion app and SDK:

```bash
aria_doctor
aria_gen2 device list
aria_gen2 auth pair
```

Only one application can own an Aria or RealSense stream. Close other viewers
before starting Edith. By default the SDK discovers USB-connected glasses.
Use `--aria-serial` if several are attached. If USB mDNS discovery is unavailable,
`--aria-ip <GLASSES_USB_NCM_IP>` selects the glasses' USB address directly; this
is neither a Wi-Fi address nor the old external camera server's IP.

## Connection and live preview

Run a read-only dependency/device check first:

```bash
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --check --headless
```

Show Aria, top, left-wrist, and right-wrist views together. The Aria tile
overlays the calibrated gaze point and 21 landmarks for each detected hand:

```bash
.venv/bin/python -m piper_teleop.scripts.live_sensor_dashboard
```

## Synchronized data collection

This records camera-only observations at 15 FPS without opening CAN:

```bash
.venv/bin/python -m piper_teleop.scripts.collect_trajectory_yam_local \
  --mode preview \
  --aria-profile mp_streaming_demo \
  --sync-camera top \
  --output-fps 15 \
  --tolerance-ms 100 \
  --output recordings/episode_001.h5
```

Use `q`, Escape, or Ctrl+C to close and flush a recording. Output files are
created exclusively and never overwritten. `--mode teleop --leaders leaders.json`
adds ABC GELLO/YAM state and commands after the station's leader configuration
has been verified. See [README_LOCAL_YAM.md](README_LOCAL_YAM.md) for detailed
station setup and operational safety notes.

The HDF5 root schema is `edith.yam.aria_gen2.v2`. Each stream has mapped capture,
source, receive, and sequence arrays. Images are JPEG byte arrays. Gaze is a
named 35-value vector; hand pose is a named 156-value vector covering presence,
confidence, 21 3D landmarks, wrist pose, and wrist/palm normals for both hands.
Read only rows below the root `complete_rows` attribute.

Synchronization is software capture-time alignment, not a shared hardware
trigger. The default 100 ms total-spread window accommodates 30 Hz robot cameras
and 10 Hz Aria perception while retaining each stream's real capture time.

## Robot evaluation and `get_observation()`

`RobotEvaluationEnv` combines any `RobotAdapter` with the synchronized sensor
backend. Existing evaluator code can keep its current call shape:

```python
wrapped = env.get_observation(
    mode="human_view_streamed_and_keyframe_cropped",
    audio_drain=True,
)
observation = wrapped["timestamp"]
```

The synchronized frame contains:

| Key | Value |
| --- | --- |
| `images["aria_rgb"]` | full Aria Gen 2 RGB view |
| `images["top_camera_rgb"]` | top robot camera |
| `images["left_wrist_camera_rgb"]` | left wrist camera |
| `images["right_wrist_camera_rgb"]` | right wrist camera |
| `gaze_point` | calibrated Aria RGB pixel `[x, y]`, NaN when invalid |
| `hand_keypoints["left"/"right"]` | calibrated `21 x 2` RGB pixels, NaN when unavailable |
| `aria_perception` | raw named eye-gaze and hand-pose vectors |
| `robot_state` | legacy per-arm evaluation dictionary |
| `robot_state_flat` | policy-ready flat joint vector |
| `capture_timestamps_ns`, `source_timestamps_ns`, `sequence_numbers` | per-stream synchronization evidence |

`aria_cropped_rgb` currently aliases the full synchronized Aria image; policies
that need a gaze crop should implement their desired crop size and boundary
behavior explicitly. New directory policies receive the canonical direct
observation through `get_policy_observation()`.

Run the included static policy with cameras and inference but no CAN access:

```bash
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --policy-dir policies/static_hold \
  --headless --duration 10 \
  --output recordings/static_dry_run.h5
```

After verifying the preflight and physical workspace, follower-only execution is
explicitly enabled with `--execute-actions`. Sensor health is checked before CAN
opens, the first command is bounded against live startup state, and an action
lease continuously refreshes the command or aborts on failure.

## Migrated policies

The supplied `aria_eval_scripts_refactored` clients are now available as
`policies/textvla`, `streamvla`, `policy_edith`, `vlm_textvla`, and `fam_hri`.
The previous `policies/edith` path is a compatibility symlink.
See [policy migration and usage](policies/README.md) for task presets, local
OpenPI inference, Gemini/FAM-HRI options, transcript input and validation limits.
Each adapter implements the directory contract below and consumes canonical
synchronized RGB observations; old hardware-owning evaluation loops are not run.

```bash
uv pip install --python .venv/bin/python -e '.[policies]'
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --policy-dir policies/streamvla --check-policy-only
```

Run the named Edith policy with synchronized local views and no motor commands:

```bash
.venv/bin/python -m piper_teleop.scripts.policy_edith --check-policy-only
.venv/bin/python -m piper_teleop.scripts.policy_edith \
  --policy-config policies/policy_edith/presets/passing_tools.json \
  --duration 60 --headless
```

The VLA endpoint defaults to `ws://127.0.0.1:8000` on this computer. Selecting
Gemini planning still needs internet for Gemini; manual/preset language and
a local VLA server do not. Voice text can be supplied by a local STT process
through `--transcript-file`.

## Policy directory contract

`policy.json` uses schema `edith.policy.v1`:

```json
{
  "schema": "edith.policy.v1",
  "name": "my-policy",
  "entrypoint": "policy.py:Policy",
  "control_hz": 15,
  "action_space": "joint_position",
  "config": {}
}
```

The class constructor receives `policy_dir` and `config`, and must implement
`act(observation)` returning one finite 1-D action (not an action chunk).
Optional methods are `reset(observation)` and `close()`. Policy-local imports
use relative syntax (`from .helper import ...`) in an isolated namespace.
Use repeatable `--policy-config` JSON files for overrides and
`--check-policy-only` for validation without devices or inference connections.
The runner's output FPS defaults to `control_hz`; an explicit different value
is rejected to preserve the checkpoint's action timing.

## Validation and GitHub hygiene

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
git status --short
```

Recordings, VRS/bag files, local environments, `.env` files, leader hardware
configuration, caches, and archived zip snapshots are ignored. API credentials
must be supplied only through `OPENAI_API_KEY` and `GEMINI_API_KEY`; copy
`.env.example` and keep the resulting `.env` local.

Important: credentials were previously embedded in historical scripts in this
working copy. Their literals have been removed, but those credentials should be
revoked and rotated before publishing or running online evaluations.
