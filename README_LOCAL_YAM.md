# Edith on one host: Aria Gen 2 + YAM

Use `piper_teleop.scripts.collect_trajectory_yam_local` for the new station.
The existing `collect_trajectory_aria` entry point now forwards to this local
collector; the former implementation is `collect_trajectory_aria_piper_legacy`.
The historical `piper_teleop` package name is retained for import compatibility.
The old Piper/remote-Aria experiment scripts remain legacy entry points; they do
not automatically become YAM policies. Do not feed Piper-trained joint actions
into YAM: the joint conventions, gripper normalization and kinematics differ.

## What runs locally

- Aria Gen 2 uses `aria.sdk_gen2` and `aria.stream_receiver`, explicitly selecting
  **USB_NCM**. Plug glasses into this computer with USB 3. No Wi-Fi, external
  computer, remote image publisher, cloud upload or inference service is needed.
  USB-NCM is an IP interface over the USB cable; the SDK's TLS receiver on port
  6768 must listen on `0.0.0.0` so the glasses can reach it.
- Robot control uses the installed ABC GELLO/YAM nodes and i2rt via local CAN.
  The `abc_station_config` profile supplies camera serials, CAN adapter mapping,
  rates and gripper type. No Piper SDK, Piper sign flips or Piper gripper scaling.
- RealSense cameras and Aria run in parallel. A bounded synchronizer selects
  complete bundles within `--tolerance-ms` (100 ms default) and rejects stale
  frames after `--max-age-ms` (500 ms). Slower non-anchor samples may be reused
  on adjacent master-camera rows without changing their timestamps or sequences.
- Aria capture time is mapped through the SDK TicSync latency API to host
  `perf_counter_ns`. RealSense global capture time is mapped from host wall time
  to the same clock. Unsynchronized clocks emit no frames. ABC state and command
  timestamps are host **publication times**, not encoder capture timestamps.
  This is software alignment, not hardware-triggered simultaneous exposures;
  USB/clock-estimation error remains. Recorded timestamps and bundle spread
  make alignment measurable. No spatial calibration between cameras is implied.
- A single HDF5 file contains paired images, state and leader commands, source
  timestamps, receive timestamps, mapped timestamps and sequence numbers.
  Use `--record-vrs` to retain the raw Gen 2 profile streams. The synchronized
  path records named eye-gaze and two-hand-pose vectors and projects valid gaze
  and hand landmarks onto RGB using the device calibration.

## Environment

A dedicated `.venv` is provided in this folder; the ABC environments are unchanged.
To recreate it (Python 3.11 and uv):

```bash
cd /home/user/Desktop/dongkyu/edith_source
uv venv .venv --python 3.11
printf 'scikit-build-core<0.10\n' > /tmp/edith-build-constraints.txt
uv pip install --python .venv/bin/python -r requirements-local-yam.txt \
  -e /home/user/dev/abc-robot-setting/i2rt \
  --build-constraint /tmp/edith-build-constraints.txt
source .venv/bin/activate
```

On the actual glasses host, complete the SDK's one-time USB setup and pairing:

```bash
aria_doctor
aria_gen2 device list
aria_gen2 auth pair
```

Do not run the Aria viewer/another stream receiver at the same time as this
collector. The collector starts and stops its own USB stream. It refuses to
replace an already-running Aria session. Stop other programs holding the three
RealSense cameras or YAM CAN interfaces before running this stack.

## Check, preview and collect

All commands below run from `edith_source` using `.venv/bin/python`.

```bash
# Read-only dependency/config check; no camera or motor initialization.
.venv/bin/python -m piper_teleop.scripts.collect_trajectory_yam_local --check

# First verify four synchronized views, without initializing the arms.
.venv/bin/python -m piper_teleop.scripts.collect_trajectory_yam_local

# Save camera-only observations for 30 seconds.
.venv/bin/python -m piper_teleop.scripts.collect_trajectory_yam_local \
  --mode preview --duration 30 --output data/camera_check.h5

# Use verified ABC GELLO controllers to teleoperate both YAM arms and record.
.venv/bin/python -m piper_teleop.scripts.collect_trajectory_yam_local \
  --mode teleop --leaders leaders.json --output data/episode_001.h5 \
  --instruction 'pick up the cup'
```

The local ABC station explicitly has **placeholder GELLO device paths** and no
leaders were present during development. `--mode teleop` requires a JSON file
with the actual installed controller configuration. Do not copy another
station's servo IDs/signs blindly. Example structure (replace values with your
verified YAM GELLO configuration):

```json
{
  "left": {"device_name": "/dev/serial/by-id/LEFT_LEADER", "servo_ids": [40,41,42,43,44,45,46], "joint_signs": [1,-1,-1,-1,1,1,1]},
  "right": {"device_name": "/dev/serial/by-id/RIGHT_LEADER", "servo_ids": [40,41,42,43,44,45,46], "joint_signs": [1,-1,-1,-1,1,1,1]}
}
```

ABC expects leaders at its calibration pose on startup and USB serial latency
set to 1 ms; see the installed ABC GELLO documentation. Teleop uses ABC's startup
interpolation and gripper calibration. First it verifies synchronized camera
frames, then launches the arm processes. Loss of synchronized bundles aborts
within `--timeout` (10 s default); loss of leader commands aborts the follower
within 500 ms. Exit uses i2rt close (zero torques), without ABC's automatic
move-to-zero. Support the arms when stopping. No live motor commands were run
as part of development.

For an already-running ABC teleoperation controller, use `--mode attach` to
subscribe to local `leader_<side>_actions` and `follower_<side>_state` IPC topics.
This mode owns only the cameras/recording; stopping it does not stop the external
controller. It must not compete with another process for the cameras.

Use `--headless` without a display; q/Esc or Ctrl+C ends a normal session.
Existing output files are never overwritten. Each invocation is one episode;
recording starts with the first complete bundle. Stream problems fail loudly
instead of recording mismatched images. `--abc-root`, `--profile`, and
`--aria-serial` and `--aria-ip` allow another local deployment/device. The IP
option is the glasses' USB-NCM address, used to bypass mDNS discovery; it is
not a Wi-Fi address or the old external Aria publisher's address.

## Modular policy sanity path

The policy evaluation path is split by ownership so that changing one device
does not require editing the policy or synchronization loop:

```text
piper_teleop/aria_utils/       Aria Gen 2 discovery/capture API
piper_teleop/robot_utils/      robot contract, registry, ABC/YAM adapter
piper_teleop/teleop_utils/     optional GELLO configuration and validation
piper_teleop/policy_utils/     single-directory policy loader
piper_teleop/eval_utils/       observation contract and action refresh
policies/static_hold/          complete smoke-test policy deployment
```

The default Aria stream profile is `mp_streaming_demo`. Every accepted bundle
contains six timestamp-aligned streams: Aria RGB, eye gaze, hand pose, and the
top/left/right station cameras. Eye gaze is stored as a 35-value vector with
combined/per-eye gaze, validity, blink, origin, and spatial gaze fields. Hand
pose is stored as a fixed 156-value vector containing presence, confidence, 21
3D landmarks, wrist pose, wrist normal, and palm normal for each hand. Missing
hands use a zero presence flag and NaN values. Each HDF5 values group has a
`value_fields` JSON attribute, so consumers should resolve fields by name.

Start a synchronized sensor-only recording (no CAN or motor commands) with:

```bash
.venv/bin/python -m piper_teleop.scripts.collect_trajectory_yam_local \
  --mode preview \
  --aria-profile mp_streaming_demo \
  --sync-camera top \
  --output-fps 15 \
  --tolerance-ms 100 \
  --output recordings/episode.h5
```

The preview header reports gaze validity, left/right hand presence, and the
actual six-stream timestamp spread. Use `q`, Escape, or Ctrl+C to close and
flush the file. This is capture-time software synchronization; it is not a
shared hardware trigger. The three robot cameras are acquired at 30 FPS and
every second frame of the selected master camera anchors a 15 FPS output row.
Aria RGB and hand pose are 10 Hz in `mp_streaming_demo`, so their nearest sample
is intentionally reused on some adjacent rows; original per-stream timestamps
and sequence numbers remain unchanged. A 100 ms total-spread window is needed
for this multi-rate 15 FPS resampling mode.

A deployable policy is one directory containing `policy.json`, its Python
entrypoint, and any checkpoints/normalizers/assets it needs. The runner only
depends on this manifest contract. The bundled static policy captures the live
joint state at reset and repeats that exact target. It does not depend on GELLO.

Run the read-only preflight first. It returns success only when the policy,
three station cameras, one Aria USB-NCM device, dependencies, and follower CAN
interfaces are all visible:

```bash
.venv/bin/python -m piper_teleop.scripts.eval_policy_local --check --headless
```

For a live 2x2 inspection window (Aria, top, left, right) without robot control,
run the dashboard below. It shows the YAM cameras immediately and retries Aria
USB-NCM discovery every two seconds, leaving a diagnostic placeholder in that
tile until the glasses connect. This UI is a latest-frame preview, not the
synchronized dataset recorder.

```bash
.venv/bin/python -m piper_teleop.scripts.live_sensor_dashboard
```

Run the full synchronized data/policy path without opening CAN (the default):

```bash
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --headless --duration 10 --output data/static_dry_run.h5
```

Only after the preflight reports `"ready": true`, run a follower-only static
hold. This launches ABC/YAM followers but no GELLO leaders. Sensor synchronization
is verified before CAN is opened, the target is derived from the live startup
state, and the default first-action guard permits at most 0.05 in any action
coordinate (the bundled policy produces exactly zero delta):

```bash
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --execute-actions --headless --duration 10 \
  --output data/static_hold_001.h5
```

To deploy a different policy, pass only `--policy-dir /path/to/policy`. To add a
different robot, implement `RobotAdapter` and register it in
`piper_teleop/robot_utils/factory.py`; Aria, policy loading, recording, and eval
code remain unchanged.

## Data and Python integration

HDF5 schema `edith.yam.aria_gen2.v2`: one group per stream (`aria`, `top`, `left`,
`right`; plus `leader_left_actions`, `follower_left_state` and right equivalents
in teleop/attach mode). Each contains aligned row arrays `timestamp_ns`,
`received_ns`, `source_timestamp_ns`, `sequence`. Images are `rgb_jpeg` byte
arrays; decoding with OpenCV returns BGR, as usual. Joint streams use `values`
with six joint radians followed by the i2rt normalized gripper value. The
`clock` attribute documents each stream's timestamp meaning. Readers must use
root attribute `complete_rows` to ignore a partially written trailing row.

`piper_teleop.eval_utils.RobotEvaluationEnv.get_observation()` exposes the
existing nested evaluator interface, including `aria_rgb`, the three robot-view
aliases, calibrated gaze pixels, projected hand landmarks, raw named Aria
perception, and per-stream timing evidence. `get_policy_observation()` returns
the canonical flat policy contract. `YAMLocalEnv.get_bundle()` exposes the
complete timestamped data. Use a YAM-trained policy/ABC deployment for
autonomous inference; Piper-trained actions are not interchangeable.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover nearest-frame selection, full-bundle tolerance, stale/missing and
out-of-order frames, bounded buffering, failure propagation, Gen 2 capture-time
mapping, policy-directory isolation, robot-adapter validation, startup ordering,
evaluation observation compatibility, cleanup, and image/state recording. A
hardware trial with paired Aria Gen 2 and the three configured D405 cameras
produced 122 robot-anchored rows in 8.069 seconds (14.996 FPS); the top-camera
sequence advanced by exactly two and total spread stayed below 100 ms. No motor
commands were required for that validation. GELLO is not required by the static
policy path. `requirements-local-yam.lock.txt` is an environment snapshot; i2rt
must be installed separately from the station-matched checkout.

SDK references:
- https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/streaming
- https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/python-sdk/streaming-example
- https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/client-sdk/api
