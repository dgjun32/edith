# Optional ABC/YAM integration

The default Edith runner is robot-independent. Aria Gen 2, RealSense capture,
audio, replay and no-motion policy checks do not require ABC or i2rt. Start with
[README.md](README.md) and a standalone station JSON. This guide applies only
to the optional ABC/YAM controller and GELLO teleoperation integration.

## Install in the robot environment

Install Edith and the pinned i2rt driver into the robot's supported Python
environment, or into a dedicated YAM environment:

```bash
cd /path/to/edith
git submodule update --init i2rt   # i2rt YAM motor driver, pinned at v1.3.5
uv pip install --python /path/to/yam-env/bin/python \
  -e '.[aria,realsense,policies,replay,yam]' \
  -e i2rt
```

The `i2rt/` submodule is upstream
[i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) pinned at v1.3.5
(`5b72c47`), the same version the robot server runs, with no local changes.
Do not edit files inside `i2rt/`. To change the driver version, check out the
new commit inside `i2rt/`, then commit the updated submodule pointer in Edith.
(i2rt's own package metadata reports version 1.1.2; the pinned commit is what
identifies the release.) The ABC deployment (`abc-robot-setting/abc`) is still
a separate checkout, passed with `--abc-root`.

The `yam` extra installs the IPC/CLI/GELLO Python dependencies, not a robot
repository or controller firmware. `requirements-local-yam*.txt` are historical
YAM environment snapshots, not requirements for core Edith capture.

`--robot abc-yam --abc-root /path/to/abc-robot-setting/abc` explicitly selects
the compatibility adapter. It loads the ABC `--profile abc_station_config`
(camera serials, follower CAN mapping, initial poses, and gripper settings)
instead of `--station`. The generic module factory interface in the main README
is preferred for new robot integrations; it requires no registry edit.

## USB observations and audio

Aria capture selects `StreamingInterface.USB_NCM` on this computer. Pair once
with `aria_gen2 auth pair`. No Wi-Fi or external Aria publisher is required.
The SDK's TLS receiver on port 6768 must be reachable by the glasses over the
USB network interface. Automatic USB discovery is the default; `--aria-serial`
selects glasses, and `--aria-ip` is the glasses' USB-NCM address if discovery
needs to be bypassed. It is not an external camera server address.

Do not run another Aria receiver or camera owner concurrently. The default
`mp_streaming_demo` profile supplies RGB, gaze, hands and microphone audio.
Aria TicSync and RealSense capture timestamps map to host `perf_counter_ns`.
The 30 Hz master camera anchors 15 Hz observations; lower-rate Aria samples may
be reused with their original sequence/timestamp. Total camera/perception
spread is bounded by `--tolerance-ms 100`; stale/missing streams fail the run.
Audio is a continuous interval through each anchor, with all channels and
per-frame timestamps. See the main README for schema and replay details.

For a YAM station whose camera serials have been copied into standalone JSON,
this records the policy path without importing ABC or opening CAN:

```bash
.venv/bin/edith-collect --station configs/station.local.json \
  --duration 20 --headless --output recordings/zero/observations.h5
.venv/bin/edith-replay recordings/zero/observations.h5 \
  --output-dir recordings/zero/replay --play
```

The diagnostic robot state is the configured pose, not live encoders. The
zero-action policy rejects `--execute-actions`; zero absolute positions are
not a hold command.

## Follower-only policy evaluation

The following commands use the YAM environment's Python, from the Edith root:

```bash
# Read-only discovery; no camera streams or motor initialization.
python -m edith.scripts.eval_policy_local \
  --robot abc-yam --abc-root /path/to/abc-robot-setting/abc --check --headless

# Full sensor/policy path, configured state, no motors.
python -m edith.scripts.eval_policy_local \
  --robot abc-yam --abc-root /path/to/abc-robot-setting/abc \
  --policy-dir static_hold --duration 10 --headless

# Explicit hardware execution: starts ABC/YAM followers and their startup routine.
python -m edith.scripts.eval_policy_local \
  --robot abc-yam --abc-root /path/to/abc-robot-setting/abc \
  --policy-dir static_hold --execute-actions \
  --duration 10 --headless --output recordings/hold.h5
```

The runner verifies sensors before starting followers, then obtains a fresh
bundle with live robot state before inference. Static hold repeats the startup
state; the first-action guard defaults to 0.05 maximum absolute delta. ABC
startup may itself interpolate/calibrate the arms. No GELLO leader is needed.
Shutdown closes i2rt with zero torques rather than moving to zero; support the
arms when stopping. ABC timestamps are publication times, not encoder hardware
capture times. The live-state adapter maps them into the local synchronizer.

Learned policies need a YAM-trained checkpoint with the documented 14-value
action order and matching normalization. Piper-trained actions are not
interchangeable. Pass `--policy-dir policy_edith` and an appropriate
preset to select that client. Audio is exposed to the policy, but voice text
still comes from `--transcript-file` unless a separate STT process is supplied.

## GELLO collection (optional legacy ABC adapter)

The dedicated collector remains available for existing ABC deployments:

```bash
python -m edith.scripts.collect_trajectory_yam_local \
  --abc-root /path/to/abc-robot-setting/abc --check
python -m edith.scripts.collect_trajectory_yam_local \
  --abc-root /path/to/abc-robot-setting/abc \
  --mode teleop --leaders leaders.json --output recordings/teleop.h5
```

Supply the actual calibrated GELLO configuration, including device paths,
servo IDs and joint signs. The example ABC station's leader paths are
placeholders. ABC expects calibrated startup poses and the documented USB
serial latency. Teleop verifies synchronized sensors first, then launches the
arm processes; missing leader commands trigger its follower timeout. Stop
other controllers owning the same cameras, IPC topics or CAN interfaces.

`--mode preview` records sensors without starting motors. `--mode attach`
subscribes to already-running ABC leader/follower IPC topics and does not own
or stop that external controller. This specialized collector uses
`YAMLocalEnv`; the default `edith-collect` uses the generic environment and
zero-action policy instead. Both use the shared audio recorder.

## Verification

```bash
python -m unittest discover -s tests -v
# Only in the installed ABC robot environment:
EDITH_TEST_ABC=1 python -m unittest discover -s tests -v
```

Core tests run with robot imports blocked, exercise an external three-joint
adapter, reject motor execution for zero-action diagnostics, and check PCM,
timestamps, mute flags and policy outputs through HDF5. Live observation tests
require no motor commands; controller startup and learned checkpoint behavior
must be validated separately on the chosen robot.

SDK references:
- [USB streaming](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/streaming)
- [Python streaming example](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/python-sdk/streaming-example)
- [Gen 2 API](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/client-sdk/api)
