# Zero-action observation diagnostic

The policy returns `zeros_like(observation["robot_state"])` for any joint count.
It performs no model inference and no network request. `observe_only: true`
makes the runner reject `--execute-actions` before opening devices. Zero
absolute joint positions are not a safe hold target for a real robot.

```bash
edith-eval --station configs/station.local.json \
  --policy-dir policies/zero_action_policy --duration 20 --headless \
  --output recordings/zero/observations.h5
edith-replay recordings/zero/observations.h5 \
  --output-dir recordings/zero/replay --play
```

`edith-collect` selects this policy by default. The no-op robot never imports
or opens the robot repository, including when a real adapter is named in the
station JSON. State is the configured initial pose, not measured encoders.

The recording includes every policy-loop observation/action, RGB views, gaze,
hand projections, raw Aria perception, all-channel int32 PCM, actual audio rate,
source/host timestamps, mute flags and any supplied transcript/operator event.
Cameras are JPEG encoded in the file. Audio remains lossless in HDF5; replay
exports microphone 0 with a documented gain and preserves timing gaps.
`verification.json` reports audio continuity, zero outputs and synchronization.
See [the main README](../../README.md) for the complete observation schema.
