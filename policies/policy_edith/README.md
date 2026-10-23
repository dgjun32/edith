# policy_edith — local Aria Gen 2 + YAM

This is the supported directory policy for
`eval_openpi_ours_with_voice_260503.py`. The original filename now forwards to
`edith/scripts/policy_edith.py`; it cannot start the old external Aria
publisher or `PiperRobotEnv`. `policies/edith` is a compatibility symlink to
this directory.

## Local connection path

- Aria: `aria.sdk_gen2.DeviceClient.usb_network_devices()` discovers the glasses
  attached to this computer; capture explicitly selects `USB_NCM`.
- Robot views: this computer's three USB RealSense cameras, selected by ABC
  station serial numbers.
- YAM state/control: the local ABC adapter and CAN interfaces.
- Synchronization: SDK capture-time mapping and RealSense timestamps feed the
  shared local synchronizer. The policy receives one aligned observation.
- VLA: `ws://127.0.0.1:8000` by default, on this same deployment computer.

USB-NCM is IP traffic over the USB cable. The SDK HTTP receiver listens on port
6768 so the glasses can send frames to this computer; replacing it with a
loopback-only receiver would prevent USB glasses from reaching it. This is
independent of the old Aria publisher at `SERVER_IP:9999`. No Wi-Fi or external
streaming computer is required. See the
[official USB streaming guide](https://facebookresearch.github.io/projectaria_tools/gen2/ark/client-sdk/streaming).

Pair/authenticate once using the SDK, then let the runner own the stream.
Automatic USB discovery is the default. `--aria-serial` selects among multiple
glasses; `--aria-ip <GLASSES_USB_NCM_IP>` can bypass mDNS using the glasses' USB
address. It is not the address of a remote host or a Wi-Fi camera.

## Run

From the repository root, with dependencies and a YAM-compatible checkpoint
server installed:

```bash
# Validate only; no inference, sensors or CAN.
.venv/bin/python -m edith.scripts.policy_edith --check-policy-only

# Discover devices without opening sensor streams or commanding motors.
.venv/bin/python -m edith.scripts.policy_edith --check --headless

# Local synchronized capture and inference, without CAN commands.
.venv/bin/python -m edith.scripts.policy_edith \
  --policy-config policies/policy_edith/presets/passing_tools.json \
  --duration 60 --headless
```

Motor execution requires `--execute-actions`. The generic runner works too:
`python -m edith.scripts.eval_policy_local --policy-dir policies/policy_edith`.
The original filename accepts these same current flags; use `--help`.
Legacy keyboard controls and experiment-specific MP4 options are not carried
over; recording uses `--output`, and queue updates use `--policy-events-file`.

Voice text is read through `--transcript-file /path/to/latest-transcript.txt`,
which a local speech-to-text process can update. This policy does not yet
capture Gen 2 microphone audio or run Whisper itself.

The default manual/preset planner and local VLA do not require internet.
Selecting a `gemini_*.json` preset explicitly enables cloud planning and still
requires internet, `GEMINI_API_KEY`, and a configured model. Removing Wi-Fi
from camera transport does not make Gemini an offline model.

See the [policy migration and usage guide](../README.md) for presets, the
14-value YAM action contract, checkpoint metadata and queue behavior.
Hardware-free tests cover USB transport selection and local launch paths.
Live Aria/model/robot validation must be done with the device and checkpoint.
