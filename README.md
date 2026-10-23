<div align="center">

<img src="assets/edith_logo.png" alt="EDITH" width="170">

# Teaching Robot to Follow Nonverbal Human Intent<br>from Egocentric Vision and Gaze

[Dongjun Lee](https://dgjun32.github.io/)<sup>\*1</sup> &middot;
[Juheon Choi](https://juheonchoi.com)<sup>\*1</sup> &middot;
[Dong Kyu Shin](https://donkeyu99.github.io/)<sup>\*1</sup> &middot;
[Sinjae Kang](https://sinjae.me/)<sup>1</sup> &middot;
[Kimin Lee](https://sites.google.com/view/kiminlee)<sup>1,2</sup>

<sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Config &nbsp;&nbsp; <sup>\*</sup>Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2606.10276-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.10276)
[![Project Page](https://img.shields.io/badge/Project-Page-4285F4.svg?logo=googlechrome&logoColor=white)](https://project-edith.github.io)
[![Project Aria Gen 2](https://img.shields.io/badge/Project_Aria-Gen_2-0866FF.svg?logo=meta&logoColor=white)](https://www.projectaria.com/)
![Bimanual YAM](https://img.shields.io/badge/Robot-Bimanual_YAM-222222.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img src="assets/overview_figure.png" alt="A language-conditioned policy needs a fully specified sentence; EDITH resolves &quot;Give me that metal profile&quot; from the wearer's egocentric view and gaze." width="100%">

</div>

## About the paper

EDITH lets a robot follow what a person **sees, looks at, and says**, captured
by smart glasses, instead of requiring a fully specified instruction.

- **About 2× the strongest baseline**: 48.6% average success vs. 25.0% for FAM-HRI, across three real-world tasks
- **Robust to distraction**: success holds when the human looks away (43.8% → 52.1%), while the baseline's nearly halves
- **Ego-view keyframes matter**: success drops from 43.8% to 2.1% without them

<details>
<summary>Main results (success / task progress %, 48 trials each)</summary>

| Method | Muffin-Serving | Tumbler-Sorting | Tool-Passing | **Average** |
| --- | :-: | :-: | :-: | :-: |
| π<sub>0.5</sub> (language only) | 0.0 / 12.2 | 0.0 / 26.2 | 12.5 / 31.3 | 4.2 / 23.2 |
| π<sub>0.5</sub> + egocentric context | 2.1 / 25.7 | 0.0 / 42.4 | 16.7 / 29.7 | 6.2 / 32.6 |
| VLM planner + π<sub>0.5</sub> (Hi Robot-style) | 2.1 / 15.6 | 4.2 / 36.5 | 14.6 / 40.6 | 6.9 / 30.9 |
| FAM-HRI + π<sub>0.5</sub> | 22.9 / 50.3 | 14.6 / 61.1 | 37.5 / 59.9 | 25.0 / 57.1 |
| **EDITH (ours)** | **43.8 / 70.8** | **39.6 / 77.8** | **62.5 / 79.2** | **48.6 / 75.9** |

</details>

## 1. Setup

Requires Python 3.11 and the station's ABC/i2rt checkout for the YAM arms.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-local-yam.txt
uv pip install --python .venv/bin/python -e /path/to/abc-robot-setting/i2rt
uv pip install --python .venv/bin/python -e '.[policies]'

aria_gen2 auth pair                                                             # once, to pair the glasses
.venv/bin/python -m edith.scripts.eval_policy_local --check --headless   # device check
.venv/bin/python -m edith.scripts.live_sensor_dashboard                  # live view
```

Station setup and safety notes: [README_LOCAL_YAM.md](README_LOCAL_YAM.md).

## 2. The environment

A bimanual YAM robot and Project Aria Gen 2 glasses, both on one computer (the
glasses connect over USB). One call returns a single time-synced observation:

```python
from edith.envs.yam_env import YAMLocalEnv
from edith.eval_utils import RobotEvaluationEnv
from edith.robot_utils.abc_config import load_station
from edith.robot_utils.abc_yam import ConfiguredNoOpRobot

config = load_station()              # ABC station config (default ~/dev/abc-robot-setting/abc; pass a path otherwise)
sensors = YAMLocalEnv(config)        # starts the 3 RealSense cameras and the Aria glasses
robot = ConfiguredNoOpRobot(config)  # never opens CAN or moves the arms
env = RobotEvaluationEnv(sensors, robot, robot.read(0))

try:
    obs = env.get_policy_observation()
finally:
    env.safe_end()
```

| Key | Shape | Contents |
| --- | --- | --- |
| `images["aria_rgb"]` | `(H, W, 3)` | Human's first-person view |
| `gaze_point` | `(2,)` | Gaze pixel in `aria_rgb` (NaN if invalid) |
| `hand_keypoints["left" / "right"]` | `(21, 2)` | Hand landmarks in `aria_rgb` |
| `audio["samples"]` | `(N, C)` int32 | Glasses microphone, all audio since the previous observation (48 kHz) |
| `aria_perception` | dict | Raw eye-gaze (35) and hand-pose (156) values with field names |
| `images["top_camera_rgb" / "left_wrist_camera_rgb" / "right_wrist_camera_rgb"]` | `(H, W, 3)` | Robot cameras |
| `robot_state` | `(14,)` | Per arm: 6 joints (rad) + gripper |
| `capture_timestamps_ns` | dict | Capture time of each stream |

All streams share one clock and are aligned within 100 ms at 15 Hz (software
sync, not a hardware trigger). Speech-to-text isn't built in: pass text with
`--transcript-file` and it appears as `obs["transcript"]`.

With `ConfiguredNoOpRobot`, `robot_state` is the station's configured initial
pose. The live arm pose requires running the follower arms
(`eval_policy_local --execute-actions`, section 4).

## 3. Data collection 🚧

> Under construction: the recording format and teleop workflow may change.

Record synchronized observations to HDF5 (cameras and glasses only, no motors):

```bash
.venv/bin/python -m edith.scripts.collect_trajectory_yam_local \
  --mode preview --output recordings/episode_001.h5
```

To also record YAM joint state and commands while teleoperating with GELLO
leaders, use `--mode teleop --leaders leaders.json`.

Each stream (`aria`, `eye_gaze`, `hand_pose`, `audio`, `top`, `left`, `right`,
plus joint streams in teleop mode) is stored as its own HDF5 group with
per-row timestamps. Read only rows below the root `complete_rows` attribute.

## 4. Training and evaluation

### Training 🚧

> Under construction: there is no training code in this repo yet.

For now, train a policy outside this repo (e.g., π<sub>0.5</sub> with OpenPI)
on YAM data, and serve it as an OpenPI-compatible server at
`ws://127.0.0.1:8000`. [policies/README.md](policies/README.md) lists the
metadata and action format the server must provide.

### Evaluation

Each policy is a directory under [`policies/`](policies):

| Policy | Inputs |
| --- | --- |
| `policy_edith` | Robot views and state, plus an ego-view keyframe and subtask (the paper's method) |
| `fam_hri` | FAM-HRI baseline: gaze- and speech-based intent inference |
| `vlm_textvla` | VLM planner baseline: robot views and planner-generated subtask text |
| `streamvla` | Ego + language baseline: robot views, state, text and the gaze-cropped Aria view |
| `textvla` | Language-only baseline: robot views, state and text |
| `static_hold` | Holds the current pose (hardware smoke test) |

Run a policy. Motors stay off unless you add `--execute-actions`:

```bash
.venv/bin/python -m edith.scripts.eval_policy_local \
  --policy-dir policies/policy_edith \
  --policy-config policies/policy_edith/presets/passing_tools.json --headless
```

To add your own policy, create a directory containing `policy.json` and
`policy.py`, then pass it with `--policy-dir`:

```json
{"schema": "edith.policy.v1", "name": "my-policy", "entrypoint": "policy.py:Policy",
 "control_hz": 15, "action_space": "joint_position", "config": {}}
```

```python
class Policy:
    def __init__(self, policy_dir, config): ...

    def act(self, obs):            # obs is the observation from section 2
        return obs["robot_state"]  # (14,) joint-position target
```

## Citation

```bibtex
@article{lee2026hierarchicalpoliciesverbalegocentric,
      title={Teaching Robot to Follow Nonverbal Human Intent from Egocentric Vision and Gaze},
      author={Dongjun Lee and Juheon Choi and Dong Kyu Shin and Sinjae Kang and Kimin Lee},
      year={2026},
      url={https://arxiv.org/abs/2606.10276},
}
```
