"""Policy-facing observations built from one synchronized sensor bundle."""

from collections.abc import Callable, Mapping

import numpy as np


IMAGE_ALIASES = {
    "aria": "aria_rgb",
    "top": "top_camera_rgb",
    "left": "left_wrist_camera_rgb",
    "right": "right_wrist_camera_rgb",
}


def _images(bundle) -> dict[str, np.ndarray]:
    images = {}
    for name, sample in bundle.items():
        data = sample.data
        if isinstance(data, np.ndarray) and data.ndim == 3:
            images[IMAGE_ALIASES.get(name, f"{name}_rgb")] = data
    # Legacy VLM evaluators request a gaze crop. Until a policy-specific crop
    # transform is configured, expose the synchronized full Aria frame under
    # that optional key as a lossless fallback.
    if "aria_rgb" in images:
        images.setdefault("aria_cropped_rgb", images["aria_rgb"])
    return images


def _projection(bundle, projector):
    empty = {"gaze": None, "left": [None] * 21, "right": [None] * 21}
    if projector is None or "eye_gaze" not in bundle or "hand_pose" not in bundle:
        return empty
    return projector(bundle["eye_gaze"].data, bundle["hand_pose"].data)


def _pixel_array(point) -> np.ndarray:
    if point is None:
        return np.full(2, np.nan, dtype=np.float64)
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.shape != (2,):
        return np.full(2, np.nan, dtype=np.float64)
    return value


def build_observation(
    bundle,
    *,
    arm_order,
    fallback_state,
    instruction="",
    projector: Callable | None = None,
) -> dict:
    """Build the stable observation contract consumed by local policies.

    Images and Aria perception originate from the same accepted bundle. Robot
    state uses synchronized ABC samples when present and otherwise falls back
    to the provided startup snapshot (the static/no-CAN evaluation path).
    """

    state_by_arm = {}
    for name in arm_order:
        key = f"follower_{name}_state"
        if key in bundle:
            state_by_arm[name] = np.asarray(bundle[key].data, dtype=np.float64)
        else:
            state_by_arm[name] = np.asarray(
                fallback_state.positions[name], dtype=np.float64
            )
    state = np.concatenate([state_by_arm[name] for name in arm_order])
    projection = _projection(bundle, projector)
    hand_pixels = {
        side: np.stack([_pixel_array(point) for point in projection[side]])
        for side in ("left", "right")
    }
    return {
        "images": _images(bundle),
        "gaze_point": _pixel_array(projection["gaze"]),
        "hand_keypoints": hand_pixels,
        "aria_perception": {
            name: {
                "values": np.asarray(bundle[name].data, dtype=np.float64),
                "fields": tuple(bundle[name].fields),
            }
            for name in ("eye_gaze", "hand_pose")
            if name in bundle
        },
        "robot_state": state,
        "arm_order": tuple(arm_order),
        "robot_state_by_arm": state_by_arm,
        "capture_timestamps_ns": {
            name: int(sample.timestamp_ns) for name, sample in bundle.items()
        },
        "source_timestamps_ns": {
            name: int(sample.source_timestamp_ns) for name, sample in bundle.items()
        },
        "sequence_numbers": {
            name: int(sample.sequence) for name, sample in bundle.items()
        },
        "instruction": instruction,
    }


def legacy_observation(observation: Mapping, arm_order) -> dict:
    """Adapt a canonical observation to the existing Piper evaluator shape."""

    frame = dict(observation)
    frame["robot_state_flat"] = observation["robot_state"]
    robot_state = {}
    for name in arm_order:
        positions = np.asarray(observation["robot_state_by_arm"][name])
        robot_state[f"{name}_arm_positions"] = positions[:6]
        robot_state[f"{name}_gripper_position"] = positions[6:]
    frame["robot_state"] = robot_state
    return {"timestamp": frame}
