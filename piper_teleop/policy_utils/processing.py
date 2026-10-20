"""Canonical RGB and explicit YAM state layout shared by the migrated policies."""

from pathlib import Path
import cv2
import numpy as np

ROBOT_IMAGES = {
    "observation.images.top_view": "top_camera_rgb",
    "observation.images.left_wrist_view": "left_wrist_camera_rgb",
    "observation.images.right_wrist_view": "right_wrist_camera_rgb",
}


def local_file(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Policy assets must stay inside the policy directory")
    return path


def rgb(image):
    image = np.asarray(image)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("Policy images must be nonempty HxWx3 uint8 RGB")
    return image


def gaze_crop(image, gaze, size=512, invalid="full"):
    image = rgb(image)
    gaze = np.asarray(gaze if gaze is not None else [np.nan, np.nan], dtype=float)
    h, w = image.shape[:2]
    if (
        gaze.shape != (2,)
        or not np.isfinite(gaze).all()
        or not (0 <= gaze[0] < w and 0 <= gaze[1] < h)
    ):
        if invalid == "error":
            raise ValueError("No valid calibrated gaze point for gaze crop")
        if invalid != "full":
            raise ValueError("invalid_gaze must be full or error")
        return image.copy()
    if type(size) is not int or size <= 0:
        raise ValueError("crop_size must be a positive integer")
    cw, ch = min(size, w), min(size, h)
    x = int(np.clip(round(gaze[0]) - cw // 2, 0, w - cw))
    y = int(np.clip(round(gaze[1]) - ch // 2, 0, h - ch))
    return image[y : y + ch, x : x + cw].copy()


def state(observation):
    q = np.asarray(observation["robot_state"], dtype=np.float64)
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError(
            "Expected 14 finite YAM positions: left[6 joints, gripper], right[6 joints, gripper]"
        )
    order = observation.get("arm_order")
    if order is not None and tuple(order) != ("left", "right"):
        raise ValueError("These checkpoints require arm_order=(left, right)")
    return q


def packet(observation, instruction, image_size=224):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    result = {
        "prompt": instruction.strip(),
        "observation.state": state(observation).copy(),
    }
    for dest, source in ROBOT_IMAGES.items():
        image = rgb(observation["images"][source])
        result[dest] = resize(image, image_size)
    return result


def resize(image, size):
    image = rgb(image)
    if size is None:
        return image.copy()
    if type(size) is not int or size <= 0:
        raise ValueError("image_size must be a positive integer or null")
    # Letterbox, matching OpenPI resize_with_pad. Never swap RGB channels.
    h, w = image.shape[:2]
    ratio = max(w / size, h / size)
    nh, nw = max(1, int(h / ratio)), max(1, int(w / ratio))
    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    result[y : y + nh, x : x + nw] = small
    return result
