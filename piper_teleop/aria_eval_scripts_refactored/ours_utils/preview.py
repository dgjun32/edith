"""OpenCV preview composition helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _draw_lines(img, lines, origin, font_scale, color, thickness=1,
                line_gap=18):
    if isinstance(lines, str):
        lines = lines.splitlines()
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(
            img,
            str(line),
            (x, y + idx * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def build_preview_frame(obs, status_text, status_color=(0, 255, 0)):
    images = obs["images"]
    human_view = images["aria_rgb"].copy()
    top_view = images["top_camera_rgb"].copy()

    gaze_point = np.asarray(obs["gaze_point"], dtype=np.float32).reshape(-1)
    if gaze_point.size >= 2 and np.all(np.isfinite(gaze_point[:2])):
        x = int(np.clip(gaze_point[0], 0, human_view.shape[1] - 1))
        y = int(np.clip(gaze_point[1], 0, human_view.shape[0] - 1))
        cv2.circle(human_view, (x, y), 6, (0, 255, 0), 2)
        cv2.line(human_view, (x - 12, y), (x + 12, y), (0, 255, 0), 2)
        cv2.line(human_view, (x, y - 12), (x, y + 12), (0, 255, 0), 2)

    if top_view.shape[:2] != human_view.shape[:2]:
        top_view = cv2.resize(
            top_view,
            (human_view.shape[1], human_view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    preview = np.hstack((human_view, top_view))
    _draw_lines(preview, "human(gaze) | top", (10, 22), 0.5,
                (255, 255, 255), thickness=1, line_gap=16)
    _draw_lines(preview, status_text, (10, 44), 0.45,
                status_color, thickness=1, line_gap=16)
    return preview


def build_eval_preview(
    obs,
    keyframe_img: np.ndarray,
    queue: Any,
    current_instruction: str = "",
):
    human_view = obs["images"]["aria_rgb"].copy()
    if keyframe_img is None:
        kf_display = np.zeros_like(human_view)
    else:
        kf_display = cv2.resize(
            keyframe_img,
            (human_view.shape[1], human_view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    preview = np.hstack((human_view, kf_display))
    total = getattr(queue, "total", 0)
    current_idx = getattr(queue, "current_idx", -1)
    current_num = min(current_idx + 1, total) if total else 0

    _draw_lines(preview, f"LIVE | KF [{current_num}/{total}]",
                (10, 22), 0.5, (255, 255, 255), thickness=1, line_gap=16)
    _draw_lines(preview, ("n=next keyframe", "e=end episode  q=quit"),
                (10, 44), 0.45, (0, 255, 255), thickness=1, line_gap=16)
    if current_instruction:
        _draw_lines(preview, f"subtask: {current_instruction[:70]}",
                    (10, 80), 0.42, (255, 255, 255),
                    thickness=1, line_gap=15)
    return preview
