"""Video-only helpers for the ours eval scripts."""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import imageio
import numpy as np

from eval_vlm_baseline import _draw_caption, _ffmpeg_reencode_h264


def write_silent_video(frames_bgr, output_path: str, fps: float):
    if not frames_bgr:
        return
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    imageio.mimwrite(output_path, frames_rgb, fps=fps)


def subsample_frames(frames: List[np.ndarray], src_fps: float,
                     target_fps: float) -> List[np.ndarray]:
    """Sub-sample a frame list from src_fps down to target_fps."""
    if target_fps >= src_fps or target_fps <= 0:
        return frames
    step = src_fps / target_fps
    result = []
    next_sample = 0.0
    for i in range(len(frames)):
        if i >= next_sample:
            result.append(frames[i])
            next_sample += step
    return result


def crop_around_gaze(frame: np.ndarray, gx: float, gy: float,
                     crop_size: int) -> np.ndarray:
    """Crop crop_size x crop_size around (gx, gy), clamped to frame bounds."""
    h, w = frame.shape[:2]
    if crop_size >= min(h, w):
        return frame
    half = crop_size // 2
    cx, cy = int(round(gx)), int(round(gy))
    x1 = max(0, min(cx - half, w - crop_size))
    y1 = max(0, min(cy - half, h - crop_size))
    return frame[y1:y1 + crop_size, x1:x1 + crop_size]


def frames_to_video_bytes(
    frames: List[np.ndarray],
    fps: float,
    resolution: Optional[Tuple[int, int]] = None,
) -> bytes:
    """Encode a list of BGR frames to H264 video bytes."""
    if not frames:
        return b""
    if resolution:
        out_w, out_h = resolution
    else:
        out_h, out_w = frames[0].shape[:2]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (out_w, out_h))
    for f in frames:
        if f.shape[1] != out_w or f.shape[0] != out_h:
            f = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        writer.write(f)
    writer.release()

    h264_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h264_tmp.close()
    _ffmpeg_reencode_h264(tmp.name, h264_tmp.name)
    os.unlink(tmp.name)

    with open(h264_tmp.name, "rb") as f:
        result = f.read()
    os.unlink(h264_tmp.name)
    return result


def draw_gaze_crosshair(img_bgr: np.ndarray) -> np.ndarray:
    """Overlay a highly-visible gaze marker at the image centre."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(12, h // 20)

    cv2.circle(out, (cx, cy), r + 4, (0, 0, 255), 3)
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 3)
    cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), (0, 255, 0), 3)
    cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), (0, 255, 0), 3)

    label = "GAZE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, h / 400.0)
    (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
    lx = cx - tw // 2
    ly = min(cy + r * 2 + th + 6, h - 4)
    cv2.rectangle(out, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 4),
                  (0, 0, 0), -1)
    cv2.putText(out, label, (lx, ly), font, scale, (0, 255, 255), 2,
                cv2.LINE_AA)
    return out


def get_frame_aligned_caption(
    words: list,
    time_sec: float,
    fps: float,
    transcript_text: str = "",
) -> str:
    """Return a word-level caption aligned to the current video frame."""
    if words:
        frame_duration_sec = 1.0 / max(float(fps), 1.0)
        half_window_sec = max(0.03, frame_duration_sec * 0.55)
        window_start = max(0.0, time_sec - half_window_sec)
        window_end = time_sec + half_window_sec

        visible_words = []
        for word in words:
            start_sec = float(word.get("start", 0.0) or 0.0)
            end_sec = float(word.get("end", start_sec) or start_sec)
            if end_sec < start_sec:
                end_sec = start_sec
            if end_sec >= window_start and start_sec <= window_end:
                token = str(word.get("word", "")).strip()
                if token:
                    visible_words.append(token)

        if visible_words:
            return " ".join(visible_words)

        hold_sec = min(0.18, max(0.04, frame_duration_sec * 0.75))
        for word in reversed(words):
            start_sec = float(word.get("start", 0.0) or 0.0)
            end_sec = float(word.get("end", start_sec) or start_sec)
            if end_sec < start_sec:
                end_sec = start_sec
            if end_sec <= time_sec and time_sec - end_sec <= hold_sec:
                token = str(word.get("word", "")).strip()
                if token:
                    return token
            if end_sec < time_sec - hold_sec:
                break
        return ""

    return (transcript_text or "").strip()


def burn_captions_on_video_bytes(
    video_bytes: bytes,
    words: list,
    transcript_text: str,
    fps: float,
) -> bytes:
    """Burn word-level captions onto video frames. Returns re-encoded H264 bytes."""
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(video_bytes)
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out.name, fourcc, vid_fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        caption = get_frame_aligned_caption(
            words,
            time_sec=frame_idx / vid_fps,
            fps=vid_fps,
            transcript_text=transcript_text,
        )
        if caption:
            _draw_caption(frame, caption)
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    os.unlink(tmp_in.name)

    h264_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h264_tmp.close()
    _ffmpeg_reencode_h264(tmp_out.name, h264_tmp.name)
    os.unlink(tmp_out.name)

    with open(h264_tmp.name, "rb") as f:
        result = f.read()
    os.unlink(h264_tmp.name)
    return result


def overlay_captions_on_frames(
    frames_bgr: List[np.ndarray],
    words: list,
    transcript_text: str,
    clip_start_sec: float,
    fps: float,
) -> List[np.ndarray]:
    """Overlay Whisper captions on in-memory frames for debug video saving."""
    captioned_frames = []
    for frame_idx, frame in enumerate(frames_bgr):
        overlay = frame.copy()
        caption = get_frame_aligned_caption(
            words,
            time_sec=frame_idx / max(float(fps), 1.0),
            fps=fps,
            transcript_text=transcript_text,
        )
        if caption:
            _draw_caption(overlay, caption)
        captioned_frames.append(overlay)
    return captioned_frames
