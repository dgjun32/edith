"""VLM_video_preprocess.py

Standalone streaming capture tool for paper figures.
No robot operation — records aria egocentric video and saves four variants:

  1. raw.mp4               — full aria_rgb, no gaze overlay
  2. raw_gaze.mp4          — full aria_rgb, gaze crosshair overlaid
  3. cropped_gaze.mp4      — gaze-cropped NxN region, gaze crosshair overlaid
  4. cropped_transcript.mp4— gaze-cropped NxN region, Whisper transcript overlaid

Controls
--------
  p  — start recording
  p  — stop recording and save all four videos
  q  — quit without saving
"""

import os
import sys
import cv2
import json
import subprocess
import tempfile
import numpy as np
from typing import List, Optional

_FFMPEG = "/home/alin/miniconda3/envs/piper/bin/ffmpeg"

# ensure the package root is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer
from common.config import SERVER_IP

from ours_utils.audio_processing import (
    transcribe_audio_records_for_whisper as _transcribe_audio_records_for_whisper,
)
from ours_utils.keyboard import (
    poll_keypress as _poll_keypress,
    terminal_keypress_mode as _terminal_keypress_mode,
)
from ours_utils.video_processing import (
    crop_around_gaze as _crop_around_gaze,
    overlay_captions_on_frames as _overlay_captions_on_frames,
)


def _write_video(frames, path, fps):
    """Write frames as H264 mp4 via ffmpeg for broad playback compatibility."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    subprocess.run(
        [_FFMPEG, "-y", "-i", tmp.name,
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        check=True, capture_output=True,
    )
    os.unlink(tmp.name)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

VIDEO_RECORD_FPS         = 15
GAZE_CALIBRATION_Y_OFFSET = 0        # px correction for upward calibration bias
OUTPUT_DIR               = os.path.join(os.path.dirname(__file__), "figure_VLM_preproc")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_gaze_crosshair(img_bgr: np.ndarray) -> np.ndarray:
    """Thin crosshair at image centre, no text label."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(8, h // 24)
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 1)
    cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), (0, 255, 0), 1)
    return out


def _draw_gaze_dot_on_full(frame: np.ndarray, gx: float, gy: float) -> np.ndarray:
    """Draw a visible gaze crosshair at the actual (gx, gy) pixel on the full frame."""
    out = frame.copy()
    cx, cy = int(round(gx)), int(round(gy))
    h, w = out.shape[:2]
    r = max(8, h // 60)
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 1)
    cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), (0, 255, 0), 1)
    cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), (0, 255, 0), 1)
    return out


def _save_four_variants(
    aria_frames:   List[np.ndarray],
    gaze_records:  List[dict],
    audio_records: List[dict],
    output_dir:    str,
    fps:           float,
):
    os.makedirs(output_dir, exist_ok=True)
    n = len(aria_frames)
    print(f"[Save] {n} frames @ {fps} FPS → {output_dir}")

    # ── 1. raw: full frame, no overlay ──────────────────────────────────────
    path1 = os.path.join(output_dir, "raw.mp4")
    _write_video(aria_frames, path1, fps=fps)
    print(f"[Save] 1/4 raw.mp4 → {path1}")

    # ── 2. raw_gaze: full frame, gaze dot at actual gaze position ───────────
    raw_gaze_frames = []
    for i, frame in enumerate(aria_frames):
        rec = gaze_records[i] if i < len(gaze_records) else {}
        gx = rec.get("gaze_x")
        gy = rec.get("gaze_y")
        if gx is not None and gy is not None:
            raw_gaze_frames.append(_draw_gaze_dot_on_full(frame, gx, gy))
        else:
            raw_gaze_frames.append(frame.copy())
    path2 = os.path.join(output_dir, "raw_gaze.mp4")
    _write_video(raw_gaze_frames, path2, fps=fps)
    print(f"[Save] 2/4 raw_gaze.mp4 → {path2}")

    # ── 3. cropped_gaze: gaze crop (1/2 image size) + centre crosshair ──────
    cropped_gaze_frames = []
    for i, frame in enumerate(aria_frames):
        h, w = frame.shape[:2]
        crop_size = min(h, w) // 2
        rec = gaze_records[i] if i < len(gaze_records) else {}
        gx = rec.get("gaze_x")
        gy = rec.get("gaze_y")
        if gx is not None and gy is not None:
            gy_adj = gy + GAZE_CALIBRATION_Y_OFFSET
            crop = _crop_around_gaze(frame, gx, gy_adj, crop_size)
        else:
            half = crop_size // 2
            cx, cy = w // 2, h // 2
            crop = frame[
                max(0, cy - half):max(0, cy - half) + crop_size,
                max(0, cx - half):max(0, cx - half) + crop_size,
            ]
        cropped_gaze_frames.append(_draw_gaze_crosshair(crop))
    path3 = os.path.join(output_dir, "cropped_gaze.mp4")
    _write_video(cropped_gaze_frames, path3, fps=fps)
    print(f"[Save] 3/4 cropped_gaze.mp4 → {path3}")

    # ── 4. cropped_transcript: gaze crop + Whisper transcript overlay ────────
    print("[Save] 4/4 Running Whisper transcription...")
    transcript_data = _transcribe_audio_records_for_whisper(
        audio_records, context="figure_preproc")
    transcript_text = transcript_data.get("text", "")
    words = transcript_data.get("words", [])
    print(f"[Save]     transcript: \"{transcript_text[:120]}\"")

    # use the raw crops (without crosshair) as the base for transcript variant
    cropped_base = []
    for i, frame in enumerate(aria_frames):
        h, w = frame.shape[:2]
        crop_size = min(h, w) // 2
        rec = gaze_records[i] if i < len(gaze_records) else {}
        gx = rec.get("gaze_x")
        gy = rec.get("gaze_y")
        if gx is not None and gy is not None:
            gy_adj = gy + GAZE_CALIBRATION_Y_OFFSET
            cropped_base.append(_crop_around_gaze(frame, gx, gy_adj, crop_size))
        else:
            half = crop_size // 2
            cx, cy = w // 2, h // 2
            cropped_base.append(frame[
                max(0, cy - half):max(0, cy - half) + crop_size,
                max(0, cx - half):max(0, cx - half) + crop_size,
            ])

    transcript_frames = _overlay_captions_on_frames(
        [_draw_gaze_crosshair(f) for f in cropped_base],
        words, transcript_text,
        clip_start_sec=0.0, fps=fps)
    path4 = os.path.join(output_dir, "cropped_transcript.mp4")
    _write_video(transcript_frames, path4, fps=fps)
    print(f"[Save] 4/4 cropped_transcript.mp4 → {path4}")

    # save gaze json alongside
    gaze_path = os.path.join(output_dir, "gaze.json")
    with open(gaze_path, "w") as f:
        json.dump({"frames": gaze_records}, f, indent=2)
    print(f"[Save] gaze.json → {gaze_path}")
    print("[Save] Done.")


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def main():
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_9999",
    }

    env = PiperRobotEnv(camera_ids=camera_ids)
    capture_timer = Timer(VIDEO_RECORD_FPS)
    window_name = "VLM Preproc  |  p=start/stop  q=quit"

    aria_frames:   List[np.ndarray] = []
    gaze_records:  List[dict]       = []
    audio_records: List[dict]       = []
    capturing = False

    def _clear():
        aria_frames.clear()
        gaze_records.clear()
        audio_records.clear()

    print("VLM video preprocessor — paper figure capture")
    print("  p = start recording")
    print("  p = stop and save 4 video variants")
    print("  q = quit")

    try:
        with _terminal_keypress_mode():
            while True:
                live_obs = env.get_observation(
                    mode='human_view_streamed_and_keyframe_cropped')
                live_obs = live_obs[next(reversed(live_obs))]

                aria_bgr = live_obs["images"]["aria_rgb"].copy()
                gaze_pt  = np.asarray(
                    live_obs.get("gaze_point", [None, None]),
                    dtype=object).reshape(-1)
                gx = float(gaze_pt[0]) if gaze_pt.size >= 1 and gaze_pt[0] is not None else None
                gy = float(gaze_pt[1]) if gaze_pt.size >= 2 and gaze_pt[1] is not None else None

                audio_obs = live_obs.get("audio", {})
                audio_chunk = np.asarray(
                    audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
                    dtype=np.int16)
                audio_ts_ns          = int(audio_obs.get("audio_ts_ns", 0) or 0)
                audio_sample_rate_hz = int(audio_obs.get("sample_rate_hz", 0) or 0)
                audio_num_samples    = int(audio_obs.get("num_samples", audio_chunk.size) or 0)

                if capturing:
                    frame_idx = len(gaze_records)
                    aria_frames.append(aria_bgr)
                    gaze_records.append({"frame_idx": frame_idx, "gaze_x": gx, "gaze_y": gy})
                    audio_records.append({
                        "frame_idx":        frame_idx,
                        "audio_ts_ns":      audio_ts_ns,
                        "sample_rate_hz":   audio_sample_rate_hz,
                        "num_samples":      audio_num_samples,
                        "chunk_pcm16":      audio_chunk.copy(),
                    })

                # live preview: show raw gaze dot so operator can verify
                preview = aria_bgr.copy()
                if gx is not None and gy is not None:
                    preview = _draw_gaze_dot_on_full(preview, gx, gy)
                label = (f"RECORDING  {len(aria_frames) / VIDEO_RECORD_FPS:.1f}s  (p=stop  q=quit)"
                         if capturing else "READY  (p=start  q=quit)")
                color = (0, 0, 255) if capturing else (0, 255, 0)
                cv2.putText(preview, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.imshow(window_name, preview)

                key = _poll_keypress()
                capture_timer.sleep()

                if key in (ord('q'), ord('Q'), 27):
                    print("Quit — no files saved.")
                    break

                if key in (ord('p'), ord('P')):
                    if not capturing:
                        _clear()
                        capturing = True
                        print("[Capture] Recording started. Press 'p' to stop.")
                    elif aria_frames:
                        capturing = False
                        print(f"[Capture] Stopped. {len(aria_frames)} frames "
                              f"({len(aria_frames) / VIDEO_RECORD_FPS:.1f}s)")
                        _save_four_variants(
                            aria_frames, gaze_records, audio_records,
                            OUTPUT_DIR, VIDEO_RECORD_FPS)
                        _clear()
                        print("[Capture] Ready for another recording. p=start  q=quit")

    finally:
        cv2.destroyAllWindows()
        env.safe_end()


if __name__ == "__main__":
    main()
