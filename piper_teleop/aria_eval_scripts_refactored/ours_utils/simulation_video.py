"""Optional keyframe-queue simulation video recording."""

from __future__ import annotations

import os
import queue as queue_lib
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import numpy as np

from eval_vlm_baseline import _ffmpeg_reencode_h264

from .audio_video_merging import mux_audio_records_into_video_file


@dataclass(frozen=True)
class SimulationVideoConfig:
    canvas_w: int
    canvas_h: int
    video_size: int
    thumb_size: tuple
    row_height: int
    rows_per_col: int
    col_width: int
    left_margin: int
    top_start: int
    fps: float


def render_sim_canvas(
    video_frame: Optional[np.ndarray],
    queue_snapshot: Optional[List[dict]],
    current_time: float,
    config: SimulationVideoConfig,
) -> np.ndarray:
    """Render one frame of the keyframe simulation visualization."""
    width, height = config.canvas_w, config.canvas_h
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    font = cv2.FONT_HERSHEY_SIMPLEX
    video_size = config.video_size

    video_x = width - video_size - 30
    video_y = 30
    if video_frame is not None:
        thumb = cv2.resize(
            video_frame, (video_size, video_size),
            interpolation=cv2.INTER_LANCZOS4,
        )
        canvas[video_y:video_y + video_size, video_x:video_x + video_size] = thumb
        cv2.rectangle(
            canvas,
            (video_x - 2, video_y - 2),
            (video_x + video_size + 1, video_y + video_size + 1),
            (180, 180, 180),
            2,
        )
    cv2.putText(
        canvas,
        f"t = {current_time:.1f}s",
        (video_x, video_y + video_size + 35),
        font,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    thumb_w, thumb_h = config.thumb_size
    cv2.putText(
        canvas,
        "Keyframe Action Queue",
        (config.left_margin, 45),
        font,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    if queue_snapshot is None:
        cv2.putText(
            canvas,
            "Initializing...",
            (config.left_margin, 150),
            font,
            1.2,
            (120, 120, 120),
            2,
            cv2.LINE_AA,
        )
        return canvas

    if not queue_snapshot:
        cv2.putText(
            canvas,
            "(queue is empty)",
            (config.left_margin, 150),
            font,
            1.0,
            (120, 120, 120),
            2,
            cv2.LINE_AA,
        )
        return canvas

    for idx, elem in enumerate(queue_snapshot):
        col = idx // config.rows_per_col
        row = idx % config.rows_per_col

        tile_x = config.left_margin + col * config.col_width
        tile_y = config.top_start + row * config.row_height + 8
        if tile_y + thumb_h > height or tile_x + thumb_w > width:
            continue

        keyframe_img = elem.get("keyframe_image")
        if keyframe_img is not None:
            resized = cv2.resize(
                keyframe_img, (thumb_w, thumb_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
            canvas[tile_y:tile_y + thumb_h, tile_x:tile_x + thumb_w] = resized
        else:
            canvas[tile_y:tile_y + thumb_h, tile_x:tile_x + thumb_w] = 40

        is_current = elem.get("is_current", False)
        border_color = (0, 255, 0) if is_current else (100, 100, 100)
        border_thick = 4 if is_current else 2
        cv2.rectangle(
            canvas,
            (tile_x - 2, tile_y - 2),
            (tile_x + thumb_w + 1, tile_y + thumb_h + 1),
            border_color,
            border_thick,
        )

        text_x = tile_x + thumb_w + 25
        instr = elem.get("instruction", "")
        color = (100, 255, 100) if instr.startswith("pick up") else (100, 200, 255)
        if is_current:
            color = (0, 255, 255)
        cv2.putText(
            canvas,
            f"[{idx}] {instr}",
            (text_x, tile_y + 40),
            font,
            1.1,
            color,
            2,
            cv2.LINE_AA,
        )

        if is_current:
            cv2.putText(
                canvas,
                ">>> EXECUTING",
                (text_x, tile_y + 80),
                font,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return canvas


class SimulationVideoRecorder:
    """Asynchronously renders and writes keyframe_simulation.mp4."""

    def __init__(self, output_path: str, config: SimulationVideoConfig):
        self._output_path = output_path
        self._config = config
        self._fps = config.fps
        self._tmp_path = output_path + ".tmp.mp4"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._tmp_path,
            fourcc,
            self._fps,
            (config.canvas_w, config.canvas_h),
        )
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._finalized = False
        self._write_error: Optional[Exception] = None
        self._tasks: queue_lib.Queue = queue_lib.Queue()
        self._worker = threading.Thread(
            target=self._run_writer,
            daemon=True,
            name="SimVideoWriter",
        )
        self._worker.start()
        print(f"[SimVideo] Recording enabled -> {self._output_path}")

    def _run_writer(self):
        while True:
            task = self._tasks.get()
            try:
                if task is None:
                    return
                aria_bgr, queue_snap, current_time = task
                canvas = render_sim_canvas(
                    aria_bgr, queue_snap, current_time, self._config)
                self._writer.write(canvas)
                self._frame_count += 1
            except Exception as exc:
                self._write_error = exc
                print(f"[SimVideo] Writer error: {exc}")
            finally:
                self._tasks.task_done()

    def _enqueue_frame(self, aria_bgr: Optional[np.ndarray],
                       queue_snap: Optional[List[dict]],
                       current_time: float):
        if self._finalized:
            return
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._tasks.put((aria_bgr, queue_snap, current_time))

    def record_frame(self, aria_bgr: Optional[np.ndarray],
                     queue: Optional[Any],
                     current_time: float):
        queue_snap = queue.snapshot() if queue is not None else None
        self._enqueue_frame(aria_bgr, queue_snap, current_time)

    def record_init_phase(self, aria_frames: List[np.ndarray],
                          queue: Optional[Any],
                          fps: float):
        step = max(1, int(fps / self._fps))
        queue_snap = queue.snapshot() if queue is not None else None
        for idx in range(0, len(aria_frames), step):
            self._enqueue_frame(aria_frames[idx], queue_snap, idx / fps)

    def finalize(self, audio_records: Optional[List[dict]] = None):
        if self._finalized:
            return
        self._finalized = True

        self._tasks.put(None)
        self._worker.join()
        self._writer.release()

        if self._write_error is not None:
            print(f"[SimVideo] Saved video may be incomplete due to writer error: "
                  f"{self._write_error}")

        if self._frame_count == 0:
            print("[SimVideo] No frames recorded, skipping.")
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            return

        if not os.path.exists(self._tmp_path):
            print(f"[SimVideo] Temp video missing, skipping finalize: {self._tmp_path}")
            return

        _ffmpeg_reencode_h264(self._tmp_path, self._output_path)
        os.unlink(self._tmp_path)

        if audio_records:
            mux_audio_records_into_video_file(
                self._output_path, audio_records, audio_bitrate="128k")

        print(f"[SimVideo] Saved: {self._output_path} "
              f"({self._frame_count} frames)")


def safe_finalize_sim_recorder(
    sim_recorder: Optional[SimulationVideoRecorder],
    video_buffer: Optional[Any] = None,
):
    if sim_recorder is None:
        return

    audio_records = None
    if video_buffer is not None:
        try:
            audio_records = video_buffer.get_audio_records()
        except Exception as exc:
            print(f"[SimVideo] Failed to collect audio for finalize: {exc}")

    try:
        sim_recorder.finalize(audio_records=audio_records)
    except Exception as exc:
        print(f"[SimVideo] Finalize failed: {exc}")
