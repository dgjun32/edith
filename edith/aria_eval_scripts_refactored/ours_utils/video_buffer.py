"""Thread-safe video, gaze, and audio buffer for async VLM workers."""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np


class VideoBuffer:
    """Append-only buffer shared between observation streamers and VLM workers."""

    def __init__(self, store_full_aria_in_vlm_buffer: bool = False):
        self._lock = threading.Lock()
        self._aria: List[np.ndarray] = []
        self._cropped: List[np.ndarray] = []
        self._gaze: List[dict] = []
        self._audio: List[dict] = []
        self._vlm_processed_up_to = 0
        self._store_full_aria_in_vlm_buffer = bool(store_full_aria_in_vlm_buffer)

    def append(self, aria_bgr: np.ndarray, cropped_bgr: np.ndarray,
               gaze_x: Optional[float], gaze_y: Optional[float],
               audio_chunk_pcm16: Optional[np.ndarray] = None,
               audio_ts_ns: int = 0,
               audio_sample_rate_hz: int = 0,
               audio_num_samples: Optional[int] = None):
        with self._lock:
            idx = len(self._aria)
            cropped_copy = cropped_bgr.copy()
            aria_copy = (
                aria_bgr.copy()
                if self._store_full_aria_in_vlm_buffer
                else cropped_copy
            )
            self._aria.append(aria_copy)
            self._cropped.append(cropped_copy)
            self._gaze.append({"frame_idx": idx, "gaze_x": gaze_x, "gaze_y": gaze_y})
            if audio_chunk_pcm16 is None:
                audio_chunk_pcm16 = np.zeros((0,), dtype=np.int16)
            else:
                audio_chunk_pcm16 = np.asarray(audio_chunk_pcm16, dtype=np.int16).copy()
            if audio_num_samples is None:
                audio_num_samples = int(audio_chunk_pcm16.size)
            self._audio.append({
                "frame_idx": idx,
                "audio_ts_ns": int(audio_ts_ns),
                "sample_rate_hz": int(audio_sample_rate_hz),
                "num_samples": int(audio_num_samples),
                "chunk_pcm16": audio_chunk_pcm16,
            })

    def unprocessed_count(self) -> int:
        with self._lock:
            return len(self._aria) - self._vlm_processed_up_to

    def skip_to_current(self):
        """Mark all currently buffered frames as already processed."""
        with self._lock:
            self._vlm_processed_up_to = len(self._aria)

    def reserve_segment(self, max_frames: Optional[int] = None) -> Tuple[int, int, int]:
        """Advance the VLM pointer without copying image frames."""
        with self._lock:
            start = self._vlm_processed_up_to
            end = len(self._aria)
            if max_frames is not None:
                end = min(end, start + max(0, int(max_frames)))
            self._vlm_processed_up_to = end
            return start, end, max(0, end - start)

    def get_range(self, start: int, end: int) -> Tuple[List, List, List]:
        """Copy an arbitrary frame range without changing the VLM pointer."""
        with self._lock:
            start = max(0, int(start))
            end = max(start, min(int(end), len(self._aria)))
            aria = [f.copy() for f in self._aria[start:end]]
            cropped = [f.copy() for f in self._cropped[start:end]]
            gaze = list(self._gaze[start:end])
            return aria, cropped, gaze

    def get_audio_records(self, start: int = 0, end: Optional[int] = None) -> List[dict]:
        with self._lock:
            if end is None:
                end = len(self._audio)
            records = []
            for record in self._audio[start:end]:
                copied = dict(record)
                copied["chunk_pcm16"] = record["chunk_pcm16"].copy()
                records.append(copied)
            return records

    def get_frame(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if 0 <= idx < len(self._aria):
                return self._aria[idx].copy(), self._cropped[idx].copy()
            return None, None
