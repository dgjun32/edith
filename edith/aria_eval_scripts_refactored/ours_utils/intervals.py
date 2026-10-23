"""Interval conversion helpers for keyframe timestamps."""

from __future__ import annotations


def mid_sec_from_interval(interval: dict) -> float:
    """Return the midpoint time, in seconds, of a keyframe interval."""
    start = float(interval.get("start", 0.0))
    end = float(interval.get("end", start))
    if start > end:
        start, end = end, start
    return (start + end) / 2.0


def mid_frame_from_interval(interval: dict, num_frames: int, fps: float) -> int:
    mid_sec = mid_sec_from_interval(interval)
    frame_idx = int(round(mid_sec * fps))
    return max(0, min(frame_idx, num_frames - 1))

