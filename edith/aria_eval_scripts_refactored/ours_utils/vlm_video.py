"""Prepare video clips for VLM consumption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .audio_processing import apply_whisper_transcript_to_frames
from .audio_video_merging import merge_audio_records_into_video_bytes
from .video_processing import (
    crop_around_gaze,
    draw_gaze_crosshair,
    frames_to_video_bytes,
    subsample_frames,
)


@dataclass(frozen=True)
class VLMVideoConfig:
    video_record_fps: float
    vlm_fps: float
    target_res: Optional[Tuple[int, int]]
    gaze_crop_enabled: bool
    gaze_crop_size: int
    gaze_calibration_y_offset: float
    draw_gaze_marker: bool
    debug_banner: bool
    video_source_label: str
    whisper_enabled: bool
    useful_voice_min_chars: int
    voiced_video: bool
    gaze_calibration_x_offset: float = 0.0


def uses_cropped_vlm_video(video_source: str) -> bool:
    return video_source == "aria_cropped_rgb"


def vlm_video_source_label(video_source: str) -> str:
    if uses_cropped_vlm_video(video_source):
        return "aria_cropped_rgb"
    return "aria_rgb"


def select_vlm_frames(aria_frames, cropped_frames, video_source: str):
    if uses_cropped_vlm_video(video_source):
        return cropped_frames
    return aria_frames


def build_vlm_video_config(
    video_record_fps: float,
    vlm_fps: float,
    target_res: Optional[Tuple[int, int]],
    gaze_crop_enabled: bool,
    gaze_crop_size: int,
    gaze_calibration_y_offset: float,
    draw_gaze_marker: bool,
    debug_banner: bool,
    video_source: str,
    whisper_enabled: bool,
    useful_voice_min_chars: int,
    voiced_video: bool,
    gaze_calibration_x_offset: float = 0.0,
) -> VLMVideoConfig:
    return VLMVideoConfig(
        video_record_fps=video_record_fps,
        vlm_fps=vlm_fps,
        target_res=target_res,
        gaze_crop_enabled=gaze_crop_enabled,
        gaze_crop_size=gaze_crop_size,
        gaze_calibration_y_offset=gaze_calibration_y_offset,
        gaze_calibration_x_offset=gaze_calibration_x_offset,
        draw_gaze_marker=draw_gaze_marker,
        debug_banner=debug_banner,
        video_source_label=vlm_video_source_label(video_source),
        whisper_enabled=whisper_enabled,
        useful_voice_min_chars=useful_voice_min_chars,
        voiced_video=voiced_video,
    )


def _add_debug_banner(
    frames_bgr: List[np.ndarray],
    video_source_label: str,
    frames_are_gaze_cropped: bool,
    clip_start_sec: float,
) -> List[np.ndarray]:
    banner_text_prefix = (
        f"VLM INPUT | src={video_source_label} | "
        f"cropped={frames_are_gaze_cropped} | clip_start={clip_start_sec:.1f}s"
    )
    banner_frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    for idx, frame in enumerate(frames_bgr):
        fh, fw = frame.shape[:2]
        bar = max(18, fh // 14)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, bar), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"{banner_text_prefix} | frame={idx}",
            (4, bar - 4), font, scale, (0, 255, 255), 1, cv2.LINE_AA,
        )
        banner_frames.append(overlay)
    return banner_frames


def _apply_gaze_crop(
    frames_bgr: List[np.ndarray],
    gaze_records: List[dict],
    config: VLMVideoConfig,
) -> List[np.ndarray]:
    processed = []
    for idx, frame in enumerate(frames_bgr):
        cropped = frame.copy()
        if config.gaze_crop_enabled and idx < len(gaze_records):
            gaze_record = gaze_records[idx]
            gaze_x = gaze_record.get("gaze_x")
            gaze_y = gaze_record.get("gaze_y")
            if gaze_x is not None and gaze_y is not None:
                cropped = crop_around_gaze(
                    cropped,
                    gaze_x + config.gaze_calibration_x_offset,
                    gaze_y + config.gaze_calibration_y_offset,
                    config.gaze_crop_size,
                )
        processed.append(cropped)
    return processed


def prepare_vlm_video(
    frames: List[np.ndarray],
    gaze_records: List[dict],
    audio_records: List[dict],
    clip_start_sec: float,
    frames_are_gaze_cropped: bool,
    config: VLMVideoConfig,
    transcript_data_override: Optional[dict] = None,
) -> Tuple[bytes, str, dict]:
    """Prepare a video segment for VLM consumption.

    Applies gaze crop, optional gaze marker, optional debug banner, FPS
    subsampling, Whisper captioning/transcript extraction, H264 encoding, and
    optional audio muxing.
    """
    if frames_are_gaze_cropped:
        processed = [frame.copy() for frame in frames]
    else:
        processed = _apply_gaze_crop(frames, gaze_records, config)

    if frames_are_gaze_cropped and config.draw_gaze_marker:
        processed = [draw_gaze_crosshair(frame) for frame in processed]

    if config.debug_banner:
        processed = _add_debug_banner(
            processed,
            config.video_source_label,
            frames_are_gaze_cropped,
            clip_start_sec,
        )

    subsampled = subsample_frames(
        processed, config.video_record_fps, config.vlm_fps)

    subsampled, transcript_section, transcript_data = apply_whisper_transcript_to_frames(
        subsampled,
        audio_records,
        config.vlm_fps,
        config.useful_voice_min_chars,
        config.whisper_enabled,
        transcript_data_override=transcript_data_override,
    )

    video_bytes = frames_to_video_bytes(
        subsampled, config.vlm_fps, config.target_res)

    if config.voiced_video:
        video_bytes = merge_audio_records_into_video_bytes(
            video_bytes, audio_records)

    return video_bytes, transcript_section, transcript_data
