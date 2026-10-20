"""Backward-compatible audio utility facade.

New code should import from:
- ours_utils.audio_processing
- ours_utils.video_processing
- ours_utils.audio_video_merging
"""

from .audio_processing import (
    apply_whisper_transcript_to_frames,
    collect_audio_pcm16,
    is_useful_whisper_transcript,
    transcribe_audio_records_for_whisper,
    whisper_transcribe_wav,
    write_pcm16_wav,
)
from .audio_video_merging import (
    merge_audio_into_video_bytes,
    merge_audio_records_into_video_bytes,
    mux_audio_file_into_video_file,
    mux_audio_records_into_video_file,
    write_video_with_audio,
)
from .video_processing import (
    burn_captions_on_video_bytes,
    crop_around_gaze,
    draw_gaze_crosshair,
    frames_to_video_bytes,
    get_frame_aligned_caption,
    overlay_captions_on_frames,
    subsample_frames,
    write_silent_video,
)
from .vlm_video import VLMVideoConfig, prepare_vlm_video

__all__ = [
    "apply_whisper_transcript_to_frames",
    "burn_captions_on_video_bytes",
    "collect_audio_pcm16",
    "crop_around_gaze",
    "draw_gaze_crosshair",
    "frames_to_video_bytes",
    "get_frame_aligned_caption",
    "is_useful_whisper_transcript",
    "merge_audio_into_video_bytes",
    "merge_audio_records_into_video_bytes",
    "mux_audio_file_into_video_file",
    "mux_audio_records_into_video_file",
    "overlay_captions_on_frames",
    "prepare_vlm_video",
    "subsample_frames",
    "transcribe_audio_records_for_whisper",
    "VLMVideoConfig",
    "whisper_transcribe_wav",
    "write_pcm16_wav",
    "write_silent_video",
    "write_video_with_audio",
]
