"""Shared helpers for the refactored ours eval scripts.

Utility layers:
- audio_processing: PCM/WAV, Whisper, transcript filtering.
- video_processing: silent video writing and caption rendering.
- audio_video_merging: ffmpeg muxing for audio + video outputs.
- vlm_video: video-source selection and VLM input preprocessing.
- gemini_requests: Gemini request/retry logic and JSON-to-command parsing.
- keyboard: OpenCV/terminal key polling helpers.
- command_queue: queue command types and thread-safe keyframe queue.
- video_buffer: shared video, gaze, and audio frame buffer.
- preview: OpenCV preview composition.
- simulation_video / rollout_video: optional debug video rendering.
- prompt_loader: task-specific VLM prompt template loading.
- parsing / intervals: small generic model-response and keyframe helpers.
"""
