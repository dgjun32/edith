"""eval_openpi_async_vlm_gaze_keyframe_queue.py

Async VLM-based keyframe queue management for dual-arm robot evaluation.

Architecture
------------
  Main thread   : robot eval loop + CV2 display.
  VLM thread    : background Gemini calls that manipulate the KeyframeQueue.

Human-view (aria_rgb) frames are continuously written into a shared
VideoBuffer starting from t=0.  The async worker fires every
VLM_INTERVAL_SEC seconds, drains accumulated frames, and produces
exactly one queue-manipulation command:

    NoOp / Append / Replace / Preempt / Remove
"""

import argparse
import hashlib
import numpy as np
import cv2
import os
import json
import threading
import time
from typing import List, Optional, Tuple

from edith.envs.piper_env_gaze import PiperRobotEnv
from edith.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from common.streamer import (
    BackgroundObservationStreamer,
    wait_for_queue_items,
)
from common.config import SERVER_IP
from ours_utils.command_queue import (
    CmdAppend,
    CmdAppendMany,
    CmdNoOp,
    CmdPreempt,
    CmdReplace,
    KeyframeQueue,
    RawAppendMany as _RawAppendMany,
    RawCommand as _RawCommand,
    format_cmd_label as _format_cmd_label,
)
from ours_utils.audio_processing import (
    transcribe_audio_records_for_whisper as _transcribe_audio_records_for_whisper,
)
from ours_utils.audio_video_merging import (
    write_video_with_audio as _write_video_with_audio,
)
from ours_utils.gemini_requests import (
    GeminiRequestConfig,
    query_gemini_async_update as _query_gemini_async_update,
    query_gemini_initial as _query_gemini_initial,
    short_error_reason as _short_error_reason,
)
from ours_utils.intervals import mid_frame_from_interval as _mid_frame_from_interval
from ours_utils.keyboard import (
    poll_keypress as _poll_keypress,
    terminal_keypress_mode as _terminal_keypress_mode,
)
from ours_utils.preview import (
    build_eval_preview as _build_eval_preview,
    build_preview_frame as _build_preview_frame,
)
from ours_utils.prompt_loader import (
    list_prompt_names as _list_prompt_names,
    load_prompt_templates as _load_prompt_templates,
)
from ours_utils.rollout_video import save_rollout_video as _save_rollout_video
from ours_utils.simulation_video import (
    SimulationVideoConfig,
    SimulationVideoRecorder,
    safe_finalize_sim_recorder as _safe_finalize_sim_recorder,
)
from ours_utils.video_processing import (
    draw_gaze_crosshair as _draw_gaze_crosshair,
    overlay_captions_on_frames as _overlay_captions_on_frames,
    write_silent_video as _write_video,
)
from ours_utils.video_buffer import VideoBuffer
from ours_utils.vlm_video import (
    build_vlm_video_config as _build_vlm_video_config,
    prepare_vlm_video as _prepare_vlm_video,
    select_vlm_frames as _select_vlm_frames,
    uses_cropped_vlm_video as _uses_cropped_vlm_video,
    vlm_video_source_label as _vlm_video_source_label,
)


# ---------------------------------------------------------------------------
# Constants — edit these instead of CLI args
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS   = 15
_MAX_INTERVALS      = 10
_TEMP_DIR           = os.path.normpath("/home/alin/projects/piper_project/temp")
# ── Task prompt selection ──
# Change this single line to switch the VLM prompt loaded from ours_prompts/.
# Examples:
#   PROMPT_TASK = "serving_muffins"    # loads ours_prompts/serving_muffins.txt
#   PROMPT_TASK = "sorting_tumblers"   # loads ours_prompts/sorting_tumblers.txt
#   PROMPT_TASK = "passing_tools"      # loads ours_prompts/passing_tools.txt
#   PROMPT_TASK = "collaborative_sorting"
#
# CLI override examples:
#   --prompt-task serving_muffins
#   --prompt-task sorting_tumblers
#   --prompt-task passing_tools
#   --prompt-task collaborative_sorting
PROMPT_TASK         = "sorting_tumblers"                # which prompt script to load (see ours_prompts/)

TASK_DEFAULT_INSTRUCTIONS = {
    "serving_muffins": "I want this muffin.",
    "sorting_tumblers": "Sort these tumblers.",
    "passing_tools": "I want this tool.",
    "collaborative_sorting": "Sort the packed box into the correct basket.",
}


def _task_default_instruction(prompt_task: Optional[str] = None) -> str:
    return TASK_DEFAULT_INSTRUCTIONS.get(prompt_task or PROMPT_TASK, "I want this.")


def _policy_language_instruction(language_instruction: str) -> str:
    if GLOBAL_INSTRUCTION:
        return _task_default_instruction(PROMPT_TASK)
    return language_instruction


def _image_digest(img: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()[:12]


def _save_queue_keyframes_debug(queue: KeyframeQueue,
                                current_keyframe_rgb: Optional[np.ndarray],
                                step: int,
                                saved_keys: set) -> None:
    """Save queue keyframes and the exact current RGB keyframe sent to VLA."""
    out_dir = os.path.join(_TEMP_DIR, "forwarded_keyframes_260503")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for idx, item in enumerate(queue.snapshot()):
        keyframe_bgr = item["keyframe_image"]
        if keyframe_bgr is None:
            continue

        digest = _image_digest(keyframe_bgr)
        queue_path = os.path.join(
            out_dir, f"queue_{idx:02d}_{digest}_stored_bgr.png")
        queue_key = ("queue", idx, digest)
        if queue_key not in saved_keys:
            cv2.imwrite(queue_path, keyframe_bgr)
            saved_keys.add(queue_key)

        forwarded_path = ""
        if item["is_current"] and current_keyframe_rgb is not None:
            forwarded_path = os.path.join(
                out_dir,
                f"step_{step:06d}_queue_{idx:02d}_{digest}_forwarded_rgb.png",
            )
            forwarded_key = ("forwarded", idx, digest)
            if forwarded_key not in saved_keys:
                cv2.imwrite(
                    forwarded_path,
                    cv2.cvtColor(current_keyframe_rgb, cv2.COLOR_RGB2BGR),
                )
                saved_keys.add(forwarded_key)

        manifest.append({
            "queue_idx": idx,
            "is_current": bool(item["is_current"]),
            "instruction": item["instruction"],
            "digest": digest,
            "stored_bgr_path": queue_path,
            "forwarded_rgb_path": forwarded_path,
        })

    with open(os.path.join(out_dir, "latest_manifest.json"), "w") as f:
        json.dump({"step": step, "keyframes": manifest}, f, indent=2)

# ── VLM pipeline settings ──
VLM_MODEL           = "gemini-3.5-flash"       # Gemini model to use
VLM_FPS             = 4                         # subsample to this FPS before sending to VLM
VLM_INTERVAL_SEC    = 8.0                       # seconds between async VLM calls
VLM_TARGET_RES      = (256, 256)                # resize video segments for VLM
VLM_THINKING_BUDGET = 2048                      # thinking token budget per call
VLM_REQUEST_TIMEOUT_SEC = 15.0                  # retry Gemini calls after this many seconds
VLM_MAX_ATTEMPTS    = 3                         # total attempts per VLM request
VLM_RETRY_BACKOFF_SEC = 0                     # brief pause between failed attempts
VLM_DEBUG_BANNER    = False                     # drawing per-frame debug text costs CPU

# Which Aria view to encode into the mp4 sent to Gemini.
# Options:
#   - "aria_rgb": full egocentric RGB frames
#   - "aria_cropped_rgb": gaze-centered cropped RGB frames
_GEMINI_VIDEO_SOURCE = "aria_cropped_rgb"

 # overlay center gaze marker on cropped VLM input
VLM_DRAW_GAZE_MARKER = True
STORE_FULL_ARIA_IN_VLM_BUFFER = True        # always keep full aria frames in buffer for saving

# ── Audio settings ──
WHISPER_ENABLED     = True                      # use Whisper STT → transcript in prompt
VOICED_VIDEO        = False                      # merge audio into mp4 → Gemini processes audio
ASYNC_VLM_REQUIRE_USEFUL_VOICE = False          # query Gemini even when Whisper transcript is empty
USEFUL_VOICE_MIN_CHARS = 2

# ── Gaze crop settings ──
GAZE_CROP_ENABLED   = True
GAZE_CROP_SIZE      =  512                      # crop NxN around gaze centre
GAZE_CALIBRATION_Y_OFFSET = 0                 # px to add to Y (corrects upward calibration bias)


# ── Visual queue caching and optimization──
VISUAL_QUEUE = False                             # attach keyframe images in VLM update calls
VISUAL_QUEUE_MAX_IMAGES = 10                    # bound async prompt growth after repeated Appends
VISUAL_QUEUE_IMAGE_SIZE = 128                   # small visual context is enough for queue identity
VISUAL_QUEUE_JPEG_QUALITY = 65                  # JPEG is much cheaper than PNG for prompt thumbnails
VISUAL_QUEUE_THUMB_CACHE_MAX = 128              # avoid re-encoding unchanged queue images every call

# ── Runtime Saving Handling ──
SAVE_ASYNC_RAW_SEGMENTS = True                  # async_raw_seg_*.mp4 before captioning
SAVE_ASYNC_CAPTIONED_SEGMENTS = True            # async_captioned_seg_*.mp4 with transcript overlay
SAVE_ASYNC_VLM_INPUT_VIDEOS = True             # vlm_input_*.mp4 sent to Gemini


# ── Simulation visualization ──
SAVE_SIMULATION_VIS = False                    # keyframe_simulation.mp4
# if SAVE_SIMULATION_VIS, the following control the layout and timing of the output video
SIM_CANVAS_W        = 4096
SIM_CANVAS_H        = 3250
SIM_VIDEO_SIZE      = 512                       # original video thumbnail (top-right)
SIM_THUMB_SIZE      = (512, 512)                # keyframe thumbnail
SIM_ROW_HEIGHT      = 620                       # px per queue row (512 + padding)
SIM_ROWS_PER_COL    = 5                         # 5×2 grid layout
SIM_COL_WIDTH       = 1950                      # px per column
SIM_LEFT_MARGIN     = 40
SIM_TOP_START       = 60
SIM_FPS             = 15                        # fps for keyframe_simulation.mp4
_VLM_EVENT_DISPLAY_FRAMES = 90                  # Number of top-view (global) steps for which a VLM command overlay stays

SIM_VIDEO_CONFIG = SimulationVideoConfig(
    canvas_w=SIM_CANVAS_W,
    canvas_h=SIM_CANVAS_H,
    video_size=SIM_VIDEO_SIZE,
    thumb_size=SIM_THUMB_SIZE,
    row_height=SIM_ROW_HEIGHT,
    rows_per_col=SIM_ROWS_PER_COL,
    col_width=SIM_COL_WIDTH,
    left_margin=SIM_LEFT_MARGIN,
    top_start=SIM_TOP_START,
    fps=SIM_FPS,
)

# ── Robot pause on voice ──
STOP_ROBOT_WHILE_RECV = False                    # pause robot control while VLM processes voice

# FPS for saving async VLM segment videos (logging).
# IMPORTANT : Keep this equal to the source capture FPS so the saved clip duration matches the real duration.
_VLM_VIDEO_SAVE_FPS = _VIDEO_RECORD_FPS

_GEMINI_USES_CROPPED_VIDEO = _uses_cropped_vlm_video(_GEMINI_VIDEO_SOURCE)
_GEMINI_VIDEO_SOURCE_LABEL = _vlm_video_source_label(_GEMINI_VIDEO_SOURCE)

VLM_VIDEO_CONFIG = _build_vlm_video_config(
    video_record_fps=_VIDEO_RECORD_FPS,
    vlm_fps=VLM_FPS,
    target_res=VLM_TARGET_RES,
    gaze_crop_enabled=GAZE_CROP_ENABLED,
    gaze_crop_size=GAZE_CROP_SIZE,
    gaze_calibration_y_offset=GAZE_CALIBRATION_Y_OFFSET,
    draw_gaze_marker=VLM_DRAW_GAZE_MARKER,
    debug_banner=VLM_DEBUG_BANNER,
    video_source=_GEMINI_VIDEO_SOURCE,
    whisper_enabled=WHISPER_ENABLED,
    useful_voice_min_chars=USEFUL_VOICE_MIN_CHARS,
    voiced_video=VOICED_VIDEO,
)

GEMINI_REQUEST_CONFIG = GeminiRequestConfig(
    model=VLM_MODEL,
    fps=VLM_FPS,
    thinking_budget=VLM_THINKING_BUDGET,
    request_timeout_sec=VLM_REQUEST_TIMEOUT_SEC,
    max_attempts=VLM_MAX_ATTEMPTS,
    retry_backoff_sec=VLM_RETRY_BACKOFF_SEC,
    voiced_video=VOICED_VIDEO,
    max_intervals=_MAX_INTERVALS,
    video_record_fps=_VIDEO_RECORD_FPS,
    visual_queue=VISUAL_QUEUE,
    visual_queue_image_size=VISUAL_QUEUE_IMAGE_SIZE,
    visual_queue_max_images=VISUAL_QUEUE_MAX_IMAGES,
    visual_queue_jpeg_quality=VISUAL_QUEUE_JPEG_QUALITY,
    visual_queue_thumb_cache_max=VISUAL_QUEUE_THUMB_CACHE_MAX,
)

# ── VLA prompt ──
GLOBAL_INSTRUCTION = False
# ---------------------------------------------------------------------------
# Runtime prompt templates
# ---------------------------------------------------------------------------
# Prompt text is loaded from ours_prompts/<PROMPT_TASK>.txt by _set_prompt_task().
# Keep the actual initial/async prompt scripts out of this eval file.

PROMPT_TEMPLATE_INITIAL = ""
PROMPT_TEMPLATE_ASYNC_UPDATE = ""

def _set_prompt_task(prompt_task: str, announce: bool = False):
    """Load runtime VLM prompts from ours_prompts/<prompt_task>.txt."""
    global PROMPT_TASK, PROMPT_TEMPLATE_INITIAL, PROMPT_TEMPLATE_ASYNC_UPDATE
    templates = _load_prompt_templates(prompt_task)
    PROMPT_TASK = templates.name
    PROMPT_TEMPLATE_INITIAL = templates.initial
    PROMPT_TEMPLATE_ASYNC_UPDATE = templates.async_update
    if announce:
        print(f"[Prompts] Loaded '{PROMPT_TASK}' from {templates.path}")


_set_prompt_task(PROMPT_TASK)


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _require_initial_keyframes_ready(queue: KeyframeQueue, context: str = "startup"):
    """Fail before policy/async startup only when the queue object is invalid.

    An empty queue is valid when the user pressed 's' to skip initial
    keyframe detection; the robot will hold pose while async VLM appends.
    """
    if queue is None:
        raise RuntimeError(
            f"[{context}] Keyframe queue is not ready; "
            "not starting policy/VLA inference or async VLM collection."
        )
    if queue.total > 0 and queue.current_keyframe is None:
        raise RuntimeError(
            f"[{context}] Keyframe queue is inconsistent; "
            "queue has items but no current keyframe."
        )


class EpisodeEnd(Exception):
    """User requested ending only the current episode."""


class ProgramQuit(Exception):
    """User requested exiting the whole program."""


def _episode_control_from_key(key: int) -> Optional[str]:
    if key in (ord('q'), ord('Q'), 27):
        return "quit"
    if key in (ord('e'), ord('E')):
        return "episode_end"
    return None


def _raise_for_episode_control(key: int):
    control = _episode_control_from_key(key)
    if control == "quit":
        raise ProgramQuit
    if control == "episode_end":
        raise EpisodeEnd


# ---------------------------------------------------------------------------
# Async VLM worker thread
# ---------------------------------------------------------------------------

class AsyncVLMWorker:
    """
    Background daemon thread.

    Timer loop (main worker thread):
      1. Wait VLM_INTERVAL_SEC wall-clock seconds.
      2. Reserve one fixed segment and run the audio/Whisper pre-check.
      3. If useful voice exists, copy/encode frames for Gemini.
      4. Fire a per-call sub-thread for the Gemini API call (non-blocking).
      5. Immediately go back to step 1 — timer resets right now, new frames
         accumulate while the previous Gemini call is still in flight.

    Per-call sub-thread:
      - Sends the prepared mp4 bytes to Gemini.
      - Resolves the returned command and applies it to the KeyframeQueue.
      - Sets preempt_event if needed.
      - Updates call_log entry with timing info.
    """

    def __init__(self, video_buffer: VideoBuffer, queue: KeyframeQueue,
                 global_instruction: str, api_key: str,
                 interval_sec: float = VLM_INTERVAL_SEC,
                 temp_dir: str = _TEMP_DIR):
        self._buf                = video_buffer
        self._queue              = queue
        self._global_instruction = global_instruction
        self._api_key            = api_key
        self._interval_sec       = interval_sec
        self._segment_frames     = max(
            1, int(round(self._interval_sec * _VIDEO_RECORD_FPS))
        )
        self._temp_dir           = temp_dir
        os.makedirs(self._temp_dir, exist_ok=True)
        self._stop_event         = threading.Event()
        self._call_count         = 0
        self._episode_start: float = 0.0   # set in start()

        # Preempt signal: main thread checks this after every env.step().
        self.preempt_event = threading.Event()

        # Robot pause signal: when STOP_ROBOT_WHILE_RECV is True and voice is
        # detected, this event is SET to pause the robot. It is CLEARED once
        # the VLM response arrives and the queue is updated.
        self.robot_pause_event = threading.Event()  # SET = robot should pause

        # Shared step counter — main thread writes vlm_worker.current_step = global_step
        # each env.step() so that event timestamps align with top_view_frames_list indices.
        self.current_step: int = 0

        # VLM event log: list of (global_step, label).
        self.event_log: List = []

        # Call log: one dict per VLM call, written by the sub-thread.
        # Keys: call_id, trigger_t, obs_window (start, end), n_frames,
        #       step_at_trigger, gemini_start_t, gemini_end_t, command.
        self.call_log: List[dict] = []

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AsyncVLMWorker")
        self._started = False

    def start(self):
        if self._started:
            return
        self._episode_start = time.monotonic()
        self._started = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.robot_pause_event.clear()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _elapsed(self) -> float:
        return time.monotonic() - self._episode_start

    # ------------------------------------------------------------------
    # Timer loop — never blocks on Gemini
    # ------------------------------------------------------------------

    def _run(self):
        print(f"[AsyncVLM] Worker started. video_source={_GEMINI_VIDEO_SOURCE_LABEL}")
        while not self._stop_event.is_set():
            # ---- wait interval_sec of real wall-clock time ----
            self._stop_event.wait(timeout=self._interval_sec)
            if self._stop_event.is_set():
                break

            # Keep segment boundaries aligned to fixed episode-time windows
            # (for example 8.0 s = 240 frames at 30 FPS), even if the
            # background streamer is a frame or two late when the timer fires.
            unprocessed = self._buf.unprocessed_count()
            if 0 < unprocessed < self._segment_frames:
                missing = self._segment_frames - unprocessed
                print(
                    f"[AsyncVLM] t={self._elapsed():.2f}s — waiting for "
                    f"{missing} more frame(s) to complete a "
                    f"{self._segment_frames / _VIDEO_RECORD_FPS:.2f}s segment"
                )
            while (not self._stop_event.is_set()
                   and self._buf.unprocessed_count() < self._segment_frames):
                self._stop_event.wait(timeout=1.0 / _VIDEO_RECORD_FPS)
            if self._stop_event.is_set():
                break

            trigger_t = self._elapsed()

            # ---- reserve one fixed-duration segment ----
            # Reserve one segment for VLM. Whisper may be skipped later by the
            # fast audio gate, but the video is still queried with an empty
            # transcript when no transcript is available.
            seg_start, seg_end, seg_n_frames = self._buf.reserve_segment(
                max_frames=self._segment_frames)

            if seg_n_frames <= 0:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames, skip")
                continue

            seg_start_sec = seg_start / _VIDEO_RECORD_FPS
            seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

            # Collect audio records for this segment
            audio_records = self._buf.get_audio_records(seg_start, seg_end)

            # ── Optional Whisper pre-check ──
            # This is only for STOP_ROBOT_WHILE_RECV pause behavior. Gemini is
            # still queried when Whisper returns an empty transcript.
            voice_detected = False
            precheck_transcript: dict = {}
            precheck_ran = False
            if WHISPER_ENABLED and STOP_ROBOT_WHILE_RECV:
                precheck_ran = True
                precheck_transcript = _transcribe_audio_records_for_whisper(
                    audio_records, context=f"precheck@{seg_start}-{seg_end}")
                voice_detected = bool(
                    (precheck_transcript.get("text", "") or "").strip())

            aria_seg, cropped_seg, gaze_seg = self._buf.get_range(seg_start, seg_end)
            gemini_seg = _select_vlm_frames(
                aria_seg, cropped_seg, _GEMINI_VIDEO_SOURCE)

            if not gemini_seg:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames in reserved range, skip")
                continue

            self._call_count += 1
            call_id = self._call_count

            print(f"[AsyncVLM] Call #{call_id} | t={trigger_t:.2f}s | "
                  f"obs frames {seg_start}–{seg_end} ({len(gemini_seg)} frames) | "
                  f"time {seg_start_sec:.1f}s–{seg_end_sec:.1f}s | "
                  f"video_source={_GEMINI_VIDEO_SOURCE_LABEL}")

            if voice_detected and STOP_ROBOT_WHILE_RECV:
                print(f"[AsyncVLM] #{call_id} Voice detected: "
                      f"\"{precheck_transcript.get('text', '')[:80]}\" → pausing robot")
                self.robot_pause_event.set()

            if SAVE_ASYNC_RAW_SEGMENTS:
                seg_video_path = os.path.join(
                    self._temp_dir,
                    f"async_raw_seg_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4")
                try:
                    _write_video_with_audio(
                        gemini_seg, audio_records, seg_video_path,
                        fps=_VLM_VIDEO_SAVE_FPS)
                    print(f"[AsyncVLM] #{call_id} Raw async segment saved → {seg_video_path}")
                except Exception as save_exc:
                    print(f"[AsyncVLM] #{call_id} Failed to save raw async segment: {save_exc}")

            # Create log entry (sub-thread will fill timing fields).
            log_entry: dict = {
                "call_id":         call_id,
                "trigger_t":       trigger_t,
                "obs_window":      (seg_start, seg_end),
                "obs_time_sec":    (seg_start_sec, seg_end_sec),
                "n_frames":        len(gemini_seg),
                "target_n_frames": self._segment_frames,
                "video_source":    _GEMINI_VIDEO_SOURCE,
                "step_at_trigger": self.current_step,
                "voice_detected":  voice_detected,
                "precheck_transcript": precheck_transcript.get("text", ""),
                "gemini_start_t":  None,
                "gemini_end_t":    None,
                "command":         None,
            }
            self.call_log.append(log_entry)

            if STOP_ROBOT_WHILE_RECV and voice_detected:
                # Run Gemini SYNCHRONOUSLY so robot stays paused until done.
                self._run_gemini_call(
                    call_id, gemini_seg, gaze_seg, audio_records,
                    seg_start, seg_end, log_entry,
                    transcript_data_override=(
                        precheck_transcript if precheck_ran else None),
                    keyframe_seg=cropped_seg)
                # Unpause robot now that queue is updated.
                self.robot_pause_event.clear()
                print(f"[AsyncVLM] #{call_id} VLM done → resuming robot")
            else:
                # Fire Gemini call in a sub-thread — timer resets immediately.
                threading.Thread(
                    target=self._run_gemini_call,
                    args=(call_id, gemini_seg, gaze_seg, audio_records,
                          seg_start, seg_end, log_entry),
                    kwargs={
                        "transcript_data_override": (
                            precheck_transcript if precheck_ran else None),
                        "keyframe_seg": cropped_seg,
                    },
                    daemon=True,
                    name=f"GeminiCall-{call_id}",
                ).start()

            # ← loop back immediately; new frames accumulate while Gemini runs.

        print("[AsyncVLM] Worker stopped.")

    # ------------------------------------------------------------------
    # Per-call sub-thread
    # ------------------------------------------------------------------

    def _run_gemini_call(self, call_id: int,
                         gemini_seg: List[np.ndarray],
                         gaze_seg: List[dict],
                         audio_records: List[dict],
                         seg_start: int, seg_end: int,
                         log_entry: dict,
                         transcript_data_override: Optional[dict] = None,
                         keyframe_seg: Optional[List[np.ndarray]] = None):

        seg_start_sec = seg_start / _VIDEO_RECORD_FPS
        seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

        def _copied_keyframe_at(local_frame_idx: int) -> Optional[np.ndarray]:
            local_frame_idx = int(local_frame_idx)
            if keyframe_seg is not None and 0 <= local_frame_idx < len(keyframe_seg):
                return keyframe_seg[local_frame_idx].copy()
            abs_idx = seg_start + local_frame_idx
            _, cropped_bgr = self._buf.get_frame(abs_idx)
            return cropped_bgr

        print(f"[AsyncVLM] #{call_id} → Preparing VLM input "
              f"(video_source={_GEMINI_VIDEO_SOURCE_LABEL})...")

        try:
            if self._stop_event.is_set():
                log_entry["command"] = "DROPPED_STOPPED"
                return

            # Prepare video: optional gaze crop → subsample → encode → whisper → audio
            video_bytes, transcript_section, transcript_data = _prepare_vlm_video(
                gemini_seg, gaze_seg, audio_records, seg_start_sec,
                frames_are_gaze_cropped=_GEMINI_USES_CROPPED_VIDEO,
                config=VLM_VIDEO_CONFIG,
                transcript_data_override=transcript_data_override,
            )

            if SAVE_ASYNC_CAPTIONED_SEGMENTS:
                seg_video_path = os.path.join(
                    self._temp_dir,
                    f"async_captioned_seg_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4",
                )
                transcript_text = transcript_data.get("text", "")
                if transcript_text:
                    captioned_seg = _overlay_captions_on_frames(
                        gemini_seg,
                        transcript_data.get("words", []),
                        transcript_text,
                        seg_start_sec,
                        _VLM_VIDEO_SAVE_FPS,
                    )
                else:
                    captioned_seg = [frame.copy() for frame in gemini_seg]
                try:
                    _write_video_with_audio(
                        captioned_seg,
                        audio_records,
                        seg_video_path,
                        fps=_VLM_VIDEO_SAVE_FPS,
                    )
                    print(f"[AsyncVLM] #{call_id} Captioned async segment saved → {seg_video_path}")
                except Exception as save_exc:
                    print(f"[AsyncVLM] #{call_id} Failed to save captioned async segment: {save_exc}")

            if SAVE_ASYNC_VLM_INPUT_VIDEOS:
                vlm_input_path = os.path.join(
                    self._temp_dir,
                    f"vlm_input_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4")
                try:
                    os.makedirs(os.path.dirname(vlm_input_path) or ".", exist_ok=True)
                    with open(vlm_input_path, "wb") as vf:
                        vf.write(video_bytes)
                    print(f"[AsyncVLM] #{call_id} VLM input saved → {vlm_input_path}")
                except Exception as save_exc:
                    print(f"[AsyncVLM] #{call_id} Failed to save VLM input: {save_exc}")

            if self._stop_event.is_set():
                log_entry["command"] = "DROPPED_STOPPED"
                return

            print(f"[AsyncVLM] #{call_id} → Gemini request "
                  f"({len(video_bytes)/1024:.1f} KB)")
            log_entry["gemini_start_t"] = self._elapsed()

            raw, _, latency = _query_gemini_async_update(
                video_bytes=video_bytes,
                queue=self._queue,
                seg_start_sec=seg_start_sec,
                seg_end_sec=seg_end_sec,
                transcript_section=transcript_section,
                api_key=self._api_key,
                prompt_template=PROMPT_TEMPLATE_ASYNC_UPDATE,
                config=GEMINI_REQUEST_CONFIG,
            )

            if self._stop_event.is_set():
                log_entry["command"] = "DROPPED_STOPPED"
                print(f"[AsyncVLM] #{call_id} response arrived after stop — dropping.")
                return

            gemini_end_t = self._elapsed()
            log_entry["gemini_end_t"] = gemini_end_t
            log_entry["latency"] = latency

            # ---- resolve raw Gemini commands → typed queue commands ----
            if isinstance(raw, _RawAppendMany):
                keyframes = []
                instructions = []
                for elem in raw.elements:
                    abs_idx = seg_start + elem.local_frame_idx
                    cropped_bgr = _copied_keyframe_at(elem.local_frame_idx)
                    if cropped_bgr is None:
                        print(f"[AsyncVLM] #{call_id} frame {abs_idx} not in buffer "
                              "for AppendMany element — skipping")
                        continue
                    keyframes.append(cropped_bgr)
                    instructions.append(elem.instruction)

                if not keyframes:
                    raw = CmdNoOp()
                elif len(keyframes) == 1:
                    raw = CmdAppend(
                        keyframe=keyframes[0],
                        instruction=instructions[0],
                    )
                else:
                    raw = CmdAppendMany(
                        keyframes=keyframes,
                        instructions=instructions,
                    )

            if isinstance(raw, _RawCommand):
                abs_idx        = seg_start + raw.local_frame_idx
                cropped_bgr = _copied_keyframe_at(raw.local_frame_idx)
                if cropped_bgr is None:
                    print(f"[AsyncVLM] #{call_id} frame {abs_idx} not in buffer — NoOp")
                    raw = CmdNoOp()
                else:
                    if raw.name == "Append":
                        raw = CmdAppend(keyframe=cropped_bgr,
                                        instruction=raw.instruction)
                    elif raw.name == "Replace":
                        if raw.target_idx is None:
                            raw = CmdNoOp()
                        else:
                            raw = CmdReplace(target_idx=raw.target_idx,
                                             keyframe=cropped_bgr,
                                             instruction=raw.instruction)
                    elif raw.name == "Preempt":
                        raw = CmdPreempt(keyframe=cropped_bgr,
                                         instruction=raw.instruction)

            cmd = raw
            log_entry["command"] = type(cmd).__name__
            queue_log = self._queue.apply(cmd)
            print(f"[AsyncVLM] #{call_id} {queue_log}")

            self.event_log.append((
                self.current_step,
                _format_cmd_label(cmd),
            ))

            if isinstance(cmd, CmdPreempt):
                self.preempt_event.set()

        except Exception as exc:
            log_entry["command"] = "ERROR"
            print(f"[AsyncVLM] #{call_id} failed: {exc}")
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Initial keyframe extraction  (interactive, before robot execution)
# ---------------------------------------------------------------------------

def extract_keyframes(env, language_instruction: str,
                      api_key: str, video_buffer: VideoBuffer,
                      sim_recorder: Optional[SimulationVideoRecorder] = None,
                      ) -> KeyframeQueue:
    """
    Stream live aria | top frames.
    Press 'p' once to start recording the initial instruction clip, then
    press 'p' again to stop recording, call Gemini, and build the initial
    KeyframeQueue. Press 's' before recording to skip the initial Gemini
    query and start with the current cropped camera view as the fallback
    keyframe. Press 'q' to quit the program.

    If *sim_recorder* is provided, writes the init phase (Initializing...)
    frames and then the queue-initialized frames into the simulation video.

    Returns a KeyframeQueue ready for use in the eval loop.
    """
    window_name       = "Keyframe Selector"
    os.makedirs(_TEMP_DIR, exist_ok=True)

    video_path         = os.path.join(_TEMP_DIR, "video.mp4")
    cropped_video_path = os.path.join(_TEMP_DIR, "cropped_video.mp4")
    gaze_path          = os.path.join(_TEMP_DIR, "gaze.json")
    keyframe_meta_path = os.path.join(_TEMP_DIR, "keyframes.json")

    aria_frames         = []
    aria_cropped_frames = []
    gaze_records        = []
    audio_records       = []
    capture_timer       = Timer(_VIDEO_RECORD_FPS)
    capturing           = False

    def _clear_capture():
        aria_frames.clear()
        aria_cropped_frames.clear()
        gaze_records.clear()
        audio_records.clear()

    def _append_frame(aria_bgr, cropped_bgr, gx, gy, audio_chunk,
                      audio_ts_ns, audio_sample_rate_hz, audio_num_samples):
        frame_idx = len(gaze_records)
        aria_frames.append(aria_bgr)
        aria_cropped_frames.append(cropped_bgr)
        gaze_records.append(
            {"frame_idx": frame_idx, "gaze_x": gx, "gaze_y": gy})
        audio_records.append({
            "frame_idx": len(audio_records),
            "audio_ts_ns": audio_ts_ns,
            "sample_rate_hz": audio_sample_rate_hz,
            "num_samples": audio_num_samples,
            "chunk_pcm16": audio_chunk.copy(),
        })

        video_buffer.append(
            aria_bgr,
            cropped_bgr,
            gx,
            gy,
            audio_chunk_pcm16=audio_chunk,
            audio_ts_ns=audio_ts_ns,
            audio_sample_rate_hz=audio_sample_rate_hz,
            audio_num_samples=audio_num_samples,
        )

    print("Initial keyframes: p=start recording, p=stop/query, "
          "s=start without initial query, q=quit.")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation(
                    mode='human_view_streamed_and_keyframe_cropped'))
            status_text = (
                ("RECORDING initial clip",
                 f"{len(aria_frames) / _VIDEO_RECORD_FPS:.1f}s  p=stop/query  q=quit")
                if capturing
                else ("READY",
                      "p=start initial  s=skip initial  q=quit")
            )
            live_frame = _build_preview_frame(
                live_obs, status_text,
                status_color=(0, 0, 255) if capturing else (0, 255, 0))

            aria_bgr    = live_obs["images"]["aria_rgb"].copy()
            cropped_bgr = live_obs["images"]["aria_cropped_rgb"].copy()
            gaze_pt     = np.asarray(
                live_obs["gaze_point"], dtype=np.float64).reshape(-1)
            gx = float(gaze_pt[0]) if gaze_pt.size >= 1 else None
            gy = float(gaze_pt[1]) if gaze_pt.size >= 2 else None
            audio_obs = live_obs.get("audio", {})
            audio_chunk = np.asarray(
                audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
                dtype=np.int16,
            )
            audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
            audio_sample_rate_hz = int(audio_obs.get("sample_rate_hz", 0) or 0)
            audio_num_samples = int(audio_obs.get("num_samples", audio_chunk.size) or 0)

            if capturing:
                _append_frame(
                    aria_bgr, cropped_bgr, gx, gy, audio_chunk,
                    audio_ts_ns, audio_sample_rate_hz, audio_num_samples)

            cv2.imshow(window_name, live_frame)
            key = _poll_keypress()
            capture_timer.sleep()   # pace the loop to _VIDEO_RECORD_FPS

            if key in (ord('q'), ord('Q'), 27):
                raise ProgramQuit

            if key in (ord('s'), ord('S')):
                queue = KeyframeQueue([], [])
                print("[keyframe] Skipping initial Gemini/keyframe detection; "
                      "starting with an empty queue for async VLM append.")
                if sim_recorder is not None:
                    sim_recorder.record_frame(aria_bgr, queue, 0.0)
                return queue

            if key in (ord('p'), ord('P')) and not capturing:
                _clear_capture()
                capturing = True
                print("[keyframe] Initial recording started. Press 'p' again to stop/query.")
                continue

            if key in (ord('p'), ord('P')) and capturing and aria_frames:
                capturing = False
                total = len(aria_frames)
                duration_sec = total / _VIDEO_RECORD_FPS
                print(f"[keyframe] Captured {total} frames @ {_VIDEO_RECORD_FPS} FPS "
                      f"({duration_sec:.2f}s)")

                # Save raw video + audio for logging
                _write_video_with_audio(
                    aria_frames, audio_records, video_path, fps=_VIDEO_RECORD_FPS)
                _write_video(
                    aria_cropped_frames, cropped_video_path,
                    fps=_VIDEO_RECORD_FPS)
                with open(gaze_path, "w") as f:
                    json.dump({"frames": gaze_records}, f, indent=2)

                gemini_frames = _select_vlm_frames(
                    aria_frames, aria_cropped_frames, _GEMINI_VIDEO_SOURCE)

                # Prepare VLM input: optional gaze crop → subsample → whisper → audio
                print("[keyframe] Preparing VLM input "
                      f"({_GEMINI_VIDEO_SOURCE_LABEL} + audio)...")
                vlm_bytes, transcript_section, _ = _prepare_vlm_video(
                    gemini_frames, gaze_records, audio_records,
                    clip_start_sec=0.0,
                    frames_are_gaze_cropped=_GEMINI_USES_CROPPED_VIDEO,
                    config=VLM_VIDEO_CONFIG,
                )
                initial_vlm_input_path = os.path.join(
                    _TEMP_DIR, f"initial_vlm_input_{_GEMINI_VIDEO_SOURCE}.mp4")
                try:
                    with open(initial_vlm_input_path, "wb") as vf:
                        vf.write(vlm_bytes)
                    print(f"[keyframe] VLM input saved → {initial_vlm_input_path}")
                except Exception as save_exc:
                    print(f"[keyframe] Failed to save VLM input: {save_exc}")

                print(f"[keyframe] VLM input: {len(vlm_bytes)/1024:.1f} KB")
                print("[keyframe] Querying Gemini for initial keyframes...")
                try:
                    parsed, latency = _query_gemini_initial(
                        video_bytes=vlm_bytes,
                        transcript_section=transcript_section,
                        api_key=api_key,
                        duration_sec=duration_sec,
                        prompt_template=PROMPT_TEMPLATE_INITIAL,
                        config=GEMINI_REQUEST_CONFIG,
                    )
                except Exception as exc:
                    print(f"[keyframe] Gemini failed after retries: "
                          f"{_short_error_reason(exc)}")
                    print("[keyframe] Press 'p' to record again, 's' to skip, or 'q' to quit.")
                    _clear_capture()
                    continue
                print(f"[keyframe] Gemini responded in {latency:.1f}s")

                # ── Parse action_queue from 2-step response ──
                action_queue = parsed.get("action_queue", [])
                if not action_queue:
                    # Fallback to legacy format
                    action_queue = []
                    for iv, instr in zip(
                        parsed.get("keyframe_intervals", []),
                        parsed.get("subgoal_instructions", []),
                    ):
                        action_queue.append({
                            "keyframe_description": "",
                            "subtask_instruction": instr,
                            "keyframe_interval": iv,
                        })

                keyframe_imgs  = []
                subtask_instrs = []
                saved_items    = []

                for i1, elem in enumerate(action_queue, start=1):
                    interval = elem.get("keyframe_interval", {})
                    instruction = elem.get("subtask_instruction", language_instruction)
                    description = elem.get("keyframe_description", "")

                    # Extract mid-frame from the original (full-fps) frames
                    frame_idx = _mid_frame_from_interval(
                        interval, total, _VIDEO_RECORD_FPS)
                    frame_idx = max(0, min(frame_idx, total - 1))

                    cropped_kf = aria_cropped_frames[frame_idx].copy()

                    cropped_path = os.path.join(
                        _TEMP_DIR, f"cropped_keyframe_{i1:02d}.png")
                    cv2.imwrite(cropped_path, cropped_kf)
                    print(f"[keyframe] #{i1} (frame {frame_idx}) "
                          f"{instruction} → {cropped_path}")

                    keyframe_imgs.append(cropped_kf)
                    subtask_instrs.append(instruction)
                    saved_items.append({
                        "order":                 i1,
                        "frame_idx":             int(frame_idx),
                        "cropped_keyframe_path": cropped_path,
                        "instruction":           instruction,
                        "description":           description,
                    })

                with open(keyframe_meta_path, "w") as f:
                    json.dump({
                        "instruction":          language_instruction,
                        "num_frames":           total,
                        "video_fps":            _VIDEO_RECORD_FPS,
                        "vlm_fps":              VLM_FPS,
                        "subtask_instructions": subtask_instrs,
                        "saved_keyframes":      saved_items,
                        "gemini_response":      parsed,
                        "latency":              latency,
                    }, f, indent=2, ensure_ascii=False)

                print(f"[keyframe] action_queue: {len(keyframe_imgs)} elements")
                for idx, instr in enumerate(subtask_instrs, 1):
                    print(f"  [{idx}] {instr}")

                if not keyframe_imgs:
                    print("[keyframe] Gemini returned no usable action queue. "
                          "Press 'p' to record again, 's' to skip, or 'q' to quit.")
                    _clear_capture()
                    continue

                result_queue = KeyframeQueue(keyframe_imgs, subtask_instrs)

                # ── Write init phase to simulation video when enabled ──
                if sim_recorder is not None:
                    print("[SimVideo] Queueing initial simulation frames asynchronously.")
                    # Phase 1: all frames before 'p' → "Initializing..."
                    sim_recorder.record_init_phase(
                        aria_frames, queue=None, fps=_VIDEO_RECORD_FPS)
                    # Phase 2: show initialized queue for ~2 seconds
                    for _ in range(int(SIM_FPS * 2)):
                        last_frame = aria_frames[-1] if aria_frames else None
                        sim_recorder.record_frame(
                            last_frame, result_queue, duration_sec)

                return result_queue
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def eval(env, client, language_instruction: str, horizon: int,
         queue: KeyframeQueue, video_buffer: VideoBuffer,
         vlm_worker: AsyncVLMWorker,
         sim_recorder: Optional[SimulationVideoRecorder] = None,
         chunksize: int = 8):
    """
    Robot execution loop.

    The async VLM worker runs concurrently and may modify `queue` at any
    time.  The loop reacts to four triggers:

      1. vlm_worker.preempt_event  — robot reset + new keyframe from VLM
      2. completion_prob >= 0.8    — auto-advance to next keyframe
      3. 'n' key                   — manual advance
      4. 'e' key                   — end current episode and reset robot
      5. 'q' / ESC                 — quit the program
    """
    _require_initial_keyframes_ready(queue, context="eval")

    pending_auto_advance = False
    top_view_frames_list  = []
    comp_prob_log         = []
    global_step          = 0
    subtask_step         = 0
    timer                = Timer(env.control_hz)
    window_name          = "Eval  |  n=next kf  e=end episode  q=quit"
    rollout_video_path   = os.path.join(_TEMP_DIR, "rollout.mp4")
    keyframe_log: List[Tuple[int, np.ndarray]] = []
    saved_queue_debug_keys = set()
    async_collection_started = False
    kf = queue.current_keyframe
    if kf is not None:
        keyframe_log.append((0, kf.copy()))
    if GLOBAL_INSTRUCTION:
        print(
            "[Eval] GLOBAL_INSTRUCTION=True; policy prompt will stay "
            f"'{_task_default_instruction(PROMPT_TASK)}' for task '{PROMPT_TASK}'."
        )

    # ------------------------------------------------------------------
    # The background streamer and async VLM worker are intentionally created
    # only after the initial queue exists, and started later at the first
    # robot action. This prevents any async model activity during initial
    # keyframe extraction.
    # ------------------------------------------------------------------
    _bg_streamer = BackgroundObservationStreamer(
        env, video_buffer, target_fps=_VIDEO_RECORD_FPS,
    )

    for step in range(horizon):
        try:
            # ----------------------------------------------------------------
            # React to async Preempt (between inference cycles)
            # ----------------------------------------------------------------
            if vlm_worker.preempt_event.is_set():
                vlm_worker.preempt_event.clear()
                print("[Eval] Preempt received — resetting robot.")
                cv2.destroyAllWindows()
                env.reset()
                client = websocket_client_policy.WebsocketClientPolicy(
                    host="localhost", port=8000)
                subtask_step = 0
                kf = queue.current_keyframe
                if kf is not None:
                    keyframe_log.append((global_step, kf.copy()))
                print(f"[Eval] Preempt → KF [{queue.current_idx + 1}/{queue.total}]:"
                      f" '{queue.current_instruction}'")

            if queue.is_done:
                # Plan-A fix #2: hold pose instead of resetting.
                # The background streamer keeps feeding the buffer so the
                # VLM worker can still observe the scene and Append new
                # subtasks. We block here until something new arrives.
                if not async_collection_started:
                    video_buffer.skip_to_current()
                    _bg_streamer.start()
                    vlm_worker.start()
                    async_collection_started = True
                    print("[Eval] Empty queue: robot holds pose; async VLM "
                          "collection started for keyframe append.")

                hold_control = {"value": None}

                def _hold_key_poll():
                    key = _poll_keypress()
                    control = _episode_control_from_key(key)
                    if control is not None:
                        hold_control["value"] = control
                        return ord('q')
                    if key in (ord('s'), ord('S')):
                        return -1
                    return key

                aborted = wait_for_queue_items(
                    queue,
                    window_name=window_name,
                    preview_builder=lambda: _build_eval_preview(
                        _extract_latest_observation(env.get_observation(
                            mode='human_view_streamed_and_keyframe_cropped',
                            audio_drain=False)),
                        queue.current_keyframe, queue,
                        queue.current_instruction or ""),
                    key_poll=_hold_key_poll,
                )
                if aborted:
                    cv2.destroyAllWindows()
                    if hold_control["value"] == "episode_end":
                        raise EpisodeEnd
                    raise ProgramQuit
                # New subtask appeared — refresh keyframe bookkeeping.
                subtask_step = 0
                kf = queue.current_keyframe
                if kf is not None:
                    keyframe_log.append((global_step, kf.copy()))
                continue

            # ----------------------------------------------------------------
            # Observe
            # ----------------------------------------------------------------
            obs         = env.get_observation(
                mode="human_view_streamed_and_keyframe_cropped",
                audio_drain=False,
            )["timestamp"]
            images      = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for v in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, v])

            keyframe_img  = queue.current_keyframe
            current_instr = queue.current_instruction
            policy_prompt = _policy_language_instruction(
                current_instr or language_instruction)

            if keyframe_img is not None:
                keyframe_img = cv2.cvtColor(keyframe_img, cv2.COLOR_BGR2RGB)
                aria_rgb     = cv2.cvtColor(images["aria_rgb"], cv2.COLOR_BGR2RGB)
            observation = {
                "observation.images.top_view":                    images["top_camera_rgb"],
                "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
                "observation.images.human_view_keyframe_cropped": keyframe_img,
                "prompt":                                         policy_prompt,
                "observation.state":                              concatenated_state,
            }

            if global_step == 0 and keyframe_img is not None:
                cv2.imwrite(
                    os.path.join(_TEMP_DIR, "test_aria_keyframe.png"),
                    cv2.cvtColor(keyframe_img, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(_TEMP_DIR, "test_aria_rgb.png"),
                    cv2.cvtColor(aria_rgb, cv2.COLOR_RGB2BGR),
                )
            _save_queue_keyframes_debug(
                queue, keyframe_img, global_step, saved_queue_debug_keys)
            # ----------------------------------------------------------------
            # Infer
            # ----------------------------------------------------------------
            out             = client.infer(observation)
            action_chunk    = out["actions"][:chunksize, :]
            completion_prob = (float(out["completion_prob"])
                               if "completion_prob" in out else None)
            print("Completion prob:",
                  f"{completion_prob:.3f}" if completion_prob is not None else "N/A")
            if completion_prob is not None:
                comp_prob_log.append((global_step, completion_prob))

            # ----------------------------------------------------------------
            # Execute action chunk
            # ----------------------------------------------------------------
            for raw_action in action_chunk:
                # ── STOP_ROBOT_WHILE_RECV: wait for VLM if voice detected ──
                # Robot control is paused, but we keep streaming observations,
                # buffering video for VLM, recording sim frames, and updating
                # the CV2 preview so nothing is lost.
                if STOP_ROBOT_WHILE_RECV and vlm_worker.robot_pause_event.is_set():
                    print("[Eval] Robot paused — waiting for VLM response "
                          "(streamer keeps feeding video_buffer)...")
                    _pause_timer = Timer(env.control_hz)
                    while vlm_worker.robot_pause_event.is_set():
                        # Streamer keeps feeding video_buffer in the background;
                        # we only need a live preview for the CV2 window.
                        _pause_obs = _extract_latest_observation(
                            env.get_observation(
                                mode='human_view_streamed_and_keyframe_cropped',
                                audio_drain=False))
                        _p_aria = _pause_obs["images"]["aria_rgb"]
                        global_step += 1
                        vlm_worker.current_step = global_step
                        # Keep recording sim video
                        if (sim_recorder is not None
                                and global_step % max(1, int(env.control_hz / SIM_FPS)) == 0):
                            sim_t = global_step / env.control_hz
                            sim_recorder.record_frame(_p_aria, queue, sim_t)
                        # Keep updating CV2 preview
                        _pause_preview = _build_eval_preview(
                            _pause_obs, keyframe_img, queue, current_instr)
                        cv2.putText(_pause_preview, "PAUSED (waiting for VLM)",
                                    (10, _pause_preview.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow(window_name, _pause_preview)
                        key = _poll_keypress()
                        _raise_for_episode_control(key)
                        _pause_timer.sleep()
                    print("[Eval] Robot resumed.")

                # Check for mid-chunk Preempt.
                #if vlm_worker.preempt_event.is_set():
                #    vlm_worker.preempt_event.clear()
                #    print("[Eval] Preempt mid-chunk — resetting robot.")
                #    cv2.destroyAllWindows()
                #    env.reset()
                #    client = websocket_client_policy.WebsocketClientPolicy(
                #        host="localhost", port=8000)
                #    subtask_step  = 0
                #    keyframe_img  = queue.current_keyframe
                #    current_instr = queue.current_instruction
                #    if keyframe_img is not None:
                #        keyframe_log.append((global_step, keyframe_img.copy()))
                #    print(f"[Eval] Preempt → KF [{queue.current_idx + 1}/{queue.total}]:"
                #          f" '{current_instr}'")
                #    break  # discard remaining actions; re-infer next iteration

                action = {
                    'left_arm_positions':    raw_action[0:6],
                    'left_gripper_position': raw_action[6:7],
                    'right_arm_positions':   raw_action[7:13],
                    'right_gripper_position': raw_action[13:14],
                }
                env.step(action)
                if not async_collection_started:
                    video_buffer.skip_to_current()
                    _bg_streamer.start()
                    vlm_worker.start()
                    async_collection_started = True
                    print("[Eval] Async VLM collection started after first robot action.")
                global_step  += 1
                subtask_step += 1
                vlm_worker.current_step = global_step

                live_obs = _extract_latest_observation(
                    env.get_observation(
                        mode='human_view_streamed_and_keyframe_cropped',
                        audio_drain=False,
                    ))
                top_view_frames_list.append(
                    live_obs["images"]["top_camera_rgb"].copy())

                # NOTE: video_buffer is now populated by
                # BackgroundObservationStreamer; no manual append here.
                live_aria_bgr    = live_obs["images"]["aria_rgb"]

                # Record sim frame (at ~control_hz, subsampled by writer)
                if sim_recorder is not None and global_step % max(1, int(env.control_hz / SIM_FPS)) == 0:
                    sim_t = global_step / env.control_hz
                    sim_recorder.record_frame(live_aria_bgr, queue, sim_t)

                preview = _build_eval_preview(
                    live_obs, keyframe_img, queue, current_instr)
                cv2.imshow(window_name, preview)
                key = _poll_keypress()

                if key in (ord('n'), ord('N')):
                    if not queue.advance():
                        # Plan-A fix #2: queue exhausted → hold pose
                        # instead of resetting. Outer loop will call
                        # wait_for_queue_items() on the next iteration.
                        print("[Eval] Manual advance — queue empty, "
                              "holding pose until VLM appends a new subtask.")
                    else:
                        subtask_step  = 0
                        keyframe_img  = queue.current_keyframe
                        current_instr = queue.current_instruction
                        if keyframe_img is not None:
                            keyframe_log.append((global_step, keyframe_img.copy()))
                        print(f"  → Manual KF [{queue.current_idx + 1}/{queue.total}]:"
                              f" '{current_instr}'")
                    break

                else:
                    _raise_for_episode_control(key)

                timer.sleep()

            # ----------------------------------------------------------------
            # Auto-advance via completion_prob
            # ----------------------------------------------------------------
            if completion_prob is not None and completion_prob >= 0.8:
                if subtask_step < 40:
                    print(f"  [Auto] cp={completion_prob:.3f} >= 0.8 but "
                          f"subtask_step={subtask_step} < 40 — skip")
                else:
                    print(f"  [Auto] cp={completion_prob:.3f} >= 0.8 "
                          f"(subtask_step={subtask_step}) — advancing...")
                    pending_auto_advance = True

            if pending_auto_advance:
                pending_auto_advance = False
                if not queue.advance():
                    # Plan-A fix #2: queue exhausted → hold pose instead of
                    # resetting. The outer loop will enter wait_for_queue_items
                    # on its next iteration.
                    print("[Auto] Queue exhausted — holding pose until VLM "
                          "appends a new subtask.")
                else:
                    client = websocket_client_policy.WebsocketClientPolicy(
                        host="localhost", port=8000)
                    subtask_step = 0
                    kf = queue.current_keyframe
                    if kf is not None:
                        keyframe_log.append((global_step, kf.copy()))
                    print(f"  [Auto] → KF [{queue.current_idx + 1}/{queue.total}]:"
                          f" '{queue.current_instruction}'")

        except (EpisodeEnd, ProgramQuit, KeyboardInterrupt) as exc:
            _bg_streamer.stop()
            vlm_worker.stop()
            cv2.destroyAllWindows()
            if isinstance(exc, EpisodeEnd):
                env.reset()
            _save_rollout_video(
                top_view_frames_list, comp_prob_log, rollout_video_path,
                fps=_VIDEO_RECORD_FPS,
                vlm_event_log=vlm_worker.event_log,
                keyframe_log=keyframe_log,
                event_display_frames=_VLM_EVENT_DISPLAY_FRAMES)
            _safe_finalize_sim_recorder(sim_recorder, video_buffer)
            if isinstance(exc, KeyboardInterrupt):
                raise ProgramQuit
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    _bg_streamer.stop()
    vlm_worker.stop()
    cv2.destroyAllWindows()
    _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path,
                        fps=_VIDEO_RECORD_FPS,
                        vlm_event_log=vlm_worker.event_log,
                        keyframe_log=keyframe_log,
                        event_display_frames=_VLM_EVENT_DISPLAY_FRAMES)
    _safe_finalize_sim_recorder(sim_recorder, video_buffer)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Async VLM DTQ eval with optional center gaze marker on cropped VLM input.")
    prompt_names = _list_prompt_names()
    parser.add_argument(
        "--vlm-draw-gaze-marker",
        dest="vlm_draw_gaze_marker",
        action="store_true",
        help="Overlay a center gaze marker on cropped frames before sending them to the VLM.",
    )
    parser.add_argument(
        "--no-vlm-draw-gaze-marker",
        dest="vlm_draw_gaze_marker",
        action="store_false",
        help="Do not draw a center gaze marker on cropped VLM input frames.",
    )
    parser.add_argument(
        "--record-sim-video",
        dest="record_sim_video",
        action="store_true",
        help="Record keyframe_simulation.mp4.",
    )
    parser.add_argument(
        "--no-record-sim-video",
        dest="record_sim_video",
        action="store_false",
        help="Do not record keyframe_simulation.mp4.",
    )
    parser.add_argument(
        "--prompt-task",
        default=PROMPT_TASK,
        choices=prompt_names or None,
        help="Task prompt file to load from ours_prompts/<prompt-task>.txt.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Policy language instruction. Defaults to the selected prompt task.",
    )
    parser.set_defaults(
        vlm_draw_gaze_marker=VLM_DRAW_GAZE_MARKER,
        record_sim_video=SAVE_SIMULATION_VIS,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_cli_args()
    _set_prompt_task(args.prompt_task, announce=True)
    VLM_DRAW_GAZE_MARKER = args.vlm_draw_gaze_marker
    VLM_VIDEO_CONFIG = _build_vlm_video_config(
        video_record_fps=_VIDEO_RECORD_FPS,
        vlm_fps=VLM_FPS,
        target_res=VLM_TARGET_RES,
        gaze_crop_enabled=GAZE_CROP_ENABLED,
        gaze_crop_size=GAZE_CROP_SIZE,
        gaze_calibration_y_offset=GAZE_CALIBRATION_Y_OFFSET,
        draw_gaze_marker=VLM_DRAW_GAZE_MARKER,
        debug_banner=VLM_DEBUG_BANNER,
        video_source=_GEMINI_VIDEO_SOURCE,
        whisper_enabled=WHISPER_ENABLED,
        useful_voice_min_chars=USEFUL_VOICE_MIN_CHARS,
        voiced_video=VOICED_VIDEO,
    )

    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env                  = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = (
        args.instruction
        or _task_default_instruction(PROMPT_TASK)
    )
    horizon              = 10000
    GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    print("Starting. Instruction:", language_instruction)
    print("VLA policy instruction:", _policy_language_instruction(language_instruction))
    print("Initial controls: p=start/stop initial recording  s=start without initial  q=quit")
    print("Eval controls: n=next keyframe  e=end episode/reset  q=quit program")
    print("Prompt task:", PROMPT_TASK)
    print("VLM video source:", _GEMINI_VIDEO_SOURCE_LABEL)
    print("VLM center gaze marker:", VLM_DRAW_GAZE_MARKER)
    print(
        "VLM optimized config:",
        f"record_fps={_VIDEO_RECORD_FPS}",
        f"vlm_fps={VLM_FPS}",
        f"interval={VLM_INTERVAL_SEC}s",
        f"res={VLM_TARGET_RES}",
        f"thinking={VLM_THINKING_BUDGET}",
        f"skip_empty_transcript={ASYNC_VLM_REQUIRE_USEFUL_VOICE}",
    )
    print("Simulation video:", "enabled" if args.record_sim_video else "disabled")

    def _options_menu(lang_instr):
        print(f"\nCurrent instruction: '{lang_instr}'")
        print("Options:")
        print("  1. Press Enter         → continue with current instruction")
        print("  2. Type new instruction → update and continue")
        print("  3. Type 'quit'          → exit")
        user_input = input("Enter your choice: ").strip()
        if user_input.lower() == "quit":
            return None
        return user_input if user_input else lang_instr

    def _new_video_buffer():
        return VideoBuffer(
            store_full_aria_in_vlm_buffer=STORE_FULL_ARIA_IN_VLM_BUFFER)

    def _new_sim_recorder():
        if not args.record_sim_video:
            return None
        sim_output_path = os.path.join(_TEMP_DIR, "keyframe_simulation.mp4")
        return SimulationVideoRecorder(sim_output_path, SIM_VIDEO_CONFIG)

    video_buffer = None
    sim_recorder = None
    vlm_worker = None

    try:
        while True:
            video_buffer = _new_video_buffer()
            sim_recorder = _new_sim_recorder()
            vlm_worker = None

            try:
                # ---- Initial keyframe extraction (interactive) ----
                with _terminal_keypress_mode():
                    queue = extract_keyframes(
                        env, language_instruction, GEMINI_API_KEY,
                        video_buffer, sim_recorder=sim_recorder)
                _require_initial_keyframes_ready(queue, context="episode startup")

                # ---- Connect to policy server ----
                client = websocket_client_policy.WebsocketClientPolicy(
                    host="localhost", port=8000)

                # ---- Create async VLM worker ----
                # Actual collection starts inside eval() after the first env.step().
                video_buffer.skip_to_current()
                vlm_worker = AsyncVLMWorker(
                    video_buffer=video_buffer,
                    queue=queue,
                    global_instruction=_policy_language_instruction(language_instruction),
                    api_key=GEMINI_API_KEY,
                    interval_sec=VLM_INTERVAL_SEC,
                )

                with _terminal_keypress_mode():
                    eval(env, client, language_instruction, horizon,
                         queue, video_buffer, vlm_worker,
                         sim_recorder=sim_recorder)
                print("Episode finished.")
            except EpisodeEnd:
                print("Episode ended. Robot reset; moving to next episode.")
                continue
            except ProgramQuit:
                print("Exiting evaluation...")
                break
            finally:
                if vlm_worker is not None:
                    vlm_worker.stop()
                _safe_finalize_sim_recorder(sim_recorder, video_buffer)

            result = _options_menu(language_instruction)
            if result is None:
                print("Exiting evaluation...")
                break
            language_instruction = result

    except KeyboardInterrupt:
        print("Exiting evaluation...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        if vlm_worker is not None:
            vlm_worker.stop()
        _safe_finalize_sim_recorder(sim_recorder, video_buffer)
        os.makedirs(_TEMP_DIR, exist_ok=True)
        print(f"Preserved temp directory: {_TEMP_DIR}")
        env.safe_end()
