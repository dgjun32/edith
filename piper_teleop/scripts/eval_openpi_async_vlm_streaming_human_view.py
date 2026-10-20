"""eval_openpi_async_vlm_gaze_keyframe_queue.py

Async VLM-based keyframe queue management for dual-arm robot evaluation.

Architecture
------------
  Main thread   : robot eval loop + CV2 display.
  VLM thread    : background Gemini calls that manipulate the KeyframeQueue.

Human-view (aria_rgb) frames are continuously written into a shared
VideoBuffer starting from t=0.  The async worker fires whenever
N_TRIGGER_FRAMES new unprocessed frames accumulate, producing exactly
one of five queue-manipulation commands:

    NoOp()
    Append(keyframe, instruction)
    Replace(target_idx, keyframe, instruction)
    Preempt(keyframe, instruction)   ← signals robot reset, inserts at front
    Remove(target_idx)
"""

import numpy as np
import cv2
import os
import json
import subprocess
import threading
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from google import genai
from google.genai import types

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from eval_vlm_baseline import PROMPT_CONFIG_NOTES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS   = 30
_MAX_INTERVALS      = 8
_PROMPT_CONFIG_NAME = "no_cap_no_gaze"
_TEMP_DIR           = os.path.normpath("/home/alin/projects/piper_project/temp")
_TEMP_DIR           = os.path.normpath("/home/alin/projects/piper_project/temp")
_BLACK_TOP_KEYFRAME = np.zeros((240, 424, 3), dtype=np.uint8)

# Which Aria view to encode into the mp4 sent to Gemini.
# Options:
#   - "aria_rgb": full egocentric RGB frames
#   - "aria_cropped_rgb": gaze-centered cropped RGB frames
_GEMINI_VIDEO_SOURCE = "aria_cropped_rgb"

# How many real-world seconds to wait between successive async VLM calls.
# The worker sleeps for this duration, then drains all frames accumulated
# during that interval regardless of how many were captured.
VLM_INTERVAL_SEC = 10.0

# Number of top-view (global) steps for which a VLM command overlay stays
# visible on the rollout video after it fires.
_VLM_EVENT_DISPLAY_FRAMES = 90

# FPS for saving async VLM segment videos (≈ control_hz; files kept for logging).
_VLM_VIDEO_SAVE_FPS = 15


# ---------------------------------------------------------------------------
# Gemini prompts
# ---------------------------------------------------------------------------

# --- Initial prompt (before robot execution, unchanged from previous version) ---
PROMPT_TEMPLATE_INITIAL = """\
You are analyzing a first-person (egocentric) video recorded in a store.
The wearer is looking at items on shelves and interacts with specific objects.

A spoken instruction was given:
    "{transcript}"

{config_note}

Your task has two steps:

**Step 1: Determine the number of keyframes.**
Analyze the spoken instruction and count the number of distinct object
references or interaction moments. Each time the instruction refers to a
target object (by pointing words like "this", "that", or by naming it),
it corresponds to one keyframe — the moment the wearer interacts with
that object.

Examples:
    - "I want this cupcake" → 1 keyframe
        subgoal_instructions: ["pick up this cupcake"]
    - "give me this cupcake and this beverage" → 2 keyframes
        subgoal_instructions: ["pick up this cupcake", "pick up this beverage"]
    - "put that one in this basket" → 2 keyframes
        subgoal_instructions: ["pick up that one", "place it in this basket"]
    - "put that one in this basket and the coca cola in this basket"
        → 4 keyframes
        subgoal_instructions: ["pick up that one", "place it in this basket",
            "pick up the coca cola", "place it in this basket"]
    - "Give me that soda" → 1 keyframe
        subgoal_instructions: ["pick up that soda"]
    - "Put the green pringles to the far left of the second shelf, and put the
        red pringles to the far right of the second shelf." → 4 keyframes
        subgoal_instructions: ["pick up the green pringles",
            "place it to the far left of the second shelf",
            "pick up the red pringles",
            "place it to the far right of the second shelf"]
    - "I want this muffin" → 1 keyframe
        subgoal_instructions: ["pick up the muffin with <specific topping description>"]

IMPORTANT: Do NOT use any other verbs (e.g., "point to", "grab", "put", "get",
    "move", "take"). Always rephrase to "pick up" or "place it".

CRITICAL — pick up vs. place it discrimination:
    - Whenever the instruction specifies a destination for an object (e.g.,
      "here", "there", "to the left", "in the basket", "on the shelf",
      "to the far right"), you MUST produce a SEPARATE keyframe for the
      placement step, using the verb "place it <destination>".
    - The keyframe for "place it" should capture the moment the wearer
      looks at or points to the DESTINATION (not the object being picked up).
    - Every "pick up" subtask that has an associated destination must be
      immediately followed by a "place it <destination>" subtask.
    - Do NOT merge pick-up and placement into a single keyframe/instruction.

**Step 2: Identify each keyframe's time range.**
For each keyframe (in chronological order), identify the precise time interval
that captures the moment of interaction with the corresponding object.

The keyframe intervals are determined by the convergence of multiple cues:
    - Utterance timing: when the instruction refers to the target object.
    - Hand gesture (pointing): when the hand/finger clearly points to a
        specific object or region.
    - Gaze fixation: when the wearer's gaze stabilises on the target object
        (if gaze information is available).

Prioritise intervals where the object indicated by hand pointing and the gaze
point are spatially aligned (same object/region). Use utterance timing to
confirm that this aligned object is the instructed target.

Each interval should be short and precise, typically 0.2–1.5 seconds.
Do not merge two clearly separate events into a single long interval.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
    "reasoning": "<brief explanation of which cues you used>",
    "num_keyframes": <int, number of distinct interaction moments>,
    "object_references": ["<object 1>", "<object 2>", ...],
    "subgoal_instructions": ["pick up <object>", "place it <destination>", ...],
    "keyframe_intervals": [
        {{"start": <float, seconds>, "end": <float, seconds>}}
    ]
}}

Important:
    - All lists must have the same length, equal to num_keyframes.
    - Maximum {max_intervals} intervals allowed.
    - Intervals must be sorted chronologically and non-overlapping.
    - Always return positive start/end values. Never return -1.
"""

# --- Async update prompt (called periodically during robot execution) ---
PROMPT_TEMPLATE_ASYNC_UPDATE = """\
You are a robot task planner monitoring live execution of a manipulation task.

The robot is following a keyframe queue: each entry is a subtask the robot
should complete before moving to the next.

**Global instruction (spoken by the human before execution):**
    "{global_instruction}"

**Current keyframe queue** (0-indexed; ">>>" marks the currently executing step):
{queue_state}

**Currently executing:** index {current_idx} — "{current_instruction}"

A new first-person (egocentric) video segment from the Aria glasses is provided:
    Segment timespan : {t_start:.1f}s – {t_end:.1f}s (relative to episode start)
    Segment duration : {segment_duration:.1f}s  (at {fps:.1f} FPS)

{config_note}

Analyze the video segment and decide whether the keyframe queue needs
to be modified.  Common reasons to modify:
  - The human points to a NEW object or location not yet in the queue.
  - The human makes a correction gesture or otherwise changes intent.
  - An unexpected event in the scene makes a queued step obsolete.
  - A previously listed step is no longer needed.

If nothing notable happens, output NoOp.

CRITICAL — when you generate a new instruction for Append, Replace, or Preempt,
make it maximally fine-grained and visually specific:
  - Use the most specific object description supported by the video segment.
  - If the human points to a muffin, identify exactly which muffin it is from
    its topping appearance instead of saying "this muffin" or "that muffin".
  - In this store scene, there are six visually distinct muffins:
      * muffin with grape topping
      * muffin with cherry topping
      * muffin with 2 strawberries topping
      * muffin with oreo and strawberry topping
      * muffin with pineapple topping
      * muffin with tangerine and strawberry topping
  - Prefer instructions like "pick up the muffin with pineapple topping"
    over generic instructions like "pick up this muffin".

**Available commands — output exactly one:**

  NoOp
      No change needed.

  Append
      Add a new (keyframe, instruction) pair at the end of the queue.
      Provide: keyframe_interval (relative to this segment), instruction.

  Replace
      Replace an existing queue entry.
      Provide: target_idx (0-based, any index), keyframe_interval,
               instruction.

  Preempt
      IMMEDIATELY interrupt the current task, reset the robot, and insert
      the new step at the front of the remaining queue (before target_idx
      is relevant here — the step is inserted at the current position).
      Provide: keyframe_interval, instruction.
      Use only when the human signals an urgent correction.

  Remove
      Delete a step that is no longer needed.
      Provide: target_idx (must be strictly greater than {current_idx},
               i.e. only future steps may be removed).

Return JSON:
{{
    "reasoning": "<brief explanation>",
    "command": "NoOp" | "Append" | "Replace" | "Preempt" | "Remove",
    "target_idx": <int or null>,
    "keyframe_interval": {{"start": <float, seconds in segment>, "end": <float, seconds in segment>}} or null,
    "instruction": "<string>" or null
}}

Important:
    - keyframe_interval times are relative to the START of this video segment
      (not the episode start).
    - target_idx is required only for Replace and Remove; null otherwise.
    - For NoOp and Remove, keyframe_interval and instruction must be null.
    - Always return positive start/end values. Never return -1.
"""


# ---------------------------------------------------------------------------
# Functional command dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CmdNoOp:
    pass


@dataclass
class CmdAppend:
    keyframe: np.ndarray
    instruction: str


@dataclass
class CmdReplace:
    target_idx: int
    keyframe: np.ndarray
    instruction: str


@dataclass
class CmdPreempt:
    keyframe: np.ndarray
    instruction: str


@dataclass
class CmdRemove:
    target_idx: int


# Internal: raw command before the frame is resolved into an ndarray.
@dataclass
class _RawCommand:
    name: str                    # "Append" | "Replace" | "Preempt"
    target_idx: Optional[int]
    local_frame_idx: int         # index within the segment
    instruction: str


# ---------------------------------------------------------------------------
# Thread-safe KeyframeQueue
# ---------------------------------------------------------------------------

class KeyframeQueue:
    """
    Thread-safe ordered list of (keyframe_bgr, instruction) pairs with a
    current-pointer that advances as the robot completes subtasks.
    """

    def __init__(self, keyframes: List[np.ndarray], instructions: List[str]):
        assert len(keyframes) == len(instructions)
        self._lock         = threading.Lock()
        self._items: List[Tuple[np.ndarray, str]] = list(zip(keyframes, instructions))
        self._current_idx  = 0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_idx(self) -> int:
        with self._lock:
            return self._current_idx

    @property
    def current_keyframe(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._current_idx < len(self._items):
                return self._items[self._current_idx][0]
            return None

    @property
    def current_instruction(self) -> Optional[str]:
        with self._lock:
            if self._current_idx < len(self._items):
                return self._items[self._current_idx][1]
            return None

    @property
    def is_done(self) -> bool:
        with self._lock:
            return self._current_idx >= len(self._items)

    @property
    def total(self) -> int:
        with self._lock:
            return len(self._items)

    def queue_state_str(self) -> str:
        """Human-readable queue dump for prompt injection."""
        with self._lock:
            if not self._items:
                return "  (empty)"
            lines = []
            for i, (_, instr) in enumerate(self._items):
                marker = ">>> " if i == self._current_idx else "    "
                lines.append(f"  [{i}] {marker}{instr}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mutations — all hold the lock
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """Move to the next item. Returns True if a next item exists."""
        with self._lock:
            if self._current_idx + 1 < len(self._items):
                self._current_idx += 1
                return True
            self._current_idx = len(self._items)  # mark exhausted
            return False

    def apply(self, cmd) -> str:
        """Apply a command and return a log string describing the change."""
        with self._lock:
            if isinstance(cmd, CmdNoOp):
                return "[Queue] NoOp."

            if isinstance(cmd, CmdAppend):
                self._items.append((cmd.keyframe, cmd.instruction))
                return f"[Queue] Append → [{len(self._items) - 1}]: '{cmd.instruction}'"

            if isinstance(cmd, CmdReplace):
                idx = cmd.target_idx
                if 0 <= idx < len(self._items):
                    self._items[idx] = (cmd.keyframe, cmd.instruction)
                    return f"[Queue] Replace [{idx}] → '{cmd.instruction}'"
                return f"[Queue] Replace: invalid idx {idx} (size={len(self._items)})"

            if isinstance(cmd, CmdPreempt):
                # Insert before the currently executing item; current_idx
                # stays the same, so it now points to the new preempting item.
                self._items.insert(self._current_idx, (cmd.keyframe, cmd.instruction))
                return f"[Queue] Preempt → inserted at [{self._current_idx}]: '{cmd.instruction}'"

            if isinstance(cmd, CmdRemove):
                idx = cmd.target_idx
                if idx <= self._current_idx or idx >= len(self._items):
                    return (f"[Queue] Remove: idx {idx} not removable "
                            f"(current={self._current_idx}, size={len(self._items)})")
                self._items.pop(idx)
                return f"[Queue] Remove [{idx}] done."

            return f"[Queue] Unknown command: {type(cmd)}"


# ---------------------------------------------------------------------------
# Thread-safe video / gaze ring buffer
# ---------------------------------------------------------------------------

class VideoBuffer:
    """
    Append-only buffer of aria frames shared between the main thread
    (writer) and the async VLM worker (reader).
    `_vlm_processed_up_to` tracks how far the worker has consumed.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._aria    : List[np.ndarray] = []
        self._cropped : List[np.ndarray] = []
        self._gaze    : List[dict]       = []
        self._audio   : List[dict]       = []
        self._vlm_processed_up_to: int   = 0

    def append(self, aria_bgr: np.ndarray, cropped_bgr: np.ndarray,
               gaze_x: Optional[float], gaze_y: Optional[float],
               audio_chunk_pcm16: Optional[np.ndarray] = None,
               audio_ts_ns: int = 0,
               audio_sample_rate_hz: int = 0,
               audio_num_samples: Optional[int] = None):
        with self._lock:
            idx = len(self._aria)
            self._aria.append(aria_bgr.copy())
            self._cropped.append(cropped_bgr.copy())
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

    def total_frames(self) -> int:
        with self._lock:
            return len(self._aria)

    def unprocessed_count(self) -> int:
        with self._lock:
            return len(self._aria) - self._vlm_processed_up_to

    def skip_to_current(self):
        """Mark all frames currently in the buffer as already processed.
        Call this before starting the async worker so pre-execution
        pointing-gesture frames are not re-analysed during rollout."""
        with self._lock:
            self._vlm_processed_up_to = len(self._aria)

    def get_segment(self) -> Tuple[List, List, List, int, int]:
        """Atomically drain all unprocessed frames and advance the pointer.
        Returns (aria_list, cropped_list, gaze_list, seg_start, seg_end).
        """
        with self._lock:
            start = self._vlm_processed_up_to
            end   = len(self._aria)
            aria    = [f.copy() for f in self._aria[start:end]]
            cropped = [f.copy() for f in self._cropped[start:end]]
            gaze    = list(self._gaze[start:end])
            self._vlm_processed_up_to = end
            return aria, cropped, gaze, start, end

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _write_video(frames_bgr, output_path, fps=_VIDEO_RECORD_FPS):
    if not frames_bgr:
        return
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    imageio.mimwrite(output_path, frames_rgb, fps=fps)


def _gemini_uses_cropped_video() -> bool:
    return _GEMINI_VIDEO_SOURCE == "aria_cropped_rgb"


def _gemini_video_source_label() -> str:
    if _gemini_uses_cropped_video():
        return "gaze-cropped aria_rgb"
    return "aria_rgb"


def _select_gemini_frames(aria_frames, cropped_frames):
    if _gemini_uses_cropped_video():
        return cropped_frames
    return aria_frames


def _collect_audio_pcm16(audio_records):
    chunks = []
    sample_rate_hz = 0
    for record in audio_records:
        chunk = np.asarray(
            record.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
            dtype=np.int16,
        ).reshape(-1)
        if chunk.size == 0:
            continue

        chunk_sample_rate_hz = int(record.get("sample_rate_hz", 0) or 0)
        if chunk_sample_rate_hz <= 0:
            continue

        if sample_rate_hz not in (0, chunk_sample_rate_hz):
            raise ValueError(
                "Audio sample rate changed while recording keyframe video: "
                f"{sample_rate_hz} -> {chunk_sample_rate_hz}"
            )

        sample_rate_hz = chunk_sample_rate_hz
        chunks.append(chunk.copy())

    if not chunks or sample_rate_hz <= 0:
        return np.zeros((0,), dtype=np.int16), 0

    return np.concatenate(chunks, axis=0), sample_rate_hz


def _write_pcm16_wav(audio_pcm16: np.ndarray, sample_rate_hz: int, output_path: str):
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(np.asarray(audio_pcm16, dtype=np.int16).tobytes())


def _write_video_with_audio(frames_bgr, audio_records, output_path,
                            fps=_VIDEO_RECORD_FPS):
    if not frames_bgr:
        return

    audio_pcm16, sample_rate_hz = _collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate_hz <= 0:
        _write_video(frames_bgr, output_path, fps=fps)
        return

    silent_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    silent_tmp_path = silent_tmp.name
    wav_tmp_path = wav_tmp.name
    silent_tmp.close()
    wav_tmp.close()

    try:
        _write_video(frames_bgr, silent_tmp_path, fps=fps)
        _write_pcm16_wav(audio_pcm16, sample_rate_hz, wav_tmp_path)

        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("[video] imageio_ffmpeg unavailable — saving silent video only.")
            os.replace(silent_tmp_path, output_path)
            silent_tmp_path = None
            return

        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    silent_tmp_path,
                    "-i",
                    wav_tmp_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_text = (exc.stderr or "").strip()
            print(
                "[video] audio mux failed — saving silent video only."
                + (f" {stderr_text}" if stderr_text else "")
            )
            os.replace(silent_tmp_path, output_path)
            silent_tmp_path = None
    finally:
        if silent_tmp_path and os.path.exists(silent_tmp_path):
            os.unlink(silent_tmp_path)
        if os.path.exists(wav_tmp_path):
            os.unlink(wav_tmp_path)


def _extract_json_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _mid_frame_from_interval(interval: dict, num_frames: int) -> int:
    start = float(interval.get("start", 0.0))
    end   = float(interval.get("end", start))
    if start > end:
        start, end = end, start
    frame_idx = int(round(((start + end) / 2.0) * _VIDEO_RECORD_FPS))
    return max(0, min(frame_idx, num_frames - 1))


def _draw_gaze_crosshair(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(8, h // 32)
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
    cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), (0, 255, 0), 2)
    cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), (0, 255, 0), 2)
    return out


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _format_cmd_label(cmd) -> str:
    """Short display string for a VLM queue command."""
    if isinstance(cmd, CmdNoOp):
        return "[VLM] NoOp"
    if isinstance(cmd, CmdAppend):
        return f"[VLM] Append: {cmd.instruction}"
    if isinstance(cmd, CmdReplace):
        return f"[VLM] Replace[{cmd.target_idx}]: {cmd.instruction}"
    if isinstance(cmd, CmdPreempt):
        return f"[VLM] PREEMPT: {cmd.instruction}"
    if isinstance(cmd, CmdRemove):
        return f"[VLM] Remove[{cmd.target_idx}]"
    return ""


def _cmd_event_color(cmd) -> tuple:
    """BGR banner color for a VLM queue command."""
    if isinstance(cmd, CmdPreempt):
        return (0, 0, 180)    # red
    if isinstance(cmd, CmdAppend):
        return (0, 140, 0)    # green
    if isinstance(cmd, CmdReplace):
        return (0, 150, 200)  # amber
    if isinstance(cmd, CmdRemove):
        return (0, 100, 200)  # orange
    return (80, 80, 80)


def _overlay_vlm_event(img_bgr: np.ndarray, label: str,
                       color_bgr: tuple) -> np.ndarray:
    """Draw a semi-transparent banner at the bottom of img_bgr."""
    out     = img_bgr.copy()
    h, w    = out.shape[:2]
    y1, y2  = h - 38, h - 4
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), color_bgr, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    cv2.putText(out, label[:90], (8, y2 - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Gemini API calls
# ---------------------------------------------------------------------------

def _call_gemini(video_bytes: bytes, prompt: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=types.Content(parts=[
            types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                video_metadata=types.VideoMetadata(fps=4),
            ),
            types.Part(text=prompt),
        ]),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return json.loads(_extract_json_text(response.text))


def _query_gemini_initial(video_path: str, instruction: str,
                          api_key: str, num_frames: int):
    """One-shot call before robot execution to build the initial queue."""
    duration_sec = num_frames / _VIDEO_RECORD_FPS
    prompt = PROMPT_TEMPLATE_INITIAL.format(
        transcript=instruction,
        duration=duration_sec,
        fps=float(_VIDEO_RECORD_FPS),
        config_note=PROMPT_CONFIG_NOTES[_PROMPT_CONFIG_NAME],
        max_intervals=_MAX_INTERVALS,
    )
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    parsed = _call_gemini(video_bytes, prompt, api_key)
    intervals = parsed.get("keyframe_intervals", [])
    frame_indices = []
    for iv in intervals[:_MAX_INTERVALS]:
        frame_indices.append(_mid_frame_from_interval(iv, num_frames))
    frame_indices = sorted(set(frame_indices))
    if not frame_indices:
        frame_indices = [num_frames // 2]
    return frame_indices, parsed


def _query_gemini_async_update(video_path: str, global_instruction: str,
                               queue: KeyframeQueue,
                               seg_start_idx: int, seg_end_idx: int,
                               api_key: str):
    """Periodic async call during robot execution — returns a queue command."""
    num_seg_frames   = seg_end_idx - seg_start_idx
    t_start          = seg_start_idx / _VIDEO_RECORD_FPS
    t_end            = seg_end_idx   / _VIDEO_RECORD_FPS

    prompt = PROMPT_TEMPLATE_ASYNC_UPDATE.format(
        global_instruction=global_instruction,
        queue_state=queue.queue_state_str(),
        current_idx=queue.current_idx,
        current_instruction=queue.current_instruction or "(none)",
        t_start=t_start,
        t_end=t_end,
        segment_duration=(t_end - t_start),
        fps=float(_VIDEO_RECORD_FPS),
        config_note=PROMPT_CONFIG_NOTES[_PROMPT_CONFIG_NAME],
    )
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    parsed       = _call_gemini(video_bytes, prompt, api_key)
    command_name = parsed.get("command", "NoOp")
    target_idx   = parsed.get("target_idx")
    interval     = parsed.get("keyframe_interval")
    instruction  = (parsed.get("instruction") or "").strip()

    print(f"[AsyncVLM] command={command_name}  "
          f"reasoning={parsed.get('reasoning', '')[:120]}")

    if command_name == "NoOp" or command_name not in (
            "Append", "Replace", "Preempt", "Remove"):
        return CmdNoOp()

    if command_name == "Remove":
        if target_idx is None:
            print("[AsyncVLM] Remove: missing target_idx — NoOp")
            return CmdNoOp()
        return CmdRemove(target_idx=int(target_idx))

    # Commands that require a keyframe image from the video segment.
    if interval is None:
        print(f"[AsyncVLM] {command_name}: missing keyframe_interval — NoOp")
        return CmdNoOp()

    local_frame_idx = _mid_frame_from_interval(interval, num_seg_frames)
    return _RawCommand(
        name=command_name,
        target_idx=int(target_idx) if target_idx is not None else None,
        local_frame_idx=local_frame_idx,
        instruction=instruction,
    )


# ---------------------------------------------------------------------------
# Async VLM worker thread
# ---------------------------------------------------------------------------

class AsyncVLMWorker:
    """
    Background daemon thread.

    Timer loop (main worker thread):
      1. Wait VLM_INTERVAL_SEC wall-clock seconds.
      2. Atomically drain all accumulated frames from VideoBuffer (= flush).
      3. Save them as async_seg_NNNN.mp4 (fast I/O).
      4. Fire a per-call sub-thread for the Gemini API call (non-blocking).
      5. Immediately go back to step 1 — timer resets right now, new frames
         accumulate while the previous Gemini call is still in flight.

    Per-call sub-thread:
      - Sends the saved mp4 to Gemini.
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
        self._temp_dir           = temp_dir
        self._stop_event         = threading.Event()
        self._call_count         = 0
        self._episode_start: float = 0.0   # set in start()

        # Preempt signal: main thread checks this after every env.step().
        self.preempt_event = threading.Event()

        # Shared step counter — main thread writes vlm_worker.current_step = global_step
        # each env.step() so that event timestamps align with top_view_frames_list indices.
        self.current_step: int = 0

        # VLM event log: list of (global_step, label, color_bgr).
        self.event_log: List = []

        # Call log: one dict per VLM call, written by the sub-thread.
        # Keys: call_id, trigger_t, obs_window (start, end), n_frames,
        #       step_at_trigger, gemini_start_t, gemini_end_t, command.
        self.call_log: List[dict] = []

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AsyncVLMWorker")

    def start(self):
        self._episode_start = time.monotonic()
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _elapsed(self) -> float:
        return time.monotonic() - self._episode_start

    # ------------------------------------------------------------------
    # Timer loop — never blocks on Gemini
    # ------------------------------------------------------------------

    def _run(self):
        print(f"[AsyncVLM] Worker started. video_source={_gemini_video_source_label()}")
        while not self._stop_event.is_set():
            # ---- wait interval_sec of real wall-clock time ----
            self._stop_event.wait(timeout=self._interval_sec)
            if self._stop_event.is_set():
                break

            trigger_t = self._elapsed()

            # ---- atomically drain all frames accumulated since last call ----
            aria_seg, cropped_seg, _, seg_start, seg_end = self._buf.get_segment()
            gemini_seg = _select_gemini_frames(aria_seg, cropped_seg)

            if not gemini_seg:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames, skip")
                continue

            self._call_count += 1
            call_id        = self._call_count
            seg_video_path = os.path.join(
                self._temp_dir,
                f"async_seg_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4",
            )

            print(f"[AsyncVLM] Call #{call_id} | t={trigger_t:.2f}s | "
                  f"obs frames {seg_start}–{seg_end} ({len(gemini_seg)} frames) | "
                  f"video_source={_gemini_video_source_label()}")

            # Save synchronously — fast I/O, does not block next timer cycle.
            _write_video(gemini_seg, seg_video_path, fps=_VLM_VIDEO_SAVE_FPS)

            # Create log entry (sub-thread will fill timing fields).
            log_entry: dict = {
                "call_id":         call_id,
                "trigger_t":       trigger_t,
                "obs_window":      (seg_start, seg_end),
                "n_frames":        len(gemini_seg),
                "video_source":    _GEMINI_VIDEO_SOURCE,
                "step_at_trigger": self.current_step,
                "gemini_start_t":  None,
                "gemini_end_t":    None,
                "command":         None,
            }
            self.call_log.append(log_entry)

            # Fire Gemini call in a sub-thread — timer resets immediately.
            threading.Thread(
                target=self._run_gemini_call,
                args=(call_id, seg_video_path, seg_start, seg_end, log_entry),
                daemon=True,
                name=f"GeminiCall-{call_id}",
            ).start()

            # ← loop back immediately; new frames accumulate while Gemini runs.

        print("[AsyncVLM] Worker stopped.")

    # ------------------------------------------------------------------
    # Per-call sub-thread
    # ------------------------------------------------------------------

    def _run_gemini_call(self, call_id: int, seg_video_path: str,
                         seg_start: int, seg_end: int, log_entry: dict):
        num_seg_frames = seg_end - seg_start

        gemini_start_t = self._elapsed()
        log_entry["gemini_start_t"] = gemini_start_t
        print(f"[AsyncVLM] #{call_id} → Gemini request at t={gemini_start_t:.2f}s")

        try:
            raw = _query_gemini_async_update(
                video_path=seg_video_path,
                global_instruction=self._global_instruction,
                queue=self._queue,
                seg_start_idx=seg_start,
                seg_end_idx=seg_end,
                api_key=self._api_key,
            )

            gemini_end_t = self._elapsed()
            log_entry["gemini_end_t"] = gemini_end_t
            print(f"[AsyncVLM] #{call_id} ← Gemini response at t={gemini_end_t:.2f}s "
                  f"(latency={gemini_end_t - gemini_start_t:.1f}s)")

            # ---- resolve _RawCommand → typed command with ndarray ----
            if isinstance(raw, _RawCommand):
                abs_idx        = seg_start + raw.local_frame_idx
                _, cropped_bgr = self._buf.get_frame(abs_idx)
                if cropped_bgr is None:
                    print(f"[AsyncVLM] #{call_id} frame {abs_idx} not in buffer — NoOp")
                    raw = CmdNoOp()
                else:
                    cropped_bgr = _draw_gaze_crosshair(cropped_bgr)
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
                _cmd_event_color(cmd),
            ))

            if isinstance(cmd, CmdPreempt):
                self.preempt_event.set()

        except Exception as exc:
            log_entry["command"] = "ERROR"
            print(f"[AsyncVLM] #{call_id} failed: {exc}")


# ---------------------------------------------------------------------------
# CV2 preview helpers
# ---------------------------------------------------------------------------

def _build_preview_frame(obs, status_text, status_color=(0, 255, 0)):
    images     = obs["images"]
    if _gemini_uses_cropped_video():
        human_view = _draw_gaze_crosshair(images["aria_cropped_rgb"])
        human_label = "human(cropped) | top"
    else:
        human_view = images["aria_rgb"].copy()
        human_label = "human(gaze) | top"
    top_view   = images["top_camera_rgb"].copy()

    if not _gemini_uses_cropped_video():
        gaze_point = np.asarray(obs["gaze_point"], dtype=np.float32).reshape(-1)
        if gaze_point.size >= 2 and np.all(np.isfinite(gaze_point[:2])):
            x = int(np.clip(gaze_point[0], 0, human_view.shape[1] - 1))
            y = int(np.clip(gaze_point[1], 0, human_view.shape[0] - 1))
            cv2.circle(human_view, (x, y), 6, (0, 255, 0), 2)
            cv2.line(human_view, (x - 12, y), (x + 12, y), (0, 255, 0), 2)
            cv2.line(human_view, (x, y - 12), (x, y + 12), (0, 255, 0), 2)

    if top_view.shape[:2] != human_view.shape[:2]:
        top_view = cv2.resize(top_view,
                              (human_view.shape[1], human_view.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    preview = np.hstack((human_view, top_view))
    cv2.putText(preview, human_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, status_text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    return preview


def _build_eval_preview(obs, keyframe_img: np.ndarray,
                        queue: KeyframeQueue, current_instruction: str = ""):
    if _gemini_uses_cropped_video():
        human_view = _draw_gaze_crosshair(obs["images"]["aria_cropped_rgb"])
        live_label = "LIVE(cropped)"
    else:
        human_view = obs["images"]["aria_rgb"].copy()
        live_label = "LIVE"
    kf_display = cv2.resize(keyframe_img,
                            (human_view.shape[1], human_view.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    preview = np.hstack((human_view, kf_display))
    cv2.putText(preview,
                f"{live_label} | KF [{queue.current_idx + 1}/{queue.total}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "n=next kf  q=quit",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if current_instruction:
        cv2.putText(preview, f"subtask: {current_instruction[:80]}",
                    (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Rollout video saver
# ---------------------------------------------------------------------------

def _save_rollout_video(top_frames, comp_prob_log, output_path,
                        fps=_VIDEO_RECORD_FPS, vlm_event_log=None,
                        keyframe_log=None):
    """
    Saves rollout.mp4 with 3 stacked rows per frame:
      Row 1 : currently executing keyframe image
      Row 2 : top-view camera frame  (VLM event as red text overlay)
      Row 3 : completion-probability progress plot
    """
    if not top_frames:
        print("[rollout] No frames to save.")
        return

    fw, fh = top_frames[0].shape[1], top_frames[0].shape[0]
    dpi    = 100

    comp_prob_log = sorted(comp_prob_log, key=lambda x: x[0])
    max_step      = max(
        (s for s, _ in comp_prob_log), default=len(top_frames))
    max_step      = max(max_step, len(top_frames))

    fig, ax = plt.subplots(figsize=(fw / dpi, fh / dpi), dpi=dpi)

    def _render(visible_log):
        ax.clear()
        xs, ys = zip(*visible_log) if visible_log else ([], [])
        ax.plot(xs, ys, color='steelblue', linewidth=1.5)
        ax.scatter(xs, ys, color='steelblue', s=12, zorder=3)
        ax.axhline(y=0.8, color='tomato', linestyle='--', linewidth=1.2,
                   alpha=0.8, label='threshold = 0.8')
        ax.set_xlim(0, max_step)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Global Step', fontsize=8)
        ax.set_ylabel('Completion Prob', fontsize=8)
        ax.set_title('Completion Probability over Time', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), (fw, fh))

    breakpoints = [-1] + [s for s, _ in comp_prob_log]
    rendered    = {bp: _render([(s, p) for s, p in comp_prob_log if s <= bp])
                   for bp in breakpoints}
    plt.close(fig)

    def _get_plot(step):
        best = max((bp for bp in breakpoints if bp <= step), default=-1)
        return rendered[best]

    # Row 1: keyframe lookup — find the keyframe active at each step
    kf_entries = sorted(keyframe_log or [], key=lambda x: x[0])
    _blank_kf  = np.zeros((fh, fw, 3), dtype=np.uint8)

    def _get_keyframe(step):
        kf_img = None
        for s, img in kf_entries:
            if s <= step:
                kf_img = img
            else:
                break
        if kf_img is None:
            return _blank_kf
        return cv2.resize(kf_img, (fw, fh), interpolation=cv2.INTER_LINEAR)

    # Row 2: VLM event as red text
    vlm_events = sorted(vlm_event_log or [], key=lambda x: x[0])

    def _get_active_event(step):
        active = [(s, lbl) for s, lbl, _ in vlm_events
                  if s <= step < s + _VLM_EVENT_DISPLAY_FRAMES]
        if not active:
            return None, None
        s_evt, lbl = max(active, key=lambda x: x[0])
        return s_evt, lbl

    print(f"[rollout] Compositing {len(top_frames)} frames "
          f"({len(vlm_events)} VLM events)...")
    combined_frames = []
    for i, f in enumerate(top_frames):
        # Row 1: currently executing keyframe
        row1 = _get_keyframe(i)

        # Row 2: top-view + red VLM event text
        row2 = f.copy()
        evt_step, lbl = _get_active_event(i)
        if lbl:
            cv2.putText(row2, f"step {evt_step}: {lbl}"[:90],
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Row 3: completion probability plot
        row3 = _get_plot(i)

        combined_frames.append(
            cv2.cvtColor(np.vstack([row1, row2, row3]), cv2.COLOR_BGR2RGB)
        )
    imageio.mimwrite(output_path, combined_frames, fps=fps)
    print(f"[rollout] Saved → {output_path}")


# ---------------------------------------------------------------------------
# Initial keyframe extraction  (interactive, before robot execution)
# ---------------------------------------------------------------------------

def extract_keyframes(env, language_instruction: str,
                      api_key: str, video_buffer: VideoBuffer) -> KeyframeQueue:
    """
    Stream live aria | top frames.
    Press 'p' to snapshot the buffer, call Gemini, and build the initial
    KeyframeQueue.  All captured frames are written into video_buffer so
    the async worker sees them in its history.

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

    print("Keyframe selection: press 'p' to capture & query VLM, 'q' to quit.")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation(
                    mode='human_view_streamed_and_keyframe_cropped'))
            live_frame = _build_preview_frame(
                live_obs, "LIVE: p=save&proceed  q=quit")

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

            aria_frames.append(aria_bgr)
            aria_cropped_frames.append(cropped_bgr)
            gaze_records.append(
                {"frame_idx": len(gaze_records), "gaze_x": gx, "gaze_y": gy})
            audio_records.append({
                "frame_idx": len(audio_records),
                "audio_ts_ns": audio_ts_ns,
                "sample_rate_hz": audio_sample_rate_hz,
                "num_samples": audio_num_samples,
                "chunk_pcm16": audio_chunk.copy(),
            })

            # Also stream into the shared video buffer.
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

            cv2.imshow(window_name, live_frame)
            key = cv2.waitKey(1) & 0xFF
            capture_timer.sleep()   # pace the loop to _VIDEO_RECORD_FPS

            if key in (ord('p'), ord('P')) and aria_frames:
                total = len(aria_frames)
                print(f"[keyframe] Captured {total} frames @ {_VIDEO_RECORD_FPS} FPS "
                      f"({total / _VIDEO_RECORD_FPS:.2f}s)")

                if _gemini_uses_cropped_video():
                    _write_video(aria_frames, video_path, fps=_VIDEO_RECORD_FPS)
                    _write_video_with_audio(
                        aria_cropped_frames,
                        audio_records,
                        cropped_video_path,
                        fps=_VIDEO_RECORD_FPS,
                    )
                    gemini_video_path = cropped_video_path
                else:
                    _write_video_with_audio(
                        aria_frames,
                        audio_records,
                        video_path,
                        fps=_VIDEO_RECORD_FPS,
                    )
                    _write_video(aria_cropped_frames, cropped_video_path, fps=_VIDEO_RECORD_FPS)
                    gemini_video_path = video_path
                with open(gaze_path, "w") as f:
                    json.dump({"frames": gaze_records}, f, indent=2)

                print("[keyframe] Querying Gemini for initial keyframes... "
                      f"video_source={_gemini_video_source_label()}")
                frame_indices, parsed = _query_gemini_initial(
                    video_path=gemini_video_path,
                    instruction=language_instruction,
                    api_key=api_key,
                    num_frames=total,
                )

                keyframe_imgs  = []
                subtask_instrs = []
                saved_items    = []

                subtask_raw = parsed.get("subgoal_instructions", [])
                if not isinstance(subtask_raw, list):
                    subtask_raw = []
                subtask_raw = [str(s).strip() for s in subtask_raw if str(s).strip()]

                for i1, frame_idx in enumerate(frame_indices, start=1):
                    full_bgr   = aria_frames[frame_idx].copy()
                    cropped_kf = aria_cropped_frames[frame_idx].copy()
                    cropped_kf = _draw_gaze_crosshair(cropped_kf)

                    full_path    = os.path.join(_TEMP_DIR, f"keyframe_{i1:02d}.png")
                    cropped_path = os.path.join(_TEMP_DIR,
                                                f"cropped_keyframe_{i1:02d}.png")
                    plt.imsave(full_path,
                               cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB))
                    plt.imsave(cropped_path,
                               cv2.cvtColor(cropped_kf, cv2.COLOR_BGR2RGB))
                    print(f"[keyframe] #{i1} (frame {frame_idx}) → {cropped_path}")

                    keyframe_imgs.append(cropped_kf)
                    saved_items.append({
                        "order":                i1,
                        "frame_idx":            int(frame_idx),
                        "keyframe_path":        full_path,
                        "cropped_keyframe_path": cropped_path,
                    })

                # Align instruction list length
                while len(subtask_raw) < len(keyframe_imgs):
                    subtask_raw.append(language_instruction)
                subtask_raw = subtask_raw[:len(keyframe_imgs)]

                with open(keyframe_meta_path, "w") as f:
                    json.dump({
                        "instruction":          language_instruction,
                        "num_frames":           total,
                        "video_fps":            _VIDEO_RECORD_FPS,
                        "gemini_video_source":  _GEMINI_VIDEO_SOURCE,
                        "gemini_video_path":    gemini_video_path,
                        "frame_indices":        [int(i) for i in frame_indices],
                        "subtask_instructions": subtask_raw,
                        "saved_keyframes":      saved_items,
                        "gemini_response":      parsed,
                    }, f, indent=2)

                print(f"[keyframe] n_keyframes: {len(keyframe_imgs)}")
                for idx, instr in enumerate(subtask_raw, 1):
                    print(f"  [{idx}] {instr}")

                return KeyframeQueue(keyframe_imgs, subtask_raw)

            elif key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def eval(env, client, language_instruction: str, horizon: int,
         queue: KeyframeQueue, video_buffer: VideoBuffer,
         vlm_worker: AsyncVLMWorker, chunksize: int = 8):
    """
    Robot execution loop.

    The async VLM worker runs concurrently and may modify `queue` at any
    time.  The loop reacts to four triggers:

      1. vlm_worker.preempt_event  — robot reset + new keyframe from VLM
      2. completion_prob >= 0.8    — auto-advance to next keyframe
      3. 'n' key                   — manual advance
      4. 'q' / ESC                 — abort episode
    """
    if queue.is_done:
        print("[Eval] Empty keyframe queue — nothing to do.")
        return

    pending_auto_advance = False
    top_view_frames_list = []
    comp_prob_log        = []
    global_step          = 0
    subtask_step         = 0
    timer                = Timer(env.control_hz)
    window_name          = "Eval  |  n=next kf  q=quit"
    rollout_video_path   = os.path.join(_TEMP_DIR, "rollout.mp4")
    keyframe_log: List[Tuple[int, np.ndarray]] = []
    kf = queue.current_keyframe
    if kf is not None:
        keyframe_log.append((0, kf.copy()))

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
                print("[Eval] Queue exhausted — episode complete.")
                break

            # ----------------------------------------------------------------
            # Observe
            # ----------------------------------------------------------------
            obs         = env.get_observation(mode="human_view_streamed_and_keyframe_cropped")["timestamp"]
            images      = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for v in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, v])

            # Buffer this frame for the async VLM worker (human view + cropped + gaze).
            aria_bgr    = images["aria_rgb"]
            cropped_bgr = images["aria_cropped_rgb"]
            gaze_pt  = np.asarray(
                obs.get("gaze_point", [0.0, 0.0]),
                dtype=np.float64).reshape(-1)
            gx = float(gaze_pt[0]) if gaze_pt.size >= 1 else None
            gy = float(gaze_pt[1]) if gaze_pt.size >= 2 else None
            audio_obs = obs.get("audio", {})
            audio_chunk = np.asarray(
                audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
                dtype=np.int16,
            )
            audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
            audio_sample_rate_hz = int(audio_obs.get("sample_rate_hz", 0) or 0)
            audio_num_samples = int(audio_obs.get("num_samples", audio_chunk.size) or 0)
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

            keyframe_img  = queue.current_keyframe
            current_instr = queue.current_instruction

            human_view_gaze_cropped_rgb = cv2.cvtColor(
                images["aria_cropped_rgb"], cv2.COLOR_BGR2RGB
            )
            if keyframe_img is not None:
                keyframe_img = cv2.cvtColor(keyframe_img, cv2.COLOR_BGR2RGB)
            # temp
            #keyframe_img = _BLACK_TOP_KEYFRAME
            #
            observation = {
                "observation.images.top_view":                    images["top_camera_rgb"],
                "observation.images.human_view_gaze_cropped":     human_view_gaze_cropped_rgb,
                "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
                #"observation.gaze_point":                         obs["gaze_point"],
                "prompt":                                         current_instr,
                "observation.state":                              concatenated_state,
            }

            if global_step == 0:
                plt.imsave(
                    os.path.join(_TEMP_DIR, "test_human_view_gaze_cropped.png"),
                    human_view_gaze_cropped_rgb,
                )
            if global_step == 0 and keyframe_img is not None:
                plt.imsave(os.path.join(_TEMP_DIR, "test_aria_keyframe.png"), keyframe_img)
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
                # Check for mid-chunk Preempt.
                if vlm_worker.preempt_event.is_set():
                    vlm_worker.preempt_event.clear()
                    print("[Eval] Preempt mid-chunk — resetting robot.")
                    cv2.destroyAllWindows()
                    env.reset()
                    client = websocket_client_policy.WebsocketClientPolicy(
                        host="localhost", port=8000)
                    subtask_step  = 0
                    keyframe_img  = queue.current_keyframe
                    current_instr = queue.current_instruction
                    if keyframe_img is not None:
                        keyframe_log.append((global_step, keyframe_img.copy()))
                    print(f"[Eval] Preempt → KF [{queue.current_idx + 1}/{queue.total}]:"
                          f" '{current_instr}'")
                    break  # discard remaining actions; re-infer next iteration

                action = {
                    'left_arm_positions':    raw_action[0:6],
                    'left_gripper_position': raw_action[6:7],
                    'right_arm_positions':   raw_action[7:13],
                    'right_gripper_position': raw_action[13:14],
                }
                env.step(action)
                global_step  += 1
                subtask_step += 1
                vlm_worker.current_step = global_step

                live_obs = _extract_latest_observation(
                    env.get_observation(mode='human_view_streamed_and_keyframe_cropped'))
                top_view_frames_list.append(
                    live_obs["images"]["top_camera_rgb"].copy())

                # micro-step마다 aria_rgb를 video_buffer에 추가해 VLM이 보는 영상이 완전해지도록.
                live_aria_bgr    = live_obs["images"]["aria_rgb"]
                live_cropped_bgr = live_obs["images"]["aria_cropped_rgb"]
                live_audio_obs = live_obs.get("audio", {})
                live_audio_chunk = np.asarray(
                    live_audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
                    dtype=np.int16,
                )
                live_audio_ts_ns = int(live_audio_obs.get("audio_ts_ns", 0) or 0)
                live_audio_sample_rate_hz = int(
                    live_audio_obs.get("sample_rate_hz", 0) or 0
                )
                live_audio_num_samples = int(
                    live_audio_obs.get("num_samples", live_audio_chunk.size) or 0
                )
                # inference-step aria와 해상도를 맞춰 video encoding 오류 방지
                _tgt_h, _tgt_w = aria_bgr.shape[:2]
                if live_aria_bgr.shape[:2] != (_tgt_h, _tgt_w):
                    live_aria_bgr = cv2.resize(live_aria_bgr, (_tgt_w, _tgt_h),
                                               interpolation=cv2.INTER_LINEAR)
                video_buffer.append(
                    live_aria_bgr,
                    live_cropped_bgr,
                    None,
                    None,
                    audio_chunk_pcm16=live_audio_chunk,
                    audio_ts_ns=live_audio_ts_ns,
                    audio_sample_rate_hz=live_audio_sample_rate_hz,
                    audio_num_samples=live_audio_num_samples,
                )

                preview = _build_eval_preview(
                    live_obs, keyframe_img, queue, current_instr)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('n'), ord('N')):
                    if not queue.advance():
                        print("No more keyframes. Resetting robot.")
                        cv2.destroyAllWindows()
                        env.reset()
                        client = websocket_client_policy.WebsocketClientPolicy(
                            host="localhost", port=8000)
                    else:
                        subtask_step  = 0
                        keyframe_img  = queue.current_keyframe
                        current_instr = queue.current_instruction
                        if keyframe_img is not None:
                            keyframe_log.append((global_step, keyframe_img.copy()))
                        print(f"  → Manual KF [{queue.current_idx + 1}/{queue.total}]:"
                              f" '{current_instr}'")
                    break

                elif key in (ord('q'), ord('Q'), 27):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt

                timer.sleep()

            # ----------------------------------------------------------------
            # Auto-advance via completion_prob
            # ----------------------------------------------------------------
            if completion_prob is not None and completion_prob >= 0.8:
                if subtask_step < 20:
                    print(f"  [Auto] cp={completion_prob:.3f} >= 0.8 but "
                          f"subtask_step={subtask_step} < 20 — skip")
                else:
                    print(f"  [Auto] cp={completion_prob:.3f} >= 0.8 "
                          f"(subtask_step={subtask_step}) — advancing...")
                    pending_auto_advance = True

            if pending_auto_advance:
                pending_auto_advance = False
                if not queue.advance():
                    print("[Auto] Queue exhausted — resetting robot.")
                    cv2.destroyAllWindows()
                    env.reset()
                    vlm_worker.stop()
                    _save_rollout_video(
                        top_view_frames_list, comp_prob_log, rollout_video_path,
                        vlm_event_log=vlm_worker.event_log,
                        keyframe_log=keyframe_log)
                    client = websocket_client_policy.WebsocketClientPolicy(
                        host="localhost", port=8000)
                    return
                else:
                    client = websocket_client_policy.WebsocketClientPolicy(
                        host="localhost", port=8000)
                    subtask_step = 0
                    kf = queue.current_keyframe
                    if kf is not None:
                        keyframe_log.append((global_step, kf.copy()))
                    print(f"  [Auto] → KF [{queue.current_idx + 1}/{queue.total}]:"
                          f" '{queue.current_instruction}'")

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(
                host="localhost", port=8000)
            _save_rollout_video(
                top_view_frames_list, comp_prob_log, rollout_video_path,
                vlm_event_log=vlm_worker.event_log,
                keyframe_log=keyframe_log)
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()
    _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path,
                        vlm_event_log=vlm_worker.event_log,
                        keyframe_log=keyframe_log)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SERVER_IP             = "137.68.92.53"
    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env                  = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "Put this tumbler to this basket."
    horizon              = 10000
    GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    print("Starting. Instruction:", language_instruction)
    print("During eval: n=next keyframe  q=stop episode")

    # Shared buffer — persists for the lifetime of the process.
    video_buffer = VideoBuffer()

    # ---- Initial keyframe extraction (interactive) ----
    queue = extract_keyframes(env, language_instruction, GEMINI_API_KEY,
                              video_buffer)

    # ---- Connect to policy server ----
    client = websocket_client_policy.WebsocketClientPolicy(
        host="localhost", port=8000)

    # ---- Start async VLM worker ----
    # Skip pre-execution frames so the worker only sees execution-time video.
    video_buffer.skip_to_current()
    vlm_worker = AsyncVLMWorker(
        video_buffer=video_buffer,
        queue=queue,
        global_instruction=language_instruction,
        api_key=GEMINI_API_KEY,
        interval_sec=VLM_INTERVAL_SEC,
    )
    vlm_worker.start()

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

    try:
        while True:
            try:
                eval(env, client, language_instruction, horizon,
                     queue, video_buffer, vlm_worker)
                print("Episode finished.")
            except KeyboardInterrupt:
                pass

            result = _options_menu(language_instruction)
            if result is None:
                print("Exiting evaluation...")
                break
            language_instruction = result

            # Restart for next episode: stop old worker, fresh buffer/queue.
            vlm_worker.stop()
            video_buffer = VideoBuffer()
            queue        = extract_keyframes(
                env, language_instruction, GEMINI_API_KEY, video_buffer)
            client = websocket_client_policy.WebsocketClientPolicy(
                host="localhost", port=8000)
            video_buffer.skip_to_current()
            vlm_worker = AsyncVLMWorker(
                video_buffer=video_buffer,
                queue=queue,
                global_instruction=language_instruction,
                api_key=GEMINI_API_KEY,
                interval_sec=VLM_INTERVAL_SEC,
            )
            vlm_worker.start()

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        vlm_worker.stop()
        if os.path.isdir(_TEMP_DIR):
            import shutil
            shutil.rmtree(_TEMP_DIR)
            os.makedirs(_TEMP_DIR, exist_ok=True)
            print(f"Cleared temp directory: {_TEMP_DIR}")
        env.safe_end()
