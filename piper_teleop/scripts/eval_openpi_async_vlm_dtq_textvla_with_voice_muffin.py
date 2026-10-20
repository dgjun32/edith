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

import contextlib
import numpy as np
import cv2
import os
import json
import re
import select
import subprocess
import sys
import termios
import threading
import tempfile
import time
import tty
import warnings
import wave
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from google import genai
from google.genai import types

from openai import OpenAI

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from eval_vlm_baseline import (
    _draw_caption,
    _get_active_caption,
    _ffmpeg_reencode_h264,
)


# ---------------------------------------------------------------------------
# Constants — edit these instead of CLI args
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS   = 30
_MAX_INTERVALS      = 10
_TEMP_DIR           = os.path.normpath("/home/alin/projects/piper_project/temp")
_BLACK_TOP_KEYFRAME = np.zeros((240, 424, 3), dtype=np.uint8)

# ── VLM pipeline settings ──
VLM_MODEL           = "gemini-3.1-flash-lite-preview"       # Gemini model to use
VLM_FPS             = 4                         # subsample to this FPS before sending to VLM
VLM_INTERVAL_SEC    = 8.0                       # seconds between async VLM calls
VLM_TARGET_RES      = (256, 256)                # resize video segments for VLM
VLM_THINKING_BUDGET = 2048                      # thinking token budget per call

# ── Audio settings ──
WHISPER_ENABLED     = True                      # use Whisper STT → transcript in prompt
VOICED_VIDEO        = False                      # merge audio into mp4 → Gemini processes audio

# ── Gaze crop settings ──
GAZE_CROP_ENABLED   = True
GAZE_CROP_SIZE      =  512                      # crop NxN around gaze centre
GAZE_CALIBRATION_Y_OFFSET = 0                 # px to add to Y (corrects upward calibration bias)

# ── Visual queue ──
VISUAL_QUEUE        = True                      # attach keyframe images in VLM update calls

# ── Simulation visualization ──
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

# ── Robot pause on voice ──
STOP_ROBOT_WHILE_RECV = False                    # pause robot control while VLM processes voice

# Number of top-view (global) steps for which a VLM command overlay stays
# visible on the rollout video after it fires.
_VLM_EVENT_DISPLAY_FRAMES = 90

# FPS for saving async VLM segment videos (logging).
_VLM_VIDEO_SAVE_FPS = 15


# ---------------------------------------------------------------------------
# Keyboard input helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _terminal_keypress_mode():
    """Allow single-key reads from the SSH terminal during interactive loops."""
    if not sys.stdin.isatty():
        yield False
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _poll_keypress() -> int:
    """Read a key from either the OpenCV window or the SSH terminal."""
    key = cv2.waitKey(1)
    if key != -1:
        return key & 0xFF

    if not sys.stdin.isatty():
        return -1

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return -1

    ch = sys.stdin.read(1)
    return ord(ch) if ch else -1


# ---------------------------------------------------------------------------
# Gemini prompts  (ported from eval_vlm_baseline_asynchronous.py)
# ---------------------------------------------------------------------------

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

CRITICAL — make object descriptions and subgoal instructions maximally fine-grained:
    - When generating object_references and subgoal_instructions, use the most
      specific visible description possible, not generic phrases like
      "this muffin" or "that muffin", if the object can be visually distinguished.
    - If the human points to a muffin, identify exactly which muffin it is from
      its topping appearance and include that description in the subgoal.
    - In this store scene, there are six visually distinct muffins:
        * muffin with grape topping
        * muffin with cherry topping
        * muffin with 2 strawberries topping
        * muffin with oreo and strawberry topping
        * muffin with pineapple topping
        * muffin with tangerine and strawberry topping
    - Prefer these exact names whenever they match the pointed muffin.
    - Example: say "pick up the muffin with oreo and strawberry topping"
      instead of "pick up this muffin".

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

PROMPT_TEMPLATE_ASYNC_UPDATE = """\
You are monitoring a robot task execution from a first-person (egocentric) \
video in a store. You are receiving a new {segment_duration:.1f}-second video clip \
showing what just happened (covering {t_start:.1f}s to \
{t_end:.1f}s of the episode).

The robot is executing a pick-and-place task. The current keyframe queue \
(ordered list of remaining subtask steps) is:

{queue_state}
{visual_queue_section}

**Currently executing:** index {current_idx} — "{current_instruction}"

CRITICAL: Each video clip covers a **disjoint, non-overlapping** time window. \
If you detect ANY human speech, gesture, or instruction in this clip, it is \
**new information that was NOT present in any previous clip**. You MUST \
update the queue accordingly — do NOT dismiss it as "already reflected" or \
"already in the queue". NoOp is ONLY correct when the clip contains NO \
human intervention at all (the person is silent and passive).

Your task: Analyze the new video clip and decide whether the queue needs to \
be modified. A human may intervene during the task to change, add, or cancel \
steps. Look for:
- Human speech or gestures indicating a change in the task
- Pointing at new objects or destinations
- Verbal corrections ("not that one", "also grab this", "forget about that")

Choose exactly ONE action:

1. **NoOp**: No change needed. The robot is working normally, no human \
intervention detected, or the person is simply observing without meaningful \
instruction.

2. **Append**: Add new element(s) to the END of the queue. Use when the \
human adds new objects/destinations to the task.

3. **Replace**: Replace a specific queue element at a given index. Use when \
the human changes a target object or destination for a specific step.

4. **Remove**: Remove one or more queue elements by index. Use when \
the human cancels a specific step. IMPORTANT: pick-and-place actions are \
PAIRED — every "pick up X" has a corresponding "place it Y". When the \
human cancels an object (e.g., "forget about that", "we don't need this"), \
you MUST remove BOTH the "pick up" element AND its matching "place it" \
element. Return ALL indices that should be removed together in \
"target_indices" (a list of ints).

5. **Preempt**: IMMEDIATELY interrupt the current task, reset the robot, \
and insert the new step at the front of the remaining queue. Use only when \
the human signals an urgent correction.

The keyframe intervals are determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object.
  - Hand gesture (pointing): when the hand/finger clearly points to a \
specific object or region.
  - Gaze fixation: when the wearer's gaze stabilises on the target object \
(if gaze information is available).

RULES:
- subtask_instruction MUST use ONLY "pick up <object>" or "place it <destination>"
- Do NOT use any other verbs (e.g., "point to", "grab", "put", "get", \
"move", "take"). Always rephrase to "psick up" or "place it".
- keyframe_description must be fine-grained with visual attributes (color, \
shape, brand name, size, shelf position, etc.)
- PERSPECTIVE: This is an egocentric (first-person) video from the \
wearer's viewpoint. All spatial references (left, right, top, bottom, \
near, far) MUST be from the **wearer's perspective** — i.e., "left" \
means to the wearer's left as seen in the video. Do NOT use the \
perspective of a person facing the camera.
- CLOSED VOCABULARY: Restrict all newly generated objects/destinations to:
    * muffin with grape topping
    * muffin with cherry topping
    * muffin with 2 strawberries topping
    * muffin with oreo and strawberry topping
    * muffin with pineapple topping
    * muffin with tangerine and strawberry topping
    * left basket
    * right basket
- Allowed new subtask_instruction strings are exactly:
    * muffin with grape topping
    * muffin with cherry topping
    * muffin with 2 strawberries topping
    * muffin with oreo and strawberry topping
    * muffin with pineapple topping
    * muffin with tangerine and strawberry topping
    * place it in the left basket
    * place it in the right basket
- Do NOT invent other object names or alternate location descriptions.
- If the clip does not clearly support one of these valid instructions,
  prefer NoOp rather than inventing a new one.
- All timestamps are **relative to this clip** (0.0 = start of this clip, \
{segment_duration:.1f} = end of this clip)
- Apply contextual recomposition: if the human says "also put this in that \
basket", append TWO elements (pick up + place it), not one

{transcript_section}
This video clip is {segment_duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<explanation of what you observed in the clip>",
  "command": "<NoOp|Append|Replace|Remove|Preempt>",
  "target_idx": <int or null>,
  "target_indices": <list of ints or null>,
  "new_elements": [
    {{
      "keyframe_description": "<detailed visual description>",
      "subtask_instruction": "<pick up X or place it Y>",
      "keyframe_interval": {{"start": <float, seconds>, "end": <float, seconds>}}
    }}
  ]
}}

For NoOp: target_idx null, new_elements []
For Append: target_idx null, new_elements = list of new elements
For Replace: target_idx = index to replace, new_elements = [single replacement]
For Remove: target_indices = list of indices to remove, target_idx null, new_elements []
For Preempt: target_idx null, new_elements = [single new element inserted at front]
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
    target_indices: List[int]    # one or more indices to remove


# Internal: raw command before the frame is resolved into an ndarray.
@dataclass
class _RawCommand:
    name: str                    # "Append" | "Replace" | "Preempt"
    target_idx: Optional[int]
    local_frame_idx: int         # index within the segment
    instruction: str
    keyframe_description: str = ""


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

    def snapshot(self) -> List[dict]:
        """Thread-safe snapshot of (image, instruction) for visualization."""
        with self._lock:
            result = []
            for i, (img, instr) in enumerate(self._items):
                result.append({
                    "keyframe_image": img.copy() if img is not None else None,
                    "instruction": instr,
                    "is_current": (i == self._current_idx),
                })
            return result

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
                removed = []
                # Sort descending so pops don't shift remaining indices
                for idx in sorted(set(cmd.target_indices), reverse=True):
                    if idx <= self._current_idx or idx >= len(self._items):
                        continue
                    self._items.pop(idx)
                    removed.append(idx)
                if removed:
                    return f"[Queue] Remove {sorted(removed)} done."
                return f"[Queue] Remove: no valid indices in {cmd.target_indices}"

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

    def get_gaze(self, idx: int) -> Optional[dict]:
        """Return gaze record at the given frame index, or None."""
        with self._lock:
            if 0 <= idx < len(self._gaze):
                return dict(self._gaze[idx])
            return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _write_video(frames_bgr, output_path, fps=_VIDEO_RECORD_FPS):
    if not frames_bgr:
        return
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    imageio.mimwrite(output_path, frames_rgb, fps=fps)


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
    """Robustly extract JSON from markdown code fences or raw text."""
    text = (raw_text or "").strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    # Legacy fallback
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _mid_sec_from_interval(interval: dict) -> float:
    """Return the midpoint time (seconds) of a keyframe interval."""
    start = float(interval.get("start", 0.0))
    end   = float(interval.get("end", start))
    if start > end:
        start, end = end, start
    return (start + end) / 2.0


def _mid_frame_from_interval(interval: dict, num_frames: int,
                              fps: float = _VIDEO_RECORD_FPS) -> int:
    mid_sec = _mid_sec_from_interval(interval)
    frame_idx = int(round(mid_sec * fps))
    return max(0, min(frame_idx, num_frames - 1))


# ---------------------------------------------------------------------------
# Audio / Whisper / Gaze crop utilities
# ---------------------------------------------------------------------------

def _whisper_transcribe_wav(wav_path: str) -> dict:
    """Transcribe a WAV file using OpenAI Whisper API.

    Returns {"text": "...", "words": [{"word": "...", "start": <float>, "end": <float>}]}
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # uses OPENAI_API_KEY env var
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        result = {"text": response.text, "words": []}
        if hasattr(response, "words") and response.words:
            for w in response.words:
                result["words"].append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })
        return result
    except Exception as exc:
        print(f"  [WARN] Whisper transcription failed: {exc}")
        return {"text": "", "words": []}


def _burn_captions_on_video_bytes(
    video_bytes: bytes,
    words: list,
    clip_start_sec: float,
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
        abs_time = clip_start_sec + frame_idx / vid_fps
        caption = _get_active_caption(words, abs_time)
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


def _merge_audio_into_video_bytes(video_bytes: bytes, wav_path: str) -> bytes:
    """Merge H264 video bytes with a WAV file → MP4 with audio track."""
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.write(video_bytes)
    tmp_video.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-i", tmp_video.name,
         "-i", wav_path,
         "-c:v", "copy",
         "-ac", "1",
         "-c:a", "aac", "-b:a", "64k",
         "-shortest",
         tmp_out.name],
        check=True,
    )

    with open(tmp_out.name, "rb") as f:
        result = f.read()

    os.unlink(tmp_video.name)
    os.unlink(tmp_out.name)
    return result


def _subsample_frames(frames: List[np.ndarray], src_fps: float,
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


def _crop_around_gaze(frame: np.ndarray, gx: float, gy: float,
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


def _frames_to_video_bytes(frames: List[np.ndarray], fps: float,
                            resolution: Optional[Tuple[int, int]] = None) -> bytes:
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


def _prepare_vlm_video(
    aria_frames: List[np.ndarray],
    gaze_records: List[dict],
    audio_records: List[dict],
    clip_start_sec: float,
) -> Tuple[bytes, str, dict]:
    """Prepare a video segment for VLM consumption.

    Applies: gaze crop → subsample → encode → whisper → captions → audio merge.
    Returns (video_bytes, transcript_section, transcript_data).
    """
    processed = []
    for i, frame in enumerate(aria_frames):
        f = frame.copy()
        # Gaze crop
        if GAZE_CROP_ENABLED and i < len(gaze_records):
            gr = gaze_records[i]
            gx = gr.get("gaze_x")
            gy = gr.get("gaze_y")
            if gx is not None and gy is not None:
                # Apply calibration offset
                gy_corrected = gy + GAZE_CALIBRATION_Y_OFFSET
                f = _crop_around_gaze(f, gx, gy_corrected, GAZE_CROP_SIZE)
        processed.append(f)

    # Subsample FPS
    subsampled = _subsample_frames(processed, _VIDEO_RECORD_FPS, VLM_FPS)

    # Encode to H264 video bytes
    video_bytes = _frames_to_video_bytes(subsampled, VLM_FPS, VLM_TARGET_RES)

    transcript_section = ""
    transcript_data: dict = {}

    # Whisper transcription
    if WHISPER_ENABLED:
        audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
        if audio_pcm16.size > 0 and sample_rate > 0:
            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_tmp.close()
            _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
            try:
                transcript_data = _whisper_transcribe_wav(wav_tmp.name)
                if transcript_data["text"]:
                    print(f"  [Whisper] \"{transcript_data['text'][:100]}\"")
                    # Burn captions onto video
                    video_bytes = _burn_captions_on_video_bytes(
                        video_bytes, transcript_data["words"],
                        clip_start_sec, VLM_FPS,
                    )
                    transcript_section = (
                        f"\nA speech transcript was detected in this clip:\n"
                        f"  \"{transcript_data['text']}\"\n"
                        f"Use this transcript to understand the human's spoken instruction.\n"
                    )
            finally:
                os.unlink(wav_tmp.name)

    # Merge audio into video (for Gemini native audio processing)
    if VOICED_VIDEO:
        audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
        if audio_pcm16.size > 0 and sample_rate > 0:
            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_tmp.close()
            _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
            try:
                video_bytes = _merge_audio_into_video_bytes(video_bytes, wav_tmp.name)
            finally:
                os.unlink(wav_tmp.name)

    return video_bytes, transcript_section, transcript_data


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
        return f"[VLM] Remove{cmd.target_indices}"
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

def _call_gemini(video_bytes: bytes, prompt: str, api_key: str,
                 extra_parts: Optional[List] = None) -> Tuple[dict, float]:
    """Call Gemini, return (parsed_json, latency_seconds).

    Handles VOICED_VIDEO (no VideoMetadata so Gemini processes audio),
    thinking_budget, and optional extra_parts (e.g. keyframe images).
    """
    client = genai.Client(api_key=api_key)

    if VOICED_VIDEO:
        video_part = types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
        )
    else:
        video_part = types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=VLM_FPS),
        )

    parts = [video_part]
    if extra_parts:
        parts.extend(extra_parts)
    parts.append(types.Part(text=prompt))

    t0 = time.time()
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=VLM_THINKING_BUDGET),
            response_mime_type="application/json",
        ),
    )
    latency = time.time() - t0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_text = response.text

    parsed = json.loads(_extract_json_text(raw_text))
    return parsed, round(latency, 3)


def _load_queue_images_as_parts(queue: KeyframeQueue,
                                 resize: int = 256) -> List:
    """Load keyframe images from the queue and return as Gemini Parts."""
    parts = []
    with queue._lock:
        for i, (kf_img, instr) in enumerate(queue._items):
            if kf_img is None:
                continue
            img = cv2.resize(kf_img, (resize, resize),
                             interpolation=cv2.INTER_LANCZOS4)
            _, png_bytes = cv2.imencode(".png", img)
            parts.append(types.Part(
                text=f"[Queue index {i}] {instr}:",
            ))
            parts.append(types.Part(
                inline_data=types.Blob(data=png_bytes.tobytes(),
                                       mime_type="image/png"),
            ))
    return parts


def _query_gemini_initial(video_bytes: bytes, transcript_section: str,
                          api_key: str, duration_sec: float):
    """One-shot call before robot execution to build the initial queue.

    Returns (parsed_response, latency).
    """
    prompt = PROMPT_TEMPLATE_INITIAL.format(
        duration=duration_sec,
        fps=float(VLM_FPS),
        transcript_section=transcript_section,
        max_intervals=_MAX_INTERVALS,
    )

    parsed, latency = _call_gemini(video_bytes, prompt, api_key)
    return parsed, latency


def _query_gemini_async_update(video_bytes: bytes,
                               queue: KeyframeQueue,
                               seg_start_sec: float, seg_end_sec: float,
                               transcript_section: str,
                               api_key: str):
    """Periodic async call during robot execution — returns a queue command.

    Returns (command, parsed_dict, latency).
    """
    # Build visual queue parts
    extra_parts = None
    visual_queue_note = ""
    if VISUAL_QUEUE and queue.total > 0:
        extra_parts = _load_queue_images_as_parts(queue, resize=256)
        visual_queue_note = (
            "\nThe keyframe images for each queue element are attached above "
            "(labelled by queue index). Use these images to visually identify "
            "which queue element corresponds to which object when deciding on "
            "Replace or Remove actions.\n"
        )

    prompt = PROMPT_TEMPLATE_ASYNC_UPDATE.format(
        queue_state=queue.queue_state_str(),
        current_idx=queue.current_idx,
        current_instruction=queue.current_instruction or "(none)",
        t_start=seg_start_sec,
        t_end=seg_end_sec,
        segment_duration=(seg_end_sec - seg_start_sec),
        fps=float(VLM_FPS),
        visual_queue_section=visual_queue_note,
        transcript_section=transcript_section,
    )

    parsed, latency = _call_gemini(video_bytes, prompt, api_key,
                                    extra_parts=extra_parts)
    command_name = parsed.get("command", "NoOp")

    print(f"[AsyncVLM] command={command_name}  latency={latency:.1f}s  "
          f"reasoning={parsed.get('reasoning', '')[:120]}")

    if command_name == "NoOp" or command_name not in (
            "Append", "Replace", "Preempt", "Remove"):
        return CmdNoOp(), parsed, latency

    if command_name == "Remove":
        # Support both target_indices (list) and legacy target_idx (single int)
        indices = parsed.get("target_indices")
        if indices and isinstance(indices, list):
            return CmdRemove(target_indices=[int(i) for i in indices]), parsed, latency
        target_idx = parsed.get("target_idx")
        if target_idx is not None:
            return CmdRemove(target_indices=[int(target_idx)]), parsed, latency
        print("[AsyncVLM] Remove: missing target_indices/target_idx — NoOp")
        return CmdNoOp(), parsed, latency

    # Commands that need a keyframe: Append, Replace, Preempt
    #
    # IMPORTANT: VLM sees a 4fps subsampled video, but returns keyframe
    # intervals in *seconds* (relative to clip start = 0).  We convert
    # the midpoint time to a frame index in the ORIGINAL 30fps buffer so
    # that `seg_start + local_frame_idx` yields the correct abs_idx.
    num_seg_frames_orig = int((seg_end_sec - seg_start_sec) * _VIDEO_RECORD_FPS)

    new_elements = parsed.get("new_elements", [])
    if not new_elements:
        # Legacy format fallback
        interval = parsed.get("keyframe_interval")
        instruction = (parsed.get("instruction") or "").strip()
        if interval is None:
            print(f"[AsyncVLM] {command_name}: missing new_elements/keyframe_interval — NoOp")
            return CmdNoOp(), parsed, latency
        local_frame_idx = _mid_frame_from_interval(
            interval, num_seg_frames_orig, _VIDEO_RECORD_FPS)
        return _RawCommand(
            name=command_name,
            target_idx=parsed.get("target_idx"),
            local_frame_idx=local_frame_idx,
            instruction=instruction,
        ), parsed, latency

    # Use first new_element for the keyframe
    elem = new_elements[0]
    interval = elem.get("keyframe_interval", {})
    instruction = elem.get("subtask_instruction", "")
    description = elem.get("keyframe_description", "")
    local_frame_idx = _mid_frame_from_interval(
        interval, num_seg_frames_orig, _VIDEO_RECORD_FPS)

    return _RawCommand(
        name=command_name,
        target_idx=parsed.get("target_idx"),
        local_frame_idx=local_frame_idx,
        instruction=instruction,
        keyframe_description=description,
    ), parsed, latency


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

        # Robot pause signal: when STOP_ROBOT_WHILE_RECV is True and voice is
        # detected, this event is SET to pause the robot. It is CLEARED once
        # the VLM response arrives and the queue is updated.
        self.robot_pause_event = threading.Event()  # SET = robot should pause

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
        print("[AsyncVLM] Worker started.")
        while not self._stop_event.is_set():
            # ---- wait interval_sec of real wall-clock time ----
            self._stop_event.wait(timeout=self._interval_sec)
            if self._stop_event.is_set():
                break

            trigger_t = self._elapsed()

            # ---- atomically drain all frames accumulated since last call ----
            aria_seg, cropped_seg, gaze_seg, seg_start, seg_end = (
                self._buf.get_segment())

            if not aria_seg:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames, skip")
                continue

            self._call_count += 1
            call_id = self._call_count

            seg_start_sec = seg_start / _VIDEO_RECORD_FPS
            seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

            print(f"[AsyncVLM] Call #{call_id} | t={trigger_t:.2f}s | "
                  f"obs frames {seg_start}–{seg_end} ({len(aria_seg)} frames) | "
                  f"time {seg_start_sec:.1f}s–{seg_end_sec:.1f}s")

            # Collect audio records for this segment
            audio_records = self._buf.get_audio_records(seg_start, seg_end)

            # Save raw segment video with audio for logging
            seg_video_path = os.path.join(
                self._temp_dir, f"async_seg_{call_id:04d}.mp4")
            _write_video_with_audio(
                aria_seg, audio_records, seg_video_path, fps=_VLM_VIDEO_SAVE_FPS)

            # ── STOP_ROBOT_WHILE_RECV: whisper pre-check ──
            # Do a quick whisper check on the audio. If voice is detected,
            # signal the main thread to pause the robot until VLM responds.
            voice_detected = False
            if STOP_ROBOT_WHILE_RECV and WHISPER_ENABLED:
                audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
                if audio_pcm16.size > 0 and sample_rate > 0:
                    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    wav_tmp.close()
                    _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
                    try:
                        quick_transcript = _whisper_transcribe_wav(wav_tmp.name)
                        if quick_transcript["text"].strip():
                            voice_detected = True
                            print(f"[AsyncVLM] #{call_id} Voice detected: "
                                  f"\"{quick_transcript['text'][:80]}\" → pausing robot")
                            self.robot_pause_event.set()
                    finally:
                        os.unlink(wav_tmp.name)

            # Create log entry (sub-thread will fill timing fields).
            log_entry: dict = {
                "call_id":         call_id,
                "trigger_t":       trigger_t,
                "obs_window":      (seg_start, seg_end),
                "obs_time_sec":    (seg_start_sec, seg_end_sec),
                "n_frames":        len(aria_seg),
                "step_at_trigger": self.current_step,
                "voice_detected":  voice_detected,
                "gemini_start_t":  None,
                "gemini_end_t":    None,
                "command":         None,
            }
            self.call_log.append(log_entry)

            if STOP_ROBOT_WHILE_RECV and voice_detected:
                # Run Gemini SYNCHRONOUSLY so robot stays paused until done.
                self._run_gemini_call(
                    call_id, aria_seg, gaze_seg, audio_records,
                    seg_start, seg_end, log_entry)
                # Unpause robot now that queue is updated.
                self.robot_pause_event.clear()
                print(f"[AsyncVLM] #{call_id} VLM done → resuming robot")
            else:
                # Fire Gemini call in a sub-thread — timer resets immediately.
                threading.Thread(
                    target=self._run_gemini_call,
                    args=(call_id, aria_seg, gaze_seg, audio_records,
                          seg_start, seg_end, log_entry),
                    daemon=True,
                    name=f"GeminiCall-{call_id}",
                ).start()

            # ← loop back immediately; new frames accumulate while Gemini runs.

        print("[AsyncVLM] Worker stopped.")

    # ------------------------------------------------------------------
    # Per-call sub-thread
    # ------------------------------------------------------------------

    def _run_gemini_call(self, call_id: int,
                         aria_seg: List[np.ndarray],
                         gaze_seg: List[dict],
                         audio_records: List[dict],
                         seg_start: int, seg_end: int,
                         log_entry: dict):

        seg_start_sec = seg_start / _VIDEO_RECORD_FPS
        seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

        print(f"[AsyncVLM] #{call_id} → Preparing VLM input...")

        try:
            # Prepare video: gaze crop → subsample → encode → whisper → audio
            video_bytes, transcript_section, _ = _prepare_vlm_video(
                aria_seg, gaze_seg, audio_records, seg_start_sec,
            )

            # Save the VLM input video (with whisper captions) to temp dir
            vlm_input_path = os.path.join(
                self._temp_dir, f"vlm_input_{call_id:04d}.mp4")
            try:
                with open(vlm_input_path, "wb") as vf:
                    vf.write(video_bytes)
                print(f"[AsyncVLM] #{call_id} VLM input saved → {vlm_input_path}")
            except Exception as save_exc:
                print(f"[AsyncVLM] #{call_id} Failed to save VLM input: {save_exc}")

            print(f"[AsyncVLM] #{call_id} → Gemini request "
                  f"({len(video_bytes)/1024:.1f} KB)")
            log_entry["gemini_start_t"] = self._elapsed()

            raw, parsed, latency = _query_gemini_async_update(
                video_bytes=video_bytes,
                queue=self._queue,
                seg_start_sec=seg_start_sec,
                seg_end_sec=seg_end_sec,
                transcript_section=transcript_section,
                api_key=self._api_key,
            )

            gemini_end_t = self._elapsed()
            log_entry["gemini_end_t"] = gemini_end_t
            log_entry["latency"] = latency

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
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Keyframe simulation video (like offline_simulation.mp4 from baseline)
# ---------------------------------------------------------------------------

def _render_sim_canvas(
    video_frame: Optional[np.ndarray],
    queue_snapshot: Optional[List[dict]],
    current_time: float,
) -> np.ndarray:
    """Render one frame of the keyframe simulation visualization.

    Layout: SIM_CANVAS_W x SIM_CANVAS_H canvas.
      - Top-right: live video thumbnail (SIM_VIDEO_SIZE x SIM_VIDEO_SIZE)
      - Left: keyframe queue in 5×2 grid
    """
    W, H = SIM_CANVAS_W, SIM_CANVAS_H
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    font = cv2.FONT_HERSHEY_SIMPLEX
    vs = SIM_VIDEO_SIZE

    # ── Video thumbnail (top-right) ──
    vx = W - vs - 30
    vy = 30
    if video_frame is not None:
        thumb = cv2.resize(video_frame, (vs, vs), interpolation=cv2.INTER_LANCZOS4)
        canvas[vy:vy + vs, vx:vx + vs] = thumb
        cv2.rectangle(canvas, (vx - 2, vy - 2), (vx + vs + 1, vy + vs + 1),
                       (180, 180, 180), 2)
    cv2.putText(canvas, f"t = {current_time:.1f}s",
                (vx, vy + vs + 35), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Queue section ──
    thumb_w, thumb_h = SIM_THUMB_SIZE

    cv2.putText(canvas, "Keyframe Action Queue",
                (SIM_LEFT_MARGIN, 45), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    if queue_snapshot is None:
        cv2.putText(canvas, "Initializing...",
                    (SIM_LEFT_MARGIN, 150), font, 1.2, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas

    if len(queue_snapshot) == 0:
        cv2.putText(canvas, "(queue is empty)",
                    (SIM_LEFT_MARGIN, 150), font, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas

    for i, elem in enumerate(queue_snapshot):
        col = i // SIM_ROWS_PER_COL
        row = i % SIM_ROWS_PER_COL

        tx = SIM_LEFT_MARGIN + col * SIM_COL_WIDTH
        ty = SIM_TOP_START + row * SIM_ROW_HEIGHT + 8
        if ty + thumb_h > H or tx + thumb_w > W:
            continue

        # Thumbnail
        kf_img = elem.get("keyframe_image")
        if kf_img is not None:
            kf_resized = cv2.resize(kf_img, (thumb_w, thumb_h),
                                     interpolation=cv2.INTER_LANCZOS4)
            canvas[ty:ty + thumb_h, tx:tx + thumb_w] = kf_resized
        else:
            canvas[ty:ty + thumb_h, tx:tx + thumb_w] = 40  # dark grey placeholder

        # Border — green if currently executing
        is_current = elem.get("is_current", False)
        border_color = (0, 255, 0) if is_current else (100, 100, 100)
        border_thick = 4 if is_current else 2
        cv2.rectangle(canvas, (tx - 2, ty - 2), (tx + thumb_w + 1, ty + thumb_h + 1),
                       border_color, border_thick)

        # Text
        text_x = tx + thumb_w + 25
        instr = elem.get("instruction", "")
        color = (100, 255, 100) if instr.startswith("pick up") else (100, 200, 255)
        if is_current:
            color = (0, 255, 255)  # yellow for currently executing
        cv2.putText(canvas, f"[{i}] {instr}",
                    (text_x, ty + 40), font, 1.1, color, 2, cv2.LINE_AA)

        if is_current:
            cv2.putText(canvas, ">>> EXECUTING",
                        (text_x, ty + 80), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return canvas


class SimulationVideoRecorder:
    """Records keyframe_simulation.mp4 in real-time as frames arrive.

    The main thread calls `record_frame(aria_bgr, queue, current_time)` at
    a steady rate; this class writes the rendered canvas to a VideoWriter.
    Audio is merged at the end via `finalize()`.
    """

    def __init__(self, output_path: str, fps: float = SIM_FPS):
        self._output_path = output_path
        self._fps = fps
        self._tmp_path = output_path + ".tmp.mp4"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._tmp_path, fourcc, fps, (SIM_CANVAS_W, SIM_CANVAS_H))
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._finalized = False

    def record_frame(self, aria_bgr: Optional[np.ndarray],
                     queue: Optional[KeyframeQueue],
                     current_time: float):
        """Render and write one simulation frame."""
        if self._start_time is None:
            self._start_time = time.monotonic()

        queue_snap = queue.snapshot() if queue is not None else None
        canvas = _render_sim_canvas(aria_bgr, queue_snap, current_time)
        self._writer.write(canvas)
        self._frame_count += 1

    def record_init_phase(self, aria_frames: List[np.ndarray],
                          queue: Optional[KeyframeQueue],
                          fps: float = _VIDEO_RECORD_FPS):
        """Write the init phase (before 'p' press) as 'Initializing...' frames,
        then switch to showing the queue once available.

        We subsample to SIM_FPS from the original capture FPS.
        """
        step = max(1, int(fps / self._fps))
        for i in range(0, len(aria_frames), step):
            t = i / fps
            # Before queue is ready, show "Initializing..."
            if queue is None:
                canvas = _render_sim_canvas(aria_frames[i], None, t)
            else:
                canvas = _render_sim_canvas(aria_frames[i], queue.snapshot(), t)
            self._writer.write(canvas)
            self._frame_count += 1

    def finalize(self, audio_records: Optional[List[dict]] = None):
        """Close writer, re-encode H264, optionally merge audio."""
        if self._finalized:
            return
        self._finalized = True

        self._writer.release()

        if self._frame_count == 0:
            print("[SimVideo] No frames recorded, skipping.")
            if os.path.exists(self._tmp_path):
                os.unlink(self._tmp_path)
            return

        if not os.path.exists(self._tmp_path):
            print(f"[SimVideo] Temp video missing, skipping finalize: {self._tmp_path}")
            return

        # Re-encode to H264
        _ffmpeg_reencode_h264(self._tmp_path, self._output_path)
        os.unlink(self._tmp_path)

        # Merge audio if available
        if audio_records:
            audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
            if audio_pcm16.size > 0 and sample_rate > 0:
                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_tmp.close()
                _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
                try:
                    # Use ffmpeg merge
                    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    tmp_out.close()
                    subprocess.run(
                        ["ffmpeg", "-y", "-v", "error",
                         "-i", self._output_path,
                         "-i", wav_tmp.name,
                         "-c:v", "copy",
                         "-ac", "1",
                         "-c:a", "aac", "-b:a", "128k",
                         "-shortest",
                         tmp_out.name],
                        check=True,
                    )
                    os.replace(tmp_out.name, self._output_path)
                finally:
                    if os.path.exists(wav_tmp.name):
                        os.unlink(wav_tmp.name)

        print(f"[SimVideo] Saved: {self._output_path} "
              f"({self._frame_count} frames)")


def _safe_finalize_sim_recorder(
    sim_recorder: Optional[SimulationVideoRecorder],
    video_buffer: Optional[VideoBuffer] = None,
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


# ---------------------------------------------------------------------------
# CV2 preview helpers
# ---------------------------------------------------------------------------

def _build_preview_frame(obs, status_text, status_color=(0, 255, 0)):
    images     = obs["images"]
    human_view = images["aria_rgb"].copy()
    top_view   = images["top_camera_rgb"].copy()

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
    cv2.putText(preview, "human(gaze) | top", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, status_text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    return preview


def _build_eval_preview(obs, keyframe_img: np.ndarray,
                        queue: KeyframeQueue, current_instruction: str = ""):
    human_view = obs["images"]["aria_rgb"].copy()
    kf_display = cv2.resize(keyframe_img,
                            (human_view.shape[1], human_view.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    preview = np.hstack((human_view, kf_display))
    cv2.putText(preview,
                f"LIVE | KF [{queue.current_idx + 1}/{queue.total}]",
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
                      api_key: str, video_buffer: VideoBuffer,
                      sim_recorder: Optional[SimulationVideoRecorder] = None,
                      ) -> KeyframeQueue:
    """
    Stream live aria | top frames.
    Press 'p' to snapshot the buffer, call Gemini, and build the initial
    KeyframeQueue.  All captured frames are written into video_buffer so
    the async worker sees them in its history.

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

    print("Keyframe selection: press 'p' to capture & query VLM, 'q' to quit.")
    print("If XQuartz misses key input, press the same key in the SSH terminal.")
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
            key = _poll_keypress()
            capture_timer.sleep()   # pace the loop to _VIDEO_RECORD_FPS

            if key in (ord('p'), ord('P')) and aria_frames:
                total = len(aria_frames)
                duration_sec = total / _VIDEO_RECORD_FPS
                print(f"[keyframe] Captured {total} frames @ {_VIDEO_RECORD_FPS} FPS "
                      f"({duration_sec:.2f}s)")

                # Save raw video + audio for logging
                _write_video_with_audio(
                    aria_frames, audio_records, video_path, fps=_VIDEO_RECORD_FPS)
                _write_video(aria_cropped_frames, cropped_video_path)
                with open(gaze_path, "w") as f:
                    json.dump({"frames": gaze_records}, f, indent=2)

                # Prepare VLM input: gaze crop → subsample → whisper → audio
                print("[keyframe] Preparing VLM input (gaze crop + audio)...")
                vlm_bytes, transcript_section, _ = _prepare_vlm_video(
                    aria_frames, gaze_records, audio_records,
                    clip_start_sec=0.0,
                )

                print(f"[keyframe] VLM input: {len(vlm_bytes)/1024:.1f} KB")
                print("[keyframe] Querying Gemini for initial keyframes...")
                parsed, latency = _query_gemini_initial(
                    video_bytes=vlm_bytes,
                    transcript_section=transcript_section,
                    api_key=api_key,
                    duration_sec=duration_sec,
                )
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
                    cropped_kf = _draw_gaze_crosshair(cropped_kf)

                    cropped_path = os.path.join(
                        _TEMP_DIR, f"cropped_keyframe_{i1:02d}.png")
                    plt.imsave(cropped_path,
                               cv2.cvtColor(cropped_kf, cv2.COLOR_BGR2RGB))
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

                result_queue = KeyframeQueue(keyframe_imgs, subtask_instrs)

                # ── Write init phase to simulation video ──
                if sim_recorder is not None:
                    # Phase 1: all frames before 'p' → "Initializing..."
                    sim_recorder.record_init_phase(
                        aria_frames, queue=None, fps=_VIDEO_RECORD_FPS)
                    # Phase 2: show initialized queue for ~2 seconds
                    for _ in range(int(SIM_FPS * 2)):
                        last_frame = aria_frames[-1] if aria_frames else None
                        sim_recorder.record_frame(
                            last_frame, result_queue, duration_sec)

                return result_queue

            elif key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def eval(env, client, language_instruction: str, horizon: int,
         queue: KeyframeQueue, video_buffer: VideoBuffer,
         vlm_worker: AsyncVLMWorker,
         sim_recorder: Optional[SimulationVideoRecorder] = None,
         chunksize: int = 16):
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

            if keyframe_img is not None:
                keyframe_img = cv2.cvtColor(keyframe_img, cv2.COLOR_BGR2RGB)
                aria_rgb     = cv2.cvtColor(images["aria_rgb"], cv2.COLOR_BGR2RGB)
            # temp
            #keyframe_img = _BLACK_TOP_KEYFRAME
            #
            observation = {
                "observation.images.top_view":                    images["top_camera_rgb"],
                #"observation.images.human_view":                  aria_rgb,
                "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
                "observation.images.human_view_keyframe_cropped": keyframe_img,
                #"observation.gaze_point":                         obs["gaze_point"],
                "prompt":                                         current_instr,
                "observation.state":                              concatenated_state,
            }

            if global_step == 0 and keyframe_img is not None:
                plt.imsave(os.path.join(_TEMP_DIR, "test_aria_keyframe.png"), keyframe_img)
                plt.imsave(os.path.join(_TEMP_DIR, "test_aria_rgb.png"), aria_rgb)
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
                          "(still streaming observations)...")
                    _pause_timer = Timer(env.control_hz)
                    while vlm_worker.robot_pause_event.is_set():
                        # Read live observation (no robot action sent)
                        _pause_obs = _extract_latest_observation(
                            env.get_observation(
                                mode='human_view_streamed_and_keyframe_cropped'))
                        _p_aria = _pause_obs["images"]["aria_rgb"]
                        _p_crop = _pause_obs["images"]["aria_cropped_rgb"]
                        _p_audio_obs = _pause_obs.get("audio", {})
                        _p_audio_chunk = np.asarray(
                            _p_audio_obs.get("chunk_pcm16",
                                             np.zeros((0,), dtype=np.int16)),
                            dtype=np.int16)
                        _p_audio_ts = int(_p_audio_obs.get("audio_ts_ns", 0) or 0)
                        _p_audio_sr = int(_p_audio_obs.get("sample_rate_hz", 0) or 0)
                        _p_audio_ns = int(
                            _p_audio_obs.get("num_samples", _p_audio_chunk.size) or 0)
                        # Resize to match expected shape
                        _tgt_h, _tgt_w = aria_bgr.shape[:2]
                        if _p_aria.shape[:2] != (_tgt_h, _tgt_w):
                            _p_aria = cv2.resize(
                                _p_aria, (_tgt_w, _tgt_h),
                                interpolation=cv2.INTER_LINEAR)
                        # Keep feeding the shared video buffer
                        video_buffer.append(
                            _p_aria, _p_crop, None, None,
                            audio_chunk_pcm16=_p_audio_chunk,
                            audio_ts_ns=_p_audio_ts,
                            audio_sample_rate_hz=_p_audio_sr,
                            audio_num_samples=_p_audio_ns)
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
                        cv2.waitKey(1)
                        _pause_timer.sleep()
                    print("[Eval] Robot resumed.")

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
                    _safe_finalize_sim_recorder(sim_recorder, video_buffer)
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
            _safe_finalize_sim_recorder(sim_recorder, video_buffer)
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()
    _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path,
                        vlm_event_log=vlm_worker.event_log,
                        keyframe_log=keyframe_log)
    _safe_finalize_sim_recorder(sim_recorder, video_buffer)


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
    language_instruction = "Pick up this muffin."
    horizon              = 10000
    GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    print("Starting. Instruction:", language_instruction)
    print("During eval: n=next keyframe  q=stop episode")

    # Shared buffer — persists for the lifetime of the process.
    video_buffer = VideoBuffer()

    # ---- Simulation video recorder ----
    sim_output_path = os.path.join(_TEMP_DIR, "keyframe_simulation.mp4")
    sim_recorder = SimulationVideoRecorder(sim_output_path)

    # ---- Initial keyframe extraction (interactive) ----
    with _terminal_keypress_mode():
        queue = extract_keyframes(env, language_instruction, GEMINI_API_KEY,
                                  video_buffer, sim_recorder=sim_recorder)

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
                with _terminal_keypress_mode():
                    eval(env, client, language_instruction, horizon,
                         queue, video_buffer, vlm_worker,
                         sim_recorder=sim_recorder)
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
            # Finalize previous simulation video and create a new one.
            _safe_finalize_sim_recorder(sim_recorder, video_buffer)
            video_buffer = VideoBuffer()
            sim_output_path = os.path.join(_TEMP_DIR, "keyframe_simulation.mp4")
            sim_recorder = SimulationVideoRecorder(sim_output_path)
            with _terminal_keypress_mode():
                queue = extract_keyframes(
                    env, language_instruction, GEMINI_API_KEY, video_buffer,
                    sim_recorder=sim_recorder)
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
        _safe_finalize_sim_recorder(sim_recorder, video_buffer)
        if os.path.isdir(_TEMP_DIR):
            import shutil
            shutil.rmtree(_TEMP_DIR)
            os.makedirs(_TEMP_DIR, exist_ok=True)
            print(f"Cleared temp directory: {_TEMP_DIR}")
        env.safe_end()
