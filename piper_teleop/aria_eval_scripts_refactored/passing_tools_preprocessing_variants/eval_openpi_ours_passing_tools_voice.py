"""eval_openpi_async_vlm_gaze_keyframe_queue.py

Async VLM-based keyframe queue management for dual-arm robot evaluation.

Architecture
------------
  Main thread   : robot eval loop + CV2 display.
  VLM thread    : background Gemini calls that manipulate the KeyframeQueue.

Human-view (aria_rgb) frames are continuously written into a shared
VideoBuffer starting from t=0.  The async worker fires every
VLM_INTERVAL_SEC seconds, drains accumulated frames, and produces
one queue-manipulation command. Append can insert multiple keyframes from
one async video clip:

    NoOp / Append / Replace / Preempt / Remove
"""

import argparse
import numpy as np
import cv2
import os
import json
import re
import subprocess
import threading
import tempfile
import textwrap
import time
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
from common.streamer import (
    BackgroundObservationStreamer,
)
from common.config import SERVER_IP


##checkout common/audio_filter if needed in th efuture

# ---------------------------------------------------------------------------
# Constants — edit these instead of CLI args
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS   = 15
_MAX_INTERVALS      = 10
_TEMP_DIR           = os.path.normpath("/home/alin/projects/piper_project/temp")
_BLACK_TOP_KEYFRAME = np.zeros((240, 424, 3), dtype=np.uint8)

# ── Runtime artifact saving ──
# Keep these off for real robot inference. They are useful for debugging, but
# they accumulate frames/audio/video and can slow long runs even when Gemini is
# skipped.
SAVE_INITIAL_CAPTURE_LOGS      = False   # video.mp4, cropped_video.mp4, gaze.json
SAVE_INITIAL_VLM_INPUT_VIDEO   = False   # initial_vlm_input_*.mp4
SAVE_ASYNC_VLM_INPUT_VIDEOS    = False   # vlm_input_*.mp4
SAVE_ASYNC_CAPTIONED_SEGMENTS  = False   # async_seg_*.mp4
SAVE_ROLLOUT_VIDEO             = False   # rollout.mp4 + top-view frame list
SAVE_SIMULATION_VIDEO          = False   # keyframe_simulation.mp4
SAVE_POLICY_DEBUG_IMAGES       = False   # test_aria_keyframe/rgb.png
SAVE_KEYFRAME_IMAGES           = False   # cropped_keyframe_*.png
SAVE_KEYFRAME_METADATA         = True    # small key_frames.json/keyframes.json
ARCHIVE_AUDIO_FOR_SIM_VIDEO    = SAVE_SIMULATION_VIDEO
STREAM_SELECTOR_TO_VLM_BUFFER  = False   # selector frames are discarded before async anyway

# ── Realtime display cost control ──
# Keep OpenCV UI lightweight so XQuartz/local canvas rendering does not steal
# time from robot control. Key polling still runs every loop.
REALTIME_DISPLAY_ENABLED       = True
REALTIME_DISPLAY_FPS           = 3.0
REALTIME_QUEUE_CANVAS_ENABLED  = False   # large queue canvas is expensive
REALTIME_HOLD_DISPLAY_FPS      = 2.0

# ── VLM pipeline settings ──
VLM_MODEL           = "gemini-3.1-flash-lite-preview"       # Gemini model to use
VLM_FPS             = 4                         # subsample to this FPS before sending to VLM
VLM_INTERVAL_SEC    = 2                         # seconds per async precheck / base VLM clip
VLM_INTERVAL_SEC_MAX = 5                        # max voice-anchored VLM clip length when speech continues
VLM_TARGET_RES      = (256, 256)                # resize video segments for VLM
VLM_THINKING_BUDGET = 2048                      # thinking token budget per call
VLM_REQUEST_TIMEOUT_SEC = 15.0                  # retry Gemini calls after this many seconds
VLM_MAX_ATTEMPTS    = 3                         # total attempts per VLM request
VLM_RETRY_BACKOFF_SEC = 0.5                     # brief pause between failed attempts
# Which Aria view to encode into the mp4 sent to Gemini.
# Options:
#   - "aria_rgb": full egocentric RGB frames
#   - "aria_cropped_rgb": gaze-centered cropped RGB frames
_GEMINI_VIDEO_SOURCE = "aria_cropped_rgb"
VLM_DRAW_GAZE_MARKER = True                   # overlay center gaze marker on cropped VLM input
STORE_FULL_ARIA_IN_VLM_BUFFER = (_GEMINI_VIDEO_SOURCE == "aria_rgb")

# ── Audio settings ──
WHISPER_ENABLED     = True                      # use Whisper STT → transcript in prompt
VOICED_VIDEO        = False                      # merge audio into mp4 → Gemini processes audio

# ── Fast audio sanity gate before Whisper ──
# This is much cheaper than Whisper. It is intentionally strict so only a
# close mic-on speaker reaches STT; background speech/noise should be rejected.
FAST_AUDIO_GATE_ENABLED = True
FAST_AUDIO_GATE_FRAME_MS = 30.0
FAST_AUDIO_GATE_NOISE_PERCENTILE = 20.0
FAST_AUDIO_GATE_NOISE_MULT = 3.5
FAST_AUDIO_GATE_MIN_RMS = 0.0020
FAST_AUDIO_GATE_MIN_PEAK = 0.012
FAST_AUDIO_GATE_MIN_ACTIVE_SEC = 0.12
FAST_AUDIO_GATE_MIN_ACTIVE_RATIO = 0.04
FAST_AUDIO_GATE_MIN_SNR_DB = 8.0
FAST_AUDIO_GATE_MIN_SPEECH_BAND_RATIO = 0.35
FAST_AUDIO_GATE_MAX_LOW_NOISE_RATIO = 0.85
FAST_AUDIO_GATE_MAX_HIGH_NOISE_RATIO = 0.90
FAST_AUDIO_GATE_EARLY_ACCEPT = True
FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_SEC = 0.18
FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_RATIO = 0.05
FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_SNR_DB = 12.0
VOICE_PRECHECK_TIMING_LOG = True
VOICE_PRECHECK_WARN_SEC = 0.75                 # warn when audio gate + Whisper blocks async cadence

# ── Async VLM dispatch gating ──
ASYNC_VLM_REQUIRE_USEFUL_VOICE = True           # original mode: skip Gemini when Whisper sees no useful speech
ASYNC_VLM_USE_VOICE_FOR_WINDOWING = True        # original mode: voice anchors/extends the VLM video window
ASYNC_VLM_USE_AUDIO_TRANSCRIPT_CONTEXT = True   # original mode: async VLM may use Whisper transcript context
ASYNC_VLM_DISCARD_INITIAL_TARGET_SEC = 0.5
ASYNC_VLM_DISCARD_INITIAL_RATIO = ASYNC_VLM_DISCARD_INITIAL_TARGET_SEC/VLM_INTERVAL_SEC           # legacy name: fraction of VLM clip kept before command cue
ASYNC_VLM_PREPEND_EMPTY_INITIAL_SEGMENT = True  # synthetic black/silent first segment to avoid startup wait
ASYNC_VLM_MAX_BACKLOG_SEGMENTS = 2.0            # if async checks fall behind, drop stale unprocessed windows
ASYNC_VLM_KEEP_LATEST_SEGMENTS = 1.0            # keep only this much newest unprocessed video after backlog flush
ASYNC_VOICE_PRECHECK_THREAD_ENABLED = True      # run audio gate + Whisper away from VLM scheduler
ASYNC_VOICE_PRECHECK_MAX_ACTIVE = 2             # cap Whisper/API pressure; newer stale segments are skipped
VOICE_COMMAND_CUE_KEYWORDS_ENABLED = True       # set False to ignore cue words like "hand"/"give"/"tool"
USEFUL_VOICE_MIN_WORDS_WITHOUT_CUE = 1          # increase to 2 to reject one-word transcripts when cues are off
USEFUL_VOICE_MIN_CHARS = 2
USEFUL_VOICE_MIN_INPUT_RMS = 0.0005             # reject only near-silent clips before Whisper

# ── Gaze crop settings ──
GAZE_CROP_ENABLED   = True
GAZE_CROP_SIZE      =  768                      # crop NxN around gaze centre
GAZE_CALIBRATION_Y_OFFSET = 0                 # px to add to Y (corrects upward calibration bias)

# ── Visual queue ──
VISUAL_QUEUE        = True                      # attach keyframe images in VLM update calls
VISUAL_QUEUE_MAX_IMAGES = 4                     # bound async prompt growth after repeated Appends
VISUAL_QUEUE_IMAGE_SIZE = 160                   # small visual context is enough for queue identity
VISUAL_QUEUE_JPEG_QUALITY = 70                  # JPEG is much cheaper than PNG for prompt thumbnails

# ── VLA policy prompt ──
# Modes:
#   - "fixed": send the same generic policy prompt for every subtask.
#   - "subtask": send the queue's current subtask instruction to the policy.
#   - "global": send the episode-level language_instruction to the policy.
VLA_PROMPT_MODE     = "fixed"
VLA_FIXED_PROMPT    = "I want this tool."

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
# Keep this aligned with the source capture FPS so saved 8-second segments
# preserve their real duration and caption timing.
_VLM_VIDEO_SAVE_FPS = _VIDEO_RECORD_FPS


# ---------------------------------------------------------------------------
# Gemini prompts  (ported from eval_vlm_baseline_asynchronous.py)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_INITIAL = """\
You are analyzing the first {duration:.1f} seconds of a first-person \
(egocentric) video recorded in a store. A human is giving instructions to a \
robot by pointing at objects and speaking.

Your task has two steps. You MUST complete them in order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Step 1 — Visual Grounding: Extract keyframes**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Watch the video and identify every moment when the human points at or \
indicates an object/location. Each such moment is a **keyframe**.

Use the **convergence of multiple cues** to pinpoint each keyframe:
  - Utterance timing: when the spoken instruction refers to a target \
(words like "this", "that", or naming an object).
  - Hand gesture: when the hand/finger clearly points to a specific \
object or region.
  - Gaze fixation: when the wearer's gaze stabilises on the target \
(if gaze overlay is available).

Prioritise intervals where hand pointing and gaze are spatially aligned \
(same object/region). Use utterance timing to confirm which object is meant.

Each keyframe interval should be short and precise (0.2–2.0 s). \
Do not merge separate pointing events into one long interval.

Record each keyframe with:
  - **keyframe_description**: fine-grained visual description of what \
the human is pointing at (color, shape, brand, size, shelf position, \
distinguishing features). E.g., "small red cupcake with white frosting \
on the 2nd shelf from top, left side".
  - **keyframe_interval**: {{"start": <float>, "end": <float>}} in seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Step 2 — Contextual Recomposition: Build action queue**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using the keyframes from Step 1, construct a robot **action queue** — \
the sequence of atomic pick-and-place steps a robot would execute.

The human instruction may reference multiple objects and destinations in \
a single sentence. You must decompose it into individual steps. Examples:

  - "Put this, this, and this to that basket" \
→ 3 keyframes (obj1, obj2, obj3) + 1 keyframe (basket) \
→ 6 action queue elements:
    [pick up obj1, place it in basket,
     pick up obj2, place it in basket,
     pick up obj3, place it in basket]

  - "Give me this cupcake and this bottle" \
→ 2 keyframes → 2 action queue elements:
    [pick up the red cupcake, pick up the green bottle]

  - "Put this here" \
→ 2 keyframes (object + destination) → 2 action queue elements:
    [pick up <object>, place it <destination>]

RULES for subtask_instruction:
  - MUST use exactly one of two verb forms:
    1. "pick up <object>"
    2. "place it <destination>"
  - Do NOT use any other verb (e.g., "point to", "grab", "put", "get", \
"move", "take"). Always rephrase to "pick up" or "place it".
  - PERSPECTIVE: This is an egocentric (first-person) video from the \
wearer's viewpoint. All spatial references (left, right, top, bottom, \
near, far) MUST be from the **wearer's perspective** — i.e., "left" \
means to the wearer's left as seen in the video. Do NOT use the \
perspective of a person facing the camera.
  - CLOSED VOCABULARY — subtask_instruction MUST be EXACTLY one of these \
five strings (character-for-character, no rephrasing, no added color / \
size / shelf / location descriptors):
    * "pick up hex screwdriver"
    * "pick up Philips screwdriver"
    * "pick up 5 cm metal support"
    * "pick up 10 cm metal support"
    * "pick up 12 cm metal support"
  - Do NOT invent variations such as "pick up the hex screwdriver", \
"pick up flathead screwdriver", "pick up Allen key", "pick up wrench", \
"pick up 7 cm metal support", "pick up metal rod", "pick up screwdriver \
on the left", etc.
  - Do NOT produce any "place it ..." elements for this task — the human \
is asking the robot to HAND tools to them, so every action queue element \
must be exactly one of the five "pick up ..." strings above.
  - If the object the human points to does not clearly match any of the \
five allowed items, pick the closest valid match from the list rather \
than inventing a new name.

Each action queue element inherits the keyframe_interval from the \
corresponding keyframe in Step 1. For recomposed elements (e.g., \
"place it in basket" repeated for multiple objects), use the \
keyframe_interval of the destination keyframe.

{transcript_section}
The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON. **You MUST populate the keys in the exact order \
shown below.** Start with `human_intent` — first infer the overall intent \
of what the human is trying to accomplish (what objects, what destinations, \
what outcome they want), then use that inferred intent to ground Step 1 \
(keyframe extraction) and Step 2 (action queue decomposition):
{{
  "human_intent": "<inferred high-level goal of the human: what they want the robot to do, which objects are involved, and which destinations/recipients. Reason about this FIRST, before extracting keyframes or building the action queue.>",
  "reasoning": "<brief explanation of cues used in Step 1>",
  "human_instruction_summary": "<one-line summary of what the human asked>",
  "keyframe_queue": [
    {{
      "keyframe_description": "<detailed visual description of what is pointed at>",
      "keyframe_interval": {{"start": <float>, "end": <float>}}
    }}
  ],
  "action_queue": [
    {{
      "keyframe_description": "<detailed visual description>",
      "subtask_instruction": "<pick up X or place it Y>",
      "keyframe_interval": {{"start": <float>, "end": <float>}}
    }}
  ]
}}

Important:
  - All timestamps are relative to this clip (0.0 = beginning of clip).
  - keyframe_queue contains ONLY the raw pointing moments (N keyframes).
  - action_queue contains the recomposed robot steps (may be longer than \
keyframe_queue due to pick-up/place-it pairing).
  - Maximum {max_intervals} action queue elements allowed.
  - Intervals must be sorted chronologically.
  - Each interval should be short and precise (0.2–2.0 seconds).
  - Always return positive start/end values.
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

Choose exactly ONE command action. The Append command may contain multiple
new_elements/keyframes from this same video clip:

1. **NoOp**: No change needed. The robot is working normally, no human \
intervention detected, or the person is simply observing without meaningful \
instruction.

2. **Append**: Add new element(s) to the END of the queue. Use when the \
human adds one or more objects/destinations to the task. If multiple \
targets are indicated in this clip, include ALL of them in one Append \
response as separate new_elements, sorted by time.

The keyframe intervals are determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object.
  - Hand gesture (pointing): when the hand/finger clearly points to a \
specific object or region.
  - Gaze fixation: when the wearer's gaze stabilises on the target object \
(if gaze information is available).

RULES:
- subtask_instruction MUST use ONLY "pick up <object>" or "place it <destination>"
- Do NOT use any other verbs (e.g., "point to", "grab", "put", "get", \
"move", "take"). Always rephrase to "pick up" or "place it".
- keyframe_description must be fine-grained with visual attributes (color, \
shape, brand name, size, shelf position, etc.)
- PERSPECTIVE: This is an egocentric (first-person) video from the \
wearer's viewpoint. All spatial references (left, right, top, bottom, \
near, far) MUST be from the **wearer's perspective** — i.e., "left" \
means to the wearer's left as seen in the video. Do NOT use the \
perspective of a person facing the camera.
- CLOSED VOCABULARY — every newly generated subtask_instruction MUST be \
EXACTLY one of these five strings (character-for-character, no rephrasing, \
no added color / size / shelf / location descriptors):
    * "pick up hex screwdriver"
    * "pick up Philips screwdriver"
    * "pick up 5 cm metal support"
    * "pick up 10 cm metal support"
    * "pick up 12 cm metal support"
- Do NOT invent variations such as "pick up the hex screwdriver", \
"pick up flathead screwdriver", "pick up Allen key", "pick up wrench", \
"pick up 7 cm metal support", "pick up metal rod", "pick up screwdriver \
on the left", etc.
- Do NOT emit any "place it ..." element for this task — the human is \
asking the robot to HAND tools to them, so every new_elements entry must \
be exactly one of the five "pick up ..." strings above.
- If the clip does not clearly support one of these five instructions, \
prefer NoOp rather than inventing a new one.
- All timestamps are **relative to this clip** (0.0 = start of this clip, \
{segment_duration:.1f} = end of this clip)
- Apply contextual recomposition: if the human points at two tools in the \
same utterance (e.g. "hand me these two"), append TWO elements — one \
"pick up ..." per tool — using the closed vocabulary above.
- Multiple keyframes in one video are allowed and expected: do NOT collapse \
separate pointing events into one keyframe, and do NOT return only the \
first or last target when several valid tools are indicated.
- Every new_elements entry must have its own short keyframe_interval. Sort \
new_elements chronologically by keyframe_interval.start.

{transcript_section}
This video clip is {segment_duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON. **You MUST populate the keys in the exact order \
shown below.** Start with `human_intent` — first infer whether (and how) \
the human is intervening in this clip and what they want to change about \
the task. Use that inferred intent to decide whether NoOp or Append is \
appropriate and what new_elements to propose:
{{
  "human_intent": "<inferred intent of the human in THIS clip: are they silent/passive, or do they want to add new objects/destinations? If intervening, describe the high-level goal (e.g., 'add a chocolate cookie to the task and place it in the right basket'). Reason about this FIRST, before choosing the command.>",
  "reasoning": "<explanation of what you observed in the clip>",
  "command": "<NoOp|Append>",
  "new_elements": [
    {{
      "keyframe_description": "<detailed visual description>",
      "subtask_instruction": "<pick up X or place it Y>",
      "keyframe_interval": {{"start": <float, seconds>, "end": <float, seconds>}}
    }}
  ]
}}

For NoOp: new_elements = []
For Append: new_elements = chronological list of ALL new elements in this \
clip, possibly more than one
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
class CmdAppendMany:
    keyframes: List[np.ndarray]
    instructions: List[str]


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
    target_indices: List[int]


# Internal: raw command before the frame is resolved into an ndarray.
@dataclass
class _RawCommand:
    name: str                    # "Append" | "Replace" | "Preempt"
    target_idx: Optional[int]
    local_frame_idx: int         # index within the segment
    instruction: str
    keyframe_description: str = ""


@dataclass
class _RawAppendElement:
    local_frame_idx: int
    instruction: str
    keyframe_description: str = ""


@dataclass
class _RawAppendMany:
    elements: List[_RawAppendElement]


class EpisodeEnded(Exception):
    """Raised when the user ends the current episode with the 'e' hotkey."""

    pass


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

            if isinstance(cmd, CmdAppendMany):
                appended = []
                for keyframe, instruction in zip(cmd.keyframes, cmd.instructions):
                    self._items.append((keyframe, instruction))
                    appended.append(f"[{len(self._items) - 1}]: '{instruction}'")
                if appended:
                    return "[Queue] AppendMany → " + ", ".join(appended)
                return "[Queue] AppendMany: no valid elements."

            if isinstance(cmd, CmdReplace):
                idx = cmd.target_idx
                if 0 <= idx < len(self._items):
                    self._items[idx] = (cmd.keyframe, cmd.instruction)
                    return f"[Queue] Replace [{idx}] → '{cmd.instruction}'"
                return f"[Queue] Replace: invalid idx {idx} (size={len(self._items)})"

            if isinstance(cmd, CmdPreempt):
                self._items.insert(self._current_idx, (cmd.keyframe, cmd.instruction))
                return f"[Queue] Preempt → inserted at [{self._current_idx}]: '{cmd.instruction}'"

            if isinstance(cmd, CmdRemove):
                removed = []
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

    Frame indices returned by this class are absolute episode-local indices.
    Old processed frames can be pruned without invalidating later absolute
    indices because `_base_frame_idx` tracks the absolute index of list[0].
    """

    def __init__(self, archive_audio: bool = ARCHIVE_AUDIO_FOR_SIM_VIDEO):
        self._lock    = threading.Lock()
        self._aria    : List[np.ndarray] = []
        self._cropped : List[np.ndarray] = []
        self._gaze    : List[dict]       = []
        self._audio   : List[dict]       = []
        self._archived_audio: List[dict] = []
        self._archive_audio: bool        = bool(archive_audio)
        self._base_frame_idx: int        = 0
        self._vlm_processed_up_to: int   = 0

    def _total_frames_locked(self) -> int:
        return self._base_frame_idx + len(self._aria)

    def _to_local_slice_locked(self, start: int, end: int) -> Tuple[int, int]:
        start = max(self._base_frame_idx, int(start))
        end = max(start, min(int(end), self._total_frames_locked()))
        return start - self._base_frame_idx, end - self._base_frame_idx

    def append(self, aria_bgr: np.ndarray, cropped_bgr: np.ndarray,
               gaze_x: Optional[float], gaze_y: Optional[float],
               audio_chunk_pcm16: Optional[np.ndarray] = None,
               audio_ts_ns: int = 0,
               audio_sample_rate_hz: int = 0,
               audio_num_samples: Optional[int] = None):
        with self._lock:
            idx = self._total_frames_locked()
            cropped_copy = cropped_bgr.copy()
            if STORE_FULL_ARIA_IN_VLM_BUFFER:
                aria_copy = aria_bgr.copy()
            else:
                # This script sends cropped Aria frames to Gemini. Keeping the
                # full Aria frame here only adds streamer copy/memory cost.
                aria_copy = cropped_copy
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

    def append_empty_segment(self,
                             num_frames: int,
                             aria_shape: Tuple[int, int, int],
                             cropped_shape: Tuple[int, int, int],
                             audio_sample_rate_hz: int = 0) -> Tuple[int, int]:
        """Append a black/silent synthetic segment and return abs (start, end)."""
        num_frames = max(0, int(num_frames))
        if num_frames <= 0:
            with self._lock:
                idx = self._total_frames_locked()
            return idx, idx

        cropped_blank = np.zeros(tuple(cropped_shape), dtype=np.uint8)
        aria_blank = (
            np.zeros(tuple(aria_shape), dtype=np.uint8)
            if STORE_FULL_ARIA_IN_VLM_BUFFER else cropped_blank
        )
        with self._lock:
            start = self._total_frames_locked()
            for offset in range(num_frames):
                idx = start + offset
                cropped_copy = cropped_blank.copy()
                self._aria.append(
                    aria_blank.copy() if STORE_FULL_ARIA_IN_VLM_BUFFER else cropped_copy)
                self._cropped.append(cropped_copy)
                self._gaze.append({
                    "frame_idx": idx,
                    "gaze_x": None,
                    "gaze_y": None,
                    "synthetic_empty": True,
                })
                self._audio.append({
                    "frame_idx": idx,
                    "audio_ts_ns": 0,
                    "sample_rate_hz": int(audio_sample_rate_hz),
                    "num_samples": 0,
                    "chunk_pcm16": np.zeros((0,), dtype=np.int16),
                    "synthetic_empty": True,
                })
            return start, start + num_frames

    def total_frames(self) -> int:
        with self._lock:
            return self._total_frames_locked()

    def unprocessed_count(self) -> int:
        with self._lock:
            return max(0, self._total_frames_locked() - self._vlm_processed_up_to)

    def skip_unprocessed_to_latest(self, keep_frames: int) -> int:
        """Mark stale unprocessed frames as consumed, preserving recent tail.

        This is a real-time safety valve: if audio/VLM checks fall behind wall
        clock, process the newest segment rather than slowly catching up on old
        audio/video that can no longer affect the robot in time.
        """
        with self._lock:
            total = self._total_frames_locked()
            target = max(
                self._vlm_processed_up_to,
                total - max(0, int(keep_frames)),
            )
            skipped = target - self._vlm_processed_up_to
            self._vlm_processed_up_to = target
            return max(0, skipped)

    def skip_to_current(self):
        """Mark all frames currently in the buffer as already processed.
        Call this before starting the async worker so pre-execution
        pointing-gesture frames are not re-analysed during rollout."""
        with self._lock:
            self._vlm_processed_up_to = self._total_frames_locked()

    def get_segment(self, max_frames: Optional[int] = None) -> Tuple[List, List, List, int, int]:
        """Atomically drain unprocessed frames and advance the pointer.

        If ``max_frames`` is provided, drain at most that many frames so the
        async worker can keep fixed-duration clip boundaries.
        Returns (aria_list, cropped_list, gaze_list, seg_start, seg_end).
        """
        with self._lock:
            start = max(self._vlm_processed_up_to, self._base_frame_idx)
            end   = self._total_frames_locked()
            if max_frames is not None:
                end = min(end, start + max(0, int(max_frames)))
            local_start, local_end = self._to_local_slice_locked(start, end)
            aria    = [f.copy() for f in self._aria[local_start:local_end]]
            cropped = [f.copy() for f in self._cropped[local_start:local_end]]
            gaze    = list(self._gaze[local_start:local_end])
            self._vlm_processed_up_to = end
            return aria, cropped, gaze, start, end

    def reserve_segment(self, max_frames: Optional[int] = None) -> Tuple[int, int, int, bool]:
        """Advance the processed pointer without copying image frames.

        The async worker uses this for the Whisper pre-check path so no-voice
        segments can be discarded using audio only.
        """
        with self._lock:
            start = max(self._vlm_processed_up_to, self._base_frame_idx)
            end = self._total_frames_locked()
            if max_frames is not None:
                end = min(end, start + max(0, int(max_frames)))
            local_start, local_end = self._to_local_slice_locked(start, end)
            gaze_slice = self._gaze[local_start:local_end]
            synthetic_empty = bool(gaze_slice) and all(
                g.get("synthetic_empty") for g in gaze_slice)
            self._vlm_processed_up_to = end
            return start, end, max(0, end - start), synthetic_empty

    def get_range(self, start: int, end: int) -> Tuple[List, List, List]:
        """Copy an arbitrary frame range without changing the processed pointer."""
        with self._lock:
            local_start, local_end = self._to_local_slice_locked(start, end)
            aria = [f.copy() for f in self._aria[local_start:local_end]]
            cropped = [f.copy() for f in self._cropped[local_start:local_end]]
            gaze = list(self._gaze[local_start:local_end])
            return aria, cropped, gaze

    def mark_processed_up_to(self, end: int):
        """Advance the VLM processed pointer past frames consumed as context."""
        with self._lock:
            self._vlm_processed_up_to = max(
                self._vlm_processed_up_to,
                min(int(end), self._total_frames_locked()),
            )

    def get_audio_records(self, start: int = 0, end: Optional[int] = None) -> List[dict]:
        with self._lock:
            full_history = (start == 0 and end is None)
            if end is None:
                end = self._total_frames_locked()
            if full_history:
                source_records = self._archived_audio + self._audio
            else:
                local_start, local_end = self._to_local_slice_locked(start, end)
                source_records = self._audio[local_start:local_end]
            records = []
            for record in source_records:
                copied = dict(record)
                copied["chunk_pcm16"] = record["chunk_pcm16"].copy()
                records.append(copied)
            return records

    def get_frame(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            local_idx = int(idx) - self._base_frame_idx
            if 0 <= local_idx < len(self._aria):
                return self._aria[local_idx].copy(), self._cropped[local_idx].copy()
            return None, None

    def get_gaze(self, idx: int) -> Optional[dict]:
        """Return gaze record at the given frame index, or None."""
        with self._lock:
            local_idx = int(idx) - self._base_frame_idx
            if 0 <= local_idx < len(self._gaze):
                return dict(self._gaze[local_idx])
            return None

    def prune_processed(self, keep_frames: int = 0,
                        protect_from: Optional[int] = None) -> int:
        """Drop old processed frames/audio while preserving absolute indices.

        `keep_frames` retains a small pre-roll window for future voice-anchored
        VLM clips. `protect_from` prevents pruning frames still needed by an
        in-flight Gemini call that has not resolved its returned keyframe yet.
        Returns the number of frames dropped.
        """
        with self._lock:
            prune_before = self._vlm_processed_up_to - max(0, int(keep_frames))
            if protect_from is not None:
                prune_before = min(prune_before, int(protect_from))
            prune_before = max(
                self._base_frame_idx,
                min(prune_before, self._total_frames_locked()),
            )
            drop = prune_before - self._base_frame_idx
            if drop <= 0:
                return 0
            if self._archive_audio:
                self._archived_audio.extend(self._audio[:drop])
            del self._aria[:drop]
            del self._cropped[:drop]
            del self._gaze[:drop]
            del self._audio[:drop]
            self._base_frame_idx = prune_before
            self._vlm_processed_up_to = max(
                self._vlm_processed_up_to,
                self._base_frame_idx,
            )
            return drop


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
        return "aria_cropped_rgb"
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


def _pcm16_audio_stats(audio_pcm16: np.ndarray, sample_rate_hz: int) -> dict:
    audio_pcm16 = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
    if audio_pcm16.size == 0:
        return {
            "input_rms": 0.0,
            "input_peak": 0.0,
            "num_samples": 0,
            "sample_rate_hz": int(sample_rate_hz or 0),
        }

    audio_float = audio_pcm16.astype(np.float32) / 32768.0
    return {
        "input_rms": float(np.sqrt(np.mean(audio_float * audio_float))),
        "input_peak": float(np.max(np.abs(audio_float))),
        "num_samples": int(audio_pcm16.size),
        "sample_rate_hz": int(sample_rate_hz or 0),
    }


def _fast_audio_gate(
    audio_pcm16: np.ndarray,
    sample_rate_hz: int,
    context: str = "whisper",
) -> Tuple[bool, dict]:
    """Cheap voice/noise sanity check before Whisper.

    The gate is intentionally strict for a close mic-on speaker. It rejects
    low-energy, low-SNR, and steady-noise clips without modifying the audio
    sent to Whisper.
    """
    stats = _pcm16_audio_stats(audio_pcm16, sample_rate_hz)
    stats["fast_gate_enabled"] = bool(FAST_AUDIO_GATE_ENABLED)
    stats["fast_gate_passed"] = True
    stats["fast_gate_reason"] = "disabled"

    if not FAST_AUDIO_GATE_ENABLED:
        return True, stats

    audio_pcm16 = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
    if audio_pcm16.size == 0 or sample_rate_hz <= 0:
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "empty"
        return False, stats

    input_rms = float(stats.get("input_rms", 0.0))
    input_peak = float(stats.get("input_peak", 0.0))
    if input_rms < FAST_AUDIO_GATE_MIN_RMS or input_peak < FAST_AUDIO_GATE_MIN_PEAK:
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "low_energy"
        return False, stats

    x = audio_pcm16.astype(np.float32) / 32768.0
    x = x - float(np.mean(x))
    if x.size >= 2:
        # Vectorized pre-emphasis: cheap high-pass proxy for activity/noise
        # detection, not used for the final Whisper audio.
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - 0.97 * x[:-1]
    else:
        y = x

    frame_len = max(1, int(round(
        sample_rate_hz * FAST_AUDIO_GATE_FRAME_MS / 1000.0)))
    pad = (-y.size) % frame_len
    padded = np.pad(y, (0, pad), mode="constant") if pad else y
    frames = padded.reshape(-1, frame_len)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_floor = float(np.percentile(
        frame_rms, FAST_AUDIO_GATE_NOISE_PERCENTILE))
    threshold = max(
        FAST_AUDIO_GATE_MIN_RMS,
        noise_floor * FAST_AUDIO_GATE_NOISE_MULT,
    )
    active = frame_rms >= threshold
    active_ratio = float(np.mean(active)) if active.size else 0.0
    active_sec = float(np.sum(active) * frame_len / float(sample_rate_hz))
    active_frame_rms = frame_rms[active]
    active_rms = (
        float(np.sqrt(np.mean(active_frame_rms * active_frame_rms)))
        if active_frame_rms.size else 0.0
    )
    snr_db = float(20.0 * np.log10(
        max(active_rms, 1e-8) / max(noise_floor, 1e-8)))
    rms_p20 = float(np.percentile(frame_rms, 20.0)) if frame_rms.size else 0.0
    rms_p90 = float(np.percentile(frame_rms, 90.0)) if frame_rms.size else 0.0
    modulation_ratio = rms_p90 / max(rms_p20, 1e-6)

    stats.update({
        "fast_gate_noise_floor": noise_floor,
        "fast_gate_threshold": float(threshold),
        "fast_gate_active_ratio": active_ratio,
        "fast_gate_active_sec": active_sec,
        "fast_gate_active_rms": active_rms,
        "fast_gate_snr_db": snr_db,
        "fast_gate_modulation_ratio": modulation_ratio,
    })

    if (
        active_sec < FAST_AUDIO_GATE_MIN_ACTIVE_SEC
        or active_ratio < FAST_AUDIO_GATE_MIN_ACTIVE_RATIO
    ):
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "no_active_region"
        return False, stats

    if snr_db < FAST_AUDIO_GATE_MIN_SNR_DB:
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "low_snr"
        return False, stats

    if (
        FAST_AUDIO_GATE_EARLY_ACCEPT
        and active_sec >= FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_SEC
        and active_ratio >= FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_RATIO
        and snr_db >= FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_SNR_DB
    ):
        stats["fast_gate_passed"] = True
        stats["fast_gate_reason"] = "early_accept_voice"
        stats["fast_gate_speech_band_ratio"] = None
        stats["fast_gate_low_noise_ratio"] = None
        stats["fast_gate_high_noise_ratio"] = None
        return True, stats

    # Spectral noise sanity check. Downsample by stride for speed; this is only
    # a coarse speech-band ratio, not a transcription-quality transform.
    stride = max(1, int(sample_rate_hz // 16000))
    sx = x[::stride]
    spectral_rate = sample_rate_hz / float(stride)
    if sx.size >= 256:
        window = np.hanning(sx.size).astype(np.float32)
        power = np.abs(np.fft.rfft(sx * window)) ** 2
        freqs = np.fft.rfftfreq(sx.size, d=1.0 / spectral_rate)

        def _band_ratio(lo_hz: float, hi_hz: float, total_power: float) -> float:
            mask = (freqs >= lo_hz) & (freqs <= min(hi_hz, spectral_rate / 2.0))
            if not np.any(mask) or total_power <= 1e-12:
                return 0.0
            return float(np.sum(power[mask]) / total_power)

        total_mask = (freqs >= 40.0) & (freqs <= min(7600.0, spectral_rate / 2.0))
        total_power = float(np.sum(power[total_mask])) if np.any(total_mask) else 0.0
        speech_ratio = _band_ratio(120.0, 4200.0, total_power)
        low_noise_ratio = _band_ratio(40.0, 100.0, total_power)
        high_noise_ratio = _band_ratio(6000.0, 7600.0, total_power)
        stats.update({
            "fast_gate_speech_band_ratio": speech_ratio,
            "fast_gate_low_noise_ratio": low_noise_ratio,
            "fast_gate_high_noise_ratio": high_noise_ratio,
        })

        if (
            speech_ratio < FAST_AUDIO_GATE_MIN_SPEECH_BAND_RATIO
            and (
                low_noise_ratio > FAST_AUDIO_GATE_MAX_LOW_NOISE_RATIO
                or high_noise_ratio > FAST_AUDIO_GATE_MAX_HIGH_NOISE_RATIO
            )
        ):
            stats["fast_gate_passed"] = False
            stats["fast_gate_reason"] = "noise_dominated"
            return False, stats

        if (
            active_ratio > 0.80
            and modulation_ratio < 1.35
            and speech_ratio < 0.55
        ):
            stats["fast_gate_passed"] = False
            stats["fast_gate_reason"] = "steady_noise"
            return False, stats

    stats["fast_gate_reason"] = "passed"
    print(
        f"  [AudioGate:{context}] pass "
        f"rms={input_rms:.4f}, active={active_sec:.2f}s/"
        f"{active_ratio:.2f}, snr={snr_db:.1f}dB, noise={noise_floor:.4f}"
    )
    return True, stats


def _write_whisper_input_wav(
    audio_pcm16: np.ndarray,
    sample_rate_hz: int,
    output_path: str,
    context: str = "whisper",
) -> dict:
    """Write raw captured audio to Whisper without denoising or gating."""
    stats = _pcm16_audio_stats(audio_pcm16, sample_rate_hz)
    _write_pcm16_wav(audio_pcm16, sample_rate_hz, output_path)

    if stats.get("num_samples", 0) > 0:
        print(
            f"  [WhisperAudio:{context}] raw "
            f"rms={stats.get('input_rms', 0.0):.4f}, "
            f"peak={stats.get('input_peak', 0.0):.4f}, "
            f"samples={stats.get('num_samples', 0)}"
        )
    return stats


def _transcribe_audio_records_for_whisper(
    audio_records: List[dict],
    context: str = "whisper",
) -> dict:
    t0 = time.monotonic()
    audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate <= 0:
        return {"text": "", "words": []}

    gate_passed, gate_stats = _fast_audio_gate(
        audio_pcm16, sample_rate, context=context)
    if not gate_passed:
        gate_stats["voice_precheck_latency_sec"] = time.monotonic() - t0
        print(
            f"  [AudioGate:{context}] skip Whisper: "
            f"{gate_stats.get('fast_gate_reason', 'unknown')} "
            f"(rms={gate_stats.get('input_rms', 0.0):.4f}, "
            f"peak={gate_stats.get('input_peak', 0.0):.4f}, "
            f"active={gate_stats.get('fast_gate_active_sec', 0.0):.2f}s, "
            f"snr={gate_stats.get('fast_gate_snr_db', 0.0):.1f}dB, "
            f"speech_band={gate_stats.get('fast_gate_speech_band_ratio', 0.0):.2f})"
        )
        return {"text": "", "words": [], "audio_stats": gate_stats}

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    stats = _write_whisper_input_wav(
        audio_pcm16, sample_rate, wav_tmp.name, context=context)
    stats.update({
        k: v for k, v in gate_stats.items()
        if k.startswith("fast_gate_")
    })
    if stats.get("input_rms", 0.0) < USEFUL_VOICE_MIN_INPUT_RMS:
        stats["voice_precheck_latency_sec"] = time.monotonic() - t0
        print(
            f"  [Whisper:{context}] skip transcription: input_rms="
            f"{stats.get('input_rms', 0.0):.4f} < {USEFUL_VOICE_MIN_INPUT_RMS:.4f}"
        )
        os.unlink(wav_tmp.name)
        return {"text": "", "words": [], "audio_stats": stats}
    try:
        result = _whisper_transcribe_wav(wav_tmp.name)
        elapsed = time.monotonic() - t0
        stats["voice_precheck_latency_sec"] = elapsed
        if VOICE_PRECHECK_TIMING_LOG:
            print(
                f"  [Whisper:{context}] precheck latency={elapsed:.2f}s "
                f"(audio={audio_pcm16.size / float(sample_rate):.2f}s)"
            )
        if elapsed > VOICE_PRECHECK_WARN_SEC:
            print(
                f"  [Whisper:{context}] WARNING: precheck took {elapsed:.2f}s; "
                "async VLM may lag behind real time."
            )
        result["audio_stats"] = stats
        return result
    finally:
        if os.path.exists(wav_tmp.name):
            os.unlink(wav_tmp.name)


_VOICE_COMMAND_CUE_WORDS = {
    "want",
    "this",
    "that",
    "it",
    "give",
    "hand",
    "pass",
    "pick",
    "grab",
    "take",
    "place",
    "tool",
    "screwdriver",
    "hex",
    "philips",
    "support",
}


def _is_useful_whisper_transcript(transcript_data: dict) -> bool:
    audio_stats = (transcript_data or {}).get("audio_stats", {}) or {}
    if audio_stats.get("input_rms", 1.0) < USEFUL_VOICE_MIN_INPUT_RMS:
        return False

    text = (transcript_data or {}).get("text", "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False

    normalized = re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()
    if len(normalized.replace(" ", "")) < USEFUL_VOICE_MIN_CHARS:
        return False

    ignored = {
        "yeah",
        "okay",
        "ok",
        "um",
        "uh",
        "hmm",
        "thank",
        "thank you",
        "thanks",
        "thanks for watching",
        "music",
        "silence",
    }
    if normalized in ignored:
        return False

    hallucination_phrases = (
        "thank you for watching",
        "thanks for watching",
        "have a nice day",
        "see you next time",
        "please subscribe",
        "like and subscribe",
        "dont forget to subscribe",
        "welcome back to my channel",
        "see you in the next video",
    )
    if any(phrase in normalized for phrase in hallucination_phrases):
        print(f"  [Whisper] rejected likely no-speech hallucination: \"{text[:100]}\"")
        return False

    words = re.findall(r"[a-z0-9]+", normalized)
    if (
        VOICE_COMMAND_CUE_KEYWORDS_ENABLED
        and any(word in _VOICE_COMMAND_CUE_WORDS for word in words)
    ):
        return True

    useful_words = [
        w for w in words
        if w not in ignored and len(w) >= USEFUL_VOICE_MIN_CHARS
    ]
    return len(useful_words) >= max(1, int(USEFUL_VOICE_MIN_WORDS_WITHOUT_CUE))


def _first_useful_voice_start_sec(transcript_data: dict) -> Optional[float]:
    if not _is_useful_whisper_transcript(transcript_data):
        return None

    ignored_words = {
        "yeah", "okay", "ok", "um", "uh", "hmm",
        "thanks", "thank", "music", "silence",
    }
    first_useful_start_sec = None
    for word in (transcript_data or {}).get("words", []) or []:
        token = re.sub(r"[^a-z0-9]+", "", str(word.get("word", "")).lower())
        if not token or token in ignored_words:
            continue
        if len(token) < USEFUL_VOICE_MIN_CHARS:
            continue
        start_sec = max(0.0, float(word.get("start", 0.0) or 0.0))
        if first_useful_start_sec is None:
            first_useful_start_sec = start_sec
        if VOICE_COMMAND_CUE_KEYWORDS_ENABLED and token in _VOICE_COMMAND_CUE_WORDS:
            return start_sec

    if first_useful_start_sec is not None:
        return first_useful_start_sec

    # Whisper sometimes returns text without word timestamps. Fall back to
    # anchoring at the beginning of the transcribed audio window.
    return 0.0


def _build_transcript_section(transcript_data: dict) -> str:
    text = (transcript_data or {}).get("text", "").strip()
    if not text:
        return ""
    return (
        f"\nA speech transcript was detected in this clip:\n"
        f"  \"{text}\"\n"
        f"Use this transcript to understand the human's spoken instruction.\n"
    )


def _build_context_transcript_section(transcript_data: Optional[dict]) -> str:
    """Prompt-only transcript context that is not time-aligned to this video."""
    text = (transcript_data or {}).get("text", "").strip()
    if not text:
        return ""
    return (
        "\nSpeech was detected immediately before this fresh VLM clip:\n"
        f"  \"{text}\"\n"
        "This text is context only; it is not overlaid because it is not "
        "time-aligned to the current video frames.\n"
    )


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


def _get_frame_aligned_caption(
    words: list,
    time_sec: float,
    fps: float,
    transcript_text: str = "",
) -> str:
    """Return a word-level caption aligned to the current video frame.

    We treat each video frame as a small temporal window rather than a
    single instant. This keeps low-FPS debug videos (for example 4 FPS VLM
    inputs) aligned with Whisper word timestamps without falling back to a
    full-sentence overlay.
    """
    if words:
        frame_duration_sec = 1.0 / max(float(fps), 1.0)
        half_window_sec = max(0.03, frame_duration_sec * 0.55)
        window_start = max(0.0, time_sec - half_window_sec)
        window_end = time_sec + half_window_sec

        visible_words = []
        for word in words:
            start_sec = float(word.get("start", 0.0) or 0.0)
            end_sec = float(word.get("end", start_sec) or start_sec)
            if end_sec < start_sec:
                end_sec = start_sec
            if end_sec >= window_start and start_sec <= window_end:
                token = str(word.get("word", "")).strip()
                if token:
                    visible_words.append(token)

        if visible_words:
            return " ".join(visible_words)

        # If the frame lands in a very small gap between words, keep the
        # most recently spoken word for a short grace window.
        hold_sec = min(0.18, max(0.04, frame_duration_sec * 0.75))
        for word in reversed(words):
            start_sec = float(word.get("start", 0.0) or 0.0)
            end_sec = float(word.get("end", start_sec) or start_sec)
            if end_sec < start_sec:
                end_sec = start_sec
            if end_sec <= time_sec and time_sec - end_sec <= hold_sec:
                token = str(word.get("word", "")).strip()
                if token:
                    return token
            if end_sec < time_sec - hold_sec:
                break
        return ""

    # Do not draw a full sentence on every frame. If Whisper did not return
    # word timestamps, keep the video uncaptioned and pass transcript text
    # through the prompt only.
    return ""


def _burn_captions_on_video_bytes(
    video_bytes: bytes,
    words: list,
    transcript_text: str,
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
        caption = _get_frame_aligned_caption(
            words,
            time_sec=frame_idx / vid_fps,
            fps=vid_fps,
            transcript_text=transcript_text,
        )
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


def _overlay_captions_on_frames(
    frames_bgr: List[np.ndarray],
    words: list,
    transcript_text: str,
    clip_start_sec: float,
    fps: float,
) -> List[np.ndarray]:
    """Overlay Whisper captions on in-memory frames for debug video saving."""
    captioned_frames = []
    for frame_idx, frame in enumerate(frames_bgr):
        overlay = frame.copy()
        caption = _get_frame_aligned_caption(
            words,
            time_sec=frame_idx / max(float(fps), 1.0),
            fps=fps,
            transcript_text=transcript_text,
        )
        if caption:
            _draw_caption(overlay, caption)
        captioned_frames.append(overlay)
    return captioned_frames


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
    frames: List[np.ndarray],
    gaze_records: List[dict],
    audio_records: List[dict],
    clip_start_sec: float,
    frames_are_gaze_cropped: bool = False,
    whisper_context: str = "vlm",
    use_whisper: Optional[bool] = None,
) -> Tuple[bytes, str, dict]:
    """Prepare a video segment for VLM consumption.

    Applies: optional gaze crop → subsample → fresh Whisper → word captions → encode.
    Trigger/precheck transcripts must be passed as prompt-only context outside
    this function so they cannot be overlaid onto non-matching video frames.
    Returns (video_bytes, transcript_section, transcript_data).
    """
    if frames_are_gaze_cropped:
        processed = [frame.copy() for frame in frames]
    else:
        processed = []
        for i, frame in enumerate(frames):
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

    if frames_are_gaze_cropped and VLM_DRAW_GAZE_MARKER:
        processed = [_draw_gaze_crosshair(frame) for frame in processed]

    # Add a "VLM INPUT" banner + frame index so the saved mp4 is trivially
    # identifiable when debugging. The banner sits in a thin strip at the
    # top of the frame so it cannot be missed.
    _banner_source = _gemini_video_source_label()
    _banner_text_prefix = (
        f"VLM INPUT | src={_banner_source} | "
        f"cropped={frames_are_gaze_cropped} | clip_start={clip_start_sec:.1f}s"
    )
    _banner_frames = []
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _scale = 0.45
    for _i, _f in enumerate(processed):
        _fh = _f.shape[0]
        _fw = _f.shape[1]
        _bar = max(18, _fh // 14)
        _overlay = _f.copy()
        cv2.rectangle(_overlay, (0, 0), (_fw, _bar), (0, 0, 0), -1)
        cv2.putText(
            _overlay,
            f"{_banner_text_prefix} | frame={_i}",
            (4, _bar - 4), _font, _scale, (0, 255, 255), 1, cv2.LINE_AA,
        )
        _banner_frames.append(_overlay)
    processed = _banner_frames

    # Subsample FPS
    subsampled = _subsample_frames(processed, _VIDEO_RECORD_FPS, VLM_FPS)

    transcript_section = ""
    transcript_data: dict = {}

    # Whisper transcription. Async VLM can disable this while still keeping
    # the audio stream buffered/available elsewhere.
    if use_whisper is None:
        use_whisper = WHISPER_ENABLED
    if use_whisper:
        transcript_data = _transcribe_audio_records_for_whisper(
            audio_records, context=whisper_context)

    if transcript_data.get("text"):
        print(f"  [Whisper] \"{transcript_data['text'][:100]}\"")
        # Overlay captions onto the actual subsampled VLM frames before
        # encoding so word timestamps and frames stay aligned.
        subsampled = _overlay_captions_on_frames(
            subsampled,
            transcript_data.get("words", []),
            transcript_data["text"],
            0.0,
            VLM_FPS,
        )
        transcript_section = _build_transcript_section(transcript_data)

    # Encode to H264 video bytes after optional caption overlay.
    video_bytes = _frames_to_video_bytes(subsampled, VLM_FPS, VLM_TARGET_RES)

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
    """Overlay a highly-visible gaze marker at the image centre.

    For ``aria_cropped_rgb`` frames the gaze is, by construction, at the
    centre of the crop. The original marker (thin green crosshair) was
    hard to see on the compressed VLM input mp4; this version adds a
    red outer ring, a thicker green crosshair, and a ``GAZE`` label so
    it is unambiguous that gaze has actually been overlaid.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(12, h // 20)

    # Outer red ring for contrast on busy backgrounds.
    cv2.circle(out, (cx, cy), r + 4, (0, 0, 255), 3)

    # Inner green ring + thick crosshair.
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 3)
    cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), (0, 255, 0), 3)
    cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), (0, 255, 0), 3)

    # "GAZE" label just below the marker.
    label = "GAZE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, h / 400.0)
    (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
    lx = cx - tw // 2
    ly = min(cy + r * 2 + th + 6, h - 4)
    # Dark backdrop so the label is readable on any video.
    cv2.rectangle(out, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 4),
                  (0, 0, 0), -1)
    cv2.putText(out, label, (lx, ly), font, scale, (0, 255, 255), 2,
                cv2.LINE_AA)
    return out


def _voice_anchored_vlm_window(
    segment_start_frame: int,
    voice_start_sec_in_segment: float,
    video_len_frames: int,
    pre_voice_ratio: float = ASYNC_VLM_DISCARD_INITIAL_RATIO,
) -> Tuple[int, int, int, int]:
    """Return a VLM window anchored around the command cue word.

    ``pre_voice_ratio`` keeps that fraction of the base VLM clip before the
    cue so words like "I want" are preserved when the cue is "this".
    """
    video_len_frames = max(1, int(video_len_frames))
    pre_frames = max(0, int(round(video_len_frames * pre_voice_ratio)))
    pre_frames = min(pre_frames, video_len_frames - 1)
    post_frames = max(1, video_len_frames - pre_frames)
    voice_frame = int(segment_start_frame + round(
        max(0.0, voice_start_sec_in_segment) * _VIDEO_RECORD_FPS))
    start_frame = max(0, voice_frame - pre_frames)
    end_frame = voice_frame + post_frames
    return start_frame, end_frame, voice_frame, pre_frames


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _resolve_vla_prompt(subtask_instruction: Optional[str],
                        global_instruction: str) -> str:
    """Return the text prompt sent to the VLA policy."""
    mode = (VLA_PROMPT_MODE or "subtask").strip().lower()
    if mode == "fixed":
        return VLA_FIXED_PROMPT
    if mode == "global":
        return global_instruction
    return subtask_instruction or global_instruction


def _format_cmd_label(cmd) -> str:
    """Short display string for a VLM queue command."""
    if isinstance(cmd, CmdNoOp):
        return "[VLM] NoOp"
    if isinstance(cmd, CmdAppend):
        return f"[VLM] Append: {cmd.instruction}"
    if isinstance(cmd, CmdAppendMany):
        preview = "; ".join(cmd.instructions[:3])
        if len(cmd.instructions) > 3:
            preview += f"; +{len(cmd.instructions) - 3} more"
        return f"[VLM] Append {len(cmd.instructions)}: {preview}"
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
        return (0, 0, 180)
    if isinstance(cmd, (CmdAppend, CmdAppendMany)):
        return (0, 140, 0)
    if isinstance(cmd, CmdReplace):
        return (0, 150, 200)
    if isinstance(cmd, CmdRemove):
        return (0, 100, 200)
    return (80, 80, 80)


def _keyframe_metadata_paths() -> Tuple[str, str]:
    """Return the legacy and underscore metadata output paths."""
    return (
        os.path.join(_TEMP_DIR, "keyframes.json"),
        os.path.join(_TEMP_DIR, "key_frames.json"),
    )


def _write_keyframe_metadata(meta: dict, log_prefix: str = "keyframe"):
    if not SAVE_KEYFRAME_METADATA:
        return
    os.makedirs(_TEMP_DIR, exist_ok=True)
    for meta_path in _keyframe_metadata_paths():
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[{log_prefix}] keyframe metadata saved → "
          + ", ".join(_keyframe_metadata_paths()))


def _save_queue_snapshot_metadata(queue: KeyframeQueue,
                                  language_instruction: str,
                                  reason: str,
                                  event_log: Optional[List] = None):
    """Save the current in-memory queue when the episode ends.

    This catches keyframes added by async VLM after the initial p/p capture,
    including the path where the user ends the episode with 'e'.
    """
    with queue._lock:
        items = list(queue._items)
        current_idx = int(queue._current_idx)

    saved_items = []
    subtask_instrs = []
    vla_instrs = []
    for i1, (kf_img, instruction) in enumerate(items, start=1):
        vla_instruction = _resolve_vla_prompt(instruction, language_instruction)
        cropped_path = None
        if SAVE_KEYFRAME_IMAGES:
            cropped_path = os.path.join(_TEMP_DIR, f"cropped_keyframe_{i1:02d}.png")
        if SAVE_KEYFRAME_IMAGES and kf_img is not None:
            plt.imsave(cropped_path, cv2.cvtColor(kf_img, cv2.COLOR_BGR2RGB))
        subtask_instrs.append(instruction)
        vla_instrs.append(vla_instruction)
        saved_items.append({
            "order":                 i1,
            "queue_idx":             i1 - 1,
            "frame_idx":             None,
            "cropped_keyframe_path": cropped_path,
            "instruction":           instruction,
            "vla_instruction":       vla_instruction,
            "description":           "",
            "is_current":            (i1 - 1 == current_idx),
        })

    keyframe_meta = {
        "instruction":          language_instruction,
        "num_frames":           None,
        "video_fps":            _VIDEO_RECORD_FPS,
        "vlm_fps":              VLM_FPS,
        "vla_prompt_mode":      VLA_PROMPT_MODE,
        "vla_fixed_prompt":     VLA_FIXED_PROMPT,
        "subtask_instructions": subtask_instrs,
        "vla_instructions":     vla_instrs,
        "saved_keyframes":      saved_items,
        "current_idx":          current_idx,
        "total_keyframes":      len(items),
        "queue_done":           current_idx >= len(items),
        "save_reason":          reason,
        "saved_at_unix":        time.time(),
        "gemini_response": {
            "source":       "queue_snapshot",
            "reason":       reason,
            "action_queue": [
                {
                    "subtask_instruction": instruction,
                    "vla_instruction": _resolve_vla_prompt(
                        instruction, language_instruction),
                    "keyframe_description": "",
                    "keyframe_interval": None,
                }
                for instruction in subtask_instrs
            ],
        },
        "latency":              None,
    }
    if event_log is not None:
        keyframe_meta["vlm_event_log"] = [
            {"step": int(step), "label": str(label)}
            for step, label, _ in event_log
        ]

    _write_keyframe_metadata(keyframe_meta, log_prefix="Eval")


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

def _call_gemini_once(video_bytes: bytes, prompt: str, api_key: str,
                      extra_parts: Optional[List] = None) -> Tuple[dict, float]:
    """Make one Gemini request and parse its JSON response."""
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
        raw_text = response.text or ""

    try:
        parsed = json.loads(_extract_json_text(raw_text))
    except json.JSONDecodeError as exc:
        snippet = re.sub(r"\s+", " ", raw_text).strip()[:240]
        raise ValueError(
            f"JSON parse failed: {exc}; response_snippet={snippet!r}"
        ) from exc

    return parsed, round(latency, 3)


def _short_error_reason(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    text = re.sub(r"\s+", " ", text)
    return text[:300]


def _call_gemini(video_bytes: bytes, prompt: str, api_key: str,
                 extra_parts: Optional[List] = None) -> Tuple[dict, float]:
    """Call Gemini with timeout/retry, return (parsed_json, latency_seconds)."""
    last_error: Optional[BaseException] = None
    last_reason = "unknown error"

    for attempt in range(1, VLM_MAX_ATTEMPTS + 1):
        result: dict = {}
        done = threading.Event()

        def _target():
            try:
                result["value"] = _call_gemini_once(
                    video_bytes, prompt, api_key, extra_parts=extra_parts)
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=_target,
            daemon=True,
            name=f"GeminiRequestAttempt-{attempt}",
        ).start()

        will_retry = attempt < VLM_MAX_ATTEMPTS
        if not done.wait(timeout=VLM_REQUEST_TIMEOUT_SEC):
            last_reason = f"timeout after {VLM_REQUEST_TIMEOUT_SEC:.1f}s"
            suffix = "retrying with a fresh request." if will_retry else "no attempts left."
            print(f"[VLM] Gemini attempt {attempt}/{VLM_MAX_ATTEMPTS} "
                  f"{last_reason}; {suffix}")
        elif "error" in result:
            last_error = result["error"]
            last_reason = _short_error_reason(last_error)
            suffix = "retrying." if will_retry else "no attempts left."
            print(f"[VLM] Gemini attempt {attempt}/{VLM_MAX_ATTEMPTS} failed: "
                  f"{last_reason}; {suffix}")
        else:
            return result["value"]

        if will_retry and VLM_RETRY_BACKOFF_SEC > 0:
            time.sleep(VLM_RETRY_BACKOFF_SEC)

    raise RuntimeError(
        f"Gemini failed after {VLM_MAX_ATTEMPTS} attempts: {last_reason}"
    ) from last_error


def _select_visual_queue_indices(total: int, current_idx: int,
                                 max_images: int) -> List[int]:
    """Pick a bounded visual subset: current item plus recent queue tail."""
    total = max(0, int(total))
    max_images = max(0, int(max_images))
    if total <= 0 or max_images <= 0:
        return []
    if total <= max_images:
        return list(range(total))

    selected: List[int] = []

    def add(idx: int):
        if 0 <= idx < total and idx not in selected and len(selected) < max_images:
            selected.append(idx)

    add(current_idx)

    # The tail is where new Append commands accumulate, so keeping it visible
    # helps Gemini avoid duplicate appends without sending the entire queue.
    tail_budget = max_images - len(selected)
    for idx in range(max(0, total - tail_budget), total):
        add(idx)

    # Fill any remaining budget with the next few active items.
    for idx in range(max(0, current_idx + 1), total):
        add(idx)
        if len(selected) >= max_images:
            break

    return sorted(selected)


def _load_queue_images_as_parts(
    queue: KeyframeQueue,
    resize: int = VISUAL_QUEUE_IMAGE_SIZE,
    max_images: int = VISUAL_QUEUE_MAX_IMAGES,
) -> Tuple[List, int, int]:
    """Load a bounded keyframe-image subset for Gemini visual context."""
    parts = []
    with queue._lock:
        total = len(queue._items)
        indices = _select_visual_queue_indices(
            total, queue._current_idx, max_images)
        snapshot = [
            (i, queue._items[i][0], queue._items[i][1])
            for i in indices
            if queue._items[i][0] is not None
        ]

    for i, kf_img, instr in snapshot:
        img = cv2.resize(kf_img, (resize, resize),
                         interpolation=cv2.INTER_AREA)
        ok, jpg_bytes = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(VISUAL_QUEUE_JPEG_QUALITY)],
        )
        if not ok:
            continue
        parts.append(types.Part(
            text=f"[Queue index {i}] {instr}:",
        ))
        parts.append(types.Part(
            inline_data=types.Blob(data=jpg_bytes.tobytes(),
                                   mime_type="image/jpeg"),
        ))

    return parts, len(snapshot), total


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
        extra_parts, visual_count, queue_total = _load_queue_images_as_parts(
            queue,
            resize=VISUAL_QUEUE_IMAGE_SIZE,
            max_images=VISUAL_QUEUE_MAX_IMAGES,
        )
        visual_queue_note = (
            "\nA bounded subset of keyframe images from the queue is attached "
            "above (labelled by queue index). The text queue_state is the "
            "complete queue; use attached images only as lightweight visual "
            "context for object identity.\n"
        )
        if visual_count < queue_total:
            print(
                f"[AsyncVLM] Visual queue context: attached "
                f"{visual_count}/{queue_total} keyframe image(s)."
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
    # the midpoint time to a frame index in the ORIGINAL recording buffer so
    # that `seg_start + local_frame_idx` yields the correct abs_idx.
    num_seg_frames_orig = int((seg_end_sec - seg_start_sec) * _VIDEO_RECORD_FPS)

    new_elements = parsed.get("new_elements", [])
    if not isinstance(new_elements, list):
        new_elements = []
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

    if command_name == "Append":
        raw_elements: List[_RawAppendElement] = []
        for elem in new_elements:
            if not isinstance(elem, dict):
                continue
            interval = elem.get("keyframe_interval")
            instruction = (elem.get("subtask_instruction")
                           or elem.get("instruction") or "").strip()
            description = (elem.get("keyframe_description") or "").strip()
            if not instruction or not isinstance(interval, dict):
                print("[AsyncVLM] Append: skipping malformed new_element "
                      f"{elem}")
                continue
            local_frame_idx = _mid_frame_from_interval(
                interval, num_seg_frames_orig, _VIDEO_RECORD_FPS)
            raw_elements.append(_RawAppendElement(
                local_frame_idx=local_frame_idx,
                instruction=instruction,
                keyframe_description=description,
            ))

        raw_elements.sort(key=lambda elem: elem.local_frame_idx)
        if not raw_elements:
            print("[AsyncVLM] Append: no valid new_elements — NoOp")
            return CmdNoOp(), parsed, latency
        if len(raw_elements) > 1:
            print(f"[AsyncVLM] Append: parsed {len(raw_elements)} keyframes "
                  "from one clip")
            return _RawAppendMany(elements=raw_elements), parsed, latency

        elem = raw_elements[0]
        return _RawCommand(
            name="Append",
            target_idx=None,
            local_frame_idx=elem.local_frame_idx,
            instruction=elem.instruction,
            keyframe_description=elem.keyframe_description,
        ), parsed, latency

    # Replace/Preempt legacy path: use first new_element for the keyframe.
    elem = new_elements[0]
    if not isinstance(elem, dict):
        print(f"[AsyncVLM] {command_name}: malformed new_element — NoOp")
        return CmdNoOp(), parsed, latency
    interval = elem.get("keyframe_interval", {})
    instruction = (elem.get("subtask_instruction")
                   or elem.get("instruction") or "").strip()
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
      3. Run Whisper pre-check; discard the segment if there is no useful voice.
      4. Build a voice-anchored VLM window and wait for future context if needed.
      5. Fire a per-call sub-thread for the Gemini API call (non-blocking).
      6. Immediately go back to step 1 — timer resets right now, new frames
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
        self._segment_frames     = max(
            1, int(round(self._interval_sec * _VIDEO_RECORD_FPS))
        )
        self._max_vlm_frames     = max(
            self._segment_frames,
            int(round(VLM_INTERVAL_SEC_MAX * _VIDEO_RECORD_FPS)),
        )
        self._temp_dir           = temp_dir
        self._stop_event         = threading.Event()
        self._queries_enabled    = threading.Event()
        self._queries_enabled.set()
        self._call_count         = 0
        self._episode_start: float = 0.0   # set in start()
        self._active_vlm_windows: dict[int, Tuple[int, int]] = {}
        self._active_vlm_lock = threading.Lock()
        self._skip_initial_wait = False
        self._call_count_lock = threading.Lock()
        self._voice_precheck_lock = threading.Lock()
        self._active_voice_prechecks = 0

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
        if not self._queries_enabled.is_set():
            print("[AsyncVLM] start() ignored because this episode has already ended.")
            return
        self._episode_start = time.monotonic()
        self._thread.start()

    def skip_initial_wait_once(self):
        """Make the next worker iteration consume an already-primed segment."""
        self._skip_initial_wait = True

    def _vlm_preroll_retention_frames(self) -> int:
        return max(0, int(round(
            self._segment_frames * ASYNC_VLM_DISCARD_INITIAL_RATIO)))

    def _register_vlm_window(self, call_id: int, start: int, end: int):
        with self._active_vlm_lock:
            self._active_vlm_windows[int(call_id)] = (int(start), int(end))

    def _unregister_vlm_window(self, call_id: int):
        with self._active_vlm_lock:
            self._active_vlm_windows.pop(int(call_id), None)

    def _oldest_active_vlm_start(self) -> Optional[int]:
        with self._active_vlm_lock:
            if not self._active_vlm_windows:
                return None
            return min(start for start, _ in self._active_vlm_windows.values())

    def _prune_consumed_buffer(self, reason: str = ""):
        dropped = self._buf.prune_processed(
            keep_frames=self._vlm_preroll_retention_frames(),
            protect_from=self._oldest_active_vlm_start(),
        )
        if dropped:
            suffix = f" ({reason})" if reason else ""
            print(f"[AsyncVLM] Pruned {dropped} processed frame(s)"
                  f" from VideoBuffer{suffix}.")

    def _flush_realtime_backlog(self, reason: str = ""):
        max_backlog = max(0, int(round(
            self._segment_frames * ASYNC_VLM_MAX_BACKLOG_SEGMENTS)))
        keep_latest = max(0, int(round(
            self._segment_frames * ASYNC_VLM_KEEP_LATEST_SEGMENTS)))
        if max_backlog <= 0:
            return

        unprocessed = self._buf.unprocessed_count()
        if unprocessed <= max_backlog:
            return

        skipped = self._buf.skip_unprocessed_to_latest(keep_latest)
        if skipped <= 0:
            return

        suffix = f" ({reason})" if reason else ""
        print(
            f"[AsyncVLM] Realtime backlog{suffix}: skipped "
            f"{skipped} stale frame(s) ({skipped / _VIDEO_RECORD_FPS:.2f}s), "
            f"keeping newest {keep_latest / _VIDEO_RECORD_FPS:.2f}s for voice check."
        )
        self._prune_consumed_buffer("realtime backlog flush")

    def _wait_for_total_frames(self, end_frame: int, call_id: int,
                               reason: str) -> bool:
        """Wait until the buffer contains end_frame absolute frames."""
        while (not self._stop_event.is_set()
               and self._queries_enabled.is_set()
               and self._buf.total_frames() < end_frame):
            remaining = end_frame - self._buf.total_frames()
            if remaining % max(1, int(_VIDEO_RECORD_FPS)) == 0:
                print(f"[AsyncVLM] #{call_id} waiting for "
                      f"{remaining} future frame(s) to finish {reason}...")
            self._stop_event.wait(timeout=1.0 / _VIDEO_RECORD_FPS)
        return not self._stop_event.is_set() and self._queries_enabled.is_set()

    def _make_context_transcript_data(self, *transcripts: dict) -> dict:
        """Combine trigger/continuation transcripts as text context only."""
        texts = []
        stats = {}
        for transcript in transcripts:
            if not transcript:
                continue
            text = re.sub(r"\s+", " ", transcript.get("text", "")).strip()
            if text and text not in texts:
                texts.append(text)
            if not stats and transcript.get("audio_stats"):
                stats = transcript.get("audio_stats", {})
        return {
            "text": " ".join(texts).strip(),
            "words": [],  # not aligned to the current VLM video
            "audio_stats": stats,
        }

    def _next_call_id(self) -> int:
        with self._call_count_lock:
            self._call_count += 1
            return self._call_count

    def _voice_precheck_needed(self) -> bool:
        return bool(
            WHISPER_ENABLED
            and (
                ASYNC_VLM_REQUIRE_USEFUL_VOICE
                or ASYNC_VLM_USE_VOICE_FOR_WINDOWING
                or STOP_ROBOT_WHILE_RECV
            )
        )

    def _try_start_voice_precheck(self,
                                  trigger_t: float,
                                  seg_start: int,
                                  seg_end: int,
                                  seg_n_frames: int,
                                  audio_records: List[dict]) -> bool:
        if not self._voice_precheck_needed():
            self._handle_voice_prechecked_segment(
                trigger_t=trigger_t,
                seg_start=seg_start,
                seg_end=seg_end,
                seg_n_frames=seg_n_frames,
                audio_records=audio_records,
                precheck_transcript={},
                voice_start_sec=None,
            )
            return True

        protect_key = -max(1, int(seg_start) + 1)
        self._register_vlm_window(protect_key, seg_start, seg_end)

        if not ASYNC_VOICE_PRECHECK_THREAD_ENABLED:
            self._voice_precheck_job(
                trigger_t, seg_start, seg_end, seg_n_frames, audio_records,
                owns_slot=False,
                protect_key=protect_key,
            )
            return True

        with self._voice_precheck_lock:
            if self._active_voice_prechecks >= ASYNC_VOICE_PRECHECK_MAX_ACTIVE:
                print(
                    "[AsyncVLM] Voice precheck busy "
                    f"({self._active_voice_prechecks}/"
                    f"{ASYNC_VOICE_PRECHECK_MAX_ACTIVE}) — dropping stale "
                    f"segment {seg_start}–{seg_end} before Whisper."
                )
                self._unregister_vlm_window(protect_key)
                self._prune_consumed_buffer("voice precheck busy")
                return False
            self._active_voice_prechecks += 1

        threading.Thread(
            target=self._voice_precheck_job,
            args=(trigger_t, seg_start, seg_end, seg_n_frames, audio_records),
            kwargs={"owns_slot": True, "protect_key": protect_key},
            daemon=True,
            name=f"VoicePrecheck-{seg_start}-{seg_end}",
        ).start()
        return True

    def _voice_precheck_job(self,
                            trigger_t: float,
                            seg_start: int,
                            seg_end: int,
                            seg_n_frames: int,
                            audio_records: List[dict],
                            owns_slot: bool = True,
                            protect_key: Optional[int] = None):
        try:
            if not self._queries_enabled.is_set():
                print("[AsyncVLM] Episode ended before voice precheck — skip segment.")
                self._prune_consumed_buffer("episode ended before voice precheck")
                return

            precheck_transcript: dict = {}
            voice_start_sec = None
            if self._voice_precheck_needed():
                precheck_transcript = _transcribe_audio_records_for_whisper(
                    audio_records, context=f"precheck@{seg_start}-{seg_end}")
                voice_start_sec = _first_useful_voice_start_sec(precheck_transcript)

            self._handle_voice_prechecked_segment(
                trigger_t=trigger_t,
                seg_start=seg_start,
                seg_end=seg_end,
                seg_n_frames=seg_n_frames,
                audio_records=audio_records,
                precheck_transcript=precheck_transcript,
                voice_start_sec=voice_start_sec,
            )
        finally:
            if protect_key is not None:
                self._unregister_vlm_window(protect_key)
            if owns_slot:
                with self._voice_precheck_lock:
                    self._active_voice_prechecks = max(
                        0, self._active_voice_prechecks - 1)

    def _handle_voice_prechecked_segment(self,
                                         trigger_t: float,
                                         seg_start: int,
                                         seg_end: int,
                                         seg_n_frames: int,
                                         audio_records: List[dict],
                                         precheck_transcript: dict,
                                         voice_start_sec: Optional[float]):
        seg_start_sec = seg_start / _VIDEO_RECORD_FPS
        seg_end_sec = seg_end / _VIDEO_RECORD_FPS
        voice_detected = voice_start_sec is not None

        if ASYNC_VLM_REQUIRE_USEFUL_VOICE and not voice_detected:
            transcript_text = (precheck_transcript.get("text", "") or "").strip()
            transcript_suffix = (
                f" transcript=\"{transcript_text[:80]}\"" if transcript_text else ""
            )
            print(
                f"[AsyncVLM] No useful voice in frames {seg_start}–{seg_end} "
                f"({seg_start_sec:.1f}s–{seg_end_sec:.1f}s){transcript_suffix} "
                f"— flushed before async VLM."
            )
            self._prune_consumed_buffer("discarded no-voice segment")
            return

        call_id = self._next_call_id()

        print(f"[AsyncVLM] Call #{call_id} | t={trigger_t:.2f}s | "
              f"obs frames {seg_start}–{seg_end} ({seg_n_frames} frames) | "
              f"time {seg_start_sec:.1f}s–{seg_end_sec:.1f}s | "
              f"video_source={_gemini_video_source_label()}")

        if voice_detected:
            print(f"[AsyncVLM] #{call_id} Useful voice detected: "
                  f"\"{precheck_transcript.get('text', '')[:80]}\" "
                  f"at +{voice_start_sec:.2f}s")
            if STOP_ROBOT_WHILE_RECV:
                print(f"[AsyncVLM] #{call_id} Pausing robot until VLM completes.")
                self.robot_pause_event.set()

        if voice_start_sec is None or not ASYNC_VLM_USE_VOICE_FOR_WINDOWING:
            vlm_start, vlm_end = seg_start, seg_end
            voice_frame = seg_start
            pre_frames = 0
            vlm_transcript_data = None
        else:
            vlm_start, vlm_end, voice_frame, pre_frames = (
                _voice_anchored_vlm_window(
                    seg_start,
                    voice_start_sec,
                    self._segment_frames,
                    ASYNC_VLM_DISCARD_INITIAL_RATIO,
                )
            )

            print(
                f"[AsyncVLM] #{call_id} Voice-anchored VLM video keeps "
                f"{pre_frames / _VIDEO_RECORD_FPS:.2f}s before command cue; "
                f"window starts at {vlm_start / _VIDEO_RECORD_FPS:.2f}s."
            )
            if not self._wait_for_total_frames(
                    vlm_end, call_id, "voice-anchored VLM window"):
                print(f"[AsyncVLM] #{call_id} Episode ended before voice-anchored VLM window completed.")
                return

            vlm_end, _continuation_transcripts = (
                self._maybe_extend_fresh_vlm_window(
                    call_id, vlm_start, vlm_end))
            vlm_transcript_data = None

        vlm_start_sec = vlm_start / _VIDEO_RECORD_FPS
        vlm_end_sec = vlm_end / _VIDEO_RECORD_FPS
        print(
            f"[AsyncVLM] #{call_id} VLM window: "
            f"voice={voice_frame / _VIDEO_RECORD_FPS:.2f}s, "
            f"included_pre_voice={pre_frames / _VIDEO_RECORD_FPS:.2f}s, "
            f"window={vlm_start_sec:.2f}s–{vlm_end_sec:.2f}s"
        )

        if not self._wait_for_total_frames(vlm_end, call_id, "VLM window"):
            print(f"[AsyncVLM] #{call_id} Episode ended before VLM window completed.")
            return

        vlm_aria, vlm_cropped, vlm_gaze = self._buf.get_range(vlm_start, vlm_end)
        vlm_audio = self._buf.get_audio_records(vlm_start, vlm_end)
        vlm_gemini_seg = _select_gemini_frames(vlm_aria, vlm_cropped)
        self._buf.mark_processed_up_to(vlm_end)

        if not vlm_gemini_seg:
            print(f"[AsyncVLM] #{call_id} no frames in VLM window — skip")
            self._prune_consumed_buffer(f"empty VLM window #{call_id}")
            return

        log_entry: dict = {
            "call_id":         call_id,
            "trigger_t":       trigger_t,
            "obs_window":      (seg_start, seg_end),
            "obs_time_sec":    (seg_start_sec, seg_end_sec),
            "vlm_window":      (vlm_start, vlm_end),
            "vlm_time_sec":    (vlm_start_sec, vlm_end_sec),
            "voice_frame":     voice_frame,
            "voice_time_sec":  voice_frame / _VIDEO_RECORD_FPS,
            "n_frames":        len(vlm_gemini_seg),
            "raw_n_frames":    seg_n_frames,
            "target_n_frames": self._segment_frames,
            "pre_voice_frames": pre_frames,
            "video_source":    _GEMINI_VIDEO_SOURCE,
            "step_at_trigger": self.current_step,
            "voice_detected":  voice_detected,
            "precheck_transcript": precheck_transcript.get("text", ""),
            "vlm_context_transcript": (
                vlm_transcript_data.get("text", "")
                if vlm_transcript_data else ""
            ),
            "gemini_start_t":  None,
            "gemini_end_t":    None,
            "command":         None,
        }
        self.call_log.append(log_entry)

        self._prune_consumed_buffer(f"dispatched VLM call #{call_id}")

        if not self._queries_enabled.is_set():
            log_entry["command"] = "SKIPPED_EPISODE_END"
            print(f"[AsyncVLM] #{call_id} Query suppressed because episode ended.")
            self._unregister_vlm_window(call_id)
            self._prune_consumed_buffer(f"suppressed VLM call #{call_id}")
            return

        if STOP_ROBOT_WHILE_RECV and voice_detected:
            self._run_gemini_call(
                call_id, vlm_gemini_seg, vlm_gaze, vlm_audio,
                vlm_start, vlm_end, log_entry,
                context_transcript_data=vlm_transcript_data,
                keyframe_seg=vlm_cropped)
            self.robot_pause_event.clear()
            print(f"[AsyncVLM] #{call_id} VLM done → resuming robot")
        else:
            threading.Thread(
                target=self._run_gemini_call,
                args=(call_id, vlm_gemini_seg, vlm_gaze, vlm_audio,
                      vlm_start, vlm_end, log_entry,
                      vlm_transcript_data),
                kwargs={"keyframe_seg": vlm_cropped},
                daemon=True,
                name=f"GeminiCall-{call_id}",
            ).start()

    def _maybe_extend_fresh_vlm_window(self, call_id: int, vlm_start: int,
                                       current_end: int) -> Tuple[int, List[dict]]:
        """Wait for future audio and extend only if speech continues.

        The base VLM clip must not be queued before we know whether the user
        continues speaking past the base interval. With VLM_INTERVAL_SEC=3 and
        VLM_INTERVAL_SEC_MAX=6, this probes the next 3 seconds first; if speech
        is present there, the VLM window becomes 6 seconds.
        """
        continuation_transcripts: List[dict] = []
        while current_end - vlm_start < self._max_vlm_frames:
            next_end = min(
                vlm_start + self._max_vlm_frames,
                current_end + self._segment_frames,
            )
            if next_end <= current_end:
                break

            print(
                f"[AsyncVLM] #{call_id} probing continued speech until "
                f"{(next_end - vlm_start) / _VIDEO_RECORD_FPS:.2f}s before "
                "queueing VLM input."
            )
            if not self._wait_for_total_frames(next_end, call_id,
                                               "continued-speech probe"):
                break

            continuation_audio = self._buf.get_audio_records(current_end, next_end)
            continuation_transcript = _transcribe_audio_records_for_whisper(
                continuation_audio, context=f"continue#{call_id}")
            if _first_useful_voice_start_sec(continuation_transcript) is None:
                print(
                    f"[AsyncVLM] #{call_id} no continued speech after "
                    f"{(current_end - vlm_start) / _VIDEO_RECORD_FPS:.2f}s; "
                    "queueing base VLM clip."
                )
                break

            continuation_transcripts.append(continuation_transcript)
            print(
                f"[AsyncVLM] #{call_id} speech continues in VLM clip — "
                f"extending VLM window to "
                f"{(next_end - vlm_start) / _VIDEO_RECORD_FPS:.2f}s "
                f"(max {self._max_vlm_frames / _VIDEO_RECORD_FPS:.2f}s)."
            )
            current_end = next_end

        return current_end, continuation_transcripts

    def stop(self):
        self._stop_event.set()
        self.robot_pause_event.clear()

    def end_episode(self):
        if not self._queries_enabled.is_set():
            self._stop_event.set()
            self.robot_pause_event.clear()
            self.preempt_event.clear()
            return

        print("[AsyncVLM] Episode end requested — suppressing new Gemini queries.")
        self._queries_enabled.clear()
        self._stop_event.set()
        self.robot_pause_event.clear()
        self.preempt_event.clear()

    def _elapsed(self) -> float:
        return time.monotonic() - self._episode_start

    # ------------------------------------------------------------------
    # Timer loop — never blocks on Gemini
    # ------------------------------------------------------------------

    def _run(self):
        print(f"[AsyncVLM] Worker started. video_source={_gemini_video_source_label()}")
        next_trigger_t = time.monotonic() + self._interval_sec
        while not self._stop_event.is_set() and self._queries_enabled.is_set():
            # ---- wait interval_sec of real wall-clock time ----
            if self._skip_initial_wait:
                self._skip_initial_wait = False
                print("[AsyncVLM] Initial wait skipped; consuming primed empty segment.")
                next_trigger_t = time.monotonic() + self._interval_sec
            else:
                wait_for = max(0.0, next_trigger_t - time.monotonic())
                self._stop_event.wait(timeout=wait_for)
                if self._stop_event.is_set() or not self._queries_enabled.is_set():
                    break
                now = time.monotonic()
                while next_trigger_t <= now:
                    next_trigger_t += self._interval_sec

            # Keep segment boundaries aligned to fixed episode-time windows
            # (for example 8.0 s = 240 frames at 30 FPS), even if the
            # background streamer is slightly late when the timer fires.
            self._flush_realtime_backlog("before segment reservation")
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
            if self._stop_event.is_set() or not self._queries_enabled.is_set():
                break

            trigger_t = self._elapsed()

            # ---- reserve one fixed-duration segment ----
            # In video-first mode, this goes straight to a fixed VLM window.
            # If voice gating/windowing is enabled, a separate voice precheck
            # thread decides whether/how to dispatch it.
            seg_start, seg_end, seg_n_frames, segment_synthetic_empty = (
                self._buf.reserve_segment(max_frames=self._segment_frames))

            if seg_n_frames <= 0:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames, skip")
                continue

            seg_start_sec = seg_start / _VIDEO_RECORD_FPS
            seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

            if segment_synthetic_empty:
                print(
                    f"[AsyncVLM] Synthetic empty segment frames {seg_start}–{seg_end} "
                    f"({seg_start_sec:.1f}s–{seg_end_sec:.1f}s) — flushed before async VLM."
                )
                self._prune_consumed_buffer("synthetic initial segment")
                continue

            # Collect audio records for this segment
            audio_records = self._buf.get_audio_records(seg_start, seg_end)

            if not self._queries_enabled.is_set():
                print("[AsyncVLM] Episode ended before async VLM dispatch — skip reserved segment.")
                self._prune_consumed_buffer("episode ended before async VLM dispatch")
                break

            # Voice precheck runs away from the scheduler thread only when it
            # is needed. In video-first mode, this immediately dispatches the
            # fixed video window; audio can still be transcribed later as
            # optional context inside the Gemini worker.
            self._try_start_voice_precheck(
                trigger_t=trigger_t,
                seg_start=seg_start,
                seg_end=seg_end,
                seg_n_frames=seg_n_frames,
                audio_records=audio_records,
            )

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
                         context_transcript_data: Optional[dict] = None,
                         keyframe_seg: Optional[List[np.ndarray]] = None):

        seg_start_sec = seg_start / _VIDEO_RECORD_FPS
        seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

        def _copied_keyframe_at(local_frame_idx: int) -> Optional[np.ndarray]:
            local_frame_idx = int(local_frame_idx)
            if keyframe_seg is not None and 0 <= local_frame_idx < len(keyframe_seg):
                return keyframe_seg[local_frame_idx].copy()

            # Fallback for debug modes that keep the VLM window in VideoBuffer.
            abs_idx = seg_start + local_frame_idx
            _, cropped_bgr = self._buf.get_frame(abs_idx)
            return cropped_bgr

        if not self._queries_enabled.is_set():
            log_entry["command"] = "SKIPPED_EPISODE_END"
            print(f"[AsyncVLM] #{call_id} Gemini call skipped because episode ended.")
            self._unregister_vlm_window(call_id)
            self._prune_consumed_buffer(f"skipped VLM call #{call_id}")
            return

        print(f"[AsyncVLM] #{call_id} → Preparing VLM input "
              f"(video_source={_gemini_video_source_label()})...")

        try:
            # Two independent transcript paths:
            #   1. context_transcript_section: prompt-only text from the
            #      discarded trigger chunk, never drawn on frames.
            #   2. transcript_section/transcript_data: fresh audio from this
            #      VLM clip, used for real-time word-level captions.
            context_transcript_section = _build_context_transcript_section(
                context_transcript_data)
            # Prepare video: optional gaze crop → subsample → encode → whisper → audio
            video_bytes, transcript_section, transcript_data = _prepare_vlm_video(
                gemini_seg, gaze_seg, audio_records, seg_start_sec,
                frames_are_gaze_cropped=_gemini_uses_cropped_video(),
                whisper_context=f"vlm#{call_id}",
                use_whisper=ASYNC_VLM_USE_AUDIO_TRANSCRIPT_CONTEXT,
            )
            transcript_section = context_transcript_section + transcript_section

            context_text = re.sub(
                r"\s+", " ", (context_transcript_data or {}).get("text", "")
            ).strip()
            if (
                ASYNC_VLM_REQUIRE_USEFUL_VOICE
                and WHISPER_ENABLED
                and not _is_useful_whisper_transcript(transcript_data)
                and not context_text
            ):
                log_entry["command"] = "SKIPPED_NO_VLM_TRANSCRIPT"
                log_entry["gemini_end_t"] = self._elapsed()
                print(
                    f"[AsyncVLM] #{call_id} prepared VLM clip has no useful "
                    "Whisper transcript — skip Gemini request."
                )
                return

            # Save the voice-anchored VLM window with captions for debugging.
            if SAVE_ASYNC_CAPTIONED_SEGMENTS and transcript_data.get("text"):
                seg_video_path = os.path.join(
                    self._temp_dir,
                    f"async_seg_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4",
                )
                captioned_seg = _overlay_captions_on_frames(
                    gemini_seg,
                    transcript_data.get("words", []),
                    transcript_data.get("text", ""),
                    seg_start_sec,
                    _VLM_VIDEO_SAVE_FPS,
                )
                _write_video_with_audio(
                    captioned_seg,
                    audio_records,
                    seg_video_path,
                    fps=_VLM_VIDEO_SAVE_FPS,
                )
                print(f"[AsyncVLM] #{call_id} Captioned async segment saved → {seg_video_path}")

            if not self._queries_enabled.is_set():
                log_entry["command"] = "SKIPPED_EPISODE_END"
                print(f"[AsyncVLM] #{call_id} Episode ended before Gemini request — skip.")
                return

            # Save the VLM input video (with whisper captions) to temp dir
            if SAVE_ASYNC_VLM_INPUT_VIDEOS:
                vlm_input_path = os.path.join(
                    self._temp_dir,
                    f"vlm_input_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4")
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

            if not self._queries_enabled.is_set():
                log_entry["command"] = "IGNORED_EPISODE_END"
                print(f"[AsyncVLM] #{call_id} Gemini response ignored because episode ended.")
                return

            # ---- resolve raw commands → typed command(s) with ndarray ----
            if isinstance(raw, _RawAppendMany):
                keyframes = []
                instructions = []
                for elem in raw.elements:
                    abs_idx        = seg_start + elem.local_frame_idx
                    cropped_bgr = _copied_keyframe_at(elem.local_frame_idx)
                    if cropped_bgr is None:
                        print(f"[AsyncVLM] #{call_id} frame {abs_idx} "
                              "not in buffer — skipping append element")
                        continue
                    keyframes.append(_draw_gaze_crosshair(cropped_bgr))
                    instructions.append(elem.instruction)

                if not keyframes:
                    raw = CmdNoOp()
                elif len(keyframes) == 1:
                    raw = CmdAppend(keyframe=keyframes[0],
                                    instruction=instructions[0])
                else:
                    raw = CmdAppendMany(keyframes=keyframes,
                                        instructions=instructions)

            if isinstance(raw, _RawCommand):
                abs_idx        = seg_start + raw.local_frame_idx
                cropped_bgr = _copied_keyframe_at(raw.local_frame_idx)
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
        finally:
            self._unregister_vlm_window(call_id)
            self._prune_consumed_buffer(f"resolved VLM call #{call_id}")


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


def _build_eval_preview(obs, keyframe_img: Optional[np.ndarray],
                        queue: KeyframeQueue, current_instruction: str = ""):
    human_view = obs["images"]["aria_rgb"].copy()
    if keyframe_img is None:
        kf_display = np.zeros_like(human_view)
        cv2.putText(kf_display, "WAITING FOR ASYNC QUEUE",
                    (20, max(30, human_view.shape[0] // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        kf_display = cv2.resize(keyframe_img,
                                (human_view.shape[1], human_view.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    preview = np.hstack((human_view, kf_display))
    queue_label = f"LIVE | KF [{queue.current_idx + 1}/{queue.total}]"
    if queue.total == 0 or keyframe_img is None:
        queue_label = "LIVE | KF [waiting for async queue]"
    cv2.putText(preview,
                queue_label,
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "n=next kf  e=end episode  q=quit",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if current_instruction:
        cv2.putText(preview, f"subtask: {current_instruction[:80]}",
                    (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


def _resize_for_display(img_bgr: np.ndarray,
                        max_w: int = 1180,
                        max_h: int = 680) -> np.ndarray:
    """Shrink large preview frames for SSH/XQuartz-style remote windows."""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return img_bgr

    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return img_bgr

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _make_queue_thumbnail(img: Optional[np.ndarray], thumb_size: int) -> np.ndarray:
    thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
    if img is None:
        cv2.putText(thumb, "N/A", (10, thumb_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        return thumb

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported queue image shape: {arr.shape}")

    return cv2.resize(arr, (thumb_size, thumb_size), interpolation=cv2.INTER_LINEAR)


def _render_queue_status_frame(queue: KeyframeQueue, note: str = "") -> np.ndarray:
    snapshot = queue.snapshot()
    current_idx = queue.current_idx
    total = queue.total
    queue_done = queue.is_done

    item_count = max(1, len(snapshot))
    width = 1700
    max_height = 1400
    header_h = 140 if note else 108
    footer_h = 28
    row_h = max(96, min(180, (max_height - header_h - footer_h) // item_count))
    thumb_size = max(72, min(144, row_h - 22))
    height = max(300, header_h + row_h * item_count + footer_h)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)

    cv2.putText(canvas, "Current Queue Status",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    if total == 0:
        summary = "Queue is empty | waiting for async VLM append"
    elif queue_done:
        summary = f"Queue exhausted | total={total}"
    else:
        summary = f"Executing index {current_idx} / {max(total - 1, 0)} | total={total}"
    cv2.putText(canvas, summary,
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (180, 220, 255), 2, cv2.LINE_AA)

    if note:
        note_lines = textwrap.wrap(note, width=110)[:2]
        for line_idx, line in enumerate(note_lines):
            cv2.putText(canvas, line,
                        (20, 104 + line_idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (0, 255, 255), 2, cv2.LINE_AA)

    if not snapshot:
        cv2.rectangle(canvas, (20, header_h + 10), (width - 20, height - 20),
                      (60, 60, 60), 2)
        cv2.putText(canvas, "No queued keyframes yet.",
                    (40, header_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "The robot will hold pose until async VLM appends a task.",
                    (40, header_h + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Controls: e=end episode, q=abort, n=manual next once a queue exists",
                    (40, header_h + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return canvas

    for idx, item in enumerate(snapshot):
        y1 = header_h + idx * row_h
        y2 = min(height - footer_h, y1 + row_h - 6)

        if queue_done:
            state = "DONE"
            row_color = (42, 42, 42)
            accent = (110, 110, 110)
        elif idx < current_idx:
            state = "DONE"
            row_color = (38, 38, 38)
            accent = (120, 120, 120)
        elif idx == current_idx:
            state = "CURRENT"
            row_color = (26, 64, 26)
            accent = (0, 255, 0)
        else:
            state = "PENDING"
            row_color = (32, 32, 48)
            accent = (255, 180, 90)

        cv2.rectangle(canvas, (18, y1), (width - 18, y2), row_color, -1)
        cv2.rectangle(canvas, (18, y1), (width - 18, y2), accent, 2)

        thumb_x = 28
        thumb_y = y1 + max(6, (row_h - thumb_size) // 2)
        img = item.get("keyframe_image")
        try:
            thumb = _make_queue_thumbnail(img, thumb_size)
        except Exception as exc:
            thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
            cv2.putText(thumb, "IMG ERR", (8, thumb_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            print(f"[QueueWindow] thumbnail render failed at idx={idx}: {exc}")
        canvas[thumb_y:thumb_y + thumb_size, thumb_x:thumb_x + thumb_size] = thumb
        cv2.rectangle(canvas,
                      (thumb_x - 1, thumb_y - 1),
                      (thumb_x + thumb_size + 1, thumb_y + thumb_size + 1),
                      (220, 220, 220), 1)

        text_x = thumb_x + thumb_size + 22
        label = f"[{idx}] {state}"
        cv2.putText(canvas, label,
                    (text_x, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    accent, 2, cv2.LINE_AA)

        instruction = str(item.get("instruction") or "")
        wrapped = textwrap.wrap(instruction, width=95)[:3]
        if not wrapped:
            wrapped = ["(no instruction)"]
        for line_idx, line in enumerate(wrapped):
            cv2.putText(canvas, line,
                        (text_x, y1 + 70 + line_idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def _show_queue_status_window(queue: KeyframeQueue, window_name: str,
                              note: str = ""):
    try:
        frame = _render_queue_status_frame(queue, note=note)
    except Exception as exc:
        print(f"[QueueWindow] render error: {exc}")
        frame = np.zeros((520, 1400, 3), dtype=np.uint8)
        cv2.putText(frame, "Queue visualization error", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, str(exc)[:120], (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)


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
                      ) -> Tuple[KeyframeQueue, bool]:
    """
    Stream live aria | top frames and choose how to start the episode.

    Controls:
      - First ``p``: start recording the clip for initial queue init.
      - Second ``p``: query Gemini with that recorded clip and build the
        initial KeyframeQueue.
      - ``s``: skip initial queue init and start with an empty queue. The
        robot will hold pose while async VLM updates populate the queue.

    If *sim_recorder* is provided, writes the init phase (Initializing...)
    frames and then the queue-initialized frames into the simulation video.

    Returns ``(queue, start_async_immediately)``.
    """
    window_name       = "Keyframe Selector"
    os.makedirs(_TEMP_DIR, exist_ok=True)

    video_path         = os.path.join(_TEMP_DIR, "video.mp4")
    cropped_video_path = os.path.join(_TEMP_DIR, "cropped_video.mp4")
    gaze_path          = os.path.join(_TEMP_DIR, "gaze.json")

    aria_frames         = []
    aria_cropped_frames = []
    gaze_records        = []
    audio_records       = []
    preview_frames      = []
    capture_timer       = Timer(_VIDEO_RECORD_FPS)
    recording_for_init  = False

    print("Keyframe selection:")
    print("  p : start recording initial VLM clip")
    print("  p : again → initialize queue from that clip")
    print("  s : skip initial queue, start episode with async-only updates")
    print("  q : quit")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation(
                    mode='human_view_streamed_and_keyframe_cropped'))
            status_text = "LIVE: p=start init capture  s=start empty  q=quit"
            status_color = (0, 255, 0)
            if recording_for_init:
                status_text = (
                    f"RECORDING INIT CLIP ({len(aria_frames)} frames): "
                    "p=initialize  s=skip-to-empty  q=quit"
                )
                status_color = (0, 255, 255)
            live_frame = _build_preview_frame(
                live_obs, status_text, status_color=status_color)

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

            if sim_recorder is not None:
                preview_frames.append(aria_bgr.copy())

            if recording_for_init:
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

            if STREAM_SELECTOR_TO_VLM_BUFFER:
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

            if key in (ord('p'), ord('P')) and not recording_for_init:
                recording_for_init = True
                aria_frames.clear()
                aria_cropped_frames.clear()
                gaze_records.clear()
                audio_records.clear()
                print("[keyframe] Recording started. Press 'p' again to initialize the queue.")
                continue

            if key in (ord('p'), ord('P')) and recording_for_init and aria_frames:
                total = len(aria_frames)
                duration_sec = total / _VIDEO_RECORD_FPS
                print(f"[keyframe] Captured {total} frames @ {_VIDEO_RECORD_FPS} FPS "
                      f"({duration_sec:.2f}s)")

                if SAVE_INITIAL_CAPTURE_LOGS:
                    _write_video_with_audio(
                        aria_frames, audio_records, video_path, fps=_VIDEO_RECORD_FPS)
                    _write_video(aria_cropped_frames, cropped_video_path)
                    with open(gaze_path, "w") as f:
                        json.dump({"frames": gaze_records}, f, indent=2)

                gemini_frames = _select_gemini_frames(
                    aria_frames, aria_cropped_frames)

                # Prepare VLM input: optional gaze crop → subsample → whisper → audio
                print("[keyframe] Preparing VLM input "
                      f"({_gemini_video_source_label()} + audio)...")
                vlm_bytes, transcript_section, _ = _prepare_vlm_video(
                    gemini_frames, gaze_records, audio_records,
                    clip_start_sec=0.0,
                    frames_are_gaze_cropped=_gemini_uses_cropped_video(),
                )
                if SAVE_INITIAL_VLM_INPUT_VIDEO:
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
                    )
                except Exception as exc:
                    print(f"[keyframe] Gemini failed after retries: "
                          f"{_short_error_reason(exc)}")
                    print("[keyframe] Press 'p' to retry, or 'q' to quit.")
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
                vla_instrs     = []
                saved_items    = []

                for i1, elem in enumerate(action_queue, start=1):
                    interval = elem.get("keyframe_interval", {})
                    instruction = elem.get("subtask_instruction", language_instruction)
                    description = elem.get("keyframe_description", "")
                    vla_instruction = _resolve_vla_prompt(
                        instruction, language_instruction)

                    # Extract mid-frame from the original (full-fps) frames
                    frame_idx = _mid_frame_from_interval(
                        interval, total, _VIDEO_RECORD_FPS)
                    frame_idx = max(0, min(frame_idx, total - 1))

                    cropped_kf = aria_cropped_frames[frame_idx].copy()
                    cropped_kf = _draw_gaze_crosshair(cropped_kf)

                    cropped_path = None
                    if SAVE_KEYFRAME_IMAGES:
                        cropped_path = os.path.join(
                            _TEMP_DIR, f"cropped_keyframe_{i1:02d}.png")
                        plt.imsave(cropped_path,
                                   cv2.cvtColor(cropped_kf, cv2.COLOR_BGR2RGB))
                    path_msg = f" → {cropped_path}" if cropped_path else ""
                    print(f"[keyframe] #{i1} (frame {frame_idx}) "
                          f"{instruction}{path_msg}")

                    keyframe_imgs.append(cropped_kf)
                    subtask_instrs.append(instruction)
                    vla_instrs.append(vla_instruction)
                    saved_items.append({
                        "order":                 i1,
                        "frame_idx":             int(frame_idx),
                        "cropped_keyframe_path": cropped_path,
                        "instruction":           instruction,
                        "vla_instruction":       vla_instruction,
                        "description":           description,
                    })

                keyframe_meta = {
                    "instruction":          language_instruction,
                    "num_frames":           total,
                    "video_fps":            _VIDEO_RECORD_FPS,
                    "vlm_fps":              VLM_FPS,
                    "vla_prompt_mode":      VLA_PROMPT_MODE,
                    "vla_fixed_prompt":     VLA_FIXED_PROMPT,
                    "subtask_instructions": subtask_instrs,
                    "vla_instructions":     vla_instrs,
                    "saved_keyframes":      saved_items,
                    "gemini_response":      parsed,
                    "latency":              latency,
                }
                _write_keyframe_metadata(keyframe_meta, log_prefix="keyframe")

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

                return result_queue, False

            elif key in (ord('s'), ord('S')):
                print("[keyframe] Skipping initial queue. Async VLM will populate it during rollout.")
                empty_queue = KeyframeQueue([], [])
                if sim_recorder is not None and preview_frames:
                    sim_recorder.record_init_phase(
                        preview_frames, queue=None, fps=_VIDEO_RECORD_FPS)
                return empty_queue, True

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
         start_async_immediately: bool = False,
         chunksize: int = 8):
    """
    Robot execution loop.

    The async VLM worker runs concurrently and may modify `queue` at any
    time.  The loop reacts to four triggers:

      1. vlm_worker.preempt_event  — robot reset + new keyframe from VLM
      2. completion_prob >= 0.8    — auto-advance to next keyframe
      3. 'n' key                   — manual advance
      4. 'e' key                   — end episode without waiting for queue
      5. 'q' / ESC                 — abort episode
    """
    pending_auto_advance = False
    top_view_frames_list = []
    comp_prob_log        = []
    global_step          = 0
    subtask_step         = 0
    timer                = Timer(env.control_hz)
    window_name          = "Eval  |  n=next kf  e=end episode  q=quit"
    queue_window_name    = "Queue Status"
    rollout_video_path   = os.path.join(_TEMP_DIR, "rollout.mp4")
    keyframe_log: List[Tuple[int, np.ndarray]] = []
    async_collection_started = False
    kf = queue.current_keyframe
    if SAVE_ROLLOUT_VIDEO and kf is not None:
        keyframe_log.append((0, kf.copy()))

    if REALTIME_DISPLAY_ENABLED:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1180, 680)
        if REALTIME_QUEUE_CANVAS_ENABLED:
            cv2.namedWindow(queue_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(queue_window_name, 1800, 1320)

    # ------------------------------------------------------------------
    # Plan-A fix #1: continuously feed video_buffer at _VIDEO_RECORD_FPS
    # from a background thread, independent of env.step(). The in-loop
    # video_buffer.append() calls below are intentionally removed; the
    # streamer is the sole writer so periodic VLM segments stay at 30 Hz.
    # ------------------------------------------------------------------
    _bg_streamer = BackgroundObservationStreamer(
        env, video_buffer, target_fps=_VIDEO_RECORD_FPS,
    )

    def _prepend_empty_initial_vlm_segment():
        if not ASYNC_VLM_PREPEND_EMPTY_INITIAL_SEGMENT:
            return
        try:
            obs = _extract_latest_observation(env.get_observation(
                mode='human_view_streamed_and_keyframe_cropped',
                audio_drain=False,
            ))
            images = obs.get("images", {})
            aria_img = images.get("aria_rgb")
            cropped_img = images.get("aria_cropped_rgb")
            if aria_img is None or cropped_img is None:
                print("[Eval] Empty initial VLM segment skipped: missing Aria frames.")
                return
            audio_obs = obs.get("audio", {}) or {}
            sample_rate_hz = int(audio_obs.get("sample_rate_hz", 0) or 0)
            segment_frames = max(
                1, int(round(VLM_INTERVAL_SEC * _VIDEO_RECORD_FPS)))
            start, end = video_buffer.append_empty_segment(
                num_frames=segment_frames,
                aria_shape=aria_img.shape,
                cropped_shape=cropped_img.shape,
                audio_sample_rate_hz=sample_rate_hz,
            )
            vlm_worker.skip_initial_wait_once()
            print(
                f"[Eval] Prepended empty initial VLM segment: "
                f"frames {start}-{end} "
                f"({segment_frames / _VIDEO_RECORD_FPS:.2f}s)."
            )
        except Exception as exc:
            print(f"[Eval] Failed to prepend empty initial VLM segment: {exc}")

    def _ensure_async_collection_started(reason: str):
        nonlocal async_collection_started
        if async_collection_started:
            return
        video_buffer.skip_to_current()
        dropped = video_buffer.prune_processed(keep_frames=0)
        if dropped:
            print(f"[Eval] Cleared {dropped} pre-async frame(s) from VideoBuffer.")
        _prepend_empty_initial_vlm_segment()
        _bg_streamer.start()
        vlm_worker.start()
        async_collection_started = True
        print(f"[Eval] Async VLM collection started ({reason}).")

    last_display_t = 0.0
    display_period = (
        1.0 / REALTIME_DISPLAY_FPS
        if REALTIME_DISPLAY_FPS > 0 else float("inf")
    )
    hold_display_period = (
        1.0 / REALTIME_HOLD_DISPLAY_FPS
        if REALTIME_HOLD_DISPLAY_FPS > 0 else float("inf")
    )

    def _display_due(period: float = display_period) -> bool:
        nonlocal last_display_t
        if not REALTIME_DISPLAY_ENABLED:
            return False
        now = time.monotonic()
        if now - last_display_t < period:
            return False
        last_display_t = now
        return True

    def _show_eval_windows(preview_frame: np.ndarray, queue_note: str = ""):
        if not REALTIME_DISPLAY_ENABLED:
            return
        cv2.imshow(window_name, _resize_for_display(preview_frame))
        if REALTIME_QUEUE_CANVAS_ENABLED:
            _show_queue_status_window(queue, queue_window_name, note=queue_note)

    def _handle_eval_key(key: int):
        if key in (ord('e'), ord('E')):
            print("[Eval] Episode end requested by user.")
            vlm_worker.end_episode()
            raise EpisodeEnded
        if key in (ord('q'), ord('Q'), 27):
            print("[Eval] Episode abort requested by user.")
            vlm_worker.end_episode()
            raise KeyboardInterrupt

    def _wait_for_queue_ready():
        print("[Eval] Queue exhausted — holding pose. Waiting for async VLM to append "
              "a new subtask (press 'e' to end episode, 'q' / ESC to abort)...")
        last_log_t = time.monotonic()
        while queue.is_done:
            frame = None
            if _display_due(hold_display_period):
                try:
                    frame = _build_eval_preview(
                        _extract_latest_observation(env.get_observation(
                            mode='human_view_streamed_and_keyframe_cropped',
                            audio_drain=False)),
                        queue.current_keyframe, queue,
                        queue.current_instruction or "",
                    )
                except Exception as exc:
                    print(f"[Eval hold-pose] preview builder error: {exc}")

                if frame is not None:
                    overlay = frame.copy()
                    cv2.putText(overlay, "HOLDING POSE — waiting for async VLM queue",
                                (10, overlay.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)
                    _show_eval_windows(
                        overlay,
                        queue_note="Holding pose while async VLM listens for the next command.",
                    )
                elif REALTIME_QUEUE_CANVAS_ENABLED:
                    _show_queue_status_window(
                        queue,
                        queue_window_name,
                        note="Holding pose while async VLM listens for the next command.",
                    )

            key = cv2.waitKey(50) & 0xFF
            _handle_eval_key(key)

            now = time.monotonic()
            if now - last_log_t > 5.0:
                last_log_t = now
                print("[Eval hold-pose] Still waiting for new subtask... "
                      f"(queue.total={queue.total}, is_done={queue.is_done})")

        print("[Eval hold-pose] New subtask available — resuming.")

    def _finalize_eval(reset_robot: bool = False):
        _bg_streamer.stop()
        vlm_worker.end_episode()
        if SAVE_KEYFRAME_METADATA:
            try:
                _save_queue_snapshot_metadata(
                    queue,
                    language_instruction,
                    reason="episode_end",
                    event_log=vlm_worker.event_log,
                )
            except Exception as exc:
                print(f"[Eval] Failed to save keyframe metadata on episode end: {exc}")
        window_names = [window_name]
        if REALTIME_QUEUE_CANVAS_ENABLED:
            window_names.append(queue_window_name)
        for name in window_names:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
        if reset_robot:
            env.reset()
        if SAVE_ROLLOUT_VIDEO:
            _save_rollout_video(
                top_view_frames_list, comp_prob_log, rollout_video_path,
                vlm_event_log=vlm_worker.event_log,
                keyframe_log=keyframe_log)
        _safe_finalize_sim_recorder(sim_recorder, video_buffer)

    if start_async_immediately or queue.is_done:
        _ensure_async_collection_started("waiting for initial async queue")

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
                if SAVE_ROLLOUT_VIDEO and kf is not None:
                    keyframe_log.append((global_step, kf.copy()))
                print(f"[Eval] Preempt → KF [{queue.current_idx + 1}/{queue.total}]:"
                      f" '{queue.current_instruction}'")

            if queue.is_done:
                _ensure_async_collection_started("queue empty")
                # Plan-A fix #2: hold pose instead of resetting.
                # The background streamer keeps feeding the buffer so the
                # VLM worker can still observe the scene and Append new
                # subtasks. We block here until something new arrives.
                _wait_for_queue_ready()
                # New subtask appeared — refresh keyframe bookkeeping.
                subtask_step = 0
                kf = queue.current_keyframe
                if SAVE_ROLLOUT_VIDEO and kf is not None:
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
            vla_prompt    = _resolve_vla_prompt(
                current_instr, language_instruction)
            debug_aria_rgb = None

            if keyframe_img is not None:
                keyframe_img = cv2.cvtColor(keyframe_img, cv2.COLOR_BGR2RGB)
                if global_step == 0:
                    debug_aria_rgb = cv2.cvtColor(
                        images["aria_rgb"], cv2.COLOR_BGR2RGB)
            # temp
            #keyframe_img = _BLACK_TOP_KEYFRAME
            #
            observation = {
                "observation.images.top_view":                    images["top_camera_rgb"],
                # Human-view/gaze are intentionally not sent to the VLA policy.
                "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
                "observation.images.human_view_keyframe_cropped": keyframe_img,
                "prompt":                                         vla_prompt,
                "observation.state":                              concatenated_state,
            }

            if SAVE_POLICY_DEBUG_IMAGES and global_step == 0 and keyframe_img is not None:
                plt.imsave(os.path.join(_TEMP_DIR, "test_aria_keyframe.png"), keyframe_img)
                if debug_aria_rgb is not None:
                    plt.imsave(os.path.join(_TEMP_DIR, "test_aria_rgb.png"),
                               debug_aria_rgb)
            # ----------------------------------------------------------------
            # Infer
            # ----------------------------------------------------------------
            out             = client.infer(observation)
            action_chunk    = out["actions"][:chunksize, :]
            completion_prob = (float(out["completion_prob"])
                               if "completion_prob" in out else None)
            print("Completion prob:",
                  f"{completion_prob:.3f}" if completion_prob is not None else "N/A")
            if SAVE_ROLLOUT_VIDEO and completion_prob is not None:
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
                        global_step += 1
                        vlm_worker.current_step = global_step
                        display_now = _display_due(hold_display_period)
                        sim_now = (
                            sim_recorder is not None
                            and global_step % max(1, int(env.control_hz / SIM_FPS)) == 0
                        )
                        if display_now or sim_now:
                            _pause_obs = _extract_latest_observation(
                                env.get_observation(
                                    mode='human_view_streamed_and_keyframe_cropped',
                                    audio_drain=False))
                            if sim_now:
                                sim_t = global_step / env.control_hz
                                sim_recorder.record_frame(
                                    _pause_obs["images"]["aria_rgb"], queue, sim_t)
                            if display_now:
                                _pause_preview = _build_eval_preview(
                                    _pause_obs, keyframe_img, queue, current_instr)
                                cv2.putText(_pause_preview, "PAUSED (waiting for VLM)",
                                            (10, _pause_preview.shape[0] - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                                _show_eval_windows(
                                    _pause_preview,
                                    queue_note="Robot paused because a voiced async VLM update is in flight.",
                                )
                        key = cv2.waitKey(1) & 0xFF
                        _handle_eval_key(key)
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
                    if SAVE_ROLLOUT_VIDEO and keyframe_img is not None:
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
                _ensure_async_collection_started("first robot action")
                env.step(action)
                global_step  += 1
                subtask_step += 1
                vlm_worker.current_step = global_step

                # NOTE: video_buffer is now populated by
                # BackgroundObservationStreamer; no manual append here.

                display_now = _display_due(display_period)
                need_live_obs = (
                    SAVE_ROLLOUT_VIDEO
                    or sim_recorder is not None
                    or display_now
                )
                if need_live_obs:
                    live_obs = _extract_latest_observation(
                        env.get_observation(
                            mode='human_view_streamed_and_keyframe_cropped',
                            audio_drain=False,
                        ))
                    if SAVE_ROLLOUT_VIDEO:
                        top_view_frames_list.append(
                            live_obs["images"]["top_camera_rgb"].copy())

                    live_aria_bgr = live_obs["images"]["aria_rgb"]
                    if (sim_recorder is not None
                            and global_step % max(1, int(env.control_hz / SIM_FPS)) == 0):
                        sim_t = global_step / env.control_hz
                        sim_recorder.record_frame(live_aria_bgr, queue, sim_t)

                    if display_now:
                        preview = _build_eval_preview(
                            live_obs, keyframe_img, queue, current_instr)
                        _show_eval_windows(
                            preview,
                            queue_note="Live queue view. Current item is highlighted in green.",
                        )
                key = cv2.waitKey(1) & 0xFF
                _handle_eval_key(key)

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
                        if SAVE_ROLLOUT_VIDEO and keyframe_img is not None:
                            keyframe_log.append((global_step, keyframe_img.copy()))
                        print(f"  → Manual KF [{queue.current_idx + 1}/{queue.total}]:"
                              f" '{current_instr}'")
                    break

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
                    if SAVE_ROLLOUT_VIDEO and kf is not None:
                        keyframe_log.append((global_step, kf.copy()))
                    print(f"  [Auto] → KF [{queue.current_idx + 1}/{queue.total}]:"
                          f" '{queue.current_instruction}'")

        except EpisodeEnded:
            _finalize_eval(reset_robot=False)
            return
        except KeyboardInterrupt:
            _finalize_eval(reset_robot=True)
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    _finalize_eval(reset_robot=False)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Async VLM DTQ eval with optional center gaze marker on cropped VLM input.")
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
        "--vla-prompt-mode",
        choices=("fixed", "subtask", "global"),
        default=VLA_PROMPT_MODE,
        help=(
            "Text prompt mode for the VLA policy. fixed always sends "
            f"'{VLA_FIXED_PROMPT}'; subtask sends the current queue "
            "instruction; global sends the episode-level language instruction."
        ),
    )
    parser.add_argument(
        "--vla-fixed-prompt",
        default=VLA_FIXED_PROMPT,
        help="Prompt text used when --vla-prompt-mode=fixed.",
    )
    parser.add_argument(
        "--record-sim-video",
        action="store_true",
        help="Record keyframe_simulation.mp4. Disabled by default to avoid runtime overhead.",
    )
    parser.set_defaults(vlm_draw_gaze_marker=VLM_DRAW_GAZE_MARKER)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_cli_args()
    VLM_DRAW_GAZE_MARKER = args.vlm_draw_gaze_marker
    VLA_PROMPT_MODE = args.vla_prompt_mode
    VLA_FIXED_PROMPT = args.vla_fixed_prompt
    SAVE_SIMULATION_VIDEO = args.record_sim_video
    ARCHIVE_AUDIO_FOR_SIM_VIDEO = args.record_sim_video

    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env                  = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "I want this tool."
    horizon              = 10000
    GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    print("Starting. Instruction:", language_instruction)
    print("During eval: n=next keyframe  e=end episode  q=stop episode")
    print("VLM video source:", _gemini_video_source_label())
    print("VLM center gaze marker:", VLM_DRAW_GAZE_MARKER)
    print("VLA prompt mode:", VLA_PROMPT_MODE)
    if VLA_PROMPT_MODE == "fixed":
        print("VLA fixed prompt:", VLA_FIXED_PROMPT)
    print("Simulation video:", "enabled" if args.record_sim_video else "disabled")

    # Shared buffer — persists for the lifetime of the process.
    video_buffer = VideoBuffer(archive_audio=args.record_sim_video)

    # ---- Optional simulation video recorder ----
    sim_output_path = os.path.join(_TEMP_DIR, "keyframe_simulation.mp4")
    sim_recorder = (
        SimulationVideoRecorder(sim_output_path)
        if SAVE_SIMULATION_VIDEO else None
    )

    # ---- Initial keyframe extraction (interactive) ----
    queue, start_async_immediately = extract_keyframes(
        env, language_instruction, GEMINI_API_KEY,
        video_buffer) #sim_recorder=sim_recorder)

    # ---- Connect to policy server ----
    client = websocket_client_policy.WebsocketClientPolicy(
        host="localhost", port=8000)

    # ---- Create async VLM worker ----
    # Actual collection starts inside eval() right before the first env.step().
    video_buffer.skip_to_current()
    vlm_worker = AsyncVLMWorker(
        video_buffer=video_buffer,
        queue=queue,
        global_instruction=language_instruction,
        api_key=GEMINI_API_KEY,
        interval_sec=VLM_INTERVAL_SEC,
    )

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
                     queue, video_buffer, vlm_worker,
                     sim_recorder=sim_recorder,
                     start_async_immediately=start_async_immediately)
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
            video_buffer = VideoBuffer(archive_audio=args.record_sim_video)
            sim_recorder = (
                SimulationVideoRecorder(sim_output_path)
                if SAVE_SIMULATION_VIDEO else None
            )
            queue, start_async_immediately = extract_keyframes(
                env, language_instruction, GEMINI_API_KEY, video_buffer)
                #sim_recorder=sim_recorder)
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
