"""eval_openpi_ours_with_voice_kor.py

Korean-speech variant of eval_openpi_ours_with_voice.

Whisper STT runs in Korean (language="ko"). The Korean transcript is burned
onto the VLM input video as captions (rendered with a CJK-capable font via
PIL) and is also passed to Gemini in the prompt. Gemini interprets the Korean
speech, but its JSON output (action queue, instructions for the VLA policy)
remains in English so the downstream VLA control pipeline is unchanged.

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
import contextlib
import numpy as np
import cv2
import os
import json
import queue as queue_lib
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
from PIL import Image, ImageDraw, ImageFont

from edith.envs.piper_env_gaze import PiperRobotEnv
from edith.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from eval_vlm_baseline import (
    _get_active_caption,
    _ffmpeg_reencode_h264,
)


# ---------------------------------------------------------------------------
# Korean-capable caption rendering (overrides baseline _draw_caption)
# ---------------------------------------------------------------------------
# OpenCV's Hershey fonts cannot render Hangul glyphs, so we draw the caption
# with PIL using a CJK-capable TTF/OTF font and composite the result back
# into the OpenCV BGR frame in-place. Falls back to OpenCV putText if no
# CJK font is found on the system.

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]


def _find_cjk_font_path() -> Optional[str]:
    for p in _CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


_CJK_FONT_PATH = _find_cjk_font_path()
_CJK_FONT_CACHE: dict = {}


def _get_cjk_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    if _CJK_FONT_PATH is None:
        return None
    if size not in _CJK_FONT_CACHE:
        try:
            _CJK_FONT_CACHE[size] = ImageFont.truetype(_CJK_FONT_PATH, size)
        except Exception:
            return None
    return _CJK_FONT_CACHE[size]


def _draw_caption(frame: np.ndarray, text: str) -> None:
    """Korean-capable caption renderer (drop-in replacement for the baseline)."""
    if not text:
        return
    h, w = frame.shape[:2]

    font_size = max(14, int(h * 0.06))
    font = _get_cjk_font(font_size)

    if font is None:
        cv_font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, h / 512)
        thickness = max(1, int(h / 256))
        (tw, th), baseline = cv2.getTextSize(text, cv_font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = int(20 * h / 256)
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (tx - 6, ty - th - 6),
                      (tx + tw + 6, ty + baseline + 6),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (tx, ty),
                    cv_font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = max(4, font_size // 4)
    tx = (w - tw) // 2
    ty = max(pad + 2, int(20 * h / 256) - th)

    rect_x0 = max(0, tx - pad)
    rect_y0 = max(0, ty - pad + bbox[1])
    rect_x1 = min(w, tx + tw + pad)
    rect_y1 = min(h, ty + th + pad + bbox[1])

    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, 153))
    draw.text((tx - bbox[0], ty), text, font=font, fill=(255, 255, 255, 255))

    out_rgb = np.array(pil_img)
    frame[:, :, :] = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
from common.streamer import (
    BackgroundObservationStreamer,
    wait_for_queue_items,
)
from common.config import SERVER_IP


# ---------------------------------------------------------------------------
# Keyboard input helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _terminal_keypress_mode():
    """Allow single-key terminal input while CV2 windows are displayed."""
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


def _poll_keypress(delay_ms: int = 1) -> int:
    """Read one key from the OpenCV window or the SSH terminal."""
    key = cv2.waitKey(max(1, int(delay_ms)))
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
# Constants — edit these instead of CLI args
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS   = 15
_MAX_INTERVALS      = 10
_TEMP_DIR           = os.path.normpath("/home/alin/projects/piper_project/temp")
_BLACK_TOP_KEYFRAME = np.zeros((240, 424, 3), dtype=np.uint8)

# ── VLM pipeline settings ──
VLM_MODEL           = "gemini-3.1-flash-lite-preview"       # Gemini model to use
VLM_FPS             = 3                         # subsample to this FPS before sending to VLM
VLM_INTERVAL_SEC    = 8.0                       # seconds between async VLM calls
VLM_TARGET_RES      = (256, 256)                # resize video segments for VLM
VLM_THINKING_BUDGET = 2048                      # thinking token budget per call
VLM_REQUEST_TIMEOUT_SEC = 15.0                  # retry Gemini calls after this many seconds
VLM_MAX_ATTEMPTS    = 3                         # total attempts per VLM request
VLM_RETRY_BACKOFF_SEC = 0.5                     # brief pause between failed attempts
VLM_DEBUG_BANNER    = False                     # drawing per-frame debug text costs CPU

# Which Aria view to encode into the mp4 sent to Gemini.
# Options:
#   - "aria_rgb": full egocentric RGB frames
#   - "aria_cropped_rgb": gaze-centered cropped RGB frames
_GEMINI_VIDEO_SOURCE = "aria_cropped_rgb"

 # overlay center gaze marker on cropped VLM input
VLM_DRAW_GAZE_MARKER = True                 
STORE_FULL_ARIA_IN_VLM_BUFFER = (_GEMINI_VIDEO_SOURCE == "aria_rgb")

# ── Audio settings ──
WHISPER_ENABLED     = True                      # use Whisper STT → transcript in prompt
VOICED_VIDEO        = False                      # merge audio into mp4 → Gemini processes audio
ASYNC_VLM_REQUIRE_USEFUL_VOICE = False           # skip Gemini when Whisper sees no useful speech
USEFUL_VOICE_MIN_CHARS = 2

# ── Gaze crop settings ──
GAZE_CROP_ENABLED   = True
GAZE_CROP_SIZE      =  512                      # crop NxN around gaze centre
GAZE_CALIBRATION_X_OFFSET = 0                 # px to add to X (set interactively at startup)
GAZE_CALIBRATION_Y_OFFSET = 0                 # px to add to Y (set interactively at startup)
GAZE_CALIBRATION_RANGE_PX = 200               # ± slider range for interactive calibration


# ── Visual queue caching and optimization──
VISUAL_QUEUE        = True                      # attach keyframe images in VLM update calls
VISUAL_QUEUE_MAX_IMAGES = 10                     # bound async prompt growth after repeated Appends
VISUAL_QUEUE_IMAGE_SIZE = 128                   # small visual context is enough for queue identity
VISUAL_QUEUE_JPEG_QUALITY = 65                  # JPEG is much cheaper than PNG for prompt thumbnails
VISUAL_QUEUE_THUMB_CACHE_MAX = 128              # avoid re-encoding unchanged queue images every call

# ── Runtime Saving Handling ──
SAVE_ASYNC_RAW_SEGMENTS = False                 # async_seg_*.mp4 before captioning
SAVE_ASYNC_CAPTIONED_SEGMENTS = False           # async_seg_*.mp4 with transcript overlay
SAVE_ASYNC_VLM_INPUT_VIDEOS = False             # vlm_input_*.mp4 sent to Gemini


# ── Simulation visualization ──
SAVE_SIMULATION_VIS = False                     # keyframe_simulation.mp4
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

# ── Robot pause on voice ──
STOP_ROBOT_WHILE_RECV = False                    # pause robot control while VLM processes voice

# FPS for saving async VLM segment videos (logging).
# IMPORTANT : Keep this equal to the source capture FPS so the saved clip duration matches the real duration.
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

  - "Put this in the bag1 and this in the bag2" \
→ 2 keyframes (obj1, obj2) + 2 keyframes (bag1, bag2) \
→ 4 action queue elements with respects the causal and natural order of instructions:
    [pick up obj1, place it in bag1,
     pick up obj2, place it in bag2]

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
video. You are receiving a new {segment_duration:.1f}-second video clip \
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
- All timestamps are **relative to this clip** (0.0 = start of this clip, \
{segment_duration:.1f} = end of this clip)
- Apply contextual recomposition: if the human says "also put this in that \
basket", append TWO elements (pick up + place it), not one

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
For Append: new_elements = list of new elements
"""




"""
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
    target_indices: List[int]


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
            cropped_copy = cropped_bgr.copy()
            aria_copy = aria_bgr.copy() if STORE_FULL_ARIA_IN_VLM_BUFFER else cropped_copy
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

    def get_segment(self, max_frames: Optional[int] = None) -> Tuple[List, List, List, int, int]:
        """Atomically drain unprocessed frames and advance the pointer.

        If ``max_frames`` is provided, drain at most that many frames so the
        async worker can keep fixed-duration clip boundaries.
        Returns (aria_list, cropped_list, gaze_list, seg_start, seg_end).
        """
        with self._lock:
            start = self._vlm_processed_up_to
            end   = len(self._aria)
            if max_frames is not None:
                end = min(end, start + max(0, int(max_frames)))
            aria    = [f.copy() for f in self._aria[start:end]]
            cropped = [f.copy() for f in self._cropped[start:end]]
            gaze    = list(self._gaze[start:end])
            self._vlm_processed_up_to = end
            return aria, cropped, gaze, start, end

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
    """Transcribe a WAV file using OpenAI Whisper API (Korean).

    Returns {"text": "...", "words": [{"word": "...", "start": <float>, "end": <float>}]}
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # uses OPENAI_API_KEY env var
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ko",
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


def _is_useful_whisper_transcript(transcript_data: Optional[dict]) -> bool:
    if not transcript_data:
        return False

    text = re.sub(r"\s+", " ", transcript_data.get("text", "") or "").strip()
    # Korean variant: count Hangul syllables/jamo as content characters too,
    # otherwise an all-Hangul transcript would be rejected by the original
    # [A-Za-z0-9]-only regex.
    content = re.sub(r"[^A-Za-z0-9가-힣ᄀ-ᇿ㄰-㆏]+", "", text)
    if len(content) < USEFUL_VOICE_MIN_CHARS:
        return False

    lower = text.lower()
    boilerplate = (
        "thank you for watching",
        "thanks for watching",
        "have a nice day",
        "subscribe",
        "시청해주셔서 감사합니다",
        "구독",
        "좋아요",
    )
    if any(phrase in lower for phrase in boilerplate):
        return False

    return True


def _transcribe_audio_records_for_whisper(
    audio_records: List[dict],
    context: str = "whisper",
) -> dict:
    audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate <= 0:
        return {"text": "", "words": []}

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
    try:
        transcript = _whisper_transcribe_wav(wav_tmp.name)
        text = (transcript.get("text", "") or "").strip()
        if text:
            print(f"  [Whisper:{context}] \"{text[:100]}\"")
        return transcript
    finally:
        os.unlink(wav_tmp.name)


def _get_frame_aligned_caption(
    words: list,
    time_sec: float,
    fps: float,
    transcript_text: str = "",
) -> str:
    """Return a word-level caption aligned to the current video frame."""
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

    return (transcript_text or "").strip()


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
    transcript_data_override: Optional[dict] = None,
) -> Tuple[bytes, str, dict]:
    """Prepare a video segment for VLM consumption.

    Applies: optional gaze crop → subsample → encode → whisper → captions → audio merge.
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
                    gx_corrected = gx + GAZE_CALIBRATION_X_OFFSET
                    gy_corrected = gy + GAZE_CALIBRATION_Y_OFFSET
                    f = _crop_around_gaze(f, gx_corrected, gy_corrected, GAZE_CROP_SIZE)
            processed.append(f)

    if frames_are_gaze_cropped and VLM_DRAW_GAZE_MARKER:
        processed = [_draw_gaze_crosshair(frame) for frame in processed]

    if VLM_DEBUG_BANNER:
        # Add a "VLM INPUT" banner + frame index for debugging saved clips.
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

    # Whisper transcription. Async worker can pass its pre-check transcript
    # here so voiced segments do not pay for Whisper twice.
    if WHISPER_ENABLED:
        if transcript_data_override is not None:
            transcript_data = transcript_data_override
        else:
            transcript_data = _transcribe_audio_records_for_whisper(
                audio_records, context="vlm")

        if _is_useful_whisper_transcript(transcript_data):
            transcript_text = transcript_data.get("text", "")
            subsampled = _overlay_captions_on_frames(
                subsampled,
                transcript_data.get("words", []),
                transcript_text,
                0.0,
                VLM_FPS,
            )
            transcript_section = (
                f"\nA speech transcript (in KOREAN) was detected in this clip:\n"
                f"  \"{transcript_text}\"\n"
                f"The same Korean transcript is also burned onto the video as on-screen captions.\n"
                f"Translate the Korean speech internally and use it to understand the human's spoken instruction.\n"
                f"IMPORTANT: Your JSON output (reasoning, instructions, descriptions, commands) MUST be in ENGLISH only — \n"
                f"the closed-vocabulary object names listed in the prompt are the canonical English terms and MUST be used verbatim.\n"
            )

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
    """No-op for kor variant: return cropped image as-is, no overlay."""
    return img_bgr.copy()


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _require_initial_keyframes_ready(queue: KeyframeQueue, context: str = "startup"):
    """Fail before any policy/async workers start if keyframe extraction failed."""
    if queue is None or queue.total <= 0 or queue.current_keyframe is None:
        raise RuntimeError(
            f"[{context}] Initial keyframes are not ready; "
            "not starting policy/VLA inference or async VLM collection."
        )


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
        return (0, 0, 180)
    if isinstance(cmd, CmdAppend):
        return (0, 140, 0)
    if isinstance(cmd, CmdReplace):
        return (0, 150, 200)
    if isinstance(cmd, CmdRemove):
        return (0, 100, 200)
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

    # Appends accumulate at the tail. Keeping the tail visible helps Gemini
    # avoid duplicate appends without sending the entire visual queue.
    tail_budget = max_images - len(selected)
    for idx in range(max(0, total - tail_budget), total):
        add(idx)

    # If the current index is already in the tail, duplicates can leave spare
    # budget. Fill it with nearest earlier entries first.
    for idx in range(current_idx - 1, -1, -1):
        add(idx)
        if len(selected) >= max_images:
            break

    return sorted(selected)


_VISUAL_QUEUE_THUMB_CACHE = {}


def _encode_visual_queue_thumb(kf_img: np.ndarray,
                               resize: int,
                               quality: int) -> Optional[bytes]:
    cache_key = (id(kf_img), int(resize), int(quality))
    cached = _VISUAL_QUEUE_THUMB_CACHE.get(cache_key)
    if cached is not None:
        return cached

    img = cv2.resize(kf_img, (resize, resize), interpolation=cv2.INTER_AREA)
    ok, jpg_bytes = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return None

    if len(_VISUAL_QUEUE_THUMB_CACHE) >= VISUAL_QUEUE_THUMB_CACHE_MAX:
        _VISUAL_QUEUE_THUMB_CACHE.clear()

    data = jpg_bytes.tobytes()
    _VISUAL_QUEUE_THUMB_CACHE[cache_key] = data
    return data


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
        jpg_data = _encode_visual_queue_thumb(
            kf_img, resize, VISUAL_QUEUE_JPEG_QUALITY)
        if jpg_data is None:
            continue
        parts.append(types.Part(
            text=f"[Queue index {i}] {instr}:",
        ))
        parts.append(types.Part(
            inline_data=types.Blob(data=jpg_data,
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
        print(f"[AsyncVLM] Worker started. video_source={_gemini_video_source_label()}")
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
            # Audio is checked before image frames are copied/encoded. Quiet
            # segments become a cheap flush instead of a full VLM call.
            seg_start, seg_end, seg_n_frames = self._buf.reserve_segment(
                max_frames=self._segment_frames)

            if seg_n_frames <= 0:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames, skip")
                continue

            seg_start_sec = seg_start / _VIDEO_RECORD_FPS
            seg_end_sec   = seg_end   / _VIDEO_RECORD_FPS

            # Collect audio records for this segment
            audio_records = self._buf.get_audio_records(seg_start, seg_end)

            # ── Whisper pre-check ──
            # When ASYNC_VLM_REQUIRE_USEFUL_VOICE is enabled, no useful speech
            # means no frame copy, no mp4 encode, and no Gemini request.
            voice_detected = False
            precheck_transcript: dict = {}
            if WHISPER_ENABLED and (
                ASYNC_VLM_REQUIRE_USEFUL_VOICE or STOP_ROBOT_WHILE_RECV
            ):
                precheck_transcript = _transcribe_audio_records_for_whisper(
                    audio_records, context=f"precheck@{seg_start}-{seg_end}")
                voice_detected = _is_useful_whisper_transcript(precheck_transcript)

            if ASYNC_VLM_REQUIRE_USEFUL_VOICE and WHISPER_ENABLED and not voice_detected:
                skipped_text = (precheck_transcript.get("text", "") or "").strip()
                suffix = f" transcript=\"{skipped_text[:80]}\"" if skipped_text else ""
                print(
                    f"[AsyncVLM] No useful voice in frames {seg_start}–{seg_end} "
                    f"({seg_start_sec:.1f}s–{seg_end_sec:.1f}s){suffix} — skip Gemini."
                )
                continue

            aria_seg, cropped_seg, gaze_seg = self._buf.get_range(seg_start, seg_end)
            gemini_seg = _select_gemini_frames(aria_seg, cropped_seg)

            if not gemini_seg:
                print(f"[AsyncVLM] t={trigger_t:.1f}s — no frames in reserved range, skip")
                continue

            self._call_count += 1
            call_id = self._call_count

            print(f"[AsyncVLM] Call #{call_id} | t={trigger_t:.2f}s | "
                  f"obs frames {seg_start}–{seg_end} ({len(gemini_seg)} frames) | "
                  f"time {seg_start_sec:.1f}s–{seg_end_sec:.1f}s | "
                  f"video_source={_gemini_video_source_label()}")

            if voice_detected and STOP_ROBOT_WHILE_RECV:
                print(f"[AsyncVLM] #{call_id} Voice detected: "
                      f"\"{precheck_transcript.get('text', '')[:80]}\" → pausing robot")
                self.robot_pause_event.set()

            if SAVE_ASYNC_RAW_SEGMENTS:
                seg_video_path = os.path.join(
                    self._temp_dir,
                    f"async_seg_{call_id:04d}_{_GEMINI_VIDEO_SOURCE}.mp4")
                _write_video_with_audio(
                    gemini_seg, audio_records, seg_video_path,
                    fps=_VLM_VIDEO_SAVE_FPS)

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
                    transcript_data_override=precheck_transcript,
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
                        "transcript_data_override": precheck_transcript,
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
              f"(video_source={_gemini_video_source_label()})...")

        try:
            # Prepare video: optional gaze crop → subsample → encode → whisper → audio
            video_bytes, transcript_section, transcript_data = _prepare_vlm_video(
                gemini_seg, gaze_seg, audio_records, seg_start_sec,
                frames_are_gaze_cropped=_gemini_uses_cropped_video(),
                transcript_data_override=transcript_data_override,
            )

            if (
                ASYNC_VLM_REQUIRE_USEFUL_VOICE
                and WHISPER_ENABLED
                and not _is_useful_whisper_transcript(transcript_data)
            ):
                log_entry["command"] = "SKIPPED_NO_USEFUL_VOICE"
                log_entry["gemini_end_t"] = self._elapsed()
                print(
                    f"[AsyncVLM] #{call_id} prepared VLM clip has no useful "
                    "Whisper transcript — skip Gemini request."
                )
                return

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

            # ---- resolve _RawCommand → typed command with ndarray ----
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
    """Records keyframe_simulation.mp4 in real time when explicitly enabled.

    The main thread queues `record_frame()` jobs; a background writer thread
    renders the large simulation canvas and writes the MP4 so robot startup
    and control are not blocked by visualization work.
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
                canvas = _render_sim_canvas(aria_bgr, queue_snap, current_time)
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
                     queue: Optional[KeyframeQueue],
                     current_time: float):
        """Queue one simulation frame for asynchronous render/write."""
        queue_snap = queue.snapshot() if queue is not None else None
        self._enqueue_frame(aria_bgr, queue_snap, current_time)

    def record_init_phase(self, aria_frames: List[np.ndarray],
                          queue: Optional[KeyframeQueue],
                          fps: float = _VIDEO_RECORD_FPS):
        """Write the init phase (before 'p' press) as 'Initializing...' frames,
        then switch to showing the queue once available.

        We subsample to SIM_FPS from the original capture FPS.
        """
        step = max(1, int(fps / self._fps))
        queue_snap = queue.snapshot() if queue is not None else None
        for i in range(0, len(aria_frames), step):
            t = i / fps
            self._enqueue_frame(aria_frames[i], queue_snap, t)

    def finalize(self, audio_records: Optional[List[dict]] = None):
        """Close writer, re-encode H264, optionally merge audio."""
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

        # Re-encode to H264.
        _ffmpeg_reencode_h264(self._tmp_path, self._output_path)
        os.unlink(self._tmp_path)

        # Merge audio if available.
        if audio_records:
            audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
            if audio_pcm16.size > 0 and sample_rate > 0:
                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_tmp.close()
                _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
                try:
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
# Interactive eye-gaze calibration  (runs once at startup)
# ---------------------------------------------------------------------------

def calibrate_gaze(env) -> Tuple[int, int]:
    """Interactive gaze calibration.

    Shows the live aria_rgb frame with the eye-tracker gaze marker (red) and
    the offset-adjusted marker (green). Two OpenCV trackbars let the user
    nudge the gaze in X and Y until the green marker lands on what they are
    actually looking at. Press ENTER to accept; 'q' / ESC to skip (offsets
    stay at the current values).

    Updates module-level ``GAZE_CALIBRATION_X_OFFSET`` /
    ``GAZE_CALIBRATION_Y_OFFSET`` and also returns ``(x_off, y_off)``.
    """
    global GAZE_CALIBRATION_X_OFFSET, GAZE_CALIBRATION_Y_OFFSET

    window_name = "Eye Gaze Calibration"
    rng = int(GAZE_CALIBRATION_RANGE_PX)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _noop(_v):  # trackbar callback (we read values each frame)
        pass

    cv2.createTrackbar("X offset (+rng)", window_name,
                       int(GAZE_CALIBRATION_X_OFFSET) + rng, 2 * rng, _noop)
    cv2.createTrackbar("Y offset (+rng)", window_name,
                       int(GAZE_CALIBRATION_Y_OFFSET) + rng, 2 * rng, _noop)

    print("[calibrate] Look at a fixed target. Adjust X/Y sliders so the "
          "GREEN crosshair lands on what you're actually looking at.")
    print("[calibrate] ENTER = accept   q / ESC = skip")

    accepted = False
    try:
        while True:
            try:
                obs = _extract_latest_observation(
                    env.get_observation(
                        mode='human_view_streamed_and_keyframe_cropped'))
            except Exception as exc:
                print(f"[calibrate] env.get_observation failed: {exc}")
                break

            frame = obs["images"]["aria_rgb"].copy()
            h, w = frame.shape[:2]

            x_off = cv2.getTrackbarPos("X offset (+rng)", window_name) - rng
            y_off = cv2.getTrackbarPos("Y offset (+rng)", window_name) - rng

            gaze_point = np.asarray(obs["gaze_point"],
                                    dtype=np.float32).reshape(-1)
            has_gaze = (gaze_point.size >= 2
                        and np.all(np.isfinite(gaze_point[:2])))
            if has_gaze:
                gx_raw = float(gaze_point[0])
                gy_raw = float(gaze_point[1])
                # Raw gaze (red)
                rx = int(np.clip(gx_raw, 0, w - 1))
                ry = int(np.clip(gy_raw, 0, h - 1))
                cv2.circle(frame, (rx, ry), 8, (0, 0, 255), 2)
                cv2.line(frame, (rx - 14, ry), (rx + 14, ry), (0, 0, 255), 2)
                cv2.line(frame, (rx, ry - 14), (rx, ry + 14), (0, 0, 255), 2)
                # Calibrated gaze (green)
                cx = int(np.clip(gx_raw + x_off, 0, w - 1))
                cy = int(np.clip(gy_raw + y_off, 0, h - 1))
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (0, 255, 0), 2)
                cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (0, 255, 0), 2)
                # Crop preview rectangle (what the VLM will see)
                half = GAZE_CROP_SIZE // 2
                x1 = max(0, min(cx - half, w - GAZE_CROP_SIZE))
                y1 = max(0, min(cy - half, h - GAZE_CROP_SIZE))
                x2 = min(w - 1, x1 + GAZE_CROP_SIZE - 1)
                y2 = min(h - 1, y1 + GAZE_CROP_SIZE - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            else:
                cv2.putText(frame, "NO GAZE SIGNAL", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 2, cv2.LINE_AA)

            banner = (
                f"x_off={x_off:+d}px  y_off={y_off:+d}px   "
                f"red=raw  green=calibrated   ENTER=accept  q/ESC=skip"
            )
            cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(frame, banner, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = _poll_keypress(delay_ms=15)
            if key in (13, 10):           # Enter / LF
                GAZE_CALIBRATION_X_OFFSET = int(x_off)
                GAZE_CALIBRATION_Y_OFFSET = int(y_off)
                accepted = True
                print(f"[calibrate] Accepted offsets: "
                      f"x={GAZE_CALIBRATION_X_OFFSET:+d}px "
                      f"y={GAZE_CALIBRATION_Y_OFFSET:+d}px")
                break
            if key in (ord('q'), ord('Q'), 27):
                print(f"[calibrate] Skipped — keeping offsets "
                      f"x={GAZE_CALIBRATION_X_OFFSET:+d}px "
                      f"y={GAZE_CALIBRATION_Y_OFFSET:+d}px")
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    if not accepted:
        return GAZE_CALIBRATION_X_OFFSET, GAZE_CALIBRATION_Y_OFFSET
    return GAZE_CALIBRATION_X_OFFSET, GAZE_CALIBRATION_Y_OFFSET


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

            elif key in (ord('q'), ord('Q'), ord('s'), ord('S'), 27):
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
         chunksize: int = 8):
    """
    Robot execution loop.

    The async VLM worker runs concurrently and may modify `queue` at any
    time.  The loop reacts to four triggers:

      1. vlm_worker.preempt_event  — robot reset + new keyframe from VLM
      2. completion_prob >= 0.8    — auto-advance to next keyframe
      3. 'n' key                   — manual advance
      4. 'q' / ESC                 — abort episode
    """
    _require_initial_keyframes_ready(queue, context="eval")

    pending_auto_advance = False
    top_view_frames_list = []
    comp_prob_log        = []
    global_step          = 0
    subtask_step         = 0
    timer                = Timer(env.control_hz)
    window_name          = "Eval  |  n=next kf  q=quit"
    rollout_video_path   = os.path.join(_TEMP_DIR, "rollout.mp4")
    keyframe_log: List[Tuple[int, np.ndarray]] = []
    async_collection_started = False
    kf = queue.current_keyframe
    if kf is not None:
        keyframe_log.append((0, kf.copy()))

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
                aborted = wait_for_queue_items(
                    queue,
                    window_name=window_name,
                    preview_builder=lambda: _build_eval_preview(
                        _extract_latest_observation(env.get_observation(
                            mode='human_view_streamed_and_keyframe_cropped',
                            audio_drain=False)),
                        queue.current_keyframe, queue,
                        queue.current_instruction or ""),
                    key_poll=_poll_keypress,
                )
                if aborted:
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt
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

            # NOTE: video_buffer is now fed by BackgroundObservationStreamer
            # at a steady _VIDEO_RECORD_FPS rate, independent of env.step().
            # We still read aria_bgr here for shape-matching downstream.
            aria_bgr    = images["aria_rgb"]

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
                        if key in (ord('q'), ord('Q'), ord('s'), ord('S'), 27):
                            cv2.destroyAllWindows()
                            raise KeyboardInterrupt
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
                if not async_collection_started:
                    video_buffer.skip_to_current()
                    _bg_streamer.start()
                    vlm_worker.start()
                    async_collection_started = True
                    print("[Eval] Async VLM collection started with first robot action.")
                env.step(action)
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

                elif key in (ord('q'), ord('Q'), ord('s'), ord('S'), 27):
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

        except KeyboardInterrupt:
            _bg_streamer.stop()
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

    _bg_streamer.stop()
    cv2.destroyAllWindows()
    _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path,
                        vlm_event_log=vlm_worker.event_log,
                        keyframe_log=keyframe_log)
    _safe_finalize_sim_recorder(sim_recorder, video_buffer)


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
    VLM_DRAW_GAZE_MARKER = args.vlm_draw_gaze_marker

    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env                  = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "I want this muffin."
    horizon              = 10000
    GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    print("Starting. Instruction:", language_instruction)
    print("During eval: n=next keyframe  q/s=stop episode")
    print("VLM video source:", _gemini_video_source_label())
    print("VLM center gaze marker:", VLM_DRAW_GAZE_MARKER)
    print(
        "VLM optimized config:",
        f"record_fps={_VIDEO_RECORD_FPS}",
        f"vlm_fps={VLM_FPS}",
        f"interval={VLM_INTERVAL_SEC}s",
        f"res={VLM_TARGET_RES}",
        f"thinking={VLM_THINKING_BUDGET}",
        f"voice_gate={ASYNC_VLM_REQUIRE_USEFUL_VOICE}",
    )
    print("Simulation video:", "enabled" if args.record_sim_video else "disabled")

    # Shared buffer — persists for the lifetime of the process.
    video_buffer = VideoBuffer()

    # ---- Optional simulation video recorder ----
    sim_output_path = os.path.join(_TEMP_DIR, "keyframe_simulation.mp4")
    sim_recorder = (
        SimulationVideoRecorder(sim_output_path)
        if args.record_sim_video else None
    )

    # ---- Eye-gaze calibration (interactive, runs once) ----
    with _terminal_keypress_mode():
        calibrate_gaze(env)

    # ---- Initial keyframe extraction (interactive) ----
    with _terminal_keypress_mode():
        queue = extract_keyframes(env, language_instruction, GEMINI_API_KEY,
                                  video_buffer, sim_recorder=sim_recorder)
    _require_initial_keyframes_ready(queue, context="startup")

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
            # Finalize previous simulation video and create a new one if enabled.
            _safe_finalize_sim_recorder(sim_recorder, video_buffer)
            video_buffer = VideoBuffer()
            sim_output_path = os.path.join(_TEMP_DIR, "keyframe_simulation.mp4")
            sim_recorder = (
                SimulationVideoRecorder(sim_output_path)
                if args.record_sim_video else None
            )
            with _terminal_keypress_mode():
                queue = extract_keyframes(
                    env, language_instruction, GEMINI_API_KEY, video_buffer,
                    sim_recorder=sim_recorder)
            _require_initial_keyframes_ready(queue, context="episode restart")
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
