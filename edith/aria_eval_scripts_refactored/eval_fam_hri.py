"""eval_fam_hri.py - FAM-HRI TCP Aria session executor.

Robot-control main script for the TCP FAM-HRI pipeline. The Aria laptop is a
dumb sensor streamer: it sends RGB + gaze + audio using the inference-v2 TCP
protocol on port 9999. This process connects to that stream through
``piper_env_gaze``, records one spoken/gaze session between local ``s`` and
``e`` keypresses, runs FAM-HRI perception/planning locally, then executes the
resulting (subtask, keyframe) pairs through the keyframe-conditioned pi0.5
policy.

  Aria laptop (:9999 TCP) --> this process --> pi0.5 policy server (ws, :8000)
                              |
                              observation: 3x RealSense + keyframe crop + prompt

Session keys (this terminal):
  s = start speech/gaze session
  e = end speech/gaze session -> Whisper + FAM-HRI pipeline enqueue subtasks
  r = reset episode   -> abort session, clear queue, reset robot, back to IDLE
  n = skip current subtask        q = quit

While the robot is executing you may press 's' again to speak the next
command; new pairs are appended to the FIFO.

The policy consumes the keyframe as
``observation.images.human_view_keyframe_cropped`` (RGB), same as
eval_openpi_ours_with_voice_260503.py - serve a keyframe-conditioned
checkpoint on :8000, not a language-only one.
"""

import argparse
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from edith.legacy.envs.piper_env_gaze import PiperRobotEnv
from edith.legacy.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from common.config import SERVER_IP
from ours_utils.audio_processing import (
    is_useful_whisper_transcript,
    transcribe_audio_records_for_whisper,
)
from ours_utils.keyboard import (
    poll_keypress as _poll_keypress,
    terminal_keypress_mode as _terminal_keypress_mode,
)

# ---------------------------------------------------------------------------
# Constants - edit these instead of CLI args
# ---------------------------------------------------------------------------

POLICY_HOST          = "localhost"
POLICY_PORT          = 8000
ARIA_HOST            = SERVER_IP
ARIA_PORT            = 9999
ARIA_ROTATION        = "none"         # FAM-HRI_tcp gaze already matches upright RGB.
ACTION_CHUNKSIZE     = 8
COMPLETION_THRESH    = 0.8           # completion_prob >= this -> subtask done
MIN_SUBTASK_STEPS    = 40            # ignore early completion spikes
HOLD_POLL_SEC        = 0.2           # queue poll period while holding pose
KEYFRAME_SIZE        = 512           # fallback blank keyframe size
TCP_OBS_MODE         = "human_view_streamed_and_keyframe_cropped"

FAMHRI_DEVICE        = "cuda"
FAMHRI_LLM_MODEL     = "gpt-4o"
FAMHRI_SCORE_THR     = 0.30
FAMHRI_NO_LLM        = False
FAMHRI_MIN_DWELL_S   = 0.5
FAMHRI_MAX_DISP_PX   = 25.0          # ~70px in the native 1408 Aria frame.
FAMHRI_COOLDOWN_S    = 0.4
USEFUL_MIN_CHARS     = 3
FAMHRI_TEXT_PROMPT   = (
    "cup . mug . bottle . bowl . plate . can . box . glass . jar . "
    "apple . banana . orange . knife . spoon . fork . book . phone . "
    "remote . marker . pen . scissors . tape . stapler . keyboard . mouse . "
    "tumbler . muffin . basket . tray . tool . screwdriver . hammer . "
    "wrench . pliers"
)
FAMHRI_DEFAULT_TASK  = "generic"

BOTTLE_TEXT_PROMPT = (
    "pink bottle . green bottle . bottle . beverage . drink . can . "
    "yellow basket . left basket . right basket . basket"
)
PASSING_TOOLS_TEXT_PROMPT = (
    "hex screwdriver . philips screwdriver . screwdriver . "
    "5 cm metal support . 10 cm metal support . 12 cm metal support . "
    "metal support . support . tool"
)

PASSING_TOOL_NAMES = (
    "hex screwdriver",
    "Philips screwdriver",
    "5 cm metal support",
    "10 cm metal support",
    "12 cm metal support",
)

BOTTLE_PLANNER_CONTEXT = """\
Bottle/beverage sorting task. The robot should decompose one utterance into
pick/place subtasks. If the command says all/these/beverages, create one pair
for each focused beverage in gaze order. Use exact policy-style language:
"pick up <object>" and "place it in <basket>". For the yellow-basket task,
use "place it in the yellow basket" unless the user explicitly indicates a
left or right basket."""

PASSING_TOOLS_PLANNER_CONTEXT = """\
Passing-tools handoff task. Emit only Pick primitives; do not create Put
actions. The final policy instructions must be one of: "pick up hex
screwdriver", "pick up Philips screwdriver", "pick up 5 cm metal support",
"pick up 10 cm metal support", or "pick up 12 cm metal support". If several
tools are requested in one utterance, include each tool in gaze order."""

TASK_PROFILES = {
    "generic": {
        "description": "generic FAM-HRI pick/place executor",
        "text_prompt": FAMHRI_TEXT_PROMPT,
        "planner_context": "",
    },
    "muffin": {
        "description": "serving muffins",
        "text_prompt": FAMHRI_TEXT_PROMPT,
        "planner_context": "",
    },
    "bottle": {
        "description": "beverage/bottle sorting to basket",
        "text_prompt": BOTTLE_TEXT_PROMPT,
        "planner_context": BOTTLE_PLANNER_CONTEXT,
    },
    "passing_tools": {
        "description": "passing tools handoff",
        "text_prompt": PASSING_TOOLS_TEXT_PROMPT,
        "planner_context": PASSING_TOOLS_PLANNER_CONTEXT,
    },
}


def _default_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "FAM-HRI_tcp").is_dir():
            return parent
    return Path("/home/alin/projects/piper_project")


FAMHRI_ROOT = _default_project_root() / "FAM-HRI_tcp"

CAMERA_IDS = {
    'top_camera':         '242322070028',
    'left_wrist_camera':  '244222076964',
    'right_wrist_camera': '244222071848',
}


class EpisodeEnd(Exception):
    """'r' - abort session, reset robot, clear queue, back to IDLE."""


class ProgramQuit(Exception):
    """'q' - exit the program."""


# ---------------------------------------------------------------------------
# (subtask, keyframe) FIFO + display
# ---------------------------------------------------------------------------

class SubtaskQueue:
    """Thread-safe FIFO of {'instruction', 'keyframe_bgr', 'referent'} items."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self.current: Optional[dict] = None
        self.done: List[str] = []
        self._accepting = False   # True while a local speech/gaze session is open.

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    @accepting.setter
    def accepting(self, value: bool):
        with self._lock:
            self._accepting = bool(value)

    def extend_items(self, items: List[dict]) -> int:
        with self._lock:
            for it in items:
                self._pending.append(it)
            return len(items)

    def pop_next(self) -> Optional[dict]:
        with self._lock:
            if self.current is not None:
                self.done.append(self.current["instruction"])
                self.current = None
            if self._pending:
                self.current = self._pending.popleft()
            return self.current

    def clear(self):
        with self._lock:
            if self.current is not None:
                self.done.append(self.current["instruction"])
            self.current = None
            self._pending.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "session_active": self._accepting,
                "current": (self.current or {}).get("instruction"),
                "pending": [p["instruction"] for p in self._pending],
                "done": list(self.done),
            }


class KeyframeDisplay:
    """CV2 viewport showing the latest session's model-input keyframes.

    The session-processing thread only stores data here; render() must be
    called from the main thread (cv2 GUI is not thread-safe). Shows one
    thumbnail per (subtask, keyframe) pair: green border = currently executing,
    dimmed = done, plain = pending.
    """

    WINDOW = "FAM-HRI keyframes (model input)"
    THUMB = 320
    CAPTION_H = 46

    def __init__(self):
        self._lock = threading.Lock()
        self._batch_id = 0
        self._thumbs: List[np.ndarray] = []
        self._labels: List[str] = []
        self._status: List[str] = []       # pending | current | done
        self._enabled = True               # disabled on first imshow failure

    def set_batch(self, items: List[dict]) -> None:
        """Replace the display with a new session's pairs; tag the items."""
        with self._lock:
            self._batch_id += 1
            self._thumbs, self._labels, self._status = [], [], []
            for i, it in enumerate(items):
                it["disp_batch"] = self._batch_id
                it["disp_idx"] = i
                kf = it.get("keyframe_bgr")
                if kf is None:
                    kf = np.zeros((KEYFRAME_SIZE, KEYFRAME_SIZE, 3), np.uint8)
                self._thumbs.append(
                    cv2.resize(kf, (self.THUMB, self.THUMB)))
                self._labels.append(it["instruction"])
                self._status.append("pending")

    def mark_current(self, item: Optional[dict]) -> None:
        with self._lock:
            if item is None or item.get("disp_batch") != self._batch_id:
                return
            for i, s in enumerate(self._status):
                if s == "current":
                    self._status[i] = "done"
            self._status[item["disp_idx"]] = "current"

    def mark_all_current_done(self) -> None:
        with self._lock:
            self._status = ["done" if s == "current" else s
                            for s in self._status]

    def clear(self) -> None:
        with self._lock:
            self._thumbs, self._labels, self._status = [], [], []
        if self._enabled:
            try:
                cv2.destroyWindow(self.WINDOW)
            except Exception:  # noqa: BLE001
                pass

    def render(self) -> None:
        """Main thread only. Draw the strip and imshow it."""
        if not self._enabled:
            return
        with self._lock:
            if not self._thumbs:
                return
            thumbs = [t.copy() for t in self._thumbs]
            labels = list(self._labels)
            status = list(self._status)

        tiles = []
        th, cap = self.THUMB, self.CAPTION_H
        for i, (img, label, st) in enumerate(zip(thumbs, labels, status)):
            if st == "done":
                img = (img * 0.35).astype(np.uint8)
            tile = np.zeros((th + cap, th, 3), dtype=np.uint8)
            tile[:th] = img
            color = ((0, 255, 0) if st == "current"
                     else (160, 160, 160) if st == "done" else (255, 255, 255))
            text = f"[{i + 1}] {label}"
            if len(text) > 34:
                text = text[:33] + "..."
            cv2.putText(tile, text, (6, th + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            cv2.putText(tile, st.upper(), (6, th + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            border = (0, 255, 0) if st == "current" else (60, 60, 60)
            cv2.rectangle(tile, (0, 0), (th - 1, th + cap - 1), border,
                          3 if st == "current" else 1)
            tiles.append(tile)

        strip = np.concatenate(tiles, axis=1)
        try:
            cv2.imshow(self.WINDOW, strip)
        except Exception as exc:  # noqa: BLE001
            print(f"[Display] disabled (no GUI available: {exc})")
            self._enabled = False


# ---------------------------------------------------------------------------
# Local TCP Aria session producer
# ---------------------------------------------------------------------------

@dataclass
class TcpKeyframe:
    ts_ns: int
    wall: float
    gaze_xy: Tuple[float, float]
    dwell_s: float
    full_bgr: np.ndarray
    crop_bgr: np.ndarray


def _load_fam_hri_pipeline(root: str):
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise RuntimeError(f"FAM-HRI root does not exist: {root_path}")
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    from fam_hri.server.bundle_pipeline import (  # noqa: WPS433
        KFInput,
        PipelineOpts,
        build_labeler,
        build_planner,
        process_utterance,
    )
    return KFInput, PipelineOpts, build_labeler, build_planner, process_utterance


def _gaze_xy_from_obs(obs: dict) -> Optional[Tuple[float, float]]:
    gaze = obs.get("gaze_point")
    if gaze is None:
        return None
    arr = np.asarray(gaze, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    return float(arr[0]), float(arr[1])


def _crop_around_gaze(frame_bgr: np.ndarray, gaze_xy: Tuple[float, float],
                      crop_size: int = KEYFRAME_SIZE) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    size = min(crop_size, h, w)
    half = size // 2
    cx = int(np.clip(gaze_xy[0], half, w - half))
    cy = int(np.clip(gaze_xy[1], half, h - half))
    x0 = max(0, min(cx - half, w - size))
    y0 = max(0, min(cy - half, h - size))
    crop = frame_bgr[y0:y0 + size, x0:x0 + size].copy()
    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size),
                          interpolation=cv2.INTER_LINEAR)
    return crop


def _clean_label(label) -> str:
    return re.sub(r"\s+", " ", str(label or "").strip().lower())


def _label_mentions(label: str, *terms: str) -> bool:
    label = _clean_label(label)
    return any(term in label for term in terms)


def _is_destination_label(label: str) -> bool:
    return _label_mentions(label, "basket", "bin", "tray", "destination")


def _text_mentions(text: str, *terms: str) -> bool:
    text = _clean_label(text)
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _keyframe_meta_label(meta: dict) -> str:
    return _clean_label(meta.get("gaze_object_label"))


def _source_keyframe_indices(keyframe_labels: list) -> List[int]:
    indices = []
    for i, meta in enumerate(keyframe_labels):
        label = _keyframe_meta_label(meta)
        if label and not _is_destination_label(label):
            indices.append(i)
    return indices


def _bottle_position(gaze_xy: Tuple[float, float],
                     frame_shape: Tuple[int, ...]) -> str:
    h, w = frame_shape[:2]
    x, y = gaze_xy
    front = y >= (h * 0.5)
    if front:
        if x < w * 0.36:
            side = "left"
        elif x > w * 0.64:
            side = "right"
        else:
            side = "center"
        return f"front {side}"
    return "back left" if x < (w * 0.5) else "back right"


def _canonical_bottle_object(label: str, gaze_xy: Tuple[float, float],
                             frame_shape: Tuple[int, ...]) -> str:
    label = _clean_label(label)
    position = _bottle_position(gaze_xy, frame_shape)
    if "pink" in label:
        base = "pink bottle"
    elif "green" in label:
        base = "green bottle"
    elif "can" in label:
        base = "can"
    elif "beverage" in label or "drink" in label:
        base = "beverage"
    else:
        base = "bottle"
    return f"{base} on the {position}"


def _bottle_destination_instruction(transcript: str,
                                    keyframe_labels: list) -> str:
    labels = [_keyframe_meta_label(meta) for meta in keyframe_labels]
    haystack = " ".join([_clean_label(transcript)] + labels)
    if "left basket" in haystack:
        return "place it in the left basket"
    if "right basket" in haystack:
        return "place it in the right basket"
    if "yellow" in haystack:
        return "place it in the yellow basket"
    return "place it in the yellow basket"


def _canonical_passing_tool(label: str, transcript: str = "") -> Optional[str]:
    label = _clean_label(label)
    haystack = f"{label} {_clean_label(transcript)}"
    if "hex" in label:
        return "hex screwdriver"
    if "philips" in label or "phillips" in label:
        return "Philips screwdriver"
    if re.search(r"\b(5|five)\s*(cm|centimeter|centimetre)?\b", label):
        return "5 cm metal support"
    if re.search(r"\b(10|ten)\s*(cm|centimeter|centimetre)?\b", label):
        return "10 cm metal support"
    if re.search(r"\b(12|twelve)\s*(cm|centimeter|centimetre)?\b", label):
        return "12 cm metal support"
    if "support" in label:
        if re.search(r"\b(5|five)\s*(cm|centimeter|centimetre)?\b", haystack):
            return "5 cm metal support"
        if re.search(r"\b(10|ten)\s*(cm|centimeter|centimetre)?\b", haystack):
            return "10 cm metal support"
        if re.search(r"\b(12|twelve)\s*(cm|centimeter|centimetre)?\b", haystack):
            return "12 cm metal support"
        return "5 cm metal support"
    if "screwdriver" in label:
        if "philips" in haystack or "phillips" in haystack:
            return "Philips screwdriver"
        if "hex" in haystack:
            return "hex screwdriver"
        return "hex screwdriver"
    return None


def _passing_tools_from_text(transcript: str) -> List[str]:
    text = _clean_label(transcript)
    patterns = (
        (r"\bhex\s+screwdriver\b|\bhex\b", "hex screwdriver"),
        (r"\bphilips\s+screwdriver\b|\bphillips\s+screwdriver\b|"
         r"\bphilips\b|\bphillips\b", "Philips screwdriver"),
        (r"\b5\s*cm\s+metal\s+support\b|\bfive\s+cm\s+metal\s+support\b|"
         r"\b5\s*cm\b|\bfive\s+cm\b", "5 cm metal support"),
        (r"\b10\s*cm\s+metal\s+support\b|\bten\s+cm\s+metal\s+support\b|"
         r"\b10\s*cm\b|\bten\s+cm\b", "10 cm metal support"),
        (r"\b12\s*cm\s+metal\s+support\b|\btwelve\s+cm\s+metal\s+support\b|"
         r"\b12\s*cm\b|\btwelve\s+cm\b", "12 cm metal support"),
    )
    matches = []
    for pattern, canonical in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), canonical))
    matches.sort(key=lambda item: item[0])

    ordered = []
    for _, canonical in matches:
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


class FamHriTcpSession:
    """Collect local s/e speech sessions from the Aria TCP stream."""

    def __init__(self, queue: SubtaskQueue, display: Optional[KeyframeDisplay],
                 args):
        self.queue = queue
        self.display = display
        self.args = args
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._active = False
        self._processing_count = 0
        self._generation = 0
        self._session_start_ns = 0
        self._last_ts_ns = 0
        self._last_frame: Optional[TcpKeyframe] = None
        self._audio_records: List[dict] = []
        self._keyframes: List[TcpKeyframe] = []
        self._run_start_ns = 0
        self._run_points: List[Tuple[float, float]] = []
        self._fired_for_run_start_ns = 0
        self._last_keyframe_wall = 0.0
        self._labeler = None
        self._planner = None
        self._pipeline = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def processing(self) -> bool:
        with self._lock:
            return self._processing_count > 0

    def start(self) -> None:
        with self._lock:
            if self._active:
                print("[Session] Already recording; press 'e' to process it.")
                return
            self._active = True
            self.queue.accepting = True
            self._reset_buffers_locked()
        print("[Session] TCP speech/gaze capture started - SPEAK NOW "
              "(press 'e' when done).")

    def finish(self) -> None:
        with self._lock:
            if not self._active:
                print("[Session] No active speech/gaze session to finish.")
                return
            self._active = False
            self.queue.accepting = False
            generation = self._generation
            utt_start_ns = self._session_start_ns or self._last_ts_ns or time.time_ns()
            utt_end_ns = self._last_ts_ns or time.time_ns()
            audio_records = list(self._audio_records)
            keyframes = list(self._keyframes)
            fallback = self._last_frame
            final_keyframe = self._pending_keyframe_at_finish_locked(
                fallback, keyframes)
        if not keyframes and fallback is not None:
            keyframes = [fallback]
            print("[Session] No dwell keyframe fired; using the latest Aria frame.")
        elif final_keyframe is not None:
            keyframes.append(final_keyframe)
            print("[Session] Added final dwell keyframe at capture end.")

        print("[Session] Capture ended - running Whisper + FAM-HRI pipeline "
              f"({len(audio_records)} audio chunks, {len(keyframes)} keyframes).")
        threading.Thread(
            target=self._process_session,
            args=(generation, utt_start_ns, utt_end_ns, audio_records, keyframes),
            daemon=True,
            name="FamHriTcpSession-Process",
        ).start()

    def _pending_keyframe_at_finish_locked(
        self,
        fallback: Optional[TcpKeyframe],
        keyframes: List[TcpKeyframe],
    ) -> Optional[TcpKeyframe]:
        if fallback is None or not keyframes:
            return None
        if self._run_start_ns == 0:
            return None
        if self._fired_for_run_start_ns == self._run_start_ns:
            return None

        dwell_s = max(0.0, (fallback.ts_ns - self._run_start_ns) / 1e9)
        if dwell_s < self.args.session_min_dwell_s:
            return None
        if fallback.ts_ns == keyframes[-1].ts_ns:
            return None

        gaze_xy = fallback.gaze_xy
        if self._run_points:
            pts = np.asarray(self._run_points, dtype=np.float64)
            centroid = pts.mean(axis=0)
            gaze_xy = (float(centroid[0]), float(centroid[1]))

        return TcpKeyframe(
            ts_ns=fallback.ts_ns,
            wall=fallback.wall,
            gaze_xy=gaze_xy,
            dwell_s=dwell_s,
            full_bgr=fallback.full_bgr.copy(),
            crop_bgr=fallback.crop_bgr.copy(),
        )

    def abort(self) -> None:
        with self._lock:
            self._active = False
            self._generation += 1
            self.queue.accepting = False
            self._reset_buffers_locked()
        print("[Session] TCP speech/gaze capture aborted.")

    def consume_observation(self, obs: dict) -> None:
        with self._lock:
            active = self._active
        if not active:
            return

        images = obs.get("images", {})
        full_bgr = images.get("aria_rgb")
        if full_bgr is None:
            return
        gaze_xy = _gaze_xy_from_obs(obs)
        if gaze_xy is None:
            return

        audio_obs = obs.get("audio", {})
        audio_chunk = np.asarray(
            audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
            dtype=np.int16,
        ).reshape(-1)
        audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
        audio_sr = int(audio_obs.get("sample_rate_hz", 0) or 0)
        audio_ns = int(audio_obs.get("num_samples", audio_chunk.size) or 0)
        ts_ns = int(obs.get("aria_timestamp_ns", 0) or audio_ts_ns or time.time_ns())

        crop_bgr = images.get("aria_cropped_rgb")
        if crop_bgr is None:
            crop_bgr = _crop_around_gaze(full_bgr, gaze_xy)

        keyframe_candidate = TcpKeyframe(
            ts_ns=ts_ns,
            wall=time.time(),
            gaze_xy=gaze_xy,
            dwell_s=0.0,
            full_bgr=full_bgr.copy(),
            crop_bgr=crop_bgr.copy(),
        )

        with self._lock:
            if not self._active:
                return
            if self._session_start_ns == 0:
                self._session_start_ns = ts_ns
            self._last_frame = keyframe_candidate
            if audio_chunk.size > 0 and audio_sr > 0:
                self._audio_records.append({
                    "frame_idx": len(self._audio_records),
                    "audio_ts_ns": audio_ts_ns,
                    "sample_rate_hz": audio_sr,
                    "num_samples": audio_ns,
                    "chunk_pcm16": audio_chunk.copy(),
                })
            if ts_ns != self._last_ts_ns:
                self._last_ts_ns = ts_ns
                self._update_keyframes_locked(keyframe_candidate)

    def _reset_buffers_locked(self) -> None:
        self._session_start_ns = 0
        self._last_ts_ns = 0
        self._last_frame = None
        self._audio_records = []
        self._keyframes = []
        self._run_start_ns = 0
        self._run_points = []
        self._fired_for_run_start_ns = 0
        self._last_keyframe_wall = 0.0

    def _update_keyframes_locked(self, sample: TcpKeyframe) -> None:
        if not self._run_points:
            self._run_start_ns = sample.ts_ns
            self._run_points = [sample.gaze_xy]
            return

        pts = np.asarray(self._run_points, dtype=np.float64)
        centroid = pts.mean(axis=0)
        dist = float(np.linalg.norm(np.asarray(sample.gaze_xy) - centroid))
        if dist > self.args.session_max_disp_px:
            self._run_start_ns = sample.ts_ns
            self._run_points = [sample.gaze_xy]
            return

        self._run_points.append(sample.gaze_xy)
        if len(self._run_points) > 90:
            self._run_points = self._run_points[-90:]

        dwell_s = max(0.0, (sample.ts_ns - self._run_start_ns) / 1e9)
        now = time.time()
        if (
            dwell_s >= self.args.session_min_dwell_s
            and self._fired_for_run_start_ns != self._run_start_ns
            and (now - self._last_keyframe_wall) >= self.args.keyframe_cooldown_s
        ):
            sample.dwell_s = dwell_s
            sample.gaze_xy = (float(centroid[0]), float(centroid[1]))
            self._keyframes.append(sample)
            self._fired_for_run_start_ns = self._run_start_ns
            self._last_keyframe_wall = now
            print(f"[Session] keyframe {len(self._keyframes)} "
                  f"ts={sample.ts_ns} dwell={dwell_s:.2f}s "
                  f"gaze=({sample.gaze_xy[0]:.0f},{sample.gaze_xy[1]:.0f})")

    def _ensure_pipeline_ready(self):
        if self._pipeline is None:
            self._pipeline = _load_fam_hri_pipeline(self.args.fam_hri_root)
        if self._labeler is None:
            _, _, build_labeler, _, _ = self._pipeline
            print(f"[FAM-HRI] Loading SAM3 labeler on {self.args.fam_hri_device}...")
            self._labeler = build_labeler(
                self.args.fam_hri_device,
                self.args.fam_hri_text_prompt,
                self.args.fam_hri_score_thr,
            )
        if self._planner is None:
            _, _, _, build_planner, _ = self._pipeline
            self._planner = build_planner(self.args.fam_hri_llm_model)

    def _is_current_generation(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _process_session(self, generation: int,
                         utt_start_ns: int, utt_end_ns: int,
                         audio_records: List[dict],
                         keyframes: List[TcpKeyframe]) -> None:
        with self._lock:
            self._processing_count += 1
        try:
            with self._process_lock:
                if not self._is_current_generation(generation):
                    print("[FAM-HRI] Dropping stale session result after reset.")
                    return
                if not keyframes:
                    print("[FAM-HRI] No Aria keyframes available; no subtasks queued.")
                    return

                transcript = transcribe_audio_records_for_whisper(
                    audio_records, context=f"fam_hri@{utt_start_ns}")
                text = (transcript.get("text", "") or "").strip()
                useful_transcript = is_useful_whisper_transcript(
                    transcript, self.args.useful_transcript_min_chars
                )
                if (
                    not useful_transcript
                    and self.args.task not in ("bottle", "passing_tools")
                ):
                    print("[FAM-HRI] Whisper produced no useful instruction; "
                          "no subtasks queued.")
                    return
                if not useful_transcript:
                    print("[FAM-HRI] Whisper produced no useful instruction; "
                          "using task-specific gaze fallback.")

                self._ensure_pipeline_ready()
                KFInput, PipelineOpts, _, _, process_utterance = self._pipeline
                kf_inputs = [
                    KFInput(
                        timestamp_ns=k.ts_ns,
                        gaze_xy=k.gaze_xy,
                        dwell_s=k.dwell_s,
                        image_rgb=cv2.cvtColor(k.full_bgr, cv2.COLOR_BGR2RGB),
                    )
                    for k in keyframes
                ]
                plan, _ = process_utterance(
                    text,
                    kf_inputs,
                    self._labeler,
                    self._planner,
                    self.args.fam_hri_text_prompt,
                    opts=PipelineOpts(
                        no_llm=self.args.fam_hri_no_llm,
                        planner_context=self.args.fam_hri_planner_context,
                    ),
                    return_annotated=False,
                )
                if not self._is_current_generation(generation):
                    print("[FAM-HRI] Dropping stale session result after reset.")
                    return
                self._enqueue_plan(plan, keyframes)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAM-HRI] Pipeline failed: {exc}")
        finally:
            with self._lock:
                self._processing_count = max(0, self._processing_count - 1)

    def _enqueue_plan(self, plan: dict, keyframes: List[TcpKeyframe]) -> None:
        labels = plan.get("object_labels", [])
        keyframe_labels = plan.get("keyframe_labels", [])
        actions = plan.get("actions", [])
        focused = [
            str(kf.get("gaze_object_label") or "(none)")
            for kf in keyframe_labels
        ]
        if focused:
            print(f"[FAM-HRI] focused keyframes: {focused}")

        items = self._items_for_task_profile(plan, keyframes)
        if items is None:
            subtasks = [
                str(s).strip() for s in plan.get("subtasks", [])
                if str(s).strip()
            ]
            if not subtasks:
                print(f"[FAM-HRI] No subtasks produced for transcript: "
                      f"'{plan.get('transcript', '')}'")
                return
            items = []
            label_use_counts = {}
            for i, instruction in enumerate(subtasks):
                action = actions[i] if i < len(actions) and isinstance(actions[i], dict) else {}
                kf = self._keyframe_for_subtask(
                    i, action, keyframe_labels, keyframes, label_use_counts)
                referent = action.get("object_label") or (labels[0] if labels else "")
                items.append({
                    "instruction": instruction,
                    "keyframe_bgr": kf.crop_bgr.copy(),
                    "referent": referent,
                })

        if not items:
            print(f"[FAM-HRI] No subtasks produced for transcript: "
                  f"'{plan.get('transcript', '')}'")
            return

        subtasks = [it["instruction"] for it in items]
        if self.display is not None:
            self.display.set_batch(items)
        queued = self.queue.extend_items(items)
        print(f"[FAM-HRI] queued {queued} subtask(s): {subtasks}")

    def _items_for_task_profile(self, plan: dict,
                                keyframes: List[TcpKeyframe]) -> Optional[List[dict]]:
        task = getattr(self.args, "task", FAMHRI_DEFAULT_TASK)
        if task == "bottle":
            return self._items_for_bottle_task(plan, keyframes)
        if task == "passing_tools":
            return self._items_for_passing_tools_task(plan, keyframes)
        return None

    def _items_for_bottle_task(self, plan: dict,
                               keyframes: List[TcpKeyframe]) -> List[dict]:
        keyframe_labels = plan.get("keyframe_labels", [])
        transcript = str(plan.get("transcript", "") or "")
        source_indices = _source_keyframe_indices(keyframe_labels)
        if not source_indices and _text_mentions(
            transcript, "all", "beverages", "beverage", "bottles", "bottle"
        ):
            source_indices = list(range(len(keyframes)))
        destination_indices = [
            i for i, meta in enumerate(keyframe_labels)
            if _is_destination_label(_keyframe_meta_label(meta))
        ]
        destination_instruction = _bottle_destination_instruction(
            transcript, keyframe_labels)
        destination_kf = (
            keyframes[destination_indices[-1]]
            if destination_indices and destination_indices[-1] < len(keyframes)
            else None
        )

        items = []
        for idx in source_indices:
            if idx >= len(keyframes):
                continue
            kf = keyframes[idx]
            label = (
                _keyframe_meta_label(keyframe_labels[idx])
                if idx < len(keyframe_labels) else ""
            )
            if label and not _label_mentions(
                label, "bottle", "beverage", "drink", "can", "cup", "tumbler"
            ):
                continue
            obj = _canonical_bottle_object(label, kf.gaze_xy, kf.full_bgr.shape)
            items.append({
                "instruction": f"pick up {obj}",
                "keyframe_bgr": kf.crop_bgr.copy(),
                "referent": obj,
            })
            items.append({
                "instruction": destination_instruction,
                "keyframe_bgr": (
                    destination_kf.crop_bgr.copy()
                    if destination_kf is not None else kf.crop_bgr.copy()
                ),
                "referent": destination_instruction.replace("place it in ", ""),
            })
        if not items and plan.get("subtasks"):
            return None
        return items

    def _items_for_passing_tools_task(self, plan: dict,
                                      keyframes: List[TcpKeyframe]) -> List[dict]:
        keyframe_labels = plan.get("keyframe_labels", [])
        transcript = str(plan.get("transcript", "") or "")
        source_indices = _source_keyframe_indices(keyframe_labels)
        mentioned_tools = _passing_tools_from_text(transcript)
        if not source_indices and mentioned_tools:
            source_indices = list(range(min(len(keyframes), len(mentioned_tools))))

        items = []
        used = set()
        for ordinal, idx in enumerate(source_indices):
            if idx >= len(keyframes):
                continue
            label = (
                _keyframe_meta_label(keyframe_labels[idx])
                if idx < len(keyframe_labels) else ""
            )
            tool = _canonical_passing_tool(label, transcript)
            if tool is None and ordinal < len(mentioned_tools):
                tool = mentioned_tools[ordinal]
            if tool is None:
                continue
            instruction = f"pick up {tool}"
            if (idx, instruction) in used:
                continue
            used.add((idx, instruction))
            items.append({
                "instruction": instruction,
                "keyframe_bgr": keyframes[idx].crop_bgr.copy(),
                "referent": tool,
            })
        if not items and plan.get("subtasks"):
            return None
        return items

    @staticmethod
    def _keyframe_for_subtask(index: int, action: dict, keyframe_labels: list,
                              keyframes: List[TcpKeyframe],
                              label_use_counts: dict) -> TcpKeyframe:
        if not keyframes:
            raise ValueError("no keyframes")
        primitive = str(action.get("primitive") or "").lower()
        source_label = str(action.get("object_label") or "").lower()
        target_label = str(action.get("target_label") or "").lower()

        role = "source"
        lookup_label = source_label
        if primitive in ("put", "pour") and target_label:
            role = "target"
            lookup_label = target_label

        if lookup_label:
            usage_key = (role, lookup_label)
            occurrence = label_use_counts.get(usage_key, 0)
            label_use_counts[usage_key] = occurrence + 1
            match = FamHriTcpSession._keyframe_for_label(
                lookup_label,
                occurrence,
                keyframe_labels,
                keyframes,
                prefer_last=(role == "target"),
            )
            if match is not None:
                return match
        return keyframes[min(index, len(keyframes) - 1)]

    @staticmethod
    def _keyframe_for_label(target_label: str, occurrence: int,
                            keyframe_labels: list,
                            keyframes: List[TcpKeyframe],
                            prefer_last: bool = False) -> Optional[TcpKeyframe]:
        if target_label:
            matches = []
            for label, keyframe in zip(keyframe_labels, keyframes):
                gaze_label = str(label.get("gaze_object_label") or "").lower()
                if FamHriTcpSession._labels_match(gaze_label, target_label):
                    matches.append(keyframe)
            if matches:
                if prefer_last:
                    return matches[-1]
                if occurrence < len(matches):
                    return matches[occurrence]
                return matches[-1]
        return None

    @staticmethod
    def _labels_match(gaze_label: str, target_label: str) -> bool:
        return bool(
            gaze_label and target_label
            and (gaze_label in target_label or target_label in gaze_label)
        )


# ---------------------------------------------------------------------------
# Session keys
# ---------------------------------------------------------------------------

def _handle_key(key: int, session: FamHriTcpSession):
    """Global key handling inside a running episode."""
    if key in (ord('q'), ord('Q'), 27):
        raise ProgramQuit
    if key in (ord('r'), ord('R')):
        raise EpisodeEnd
    if key in (ord('s'), ord('S')):
        session.start()
    if key in (ord('e'), ord('E')):
        session.finish()


def wait_for_start(session: FamHriTcpSession):
    """IDLE state: block until 's' is pressed ('q' quits)."""
    print("[Eval] IDLE - press 's' to start TCP speech/gaze capture "
          "(then speak, 'e' when done). 'q' quits.")
    while True:
        key = _poll_keypress(delay_ms=int(HOLD_POLL_SEC * 1000))
        if key in (ord('s'), ord('S')):
            session.start()
            return
        if key in (ord('q'), ord('Q'), 27):
            raise ProgramQuit
        time.sleep(HOLD_POLL_SEC)


# ---------------------------------------------------------------------------
# Robot control loop
# ---------------------------------------------------------------------------

def _new_policy_client():
    return websocket_client_policy.WebsocketClientPolicy(
        host=POLICY_HOST, port=POLICY_PORT)


def _policy_port_open(timeout: float = 0.3) -> bool:
    import socket
    try:
        with socket.create_connection((POLICY_HOST, POLICY_PORT),
                                      timeout=timeout):
            return True
    except OSError:
        return False


def _connect_policy_responsive(env, session: FamHriTcpSession, render) -> object:
    """Connect to the policy server WITHOUT freezing the GUI/key loop.

    WebsocketClientPolicy blocks until the server is reachable, so we probe
    the TCP port in a loop first - rendering the keyframe viewport and
    handling s/e/r/q keys the whole time - and only construct the client
    once the port accepts connections.
    """
    announced = False
    while not _policy_port_open():
        if not announced:
            print(f"[Eval] policy ws://{POLICY_HOST}:{POLICY_PORT} is NOT up - "
                  "waiting for it (viewport + keys stay active)...")
            announced = True
        render()
        if session.active:
            _observe_tcp(env, session)
        _handle_key(_poll_keypress(delay_ms=200), session)
        time.sleep(0.1)
    if announced:
        print("[Eval] policy server is up - connecting.")
    return _new_policy_client()


def _blank_keyframe_rgb() -> np.ndarray:
    return np.zeros((KEYFRAME_SIZE, KEYFRAME_SIZE, 3), dtype=np.uint8)


def _build_observation(obs: dict, prompt: str,
                       keyframe_rgb: np.ndarray) -> dict:
    images = obs["images"]
    state = np.concatenate([v for v in obs["robot_state"].values()])
    return {
        "observation.images.top_view":                    images["top_camera_rgb"],
        "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
        "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
        "observation.images.human_view_keyframe_cropped": keyframe_rgb,
        "prompt":                                         prompt,
        "observation.state":                              state,
    }


def _observe_tcp(env, session: FamHriTcpSession,
                 audio_drain: bool = True) -> dict:
    obs = env.get_observation(
        mode=TCP_OBS_MODE,
        audio_drain=audio_drain,
    )["timestamp"]
    session.consume_observation(obs)
    return obs


def run_episode(env, queue: SubtaskQueue, args,
                session: FamHriTcpSession,
                display: Optional[KeyframeDisplay] = None):
    """Pop (subtask, keyframe) pairs and execute until EpisodeEnd/ProgramQuit."""
    # Lazy policy connection: WebsocketClientPolicy BLOCKS until the policy
    # server is reachable, so connecting eagerly would freeze the key loop
    # (s/e/r/q dead) whenever :8000 is down. Connect only when there is
    # actually a subtask to execute.
    client       = None
    timer        = Timer(env.control_hz)
    subtask_step = 0
    item         = queue.pop_next()
    keyframe_rgb = _blank_keyframe_rgb()
    waiting_announced = False

    def _render():
        if display is not None:
            display.render()

    def _activate(it: dict):
        nonlocal keyframe_rgb, subtask_step
        subtask_step = 0
        kf = it.get("keyframe_bgr")
        keyframe_rgb = (cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)
                        if kf is not None else _blank_keyframe_rgb())
        if display is not None:
            display.mark_current(it)
        snap = queue.snapshot()
        ref = f" (referent: {it['referent']})" if it.get("referent") else ""
        print(f"[Eval] Subtask: '{it['instruction']}'{ref} "
              f"(pending: {len(snap['pending'])})")

    if item is not None:
        _activate(item)

    while True:
        # ---- no subtask: hold pose, wait for pairs ----
        if item is None:
            if not waiting_announced:
                print("[Eval] Holding pose - waiting for (subtask, keyframe) "
                      "pairs (s/e/r/n/q keys active)...")
                waiting_announced = True
            _render()
            if session.active:
                _observe_tcp(env, session)
                _handle_key(_poll_keypress(), session)
                timer.sleep()
            else:
                _handle_key(_poll_keypress(delay_ms=int(HOLD_POLL_SEC * 1000)),
                            session)
                time.sleep(HOLD_POLL_SEC)
            item = queue.pop_next()
            if item is not None:
                waiting_announced = False
                _activate(item)
            continue

        # ---- observe -> infer ----
        _render()
        if client is None:
            client = _connect_policy_responsive(env, session, _render)
        obs = _observe_tcp(env, session)
        out = client.infer(_build_observation(
            obs, item["instruction"], keyframe_rgb))
        action_chunk    = out["actions"][:args.chunksize, :]
        completion_prob = (float(out["completion_prob"])
                           if "completion_prob" in out else None)
        cp_text = f"{completion_prob:.3f}" if completion_prob is not None else "N/A"
        print(f"[Eval] '{item['instruction']}' step={subtask_step} cp={cp_text}")

        # ---- execute chunk ----
        skip_requested = False
        for raw_action in action_chunk:
            env.step({
                'left_arm_positions':     raw_action[0:6],
                'left_gripper_position':  raw_action[6:7],
                'right_arm_positions':    raw_action[7:13],
                'right_gripper_position': raw_action[13:14],
            })
            subtask_step += 1
            key = _poll_keypress()
            if key in (ord('n'), ord('N')):
                skip_requested = True
                break
            _handle_key(key, session)
            if session.active:
                _observe_tcp(env, session)
            timer.sleep()

        # ---- advance decision ----
        advance = False
        if skip_requested:
            print(f"[Eval] Manual skip of '{item['instruction']}'")
            advance = True
        elif completion_prob is not None and completion_prob >= args.completion_thresh:
            if subtask_step < args.min_steps:
                print(f"  [Auto] cp={completion_prob:.3f} >= "
                      f"{args.completion_thresh} but step={subtask_step} < "
                      f"{args.min_steps} - skip")
            else:
                print(f"  [Auto] cp={completion_prob:.3f} - subtask done.")
                advance = True

        if advance:
            item = queue.pop_next()
            waiting_announced = False
            # Fresh websocket per subtask, mirroring the reference eval scripts
            # (clears any policy-server-side chunk state for the new prompt).
            # Lazy: reconnects on the next infer, keeping keys responsive.
            client = None
            if item is not None:
                _activate(item)
            elif display is not None:
                display.mark_all_current_done()
                _render()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _apply_task_profile(args) -> None:
    profile = TASK_PROFILES[args.task]
    if args.fam_hri_text_prompt is None:
        args.fam_hri_text_prompt = profile["text_prompt"]
    args.fam_hri_planner_context = profile["planner_context"]
    args.fam_hri_task_description = profile["description"]


def _parse_args(default_task: str = FAMHRI_DEFAULT_TASK,
                allow_task_override: bool = True):
    if default_task not in TASK_PROFILES:
        raise ValueError(f"Unknown FAM-HRI task preset: {default_task}")

    p = argparse.ArgumentParser(description="FAM-HRI TCP Aria pair executor.")
    if allow_task_override:
        p.add_argument("--task", choices=sorted(TASK_PROFILES),
                       default=default_task,
                       help="Task preset for SAM labels and policy prompts.")
    p.add_argument("--aria-host", default=ARIA_HOST,
                   help="Laptop IP running FAM-HRI TCP streamer.")
    p.add_argument("--aria-port", type=int, default=ARIA_PORT,
                   help="Laptop TCP streamer port.")
    p.add_argument("--aria-rotation", default=ARIA_ROTATION,
                   help="Gaze coordinate rotation for AriaCamera; FAM-HRI_tcp uses none.")
    p.add_argument("--policy-port", type=int, default=POLICY_PORT)
    p.add_argument("--chunksize", type=int, default=ACTION_CHUNKSIZE)
    p.add_argument("--completion-thresh", type=float, default=COMPLETION_THRESH)
    p.add_argument("--min-steps", type=int, default=MIN_SUBTASK_STEPS,
                   help="Min steps per subtask before completion_prob can advance it.")
    p.add_argument("--fam-hri-root", default=str(FAMHRI_ROOT),
                   help="Path containing the FAM-HRI_tcp fam_hri package.")
    p.add_argument("--fam-hri-device", default=FAMHRI_DEVICE,
                   help="Device for FAM-HRI SAM3 labeler.")
    p.add_argument("--fam-hri-llm-model", default=FAMHRI_LLM_MODEL)
    p.add_argument("--fam-hri-text-prompt", default=None,
                   help="Override the task preset SAM3 concept prompt.")
    p.add_argument("--fam-hri-score-thr", type=float, default=FAMHRI_SCORE_THR)
    p.add_argument("--fam-hri-no-llm", action="store_true", default=FAMHRI_NO_LLM,
                   help="Run FAM-HRI object labeling but skip LLM planning.")
    p.add_argument("--session-min-dwell-s", type=float, default=FAMHRI_MIN_DWELL_S,
                   help="Gaze dwell time that stores a keyframe.")
    p.add_argument("--session-max-disp-px", type=float, default=FAMHRI_MAX_DISP_PX,
                   help="Max gaze drift inside one dwell run.")
    p.add_argument("--keyframe-cooldown-s", type=float, default=FAMHRI_COOLDOWN_S,
                   help="Minimum wall-clock seconds between stored keyframes.")
    p.add_argument("--useful-transcript-min-chars", type=int,
                   default=USEFUL_MIN_CHARS)
    p.add_argument("--show-viewport", action="store_true",
                   help="Show the CV2 keyframe viewport.")
    args = p.parse_args()
    if not allow_task_override:
        args.task = default_task
    _apply_task_profile(args)
    return args


def main(default_task: str = FAMHRI_DEFAULT_TASK,
         allow_task_override: bool = True) -> None:
    args = _parse_args(default_task, allow_task_override=allow_task_override)
    global POLICY_PORT
    POLICY_PORT = args.policy_port

    camera_ids = dict(CAMERA_IDS)
    camera_ids["aria"] = (
        f"{args.aria_host}_{args.aria_port}_{args.aria_rotation}"
    )

    queue   = SubtaskQueue()
    display = KeyframeDisplay() if args.show_viewport else None
    session = FamHriTcpSession(queue, display, args)
    env     = PiperRobotEnv(camera_ids=camera_ids)

    print("[Eval] FAM-HRI pair executor ready. "
          f"policy=ws://{POLICY_HOST}:{POLICY_PORT}  "
          f"aria=tcp://{args.aria_host}:{args.aria_port}  "
          f"completion_thresh={args.completion_thresh}  min_steps={args.min_steps}")
    print(f"[Eval] Task preset: {args.task} - {args.fam_hri_task_description}")
    print("[Eval] Keys: s=start session(speak)  e=end of speech(process)  "
          "n=skip  r=reset episode  q=quit")

    try:
        with _terminal_keypress_mode():
            while True:
                try:
                    wait_for_start(session)     # IDLE until 's'
                    run_episode(env, queue, args, session, display=display)
                except EpisodeEnd:
                    print("[Eval] Episode reset - aborting TCP session, clearing "
                          "queue, back to IDLE.")
                    session.abort()
                    queue.clear()
                    if display is not None:
                        display.clear()
                    env.reset()
                except ProgramQuit:
                    print("[Eval] Quit requested.")
                    session.abort()
                    break
    except KeyboardInterrupt:
        print("\n[Eval] Interrupted.")
        session.abort()
    finally:
        print("[Eval] Cleaning up...")
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        snap = queue.snapshot()
        if snap["done"]:
            print("[Eval] Executed subtasks:")
            for i, s in enumerate(snap["done"], 1):
                print(f"  [{i}] {s}")
        env.safe_end()


if __name__ == "__main__":
    main()
