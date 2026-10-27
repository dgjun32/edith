"""eval_openpi_ours_with_manual_keyframe_realtime.py

Prototype: human-selected keyframes with realtime manual append (no VLM).

Flow per episode
----------------
1. Live preview (same as oracle/VLM flow).
2. Press 'p' to start recording an instruction clip (egocentric aria + audio
   are buffered locally).
3. Press 'p' again to stop. The recorded clip is then opened in an
   interactive scrubber where the operator:
     - navigates frames with a/d (and A/D for x10 jumps), or 'h'/'l'
     - marks a frame as a keyframe with SPACE or 'm'
       (terminal then prompts for the subtask instruction)
     - removes the most recent keyframe with 'u'
     - clears all keyframes with 'c'
     - re-records the clip with 'r'
     - confirms the queue with ENTER ('g') and starts robot execution
     - 'q' quits the program
4. Robot eval loop runs against the chosen keyframe queue. No async VLM
   worker is launched. During the episode, press 'm' or 'a' to append a
   golden keyframe from the current live human-view crop and select its
   instruction from the palette. Manual ('n') advance and completion-prob
   auto advance behave the same as before. 'e' ends the episode.
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from edith.legacy.envs.piper_env_gaze import PiperRobotEnv
from edith.legacy.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from common.config import SERVER_IP

from ours_utils.audio_video_merging import (
    write_video_with_audio as _write_video_with_audio,
)
from ours_utils.command_queue import CmdAppend, KeyframeQueue
from ours_utils.keyboard import (
    poll_keypress as _poll_keypress,
    terminal_keypress_mode as _terminal_keypress_mode,
)
from ours_utils.preview import (
    build_eval_preview as _build_eval_preview,
    build_preview_frame as _build_preview_frame,
)
from ours_utils.rollout_video import save_rollout_video as _save_rollout_video
from ours_utils.video_processing import (
    draw_gaze_crosshair as _draw_gaze_crosshair,
    write_silent_video as _write_video,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS  = 15
_TEMP_DIR          = os.path.normpath("/home/alin/projects/piper_project/temp")
_VLA_KEYFRAME_SIZE = (256, 256)
_DISPLAY_LONG_SIDE = 720   # upscale scrubber/picker frames so longest side = this
_SAVE_FORWARDED_KEYFRAMES = True
_AUTO_ADVANCE_COMPLETION_THRESHOLD = 0.85
_AUTO_ADVANCE_MIN_SUBTASK_STEPS = 40
_AUTO_ADVANCE_REQUIRED_HITS = 1
_REALTIME_APPEND_KEYS = (ord('m'), ord('M'), ord('a'), ord('A'))
_REALTIME_DIRECT_APPEND_KEYS = tuple(ord(str(i)) for i in range(1, 10))

# ---------------------------------------------------------------------------
# Instruction palette — fill these in before each session.
# Shown as clickable buttons when the operator presses SPACE to mark a frame.
# Key 1–9 selects slot 1–9; key 0 selects slot 10.
# ---------------------------------------------------------------------------
INSTRUCTION_PALETTE: List[str] = [
    "Pick up this muffin.",
    "Pick up this bottle.",
    "Put the bottle here.",
    "Pick up this one and pass it to the user.",
    "Give me the metal stand.",
    "Put the green tumbler to the basket near the robot.",
    "Put the pink tumbler to the basket near the human.",
    "Instruction 8",
    "Instruction 9",
    "Instruction 10",
]

PROMPT_TASK = "collaborative_sorting"
TASK_DEFAULT_INSTRUCTIONS = {
    "serving_muffins": "I want this muffin.",
    "sorting_tumblers": "Sort these tumblers.",
    "passing_tools": "I want this tool.",
    "collaborative_sorting": "Sort the packed box into the correct basket.",
}
GLOBAL_INSTRUCTION = False


def _task_default_instruction(prompt_task: Optional[str] = None) -> str:
    return TASK_DEFAULT_INSTRUCTIONS.get(
        prompt_task or PROMPT_TASK, "I want this.")


def _policy_language_instruction(language_instruction: str) -> str:
    if GLOBAL_INSTRUCTION:
        return _task_default_instruction(PROMPT_TASK)
    return language_instruction


def _make_vla_keyframe_bgr(frame_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Use full Aria view for VLA keyframes, resized without center crop."""
    if frame_bgr is None:
        return None
    return cv2.resize(frame_bgr, _VLA_KEYFRAME_SIZE, interpolation=cv2.INTER_AREA)


def _save_forwarded_keyframe_debug(keyframe_img_rgb: Optional[np.ndarray],
                                   step: int,
                                   queue_idx: int) -> None:
    if keyframe_img_rgb is None or not _SAVE_FORWARDED_KEYFRAMES:
        return
    out_dir = os.path.join(_TEMP_DIR, "forwarded_keyframes")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"step_{step:06d}_queue_{queue_idx:02d}_forwarded_rgb.png",
    )
    plt.imsave(out_path, keyframe_img_rgb)


# ---------------------------------------------------------------------------
# Episode control exceptions (mirrors oracle script)
# ---------------------------------------------------------------------------

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


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


# ---------------------------------------------------------------------------
# Recording phase
# ---------------------------------------------------------------------------

def _record_clip(env) -> Tuple[List[np.ndarray], List[np.ndarray],
                                List[dict], List[dict]]:
    """Live preview with a press-to-record toggle.

    Returns four parallel lists:
        aria_frames, aria_cropped_frames, gaze_records, audio_records
    """
    window_name = "Record Clip  |  p=start/stop  q=quit"
    aria_frames: List[np.ndarray] = []
    aria_cropped_frames: List[np.ndarray] = []
    gaze_records: List[dict] = []
    audio_records: List[dict] = []
    capture_timer = Timer(_VIDEO_RECORD_FPS)
    capturing = False

    print("[record] p=start recording  p=stop  q=quit")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation(
                    mode='human_view_streamed_and_keyframe_cropped'))
            status_text = (
                ("RECORDING",
                 f"{len(aria_frames) / _VIDEO_RECORD_FPS:.1f}s  p=stop  q=quit")
                if capturing
                else ("READY", "p=start  q=quit")
            )
            preview = _build_preview_frame(
                live_obs, status_text,
                status_color=(0, 0, 255) if capturing else (0, 255, 0))

            aria_bgr = live_obs["images"]["aria_rgb"].copy()
            cropped_bgr = live_obs["images"]["aria_cropped_rgb"].copy()
            gaze_pt = np.asarray(
                live_obs["gaze_point"], dtype=np.float64).reshape(-1)
            gx = float(gaze_pt[0]) if gaze_pt.size >= 1 else None
            gy = float(gaze_pt[1]) if gaze_pt.size >= 2 else None
            audio_obs = live_obs.get("audio", {})
            audio_chunk = np.asarray(
                audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
                dtype=np.int16,
            )
            audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
            audio_sample_rate_hz = int(
                audio_obs.get("sample_rate_hz", 0) or 0)
            audio_num_samples = int(
                audio_obs.get("num_samples", audio_chunk.size) or 0)

            if capturing:
                idx = len(gaze_records)
                aria_frames.append(aria_bgr)
                aria_cropped_frames.append(cropped_bgr)
                gaze_records.append(
                    {"frame_idx": idx, "gaze_x": gx, "gaze_y": gy})
                audio_records.append({
                    "frame_idx": idx,
                    "audio_ts_ns": audio_ts_ns,
                    "sample_rate_hz": audio_sample_rate_hz,
                    "num_samples": audio_num_samples,
                    "chunk_pcm16": audio_chunk.copy(),
                })

            cv2.imshow(window_name, preview)
            key = _poll_keypress()
            capture_timer.sleep()

            if key in (ord('q'), ord('Q'), 27):
                raise ProgramQuit

            if key in (ord('p'), ord('P')):
                if not capturing:
                    aria_frames.clear()
                    aria_cropped_frames.clear()
                    gaze_records.clear()
                    audio_records.clear()
                    capturing = True
                    print("[record] capturing...")
                elif aria_frames:
                    capturing = False
                    duration = len(aria_frames) / _VIDEO_RECORD_FPS
                    print(f"[record] captured {len(aria_frames)} frames "
                          f"({duration:.2f}s)")
                    return (aria_frames, aria_cropped_frames,
                            gaze_records, audio_records)
                # else: pressed p with no frames yet — ignore.
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Unified keyframe selection window
# ---------------------------------------------------------------------------
#
# Single 1920×1080 canvas, split into zones:
#
#  ┌──────────────────────────────────────────┬─────────────────────────┐  ← TOP_H
#  │  egocentric view  │  gaze crop           │  instruction buttons    │
#  │                   │                      │  (click to mark frame)  │
#  │       FRAME_W // 2 each                  │       INSTR_W           │
#  ├──────────────────────────────────────────┴─────────────────────────┤  ← MAIN_H
#  │  TIMELINE  ──────●──────▲──────▲──────────────────────────────── │
#  ├────────────────────────────────────────────────────────────────────┤  ← TL_H
#  │  marked keyframe thumbnails  (click→jump, [X]→delete)             │
#  ├────────────────────────────────────────────────────────────────────┤  ← STRIP_H
#  │  controls hint bar                                                 │
#  └────────────────────────────────────────────────────────────────────┘  ← BOT_H

# Canvas dimensions
_U_W        = 1920
_U_H        = 1080
_U_TOP_H    = 55       # title / status bar
_U_INSTR_W  = 650      # right instruction panel
_U_FRAME_W  = _U_W - _U_INSTR_W     # 1270 px for the two camera frames
_U_MAIN_H   = 665      # height shared by frame viewer + instruction panel
_U_TL_H     = 105      # timeline section
_U_STRIP_H  = 190      # marked-keyframe strip
_U_BOT_H    = 65       # controls hint
# total: 55+665+105+190+65 = 1080 ✓

# Timeline geometry (absolute canvas y)
_U_TL_Y0    = _U_TOP_H + _U_MAIN_H                   # 720
_U_TL_BAR_Y = _U_TL_Y0 + 62                           # centre of the bar track
_U_TL_BAR_H = 26                                      # track height
_U_TL_X0    = 90                                      # bar left x
_U_TL_X1    = _U_W - 90                               # bar right x

# Instruction button slot geometry (within instruction panel)
_U_BTN_AREA_Y0 = _U_TOP_H + 54     # first button top y
_U_BTN_SLOT_H  = (_U_MAIN_H - 58) // 10   # ~60 px per slot
_U_BTN_H       = _U_BTN_SLOT_H - 8        # button height

# Colours
_C_BTN_BG     = (42, 42, 42)
_C_BTN_HOVER  = (50, 115, 50)
_C_BTN_BORDER = (90, 170, 90)
_C_BTN_TEXT   = (225, 225, 225)
_C_TL_TRACK   = (50, 50, 50)
_C_TL_FILL    = (35, 75, 75)
_C_TL_HEAD    = (0, 230, 230)
_C_KF_MARKER  = (60, 200, 60)
_C_SEL        = (60, 190, 60)

# Unified mouse state
_um: dict = {"x": 0, "y": 0, "click": False, "drag_x": -1}


def _um_cb(event, x, y, flags, param):
    _um["x"], _um["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        _um["click"] = True
    if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        _um["drag_x"] = x
    if event == cv2.EVENT_LBUTTONUP:
        _um["drag_x"] = -1


def _fit_in_box(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Letterbox-resize img into a box_w × box_h black canvas."""
    h, w = img.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    yo, xo = (box_h - nh) // 2, (box_w - nw) // 2
    out[yo:yo + nh, xo:xo + nw] = resized
    return out


def _fidx_to_x(fidx: int, total: int) -> int:
    if total <= 1:
        return _U_TL_X0
    return int(_U_TL_X0 + fidx / (total - 1) * (_U_TL_X1 - _U_TL_X0))


def _x_to_fidx(px: int, total: int) -> int:
    frac = (px - _U_TL_X0) / max(1, _U_TL_X1 - _U_TL_X0)
    return int(np.clip(frac * (total - 1), 0, total - 1))


def _draw_gaze_at(img: np.ndarray,
                  gx: Optional[float], gy: Optional[float]) -> np.ndarray:
    """Draw gaze crosshair at (gx, gy) on the full egocentric frame."""
    out = img.copy()
    h, w = out.shape[:2]
    if gx is None or gy is None:
        return out
    cx = int(np.clip(gx, 0, w - 1))
    cy = int(np.clip(gy, 0, h - 1))
    r  = max(14, h // 22)
    cv2.circle(out, (cx, cy), r + 5, (0, 0, 200), 3)
    cv2.circle(out, (cx, cy), r,     (0, 230, 0), 3)
    cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), (0, 230, 0), 3)
    cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), (0, 230, 0), 3)
    label = "GAZE"
    scale = max(0.5, h / 380.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    lx = cx - tw // 2
    ly = min(cy + r * 2 + th + 8, h - 4)
    cv2.rectangle(out, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 4),
                  (0, 0, 0), -1)
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 230, 230), 2, cv2.LINE_AA)
    return out


def _choose_realtime_instruction(keyframe_bgr: np.ndarray,
                                 default_instruction: str) -> Optional[str]:
    """Pause eval and let the operator choose an instruction for this keyframe."""
    win = "Append Realtime Keyframe"
    state = {"click": None}

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, _cb)

    button_rects = []
    try:
        while True:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            canvas[:] = (24, 24, 24)
            preview = _fit_in_box(keyframe_bgr, 720, 640)
            canvas[60:700, 30:750] = preview
            cv2.putText(canvas, "Realtime golden keyframe",
                        (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (180, 230, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Choose instruction: click button or press 1-9/0",
                        (790, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, "d = default/current instruction    ESC = cancel    q = quit",
                        (790, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (150, 150, 150), 1, cv2.LINE_AA)

            button_rects = []
            y0 = 95
            for i, label in enumerate(INSTRUCTION_PALETTE):
                by0 = y0 + i * 55
                by1 = by0 + 44
                bx0, bx1 = 790, 1250
                button_rects.append((bx0, by0, bx1, by1, label))
                cv2.rectangle(canvas, (bx0, by0), (bx1, by1),
                              (42, 58, 42), -1)
                cv2.rectangle(canvas, (bx0, by0), (bx1, by1),
                              (95, 145, 95), 1)
                key_label = str(i + 1) if i < 9 else "0"
                cv2.putText(canvas, f"[{key_label}] {label[:46]}",
                            (bx0 + 12, by0 + 29),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (235, 235, 235), 1, cv2.LINE_AA)

            cv2.imshow(win, canvas)

            if state["click"] is not None:
                mx, my = state["click"]
                state["click"] = None
                for bx0, by0, bx1, by1, label in button_rects:
                    if bx0 <= mx <= bx1 and by0 <= my <= by1:
                        return label

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (27,):
                return None
            if key in (ord('q'), ord('Q')):
                raise ProgramQuit
            if key in (ord('d'), ord('D')):
                return default_instruction
            if ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(INSTRUCTION_PALETTE):
                    return INSTRUCTION_PALETTE[idx]
            if key == ord('0') and len(INSTRUCTION_PALETTE) >= 10:
                return INSTRUCTION_PALETTE[9]
    finally:
        cv2.destroyWindow(win)


def _append_realtime_keyframe(queue: KeyframeQueue,
                              keyframe_bgr: np.ndarray,
                              instruction: str,
                              step: int) -> str:
    keyframe = keyframe_bgr.copy()
    log = queue.apply(CmdAppend(keyframe=keyframe, instruction=instruction))

    out_dir = os.path.join(_TEMP_DIR, "realtime_manual_keyframes")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step:06d}_append.png")
    cv2.imwrite(path, keyframe)
    with open(os.path.join(out_dir, "events.jsonl"), "a") as f:
        f.write(json.dumps({
            "step": int(step),
            "queue_idx": int(queue.total - 1),
            "instruction": instruction,
            "keyframe_path": path,
        }, ensure_ascii=False) + "\n")

    print(f"[realtime] appended golden keyframe at step {step}: "
          f"'{instruction}' -> {path}")
    return log


def _instruction_from_direct_append_key(key: int) -> Optional[str]:
    if ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(INSTRUCTION_PALETTE):
            return INSTRUCTION_PALETTE[idx]
    return None


def _show_realtime_instruction_choices() -> None:
    win = "Realtime Instruction Choices"
    canvas = np.zeros((520, 780, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)

    cv2.putText(canvas, "Press 1-9 to append current keyframe",
                (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (180, 230, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "m/a = picker   n = next   e = end episode   q = quit",
                (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (170, 170, 170), 1, cv2.LINE_AA)

    y = 110
    for i, label in enumerate(INSTRUCTION_PALETTE[:9]):
        key_label = str(i + 1)
        cv2.rectangle(canvas, (18, y - 28), (760, y + 16),
                      (38, 50, 38), -1)
        cv2.rectangle(canvas, (18, y - 28), (760, y + 16),
                      (80, 130, 80), 1)
        cv2.putText(canvas, f"{key_label}. {label[:78]}",
                    (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (235, 235, 235), 1, cv2.LINE_AA)
        y += 45

    cv2.imshow(win, canvas)


def _draw_unified(aria_bgr: np.ndarray,
                  cropped_bgr: np.ndarray,
                  idx: int,
                  total: int,
                  keyframes: List[np.ndarray],
                  instructions: List[str],
                  frame_indices: List[int],
                  sel_kf: int = -1,
                  gaze_pt: Tuple[Optional[float], Optional[float]] = (None, None),
                  ) -> np.ndarray:
    canvas = np.full((_U_H, _U_W, 3), 18, dtype=np.uint8)
    mx, my = _um["x"], _um["y"]
    n_kf   = len(keyframes)
    t_sec  = idx / max(_VIDEO_RECORD_FPS, 1)
    dur    = total / _VIDEO_RECORD_FPS

    # ── TOP BAR ────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (_U_W, _U_TOP_H - 1), (28, 28, 28), -1)
    cv2.putText(canvas, "KEYFRAME SELECTOR",
                (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (80, 200, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"frame {idx + 1}/{total}   t={t_sec:.2f}s / {dur:.1f}s",
                (290, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"marked: {n_kf}",
                (_U_W - 220, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                _C_SEL, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, _U_TOP_H), (_U_W, _U_TOP_H), (50, 50, 50), 1)

    # ── FRAME VIEWER ───────────────────────────────────────────────────────
    half_w  = _U_FRAME_W // 2
    fy0, fy1 = _U_TOP_H, _U_TOP_H + _U_MAIN_H
    
    #aria_bgr    = _draw_gaze_at(aria_bgr, gaze_pt[0], gaze_pt[1])
    cropped_bgr = _draw_gaze_crosshair(cropped_bgr)
    aria_box    = _fit_in_box(aria_bgr,    half_w, _U_MAIN_H)
    cropped_box = _fit_in_box(cropped_bgr, half_w, _U_MAIN_H)
    canvas[fy0:fy1, 0:half_w]          = aria_box
    canvas[fy0:fy1, half_w:_U_FRAME_W] = cropped_box
    cv2.putText(canvas, "Egocentric view",
                (12, fy0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Gaze crop",
                (half_w + 12, fy0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(canvas, (half_w, fy0), (half_w, fy1), (55, 55, 55), 2)
    cv2.line(canvas, (_U_FRAME_W, fy0), (_U_FRAME_W, fy1), (55, 55, 55), 2)

    # ── INSTRUCTION PANEL ──────────────────────────────────────────────────
    ix0 = _U_FRAME_W
    cv2.rectangle(canvas, (ix0, fy0), (_U_W, fy1), (22, 22, 22), -1)
    cv2.putText(canvas, "Click to mark this frame:",
                (ix0 + 14, fy0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (140, 200, 140), 1, cv2.LINE_AA)

    for i, label in enumerate(INSTRUCTION_PALETTE):
        by0 = _U_BTN_AREA_Y0 + i * _U_BTN_SLOT_H
        by1 = by0 + _U_BTN_H
        bx0, bx1 = ix0 + 10, _U_W - 10
        hov = bx0 <= mx <= bx1 and by0 <= my <= by1
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1),
                      _C_BTN_HOVER if hov else _C_BTN_BG, -1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), _C_BTN_BORDER, 1)
        key_label = str(i + 1) if i < 9 else "0"
        cv2.putText(canvas, f"[{key_label}]  {label}",
                    (bx0 + 16, by0 + _U_BTN_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, _C_BTN_TEXT,
                    1, cv2.LINE_AA)

    # ── TIMELINE ───────────────────────────────────────────────────────────
    tl_y0 = _U_TL_Y0
    tl_y1 = tl_y0 + _U_TL_H
    cv2.rectangle(canvas, (0, tl_y0), (_U_W, tl_y1), (22, 22, 22), -1)
    cv2.line(canvas, (0, tl_y0), (_U_W, tl_y0), (50, 50, 50), 1)
    cv2.putText(canvas, "TIMELINE",
                (14, tl_y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (120, 120, 120), 1, cv2.LINE_AA)

    bar_top = _U_TL_BAR_Y - _U_TL_BAR_H // 2
    bar_bot = _U_TL_BAR_Y + _U_TL_BAR_H // 2

    # Track + filled region
    cv2.rectangle(canvas, (_U_TL_X0, bar_top), (_U_TL_X1, bar_bot),
                  _C_TL_TRACK, -1)
    cur_x = _fidx_to_x(idx, total)
    cv2.rectangle(canvas, (_U_TL_X0, bar_top), (cur_x, bar_bot),
                  _C_TL_FILL, -1)
    cv2.rectangle(canvas, (_U_TL_X0, bar_top), (_U_TL_X1, bar_bot),
                  (75, 75, 75), 1)

    # Time tick labels
    n_ticks = min(10, total)
    for ti in range(n_ticks + 1):
        fi = int(ti / n_ticks * (total - 1))
        lx = _fidx_to_x(fi, total)
        cv2.line(canvas, (lx, bar_bot), (lx, bar_bot + 7), (90, 90, 90), 1)
        cv2.putText(canvas, f"{fi / _VIDEO_RECORD_FPS:.1f}s",
                    (lx - 14, bar_bot + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110),
                    1, cv2.LINE_AA)

    # Keyframe markers: filled triangle above the bar + index number
    for ki, fidx in enumerate(frame_indices):
        kx = _fidx_to_x(fidx, total)
        tri = np.array([
            [kx,      bar_top - 2],
            [kx - 9,  bar_top - 20],
            [kx + 9,  bar_top - 20],
        ])
        cv2.fillPoly(canvas, [tri], _C_KF_MARKER)
        cv2.putText(canvas, str(ki + 1),
                    (kx - 5, bar_top - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180),
                    1, cv2.LINE_AA)

    # Playhead: vertical line + diamond
    cv2.line(canvas, (cur_x, tl_y0 + 4), (cur_x, tl_y1 - 4),
             _C_TL_HEAD, 3)
    diamond = np.array([
        [cur_x,      bar_top - 5],
        [cur_x - 8,  _U_TL_BAR_Y],
        [cur_x,      bar_bot + 5],
        [cur_x + 8,  _U_TL_BAR_Y],
    ])
    cv2.fillPoly(canvas, [diamond], _C_TL_HEAD)

    # Hover time tooltip when mouse is over timeline
    if tl_y0 <= my <= tl_y1 and _U_TL_X0 <= mx <= _U_TL_X1:
        hf = _x_to_fidx(mx, total)
        cv2.putText(canvas, f"{hf / _VIDEO_RECORD_FPS:.2f}s",
                    (mx + 10, _U_TL_BAR_Y - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210),
                    1, cv2.LINE_AA)

    # ── MARKED KEYFRAME STRIP ──────────────────────────────────────────────
    st_y0 = tl_y1
    st_y1 = st_y0 + _U_STRIP_H
    cv2.rectangle(canvas, (0, st_y0), (_U_W, st_y1), (26, 26, 26), -1)
    cv2.line(canvas, (0, st_y0), (_U_W, st_y0), (50, 50, 50), 1)

    if n_kf == 0:
        cv2.putText(canvas,
                    "No keyframes marked yet — "
                    "click an instruction button on the right to mark the current frame",
                    (20, st_y0 + _U_STRIP_H // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (90, 90, 90),
                    1, cv2.LINE_AA)
    else:
        pad     = 10
        cell_w  = min(210, (_U_W - pad) // n_kf - pad)
        thumb_h = _U_STRIP_H - 46

        for ki, (kf, instr, fidx) in enumerate(
                zip(keyframes, instructions, frame_indices)):
            cx0 = pad + ki * (cell_w + pad)
            cx1 = cx0 + cell_w
            cy0 = st_y0 + 8
            cy1 = cy0 + thumb_h

            is_selected = (ki == sel_kf)
            is_active   = (fidx == idx)
            if is_selected:
                border_col, thickness = (0, 200, 255), 4   # orange-ish = selected
            elif is_active:
                border_col, thickness = (0, 220, 220), 3   # cyan = playhead here
            else:
                border_col, thickness = _C_SEL, 1

            thumb = cv2.resize(kf, (cell_w, thumb_h),
                               interpolation=cv2.INTER_AREA)
            canvas[cy0:cy0 + thumb_h, cx0:cx0 + cell_w] = thumb
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), border_col, thickness)
            if is_selected:
                cv2.putText(canvas, "SELECTED",
                            (cx0 + 4, cy0 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 200, 255), 1, cv2.LINE_AA)

            # instruction label
            cv2.putText(canvas, f"#{ki + 1} {instr[:22]}",
                        (cx0 + 4, cy1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                        1, cv2.LINE_AA)

            # [X] delete button (top-right corner of thumb)
            dx0, dx1 = cx1 - 26, cx1
            dy0, dy1 = cy0, cy0 + 24
            del_hov = dx0 <= mx <= dx1 and dy0 <= my <= dy1
            cv2.rectangle(canvas, (dx0, dy0), (dx1, dy1),
                          (180, 50, 50) if del_hov else (90, 25, 25), -1)
            cv2.putText(canvas, "X", (dx0 + 6, dy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 120, 120),
                        1, cv2.LINE_AA)

    # ── CONTROLS BAR ───────────────────────────────────────────────────────
    bot_y0 = st_y1
    cv2.rectangle(canvas, (0, bot_y0), (_U_W, _U_H), (20, 20, 20), -1)
    cv2.line(canvas, (0, bot_y0), (_U_W, bot_y0), (50, 50, 50), 1)
    cv2.putText(canvas,
                "← → / a d = prev/next frame    "
                "click thumbnail = select    u / Del = delete selected (or last)    "
                "c = clear all    r = re-record    ENTER = proceed    q = quit",
                (16, bot_y0 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140),
                1, cv2.LINE_AA)

    return canvas


def manual_keyframe_selection(aria_frames: List[np.ndarray],
                              aria_cropped_frames: List[np.ndarray],
                              gaze_records: List[dict],
                              default_instruction: str,
                              ) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """Unified single-window scrubber. Returns (keyframes, instructions,
    frame_indices). Raises _ReRecord or ProgramQuit."""
    if not aria_frames:
        raise ValueError("no frames to select from")

    total = len(aria_frames)
    idx   = 0
    keyframes:     List[np.ndarray] = []
    instructions:  List[str]        = []
    frame_indices: List[int]        = []

    WIN = "Keyframe Selector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, _U_W, _U_H)
    cv2.moveWindow(WIN, 0, 0)
    cv2.setMouseCallback(WIN, _um_cb)
    _um["click"] = False
    _um["drag_x"] = -1

    sel_kf: int = -1   # index of the selected keyframe in the strip (-1 = none)

    def _mark(instr: str):
        nonlocal sel_kf
        kf = _make_vla_keyframe_bgr(aria_frames[idx])
        keyframes.append(kf)
        instructions.append(instr)
        frame_indices.append(idx)
        sel_kf = len(keyframes) - 1   # auto-select the newly marked keyframe
        print(f"[manual] KF #{len(keyframes)} @ frame {idx}  → '{instr}'")

    def _delete(ki: int):
        nonlocal sel_kf
        keyframes.pop(ki)
        instructions.pop(ki)
        frame_indices.pop(ki)
        sel_kf = min(ki, len(keyframes) - 1) if keyframes else -1
        print(f"[manual] deleted KF #{ki + 1}")

    print("[manual] Unified scrubber ready. "
          "Navigate with ← →, mark by clicking an instruction button.")
    try:
        while True:
            gr     = gaze_records[idx] if idx < len(gaze_records) else {}
            gaze_pt = (gr.get("gaze_x"), gr.get("gaze_y"))
            canvas = _draw_unified(
                aria_frames[idx], aria_cropped_frames[idx],
                idx, total, keyframes, instructions, frame_indices,
                sel_kf, gaze_pt)
            cv2.imshow(WIN, canvas)

            # ── Timeline drag-to-scrub ──────────────────────────────────
            drag_x = _um["drag_x"]
            if (drag_x != -1
                    and _U_TL_X0 <= drag_x <= _U_TL_X1
                    and _U_TL_Y0 <= _um["y"] <= _U_TL_Y0 + _U_TL_H):
                idx = _x_to_fidx(drag_x, total)

            # ── Mouse click dispatch ────────────────────────────────────
            if _um["click"]:
                _um["click"] = False
                mx, my = _um["x"], _um["y"]

                # Instruction button?
                if mx >= _U_FRAME_W:
                    for i, label in enumerate(INSTRUCTION_PALETTE):
                        by0 = _U_BTN_AREA_Y0 + i * _U_BTN_SLOT_H
                        by1 = by0 + _U_BTN_H
                        if (_U_FRAME_W + 10 <= mx <= _U_W - 10
                                and by0 <= my <= by1):
                            _mark(label)
                            break

                # Timeline click?
                elif (_U_TL_Y0 <= my <= _U_TL_Y0 + _U_TL_H
                      and _U_TL_X0 <= mx <= _U_TL_X1):
                    idx = _x_to_fidx(mx, total)

                # Keyframe strip click — select or delete
                elif (_U_TL_Y0 + _U_TL_H <= my
                      <= _U_TL_Y0 + _U_TL_H + _U_STRIP_H
                      and len(keyframes) > 0):
                    n_kf    = len(keyframes)
                    pad     = 10
                    cell_w  = min(210, (_U_W - pad) // n_kf - pad)
                    thumb_h = _U_STRIP_H - 46
                    st_y0   = _U_TL_Y0 + _U_TL_H
                    for ki in range(n_kf):
                        cx0 = pad + ki * (cell_w + pad)
                        cx1 = cx0 + cell_w
                        cy0 = st_y0 + 8
                        if cx0 <= mx <= cx1:
                            dx0, dy0, dy1 = cx1 - 26, cy0, cy0 + 24
                            if dx0 <= mx <= cx1 and dy0 <= my <= dy1:
                                _delete(ki)        # [X] hit
                            else:
                                sel_kf = ki        # select + jump
                                idx = frame_indices[ki]
                            break

            # ── Keyboard ───────────────────────────────────────────────
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord('q'), ord('Q')):
                raise ProgramQuit
            if key in (ord('r'), ord('R')):
                cv2.destroyAllWindows()
                raise _ReRecord
            if key in (10, 13, ord('g'), ord('G')):
                if not keyframes:
                    print("[manual] Mark at least one frame first.")
                    continue
                return keyframes, instructions, frame_indices
            if key in (81, ord('a'), ord('h')):
                idx = max(0, idx - 1)
            elif key in (83, ord('d'), ord('l')):
                idx = min(total - 1, idx + 1)
            elif key in (ord('u'), ord('U'), 127, 255 & 0x7F):  # u / Del
                if keyframes:
                    target = sel_kf if 0 <= sel_kf < len(keyframes) else len(keyframes) - 1
                    _delete(target)
            elif key in (ord('c'), ord('C')):
                keyframes.clear(); instructions.clear()
                frame_indices.clear(); sel_kf = -1
                print("[manual] cleared all keyframes")
            elif ord('1') <= key <= ord('9'):
                i = key - ord('1')
                if i < len(INSTRUCTION_PALETTE):
                    _mark(INSTRUCTION_PALETTE[i])
            elif key == ord('0') and len(INSTRUCTION_PALETTE) >= 10:
                _mark(INSTRUCTION_PALETTE[9])
    finally:
        cv2.destroyAllWindows()


class _ReRecord(Exception):
    """Internal sentinel: scrubber wants to re-record the clip."""


# ---------------------------------------------------------------------------
# Queue review screen
# ---------------------------------------------------------------------------

_REVIEW_W        = 1920
_REVIEW_H        = 1080
_LIST_W          = 520       # left panel width
_BOTTOM_H        = 100       # height of the proceed/back bar
_ROW_H           = 150       # height of each queue row in the list
_THUMB_W         = 120
_THUMB_H         = 120
_SEL_COLOR       = (60, 180, 60)
_SEL_BG          = (30, 60, 30)
_NORMAL_BG       = (30, 30, 30)
_DEL_COLOR       = (50, 50, 200)
_PROCEED_COLOR   = (30, 160, 30)
_BACK_COLOR      = (80, 80, 80)

# Shared review mouse state (re-used across renders)
_rev_mouse_pos: List[int]  = [0, 0]
_rev_mouse_click: List[bool] = [False]


def _review_mouse_cb(event, x, y, flags, param):
    _rev_mouse_pos[0], _rev_mouse_pos[1] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        _rev_mouse_click[0] = True


def _draw_review_canvas(keyframes: List[np.ndarray],
                         instructions: List[str],
                         sel: int) -> np.ndarray:
    """Render the full queue-review canvas (1400×900)."""
    canvas = np.full((_REVIEW_H, _REVIEW_W, 3), 18, dtype=np.uint8)
    right_x = _LIST_W
    content_h = _REVIEW_H - _BOTTOM_H

    # ── Left panel: scrollable list ─────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (_LIST_W - 1, content_h),
                  (28, 28, 28), -1)
    cv2.putText(canvas, f"Queue  ({len(keyframes)} keyframes)",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (200, 200, 200), 1, cv2.LINE_AA)

    mx, my = _rev_mouse_pos
    for i, (kf, instr) in enumerate(zip(keyframes, instructions)):
        y0 = 46 + i * _ROW_H
        y1 = y0 + _ROW_H - 4
        if y1 > content_h:
            break

        is_sel   = (i == sel)
        is_hover = (0 <= mx < _LIST_W and y0 <= my <= y1)
        bg = _SEL_BG if is_sel else (_NORMAL_BG if not is_hover
                                      else (40, 40, 40))
        cv2.rectangle(canvas, (2, y0), (_LIST_W - 2, y1), bg, -1)
        if is_sel:
            cv2.rectangle(canvas, (2, y0), (_LIST_W - 2, y1),
                          _SEL_COLOR, 3)

        # thumbnail
        thumb = cv2.resize(kf, (_THUMB_W, _THUMB_H),
                           interpolation=cv2.INTER_AREA)
        ty = y0 + (_ROW_H - _THUMB_H) // 2
        canvas[ty:ty + _THUMB_H, 10:10 + _THUMB_W] = thumb

        # slot number + instruction text
        tx = 10 + _THUMB_W + 12
        cv2.putText(canvas, f"#{i + 1}",
                    (tx, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    _SEL_COLOR if is_sel else (160, 200, 160),
                    1, cv2.LINE_AA)
        # word-wrap instruction into two lines of ≤22 chars
        words = instr.split()
        line1, line2 = "", ""
        for w in words:
            if len(line1) + len(w) + 1 <= 22:
                line1 = (line1 + " " + w).strip()
            else:
                line2 = (line2 + " " + w).strip()
        cv2.putText(canvas, line1,
                    (tx, y0 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (220, 220, 220), 1, cv2.LINE_AA)
        if line2:
            cv2.putText(canvas, line2[:24],
                        (tx, y0 + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (220, 220, 220), 1, cv2.LINE_AA)

        # [X] delete button (top-right of row)
        dx0 = _LIST_W - 46
        dx1 = _LIST_W - 8
        dy0, dy1 = y0 + 6, y0 + 38
        del_hover = (dx0 <= mx <= dx1 and dy0 <= my <= dy1)
        cv2.rectangle(canvas, (dx0, dy0), (dx1, dy1),
                      (160, 60, 60) if del_hover else (90, 30, 30), -1)
        cv2.putText(canvas, "X", (dx0 + 8, dy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (220, 120, 120), 1, cv2.LINE_AA)

    # vertical divider
    cv2.line(canvas, (_LIST_W, 0), (_LIST_W, content_h),
             (60, 60, 60), 1)

    # ── Right panel: large preview of selected keyframe ─────────────────
    preview_x = right_x + 20
    preview_w = _REVIEW_W - right_x - 40
    preview_h = content_h - 120

    if keyframes and 0 <= sel < len(keyframes):
        kf = keyframes[sel]
        # scale to fit preview area maintaining aspect ratio
        kf_h, kf_w = kf.shape[:2]
        scale = min(preview_w / kf_w, preview_h / kf_h)
        disp_w = int(kf_w * scale)
        disp_h = int(kf_h * scale)
        disp = cv2.resize(kf, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        px = preview_x + (preview_w - disp_w) // 2
        py = 50
        canvas[py:py + disp_h, px:px + disp_w] = disp
        cv2.rectangle(canvas, (px - 2, py - 2),
                      (px + disp_w + 2, py + disp_h + 2),
                      _SEL_COLOR, 2)

        # slot label
        cv2.putText(canvas,
                    f"Keyframe {sel + 1} / {len(keyframes)}",
                    (preview_x, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (180, 220, 180), 1, cv2.LINE_AA)

        # instruction text (large)
        instr_text = instructions[sel]
        cv2.putText(canvas, instr_text[:70],
                    (preview_x, py + disp_h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if len(instr_text) > 70:
            cv2.putText(canvas, instr_text[70:140],
                        (preview_x, py + disp_h + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "(no keyframes)", (preview_x + 40, content_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (120, 120, 120), 1, cv2.LINE_AA)

    # controls hint (right panel, bottom)
    cv2.putText(canvas,
                "j/k or Up/Down = navigate    Del/x = remove    "
                "b/ESC = back to scrubber    ENTER = proceed",
                (right_x + 10, content_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (140, 140, 140), 1, cv2.LINE_AA)

    # ── Bottom bar: BACK and PROCEED buttons ────────────────────────────
    bar_y = content_h
    cv2.rectangle(canvas, (0, bar_y), (_REVIEW_W, _REVIEW_H),
                  (22, 22, 22), -1)
    cv2.line(canvas, (0, bar_y), (_REVIEW_W, bar_y), (60, 60, 60), 1)

    # BACK button
    bk_x0, bk_y0 = 24, bar_y + 14
    bk_x1, bk_y1 = 320, bar_y + _BOTTOM_H - 14
    bk_hover = (bk_x0 <= mx <= bk_x1 and bk_y0 <= my <= bk_y1)
    cv2.rectangle(canvas, (bk_x0, bk_y0), (bk_x1, bk_y1),
                  (120, 120, 120) if bk_hover else _BACK_COLOR, -1)
    cv2.putText(canvas, "< Back to Scrubber",
                (bk_x0 + 18, bk_y0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (220, 220, 220), 1, cv2.LINE_AA)

    # PROCEED button
    pr_x0 = _REVIEW_W - 420
    pr_x1 = _REVIEW_W - 24
    pr_y0, pr_y1 = bar_y + 12, bar_y + _BOTTOM_H - 12
    pr_hover = (pr_x0 <= mx <= pr_x1 and pr_y0 <= my <= pr_y1)
    pr_enabled = len(keyframes) > 0
    pr_col = ((50, 220, 50) if pr_hover
               else (_PROCEED_COLOR if pr_enabled else (40, 80, 40)))
    cv2.rectangle(canvas, (pr_x0, pr_y0), (pr_x1, pr_y1), pr_col, -1)
    cv2.putText(canvas,
                "PROCEED  (start robot)" if pr_enabled else "(add keyframes first)",
                (pr_x0 + 24, pr_y0 + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (230, 255, 230) if pr_enabled else (100, 100, 100),
                1, cv2.LINE_AA)

    return canvas


def _review_queue(keyframes: List[np.ndarray],
                  instructions: List[str],
                  frame_indices: List[int],
                  ) -> Tuple[List[np.ndarray], List[str], List[int], bool]:
    """Interactive queue review screen.

    Returns (keyframes, instructions, frame_indices, proceed).
      proceed=True  → start robot inference
      proceed=False → return to scrubber
    Raises ProgramQuit on 'q'.
    """
    # Work on mutable copies so removals don't affect the caller on Back.
    kfs    = list(keyframes)
    instrs = list(instructions)
    fidxs  = list(frame_indices)
    sel    = 0

    win = "Queue Review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, _REVIEW_W, _REVIEW_H)
    cv2.moveWindow(win, 0, 0)
    cv2.setMouseCallback(win, _review_mouse_cb)
    _rev_mouse_click[0] = False

    print("[review] Queue review open. "
          "ENTER or click PROCEED to start robot, b/ESC to go back.")
    try:
        while True:
            canvas = _draw_review_canvas(kfs, instrs, sel)
            cv2.imshow(win, canvas)

            n   = len(kfs)
            mx  = _rev_mouse_pos[0]
            my  = _rev_mouse_pos[1]

            if _rev_mouse_click[0]:
                _rev_mouse_click[0] = False
                content_h = _REVIEW_H - _BOTTOM_H
                bar_y     = content_h

                # PROCEED button
                pr_x0 = _REVIEW_W - 260
                pr_x1 = _REVIEW_W - 20
                pr_y0 = bar_y + 10
                pr_y1 = bar_y + _BOTTOM_H - 10
                if pr_x0 <= mx <= pr_x1 and pr_y0 <= my <= pr_y1 and n > 0:
                    return kfs, instrs, fidxs, True

                # BACK button
                bk_x0, bk_y0 = 20, bar_y + 12
                bk_x1, bk_y1 = 200, bar_y + _BOTTOM_H - 12
                if bk_x0 <= mx <= bk_x1 and bk_y0 <= my <= bk_y1:
                    return kfs, instrs, fidxs, False

                # List row click (select or delete)
                if mx < _LIST_W:
                    for i in range(n):
                        y0 = 46 + i * _ROW_H
                        y1 = y0 + _ROW_H - 4
                        if y1 > content_h:
                            break
                        if y0 <= my <= y1:
                            # [X] delete zone
                            dx0 = _LIST_W - 46
                            dy0, dy1 = y0 + 6, y0 + 38
                            if dx0 <= mx <= _LIST_W - 6 and dy0 <= my <= dy1:
                                kfs.pop(i)
                                instrs.pop(i)
                                fidxs.pop(i)
                                sel = min(sel, max(0, len(kfs) - 1))
                                print(f"[review] removed keyframe #{i + 1}")
                            else:
                                sel = i
                            break

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord('q'), ord('Q')):
                raise ProgramQuit
            if key in (27, ord('b'), ord('B')):          # ESC / b → back
                return kfs, instrs, fidxs, False
            if key in (10, 13):                           # ENTER → proceed
                if n > 0:
                    return kfs, instrs, fidxs, True
            if key in (82, ord('k'), ord('K')):           # up
                sel = max(0, sel - 1)
            if key in (84, ord('j'), ord('J')):           # down
                sel = min(max(0, n - 1), sel + 1)
            if key in (255 & 127, ord('x'), ord('X'), 8):  # DEL / x / BS
                if 0 <= sel < n:
                    kfs.pop(sel)
                    instrs.pop(sel)
                    fidxs.pop(sel)
                    sel = min(sel, max(0, len(kfs) - 1))
                    print(f"[review] removed selected keyframe")
    finally:
        cv2.destroyWindow(win)


# ---------------------------------------------------------------------------
# End-to-end keyframe extraction (no VLM)
# ---------------------------------------------------------------------------

def extract_keyframes_manual(env, language_instruction: str,
                             ) -> KeyframeQueue:
    """Record → scrub → mark → review → proceed. Returns queue."""
    os.makedirs(_TEMP_DIR, exist_ok=True)
    while True:
        aria_frames, aria_cropped_frames, gaze_records, audio_records = \
            _record_clip(env)

        # Persist clip for debugging
        try:
            _write_video_with_audio(
                aria_frames, audio_records,
                os.path.join(_TEMP_DIR, "video.mp4"),
                fps=_VIDEO_RECORD_FPS)
            _write_video(
                aria_cropped_frames,
                os.path.join(_TEMP_DIR, "cropped_video.mp4"),
                fps=_VIDEO_RECORD_FPS)
            with open(os.path.join(_TEMP_DIR, "gaze.json"), "w") as f:
                json.dump({"frames": gaze_records}, f, indent=2)
        except Exception as exc:
            print(f"[manual] Failed to persist clip: {exc}")

        try:
            kfs, instrs, frame_indices = manual_keyframe_selection(
                aria_frames, aria_cropped_frames, gaze_records,
                default_instruction=language_instruction,
            )
        except _ReRecord:
            print("[manual] re-recording...")
            continue

        # Queue review — loop until Proceed or Back-to-scrubber
        while True:
            kfs, instrs, frame_indices, proceed = _review_queue(
                kfs, instrs, frame_indices)
            if proceed:
                break
            # Back → re-open scrubber on the same clip (no re-record)
            try:
                kfs, instrs, frame_indices = manual_keyframe_selection(
                    aria_frames, aria_cropped_frames, gaze_records,
                    default_instruction=language_instruction,
                )
            except _ReRecord:
                # User hit 'r' inside scrubber → break inner loop,
                # outer loop re-records.
                break

        if not kfs:
            # Can happen if user deleted everything then backed out.
            print("[manual] no keyframes — re-recording.")
            continue

        # Save keyframe images + metadata
        saved_items = []
        for i1, (kf, instr, fidx) in enumerate(
                zip(kfs, instrs, frame_indices), start=1):
            path = os.path.join(_TEMP_DIR, f"manual_keyframe_{i1:02d}.png")
            cv2.imwrite(path, kf)
            saved_items.append({
                "order": i1,
                "frame_idx": int(fidx),
                "keyframe_path": path,
                "keyframe_source": "aria_rgb_full_resized_256",
                "instruction": instr,
            })
            print(f"[manual] #{i1} (frame {fidx}) '{instr}' → {path}")

        with open(os.path.join(_TEMP_DIR, "keyframes.json"), "w") as f:
            json.dump({
                "instruction": language_instruction,
                "num_frames": len(aria_frames),
                "video_fps": _VIDEO_RECORD_FPS,
                "subtask_instructions": instrs,
                "saved_keyframes": saved_items,
                "source": "manual",
            }, f, indent=2, ensure_ascii=False)

        return KeyframeQueue(kfs, instrs)


# ---------------------------------------------------------------------------
# Eval loop (no async VLM worker)
# ---------------------------------------------------------------------------

def eval(env, client, language_instruction: str, horizon: int,
         queue: KeyframeQueue, chunksize: int = 8):
    if queue is None:
        raise RuntimeError("[eval] queue is not initialized.")

    top_view_frames_list: List[np.ndarray] = []
    comp_prob_log: List[Tuple[int, float]] = []
    keyframe_log: List[Tuple[int, np.ndarray]] = []
    global_step = 0
    subtask_step = 0
    completion_hit_count = 0
    timer = Timer(env.control_hz)
    window_name = "Eval  |  1-9=append kf  m/a=picker  n=next  e=end  q=quit"
    rollout_video_path = os.path.join(_TEMP_DIR, "rollout.mp4")
    queue_empty_notified = False

    kf = queue.current_keyframe
    if kf is not None:
        keyframe_log.append((0, kf.copy()))

    def _manual_append_from_live(live_obs, default_instr: str) -> bool:
        keyframe_bgr = _make_vla_keyframe_bgr(
            live_obs["images"]["aria_rgb"])
        instr = _choose_realtime_instruction(keyframe_bgr, default_instr)
        if not instr:
            print("[realtime] append cancelled.")
            return False
        _append_realtime_keyframe(queue, keyframe_bgr, instr, global_step)
        return True

    def _direct_append_from_live(live_obs, instruction: str) -> bool:
        keyframe_bgr = _make_vla_keyframe_bgr(
            live_obs["images"]["aria_rgb"])
        _append_realtime_keyframe(queue, keyframe_bgr, instruction, global_step)
        return True

    for step in range(horizon):
        try:
            if queue.is_done:
                if not queue_empty_notified:
                    print("[eval] queue empty — robot holding. "
                          "Press 1-9 to append, m/a for picker, e=end, q=quit.")
                    queue_empty_notified = True
                live_obs = _extract_latest_observation(
                    env.get_observation(
                        mode='human_view_streamed_and_keyframe_cropped',
                        audio_drain=False,
                    ))
                top_view_frames_list.append(
                    live_obs["images"]["top_camera_rgb"].copy())
                preview = _build_eval_preview(
                    live_obs, None, queue, "QUEUE EMPTY - press 1-9 or m/a to append")
                cv2.imshow(window_name, preview)
                _show_realtime_instruction_choices()
                key = _poll_keypress()
                direct_instr = _instruction_from_direct_append_key(key)
                if direct_instr:
                    _direct_append_from_live(live_obs, direct_instr)
                    subtask_step = 0
                    completion_hit_count = 0
                    queue_empty_notified = False
                    kf = queue.current_keyframe
                    if kf is not None:
                        keyframe_log.append((global_step, kf.copy()))
                    continue
                if key in _REALTIME_APPEND_KEYS:
                    if _manual_append_from_live(live_obs, language_instruction):
                        subtask_step = 0
                        completion_hit_count = 0
                        queue_empty_notified = False
                        kf = queue.current_keyframe
                        if kf is not None:
                            keyframe_log.append((global_step, kf.copy()))
                    continue
                _raise_for_episode_control(key)
                timer.sleep()
                continue

            queue_empty_notified = False

            obs = env.get_observation(
                mode="human_view_streamed_and_keyframe_cropped",
                audio_drain=False,
            )["timestamp"]
            images = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for v in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, v])

            keyframe_img = queue.current_keyframe
            current_instr = queue.current_instruction
            policy_prompt = _policy_language_instruction(
                current_instr or language_instruction)

            if keyframe_img is not None:
                keyframe_img_rgb = cv2.cvtColor(keyframe_img,
                                                cv2.COLOR_BGR2RGB)
            else:
                keyframe_img_rgb = None

            observation = {
                "observation.images.top_view":
                    images["top_camera_rgb"],
                "observation.images.left_wrist_view":
                    images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":
                    images["right_wrist_camera_rgb"],
                "observation.images.human_view_keyframe":
                    keyframe_img_rgb,
                "prompt": policy_prompt,
                "observation.state": concatenated_state,
            }

            _save_forwarded_keyframe_debug(
                keyframe_img_rgb, global_step, queue.current_idx)

            out = client.infer(observation)
            action_chunk = out["actions"][:chunksize, :]
            completion_prob = (float(out["completion_prob"])
                               if "completion_prob" in out else None)
            print("Completion prob:",
                  f"{completion_prob:.3f}"
                  if completion_prob is not None else "N/A")
            if completion_prob is not None:
                comp_prob_log.append((global_step, completion_prob))

            # Check completion BEFORE executing actions.
            # If threshold is met, skip the action chunk and re-infer
            # immediately with the next subtask's keyframe.
            if completion_prob is not None:
                if completion_prob >= _AUTO_ADVANCE_COMPLETION_THRESHOLD:
                    completion_hit_count += 1
                else:
                    completion_hit_count = 0

                if subtask_step < _AUTO_ADVANCE_MIN_SUBTASK_STEPS:
                    if completion_prob >= _AUTO_ADVANCE_COMPLETION_THRESHOLD:
                        print(
                            f"  [auto] cp={completion_prob:.3f} >= "
                            f"{_AUTO_ADVANCE_COMPLETION_THRESHOLD:.2f} but "
                            f"subtask_step={subtask_step} < "
                            f"{_AUTO_ADVANCE_MIN_SUBTASK_STEPS} — skip")
                elif completion_hit_count < _AUTO_ADVANCE_REQUIRED_HITS:
                    print(
                        f"  [auto] cp={completion_prob:.3f} hit "
                        f"{completion_hit_count}/"
                        f"{_AUTO_ADVANCE_REQUIRED_HITS} — wait")
                else:
                    print(f"  [auto] cp={completion_prob:.3f} "
                          f"hit {completion_hit_count}/"
                          f"{_AUTO_ADVANCE_REQUIRED_HITS} "
                          f"(subtask_step={subtask_step}) — advancing "
                          f"without executing actions...")
                    if not queue.advance():
                        print("[auto] queue exhausted — holding until realtime append.")
                        subtask_step = 0
                        completion_hit_count = 0
                        continue
                    client = websocket_client_policy.WebsocketClientPolicy(
                        host="localhost", port=8000)
                    subtask_step = 0
                    completion_hit_count = 0
                    kf = queue.current_keyframe
                    if kf is not None:
                        keyframe_log.append((global_step, kf.copy()))
                    print(f"  [auto] → KF "
                          f"[{queue.current_idx + 1}/{queue.total}]:"
                          f" '{queue.current_instruction}'")
                    continue  # re-infer with updated subtask, skip action execution

            for raw_action in action_chunk:
                action = {
                    'left_arm_positions':     raw_action[0:6],
                    'left_gripper_position':  raw_action[6:7],
                    'right_arm_positions':    raw_action[7:13],
                    'right_gripper_position': raw_action[13:14],
                }
                env.step(action)
                global_step += 1
                subtask_step += 1

                live_obs = _extract_latest_observation(
                    env.get_observation(
                        mode='human_view_streamed_and_keyframe_cropped',
                        audio_drain=False,
                    ))
                top_view_frames_list.append(
                    live_obs["images"]["top_camera_rgb"].copy())

                preview = _build_eval_preview(
                    live_obs, keyframe_img, queue, current_instr)
                cv2.imshow(window_name, preview)
                _show_realtime_instruction_choices()
                key = _poll_keypress()

                direct_instr = _instruction_from_direct_append_key(key)
                if direct_instr:
                    _direct_append_from_live(live_obs, direct_instr)
                elif key in _REALTIME_APPEND_KEYS:
                    _manual_append_from_live(
                        live_obs, current_instr or language_instruction)
                elif key in (ord('n'), ord('N')):
                    if not queue.advance():
                        print("[eval] manual advance — queue empty.")
                    else:
                        subtask_step = 0
                        completion_hit_count = 0
                        kf = queue.current_keyframe
                        if kf is not None:
                            keyframe_log.append((global_step, kf.copy()))
                        print(f"  → manual KF "
                              f"[{queue.current_idx + 1}/{queue.total}]:"
                              f" '{queue.current_instruction}'")
                    break
                else:
                    _raise_for_episode_control(key)

                timer.sleep()

        except (EpisodeEnd, ProgramQuit, KeyboardInterrupt) as exc:
            cv2.destroyAllWindows()
            if isinstance(exc, EpisodeEnd):
                env.reset()
            _save_rollout_video(
                top_view_frames_list, comp_prob_log, rollout_video_path,
                fps=_VIDEO_RECORD_FPS,
                vlm_event_log=[],
                keyframe_log=keyframe_log,
                event_display_frames=0)
            if isinstance(exc, KeyboardInterrupt):
                raise ProgramQuit
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()
    _save_rollout_video(top_view_frames_list, comp_prob_log,
                        rollout_video_path,
                        fps=_VIDEO_RECORD_FPS,
                        vlm_event_log=[],
                        keyframe_log=keyframe_log,
                        event_display_frames=0)


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------

def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Manual keyframe eval with realtime golden-keyframe append — no VLM.")
    parser.add_argument(
        "--prompt-task", default=PROMPT_TASK,
        help="Default task name used to pick the fallback instruction.")
    parser.add_argument(
        "--instruction", default=None,
        help="Top-level language instruction (default: task default).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    PROMPT_TASK = args.prompt_task

    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = (
        args.instruction or _task_default_instruction(PROMPT_TASK))
    horizon = 10000

    print("Starting realtime manual-keyframe eval.")
    print("Top-level instruction:", language_instruction)
    print("VLA keyframe source: aria_rgb full frame resized to",
          _VLA_KEYFRAME_SIZE)
    print("Record controls : p=start/stop  q=quit")
    print("Scrubber controls: a/d navigate, A/D x10, SPACE mark, "
          "u undo, c clear, r re-record, ENTER go, q quit")
    print("Eval controls   : 1-9=append matching palette instruction  "
          "m/a=append with picker  n=next keyframe  e=end episode  q=quit")

    def _options_menu(lang_instr: str) -> Optional[str]:
        print(f"\nCurrent instruction: '{lang_instr}'")
        print("Options:")
        print("  1. Press Enter         → continue with current instruction")
        print("  2. Type new instruction → update and continue")
        print("  3. Type 'quit'          → exit")
        try:
            user_input = input("Enter your choice: ").strip()
        except EOFError:
            return None
        if user_input.lower() == "quit":
            return None
        return user_input if user_input else lang_instr

    try:
        while True:
            try:
                with _terminal_keypress_mode():
                    queue = extract_keyframes_manual(
                        env, language_instruction)

                client = websocket_client_policy.WebsocketClientPolicy(
                    host="localhost", port=8000)

                with _terminal_keypress_mode():
                    eval(env, client, language_instruction, horizon, queue)
                print("Episode finished.")
            except EpisodeEnd:
                print("Episode ended. Robot reset; moving to next episode.")
                continue
            except ProgramQuit:
                print("Exiting evaluation...")
                break

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
        os.makedirs(_TEMP_DIR, exist_ok=True)
        print(f"Preserved temp directory: {_TEMP_DIR}")
        env.safe_end()
