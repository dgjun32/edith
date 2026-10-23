"""Queue command types and thread-safe keyframe queue."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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


@dataclass
class RawCommand:
    name: str
    target_idx: Optional[int]
    local_frame_idx: int
    instruction: str
    keyframe_description: str = ""


@dataclass
class RawAppendElement:
    local_frame_idx: int
    instruction: str
    keyframe_description: str = ""


@dataclass
class RawAppendMany:
    elements: List[RawAppendElement]


class KeyframeQueue:
    """Thread-safe ordered list of (keyframe_bgr, instruction) pairs."""

    def __init__(self, keyframes: List[np.ndarray], instructions: List[str]):
        assert len(keyframes) == len(instructions)
        self._lock = threading.Lock()
        self._items: List[Tuple[np.ndarray, str]] = list(zip(keyframes, instructions))
        self._current_idx = 0

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
        with self._lock:
            result = []
            for idx, (img, instr) in enumerate(self._items):
                result.append({
                    "keyframe_image": img.copy() if img is not None else None,
                    "instruction": instr,
                    "is_current": (idx == self._current_idx),
                })
            return result

    def queue_state_str(self) -> str:
        with self._lock:
            if not self._items:
                return "  (empty)"
            lines = []
            for idx, (_, instr) in enumerate(self._items):
                marker = ">>> " if idx == self._current_idx else "    "
                lines.append(f"  [{idx}] {marker}{instr}")
            return "\n".join(lines)

    def advance(self) -> bool:
        """Move to the next item. Returns True if a next item exists."""
        with self._lock:
            if self._current_idx + 1 < len(self._items):
                self._current_idx += 1
                return True
            self._current_idx = len(self._items)
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


def format_cmd_label(cmd) -> str:
    """Short display string for a VLM queue command."""
    if isinstance(cmd, CmdNoOp):
        return "[VLM] NoOp"
    if isinstance(cmd, CmdAppend):
        return f"[VLM] Append: {cmd.instruction}"
    if isinstance(cmd, CmdAppendMany):
        preview = ", ".join(cmd.instructions[:3])
        if len(cmd.instructions) > 3:
            preview += ", ..."
        return f"[VLM] Append {len(cmd.instructions)}: {preview}"
    if isinstance(cmd, CmdReplace):
        return f"[VLM] Replace[{cmd.target_idx}]: {cmd.instruction}"
    if isinstance(cmd, CmdPreempt):
        return f"[VLM] PREEMPT: {cmd.instruction}"
    if isinstance(cmd, CmdRemove):
        return f"[VLM] Remove{cmd.target_indices}"
    return ""
