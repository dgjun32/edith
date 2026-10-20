"""Validated queue edits and timestamp-based keyframe selection."""

from dataclasses import dataclass
import numpy as np
from .processing import rgb


@dataclass
class Frame:
    timestamp_ns: int
    image: np.ndarray
    keyframe: np.ndarray
    gaze_xy: object = None
    full_rgb: object = None


@dataclass
class Task:
    instruction: str
    keyframe: np.ndarray | None


def element_task(element, frames):
    instruction = element.get("subtask_instruction", element.get("instruction", ""))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Queue item needs a nonempty subtask_instruction")
    if not frames:
        raise ValueError("Planner result has no matching captured frames")
    interval = element.get("keyframe_interval")
    if interval is None:
        raise ValueError("Planner item needs a keyframe_interval")
    start, end = float(interval["start"]), float(interval["end"])
    duration = (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1e9
    if not np.isfinite([start, end]).all() or not 0 <= start <= end <= duration + 0.001:
        raise ValueError(
            f"Keyframe interval must lie in the captured clip [0,{duration:.3f}]"
        )
    target = frames[0].timestamp_ns + (start + end) * 0.5e9
    selected = min(frames, key=lambda f: abs(f.timestamp_ns - target))
    return Task(instruction.strip(), rgb(selected.keyframe).copy())


class TaskQueue:
    def __init__(self, tasks=(), max_items=64):
        self.items = list(tasks)
        self.index = 0
        self.revision = 0
        self.version = 0
        self.max_items = max_items

    @property
    def current(self):
        return self.items[self.index] if self.index < len(self.items) else None

    def advance(self):
        if self.current is not None:
            self.index += 1
            self.revision += 1
            self.version += 1

    def apply(self, response, frames):
        if not isinstance(response, dict):
            raise ValueError("Planner response must be an object")
        # Validate the entire operation before modifying the queue.
        command = response.get(
            "command", "Append" if "action_queue" in response else None
        )
        if command == "NoOp":
            return
        values = response.get("new_elements", response.get("action_queue", []))
        if "new_element" in response:
            values = [response["new_element"]]
        if not isinstance(values, list):
            raise ValueError("new_elements/action_queue must be a list")
        tasks = [element_task(v, frames) for v in values]
        items = list(self.items)
        if command == "Append":
            items.extend(tasks)
        elif command == "Preempt":
            if len(tasks) != 1:
                raise ValueError("Preempt requires one new element")
            items[self.index : self.index] = tasks
        elif command == "Replace":
            index = response.get("target_index", response.get("target_idx"))
            if (
                type(index) is not int
                or not self.index <= index < len(items)
                or len(tasks) != 1
            ):
                raise ValueError(
                    "Replace requires one valid unfinished target and one element"
                )
            items[index] = tasks[0]
        elif command == "Remove":
            indices = response.get("target_indices", [])
            if not isinstance(indices, list) or any(
                type(i) is not int or not self.index < i < len(items) for i in indices
            ):
                raise ValueError("Remove can target only future items")
            for index in sorted(set(indices), reverse=True):
                items.pop(index)
        else:
            raise ValueError(f"Unsupported planner command: {command!r}")
        if len(items) > self.max_items:
            raise ValueError("Planner queue exceeds max_queue_items")
        # Unchanged current item can keep executing an existing action chunk.
        previous = self.current
        self.items = items
        self.version += 1
        if self.current is not previous:
            self.revision += 1

    def snapshot(self):
        return [
            {
                "index": i,
                "instruction": t.instruction,
                "current": i == self.index,
                "keyframe": None if t.keyframe is None else t.keyframe.copy(),
            }
            for i, t in enumerate(self.items)
        ]
