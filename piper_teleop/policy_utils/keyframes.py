"""Shared keyframe policy state machine for Edith and VLM-TextVLA variants."""

from collections import deque
import numpy as np
import cv2
from .chunked import ChunkedPolicy, Job
from .processing import gaze_crop, local_file, packet, resize, rgb
from .tasks import Frame, Task, TaskQueue


class KeyframePolicy(ChunkedPolicy):
    use_keyframe = True
    planner_source = "aria"

    def __init__(self, policy_dir, config, *, client=None, planner=None):
        config = dict(config)
        config.setdefault("planner_source", self.planner_source)
        self.use_keyframe = bool(config.get("use_keyframe", self.use_keyframe))
        super().__init__(policy_dir, config, client=client)
        self.planner = planner
        self.plan_job = None
        self.plan_generation = 0
        self.window_s = float(config.get("planner_interval_s", 8))
        self.planner_fps = float(config.get("planner_fps", 4))
        self.planner_timeout = float(config.get("planner_timeout_s", 30))
        self.threshold = float(config.get("completion_threshold", 0.8))
        self.min_steps = int(config.get("min_subtask_steps", 40))
        self.queue_limit = int(config.get("max_queue_items", 64))
        if (
            not all(
                np.isfinite(x) and x > 0
                for x in (self.window_s, self.planner_fps, self.planner_timeout)
            )
            or not 0 <= self.threshold <= 1
            or self.min_steps < 0
            or not 1 <= self.queue_limit <= 256
        ):
            raise ValueError("Invalid planner timing, completion or queue settings")
        self.frames = deque(
            maxlen=min(256, max(2, int(self.window_s * self.planner_fps) + 2))
        )
        self.task_queue = TaskQueue(max_items=self.queue_limit)
        self.subtask_steps = 0
        self.initial_plan = True
        self.last_transcript = ""
        self.active_revision = -1
        self.last_frame_ns = -1
        self.manual_instruction = None
        self.last_event_id = None
        backend = config.get("planner", "manual")
        if backend not in ("manual", "gemini", "fam_hri"):
            raise ValueError("planner must be manual, gemini or fam_hri")
        if self.planner is None and backend == "gemini":
            from .planning import GeminiPlanner

            self.planner = GeminiPlanner(policy_dir, config)
        elif self.planner is None and backend == "fam_hri":
            from .fam_hri import FamHRIPlanner

            self.planner = FamHRIPlanner(config)

    def reset(self, observation):
        super().reset(observation)
        self.plan_generation += 1
        self.task_queue = TaskQueue(max_items=self.queue_limit)
        self.frames.clear()
        self.last_frame_ns = -1
        self.manual_instruction = None
        self.last_event_id = None
        self.active_revision = -1
        self.subtask_steps = 0
        self.initial_plan = True
        self.last_transcript = ""
        tasks = []
        for item in self.config.get("subtasks", []):
            instruction = item.get("instruction", "").strip()
            if not instruction:
                raise ValueError("Each configured subtask needs instruction")
            frame = None
            if item.get("keyframe"):
                path = local_file(self.policy_dir, item["keyframe"])
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot read keyframe: {path}")
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif self.use_keyframe:
                frame = self._keyframe(observation)
            tasks.append(Task(instruction, frame))
        if len(tasks) > self.queue_limit:
            raise ValueError("Too many configured subtasks")
        self.task_queue.items = tasks

    def _keyframe(self, observation):
        image = rgb(observation["images"]["aria_rgb"])
        if self.config.get("gaze_crop", True):
            image = gaze_crop(
                image,
                observation.get("gaze_point"),
                self.config.get("crop_size", 512),
                self.config.get("invalid_gaze", "full"),
            )
        return resize(image, self.image_size)

    def _record_frame(self, observation):
        source = self.config.get("planner_source", self.planner_source)
        if source not in ("aria", "top"):
            raise ValueError("planner_source must be aria or top")
        timestamp = int(observation["capture_timestamps_ns"][source])
        if timestamp <= self.last_frame_ns or (
            self.last_frame_ns >= 0
            and timestamp - self.last_frame_ns < 1e9 / self.planner_fps
        ):
            return
        image = (
            self._keyframe(observation)
            if source == "aria"
            else resize(observation["images"]["top_camera_rgb"], self.image_size)
        )
        self.frames.append(
            Frame(
                timestamp,
                image,
                self._keyframe(observation),
                observation.get("gaze_point"),
                rgb(observation["images"]["aria_rgb"]).copy()
                if self.config.get("planner") == "fam_hri"
                else None,
            )
        )
        self.last_frame_ns = timestamp

    def _update_plan(self, observation):
        self._record_frame(observation)
        if self.plan_job is not None:
            epoch, revision, frames, initial, job = self.plan_job
            done, response = job.poll(self.planner_timeout + 1)
            if done:
                self.plan_job = None
                if epoch == self.plan_generation:
                    if not isinstance(response, dict):
                        raise ValueError("Planner response must be an object")
                    command = response.get("command", "Append")
                    # Index-sensitive edits are invalid after execution advanced.
                    if revision == self.task_queue.version or command in (
                        "Append",
                        "NoOp",
                    ):
                        self.task_queue.apply(response, frames)
                        self.initial_plan = False
        if self.planner is not None and self.plan_job is None and len(self.frames) > 1:
            duration = (
                self.frames[-1].timestamp_ns - self.frames[0].timestamp_ns
            ) / 1e9
            if duration >= self.window_s:
                frames = list(self.frames)
                self.frames.clear()  # Non-overlapping clips, even across reused camera samples.
                text = str(
                    observation.get("transcript")
                    or observation.get("instruction")
                    or self.config.get("instruction", "")
                )
                # A repeated live transcription is not a new spoken intervention.
                transcript = (
                    text if self.initial_plan or text != self.last_transcript else ""
                )
                self.last_transcript = text
                snapshot = self.task_queue.snapshot()
                job = Job(
                    self.planner.plan, frames, snapshot, transcript, self.initial_plan
                )
                self.plan_job = (
                    self.plan_generation,
                    self.task_queue.version,
                    frames,
                    self.initial_plan,
                    job,
                )

    def _prepare(self, observation):
        if self.planner is None and not self.config.get("subtasks"):
            instruction = self._instruction(observation)
            if instruction and instruction != self.manual_instruction:
                self.manual_instruction = instruction
                self.task_queue.items[self.task_queue.index :] = [
                    Task(instruction, self._keyframe(observation))
                ]
                self.task_queue.revision += 1
                self.task_queue.version += 1
        event = observation.get("policy_event")
        if event is not None and event.get("id") != self.last_event_id:
            if not event.get("id"):
                raise ValueError("policy_event needs a unique nonempty id")
            if event.get("command") == "next":
                self.task_queue.advance()
            elif event.get("command") in (
                "Append",
                "Preempt",
                "Replace",
                "Remove",
                "NoOp",
            ):
                # An explicit operator edit uses this synchronized keyframe.
                frame = Frame(
                    0, self._keyframe(observation), self._keyframe(observation)
                )
                self.task_queue.apply(event, [frame])
            else:
                raise ValueError("Unsupported policy_event command")
            self.last_event_id = event["id"]
        self._update_plan(observation)
        if self.task_queue.revision != self.active_revision:
            self.active_revision = self.task_queue.revision
            self.generation += 1
            self.actions.clear()
            self.subtask_steps = 0
        task = self.task_queue.current
        if task is None:
            self.current_prompt = None
            self.status = {"phase": "waiting_for_subtasks"}
            return None
        self.current_prompt = task.instruction
        request = packet(observation, task.instruction, self.image_size)
        if self.use_keyframe:
            if task.keyframe is None:
                raise ValueError("A visual keyframe is required for this policy")
            keyframe = task.keyframe
            if self.config.get("black_keyframe", False):
                keyframe = np.zeros_like(keyframe)
            request["observation.images.human_view_keyframe_cropped"] = resize(
                keyframe, self.image_size
            )
        return request

    def act(self, observation):
        action = super().act(observation)
        if self.status.get("phase") == "executing":
            self.subtask_steps += 1
        self.status.update(
            queue_index=self.task_queue.index, queue_size=len(self.task_queue.items)
        )
        return action

    def _chunk_finished(self):
        # Include the final action, whose act() has not incremented the count yet.
        if (
            self.completion is not None
            and self.completion >= self.threshold
            and self.subtask_steps + 1 >= self.min_steps
        ):
            self.task_queue.advance()

    def close(self):
        try:
            super().close()
        finally:
            if self.planner is not None:
                self.planner.close()
