"""Nonblocking, chunked inference adapted from the legacy evaluation loops."""

from collections import deque
import queue
import threading
import time
import numpy as np
from ..base import BasePolicy
from .openpi import OpenPIClient
from .processing import gaze_crop, packet, resize, state


class Job:
    """One bounded background request. The caller alone mutates policy state."""

    def __init__(self, function, *args):
        self.started = time.monotonic()
        self.result = queue.Queue(maxsize=1)

        def run():
            try:
                self.result.put((True, function(*args)))
            except BaseException as error:
                self.result.put((False, error))

        self.thread = threading.Thread(target=run, daemon=True, name="policy-inference")
        self.thread.start()

    def poll(self, timeout):
        try:
            ok, value = self.result.get_nowait()
        except queue.Empty:
            if time.monotonic() - self.started > timeout:
                raise TimeoutError("Policy inference exceeded configured timeout")
            return False, None
        if not ok:
            raise RuntimeError("Policy inference failed") from value
        return True, value


class ChunkedPolicy(BasePolicy):
    """One action per act(); runner owns all cameras, CAN, rate and recording."""

    view = "text"

    def __init__(self, policy_dir, config, *, client=None):
        self.policy_dir, self.config = policy_dir, dict(config)
        self.chunk_size = int(config.get("chunk_size", 8))
        self.image_size = config.get("image_size", 224)
        self.timeout = float(config.get("timeout_s", 10))
        self.max_age = float(config.get("max_action_age_s", 2))
        self.max_step = float(config.get("max_joint_step_rad", 0.1))
        self.max_gripper_step = float(config.get("max_gripper_step", 0.2))
        if not 1 <= self.chunk_size <= 256 or not all(
            np.isfinite(x) and x > 0
            for x in (self.timeout, self.max_age, self.max_step, self.max_gripper_step)
        ):
            raise ValueError("Invalid chunk size, timeout, action age or step limit")
        self.client = client if client is not None else OpenPIClient(config)
        self.pending = None
        self.actions = deque()
        self.generation = 0
        self.last_action = None
        self.current_prompt = None
        self.completion = None
        self.closed = False
        self.status = {"phase": "idle"}

    def reset(self, observation):
        self.generation += 1
        self.actions.clear()
        self.last_action = state(observation).copy()
        self.current_prompt = None
        self.completion = None
        self.status = {"phase": "ready"}

    def _instruction(self, observation):
        return str(
            observation.get("transcript")
            or observation.get("instruction")
            or self.config.get("instruction", "")
        ).strip()

    def _prepare(self, observation):
        instruction = self._instruction(observation)
        if instruction != self.current_prompt:
            self.current_prompt = instruction
            self.generation += 1
            self.actions.clear()
        if not instruction:
            self.status = {"phase": "waiting_for_instruction"}
            return None
        request = packet(observation, instruction, self.image_size)
        if self.view == "stream":
            image = gaze_crop(
                observation["images"]["aria_rgb"],
                observation.get("gaze_point"),
                self.config.get("crop_size", 512),
                self.config.get("invalid_gaze", "full"),
            )
            request["observation.images.human_view_gaze_cropped"] = resize(
                image, self.image_size
            )
        return request

    def _chunk_finished(self):
        pass

    def _validate_response(self, result):
        actions = np.asarray(result.get("actions"), dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "Inference actions must be a nonempty finite (T,14) YAM joint-position array"
            )
        if np.any((actions[:, [6, 13]] < 0) | (actions[:, [6, 13]] > 1)):
            raise ValueError("YAM gripper actions must be normalized to [0,1]")
        cp = result.get("completion_prob")
        if cp is not None:
            cp = float(cp)
            if not np.isfinite(cp) or not 0 <= cp <= 1:
                raise ValueError("completion_prob must be in [0,1]")
        self.completion = cp
        return actions[: self.chunk_size].copy()

    def act(self, observation):
        if self.closed:
            raise RuntimeError("Policy is closed")
        state(observation)
        if self.last_action is None:
            self.reset(observation)
        request = self._prepare(observation)
        if self.pending is not None:
            generation, job = self.pending
            done, result = job.poll(self.timeout + 1)
            if done:
                self.pending = None
                if generation == self.generation:
                    if time.monotonic() - job.started > self.max_age:
                        raise TimeoutError("Inference result is too old to execute")
                    self.actions.extend(self._validate_response(result))
                    self.chunk_started = job.started
        if request is None:
            return self.last_action.copy()
        if self.actions:
            if time.monotonic() - self.chunk_started > self.max_age:
                raise TimeoutError("Buffered action chunk expired")
            action = self.actions.popleft()
            delta = np.abs(action - self.last_action)
            limits = np.full(14, self.max_step)
            limits[[6, 13]] = self.max_gripper_step
            if np.any(delta > limits):
                raise ValueError(
                    "YAM action exceeds configured per-step delta; verify checkpoint, joint order and rate"
                )
            self.last_action = action.copy()
            self.status = {
                "phase": "executing",
                "prompt": self.current_prompt,
                "completion_prob": self.completion,
                "buffered_actions": len(self.actions),
            }
            if not self.actions:
                self._chunk_finished()
            return action.copy()
        if self.pending is None:
            self.pending = (self.generation, Job(self.client.infer, request))
        self.status = {"phase": "inference", "prompt": self.current_prompt}
        return self.last_action.copy()

    def close(self):
        self.closed = True
        self.actions.clear()
        self.client.close()
