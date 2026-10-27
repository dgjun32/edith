import csv
import datetime as dt
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

try:
    from droid.misc.time import time_ms
    from droid.trajectory_utils.trajectory_writer import TrajectoryWriter
except ModuleNotFoundError:
    # Fallback for local workspace layout (/droid/droid/...)
    from droid.droid.misc.time import time_ms
    from droid.droid.trajectory_utils.trajectory_writer import TrajectoryWriter


def check_save_filepath(save_filepath: str, instruction: str):
    x = dt.datetime.now()
    date = f"{x.year}-{x.month}-{x.day}"

    if save_filepath is None:
        save_filepath = os.path.join("../../demos/" + date, instruction.replace(" ", "_"))

    if os.path.exists(save_filepath):
        print("++++++++++ Demo Already Exists. Appending... ++++++++++")
        indices = []
        for file in os.listdir(save_filepath):
            match = re.match(r"demo(\d+)\.hdf", file)
            if match:
                indices.append(int(match.group(1)))
        demo_count = (max(indices) + 1) if indices else 0
    else:
        os.makedirs(save_filepath)
        demo_count = 0

    return save_filepath, demo_count


def _load_instruction_items(csv_path: str, instruction_col: str, id_col: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Instruction CSV not found: {csv_path}")

    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if instruction_col not in reader.fieldnames:
            raise ValueError(
                f"Missing instruction column '{instruction_col}' in CSV. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            instruction = (row.get(instruction_col) or "").strip()
            if not instruction:
                continue
            instruction_id = (row.get(id_col) or "").strip()
            items.append(
                {
                    "id": instruction_id,
                    "instruction": instruction,
                }
            )

    if not items:
        raise ValueError(f"No valid instructions in CSV: {csv_path}")
    return items


class InstructionCycleSampler:
    """Samples without replacement; reshuffles after one full cycle."""

    def __init__(self, items, seed=None):
        self.items = list(items)
        self.rng = random.Random(seed)
        self.remaining = []

    def sample(self):
        if not self.remaining:
            self.remaining = self.items.copy()
            self.rng.shuffle(self.remaining)
        return self.remaining.pop()


class TaskGuidNotifier:
    """
    Sends task collection status to pi-task-guide API.

    API (from USAGE_GUIDE.md):
      - POST /api/ready {"task": "...", "emotion": "neutral"}
      - POST /api/start {"emotion": "happy|excited|..."}
      - POST /api/end {"emotion": "success|failure|..."}
    """

    def __init__(self, config_path: Optional[str]):
        self.enabled = False

        # pi-task-guide defaults
        self.base_url = ""
        self.ready_endpoint = "/api/ready"
        self.start_endpoint = "/api/start"
        self.end_endpoint = "/api/end"
        self.reset_endpoint = "/api/reset"
        self.ready_task_field = "task"
        self.emotion_field = "emotion"
        self.ready_emotion = "neutral"
        self.progress_emotion = "happy"
        self.stop_emotion_success = "success"
        self.stop_emotion_failure = "failure"

        # generic fallback (old format)
        self.use_generic = False
        self.server_url = None
        self.endpoints = {}
        self.status_field = "status"
        self.task_field = "task_instruction"

        self.timeout_sec = 1.0
        self.ready_delay_sec = 1.0
        self.method = "POST"
        self.headers = {}

        if not config_path:
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[TaskGuid] Failed to load config: {config_path}, error: {e}")
            return

        self.enabled = bool(cfg.get("enabled", False))
        self.timeout_sec = float(cfg.get("timeout_sec", 1.0))
        self.ready_delay_sec = float(cfg.get("ready_delay_sec", 1.0))
        self.method = str(cfg.get("method", "POST")).upper()
        self.headers = dict(cfg.get("headers", {}) or {})

        robot_ip = cfg.get("robot_ip")
        port = int(cfg.get("port", 5001))
        configured_base_url = cfg.get("base_url")
        if configured_base_url:
            self.base_url = str(configured_base_url).rstrip("/")
        elif robot_ip:
            self.base_url = f"http://{robot_ip}:{port}"

        self.ready_endpoint = str(cfg.get("ready_endpoint", self.ready_endpoint))
        self.start_endpoint = str(cfg.get("start_endpoint", self.start_endpoint))
        self.end_endpoint = str(cfg.get("end_endpoint", self.end_endpoint))
        self.reset_endpoint = str(cfg.get("reset_endpoint", self.reset_endpoint))
        self.ready_task_field = str(cfg.get("ready_task_field", self.ready_task_field))
        self.emotion_field = str(cfg.get("emotion_field", self.emotion_field))
        self.ready_emotion = str(cfg.get("ready_emotion", self.ready_emotion))
        self.progress_emotion = str(cfg.get("progress_emotion", self.progress_emotion))
        self.stop_emotion_success = str(cfg.get("stop_emotion_success", self.stop_emotion_success))
        self.stop_emotion_failure = str(cfg.get("stop_emotion_failure", self.stop_emotion_failure))

        # Backward-compatible fallback configuration mode
        self.server_url = cfg.get("server_url")
        self.endpoints = cfg.get("endpoints", {}) or {}
        self.status_field = str(cfg.get("status_field", self.status_field))
        self.task_field = str(cfg.get("task_field", self.task_field))
        self.use_generic = bool(self.server_url or self.endpoints)

    def _join_url(self, endpoint: str):
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"{self.base_url}{endpoint}"

    def _resolve_generic_url(self, status: str):
        if self.endpoints and status in self.endpoints:
            endpoint = self.endpoints[status]
            if endpoint.startswith("http://") or endpoint.startswith("https://"):
                return endpoint
            return f"{self.base_url}{endpoint}"
        return self.server_url

    def _send_http(self, url: str, payload: Optional[dict]):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = dict(self.headers)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        payload_preview = payload if payload is not None else {}
        print(f"[TaskGuid] Request {self.method} {url} payload={payload_preview}")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method=self.method,
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return getattr(resp, "status", "unknown")

    def send(
        self,
        status: str,
        task_instruction: Optional[str] = None,
        emotion: Optional[str] = None,
        success: Optional[bool] = None,
    ):
        if not self.enabled:
            return

        try:
            if self.use_generic:
                url = self._resolve_generic_url(status)
                if not url:
                    print(f"[TaskGuid] No URL configured for status='{status}'")
                    return
                payload = {self.status_field: status}
                if task_instruction is not None:
                    payload[self.task_field] = task_instruction
                code = self._send_http(url, payload)
                print(f"[TaskGuid] Sent {status} -> {url} ({code})")
                return

            # pi-task-guide mode
            if not self.base_url:
                print("[TaskGuid] base_url/robot_ip is not configured.")
                return
            if status == "progress":
                url = self._join_url(self.start_endpoint)
                payload = {self.emotion_field: emotion or self.progress_emotion}
            elif status == "ready":
                url = self._join_url(self.ready_endpoint)
                payload = {
                    self.ready_task_field: task_instruction or "",
                    self.emotion_field: emotion or self.ready_emotion,
                }
            elif status == "stop":
                if emotion is None:
                    if success is False:
                        emotion = self.stop_emotion_failure
                    else:
                        emotion = self.stop_emotion_success
                url = self._join_url(self.end_endpoint)
                payload = {self.emotion_field: emotion}
            elif status == "reset":
                url = self._join_url(self.reset_endpoint)
                payload = None
            else:
                print(f"[TaskGuid] Unknown status: {status}")
                return

            code = self._send_http(url, payload)
            print(f"[TaskGuid] Sent {status} -> {url} ({code})")
        except urllib.error.URLError as e:
            print(f"[TaskGuid] Failed {status}: {e}")
        except Exception as e:
            print(f"[TaskGuid] Unexpected error while sending {status}: {e}")


def _wait_for_start_signal(controller):
    """Wait until user presses S (success) to start the next collection."""
    if controller is None:
        return

    print("[Collector] Waiting for next start trigger. Press S to start next task.")
    while True:
        controller_info = controller.get_info()
        if controller_info.get("success", False):
            print("[Collector] Start trigger received (S).")
            return
        time.sleep(0.05)


def collect_trajectory(
    env,
    instruction=None,
    controller=None,
    policy=None,
    horizon=None,
    save_filepath=None,
    metadata=None,
    wait_for_controller=False,
    obs_pointer=None,
    save_images=False,
    recording_folderpath=False,
    randomize_reset=False,
    reset_robot=True,
    reset_time=10,
    instruction_csv_path=None,
    instruction_csv_instruction_col="task_instruction",
    instruction_csv_id_col="id",
    max_trajectories=None,
    instruction_random_seed=None,
    task_guid_config_path=None,
):
    """
    Collect trajectories with optional CSV-based instruction sampling.

    Existing behavior is preserved when `instruction_csv_path` is not provided.
    """

    assert (controller is not None) or (policy is not None)
    assert (controller is not None) or (horizon is not None)
    if wait_for_controller:
        assert controller is not None
    if obs_pointer is not None:
        assert isinstance(obs_pointer, dict)

    instruction_sampler = None
    current_instruction_id = None
    use_csv_instruction_mode = instruction_csv_path is not None
    fixed_save_filepath = save_filepath

    if use_csv_instruction_mode:
        instruction_items = _load_instruction_items(
            instruction_csv_path,
            instruction_col=instruction_csv_instruction_col,
            id_col=instruction_csv_id_col,
        )
        instruction_sampler = InstructionCycleSampler(
            instruction_items, seed=instruction_random_seed
        )
        sampled = instruction_sampler.sample()
        instruction = sampled["instruction"]
        current_instruction_id = sampled["id"]
        print(
            f"[InstructionCSV] Loaded {len(instruction_items)} instructions. "
            f"Start -> id={current_instruction_id or 'N/A'} "
            f"instruction='{instruction}'"
        )

    if instruction is None:
        instruction = "No instruction"

    task_guid_notifier = TaskGuidNotifier(task_guid_config_path)
    save_filepath, demo_count = check_save_filepath(save_filepath, instruction)

    success_count = 0
    failure_count = 0
    collected_count = 0

    while True:
        if controller is not None:
            controller.reset_state()

        if save_filepath:
            episode_metadata = {"instruction": instruction}
            if current_instruction_id is not None:
                episode_metadata["instruction_id"] = current_instruction_id
            if metadata:
                episode_metadata.update(metadata)
            traj_writer = TrajectoryWriter(
                os.path.join(save_filepath, f"demo{demo_count}.hdf"),
                metadata=episode_metadata,
                save_images=save_images,
            )

        num_steps = 0
        if reset_robot:
            env.reset(randomize=randomize_reset)
        task_guid_notifier.send("progress", task_instruction=instruction)

        last_controller_info = {"success": False, "failure": False}
        while True:
            controller_info = {} if (controller is None) else controller.get_info()
            skip_action = wait_for_controller and (not controller_info["movement_enabled"])
            control_timestamps = {"step_start": time_ms()}

            obs = env.get_observation()
            if obs_pointer is not None:
                obs_pointer.update(obs)
            obs["controller_info"] = controller_info
            obs["timestamp"]["skip_action"] = skip_action

            control_timestamps["policy_start"] = time_ms()
            if policy is None:
                action, controller_action_info = controller.forward(obs, include_info=True)
            else:
                action = policy.forward(obs)
                controller_action_info = {}

            control_timestamps["sleep_start"] = time_ms()
            comp_time = time_ms() - control_timestamps["step_start"]
            sleep_left = (1 / env.control_hz) - (comp_time / 1000)
            if sleep_left > 0:
                time.sleep(sleep_left)

            control_timestamps["control_start"] = time_ms()
            if skip_action:
                action_info = env.create_action_dict(np.zeros_like(action))
            else:
                action_info = env.step(action)
            action_info.update(controller_action_info)

            control_timestamps["step_end"] = time_ms()
            obs["timestamp"]["control"] = control_timestamps
            timestep = {"observation": obs, "action": action_info}
            if save_filepath:
                traj_writer.write_timestep(timestep)

            num_steps += 1
            if horizon is not None:
                end_traj = horizon == num_steps
            else:
                end_traj = controller_info["success"] or controller_info["failure"]

            if controller_info["success"]:
                success_count += 1
            if controller_info["failure"]:
                failure_count += 1
            print(f"Current collected data: {success_count} / {success_count + failure_count}")

            if end_traj:
                last_controller_info = controller_info.copy()
                task_guid_notifier.send(
                    "stop",
                    task_instruction=instruction,
                    success=last_controller_info.get("success"),
                )
                if recording_folderpath:
                    env.camera_reader.stop_recording()
                if save_filepath:
                    traj_writer.close(metadata=controller_info)
                env.reset()
                break

        collected_count += 1
        if max_trajectories is not None and collected_count >= max_trajectories:
            print(
                f"[Collector] Reached max_trajectories={max_trajectories}. "
                "Stopping collection loop."
            )
            return

        if use_csv_instruction_mode:
            # Rotate instruction on success key.
            if last_controller_info.get("success", False):
                sampled = instruction_sampler.sample()
                instruction = sampled["instruction"]
                current_instruction_id = sampled["id"]
                print(
                    "[InstructionCSV] Next instruction -> "
                    f"id={current_instruction_id or 'N/A'} "
                    f"instruction='{instruction}'"
                )

                if fixed_save_filepath is None:
                    save_filepath, demo_count = check_save_filepath(None, instruction)
                else:
                    demo_count += 1
            else:
                demo_count += 1
                print(
                    "[InstructionCSV] Trajectory ended without success. "
                    "Keeping current instruction."
                )

            time.sleep(task_guid_notifier.ready_delay_sec)
            task_guid_notifier.send("ready", task_instruction=instruction)
            _wait_for_start_signal(controller)
            continue

        demo_count += 1
        time.sleep(0.2)

        print("Next: Press B to input new instruction or A to go with the current instruction!")
        controller_info = {} if (controller is None) else controller.get_info()
        while not controller_info["success"] and not controller_info["failure"]:
            controller_info = {} if (controller is None) else controller.get_info()
            if controller_info["failure"]:
                instruction = input("Please enter new instruction: ")
                save_filepath, demo_count = check_save_filepath(None, instruction)
                break
            if controller_info["success"]:
                print("Going with the previous instruction..")
                break

        time.sleep(task_guid_notifier.ready_delay_sec)
        task_guid_notifier.send("ready", task_instruction=instruction)
