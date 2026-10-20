import csv
import datetime as dt
import io
import json
import os
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
import wave
from queue import Queue
from typing import List, Optional

import cv2
import h5py
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


def _normalize_gcs_uri(uri: str) -> str:
    uri = uri.strip()
    if not uri:
        raise ValueError("Cloud upload URI cannot be empty.")
    if not uri.startswith("gs://"):
        uri = f"gs://{uri.lstrip('/')}"
    return uri.rstrip("/")


def _join_gcs_uri(base_uri: str, *parts: str) -> str:
    uri = base_uri.rstrip("/")
    for part in parts:
        cleaned = str(part).strip("/")
        if cleaned:
            uri = f"{uri}/{cleaned}"
    return uri


class EpisodeCloudUploader:
    def __init__(
        self,
        base_uri: str,
        dataset_subdir: Optional[str] = None,
        mode: str = "async",
        tool: str = "auto",
        delete_local_after_upload: bool = False,
        log_path: Optional[str] = None,
    ):
        self.base_uri = _normalize_gcs_uri(base_uri)
        self.dataset_subdir = (dataset_subdir or "").strip("/")
        self.mode = mode
        self.tool = self._resolve_tool(tool)
        self.delete_local_after_upload = delete_local_after_upload
        self.log_path = log_path
        self._log_lock = threading.Lock()
        self._queue = Queue() if mode == "async" else None
        self._worker = None

        if mode not in {"async", "sync"}:
            raise ValueError(f"Unsupported cloud upload mode: {mode}")

        if self.mode == "async":
            self._worker = threading.Thread(target=self._worker_loop, daemon=False)
            self._worker.start()

        print(
            "[CloudUpload] Enabled "
            f"(tool={self.tool}, mode={self.mode}, target={_join_gcs_uri(self.base_uri, self.dataset_subdir)})"
        )

    def _resolve_tool(self, tool: str) -> str:
        if tool == "auto":
            for candidate in ("gcloud", "gsutil"):
                if shutil.which(candidate):
                    return candidate
            raise FileNotFoundError("Neither 'gcloud' nor 'gsutil' was found in PATH.")

        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Cloud upload tool '{tool}' was not found in PATH.")
        return tool

    def _build_command(self, local_path: str, remote_uri: str) -> List[str]:
        if self.tool == "gcloud":
            return ["gcloud", "storage", "cp", local_path, remote_uri]
        if self.tool == "gsutil":
            return ["gsutil", "cp", local_path, remote_uri]
        raise ValueError(f"Unsupported cloud upload tool: {self.tool}")

    def enqueue_paths(self, local_paths: List[str]) -> None:
        for local_path in local_paths:
            if not local_path or not os.path.exists(local_path):
                continue

            job = {
                "local_path": local_path,
                "remote_uri": _join_gcs_uri(
                    self.base_uri,
                    self.dataset_subdir,
                    os.path.basename(local_path),
                ),
                "queued_at": time.time(),
            }

            if self.mode == "async":
                self._queue.put(job)
                print(
                    "[CloudUpload] Queued "
                    f"{job['local_path']} -> {job['remote_uri']}"
                )
            else:
                self._upload_job(job)

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._upload_job(job)
            finally:
                self._queue.task_done()

    def _write_log(self, payload: dict) -> None:
        if not self.log_path:
            return

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with self._log_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _upload_job(self, job: dict) -> None:
        local_path = job["local_path"]
        remote_uri = job["remote_uri"]

        if not os.path.exists(local_path):
            print(f"[CloudUpload] Skipping missing file: {local_path}")
            return

        size_bytes = os.path.getsize(local_path)
        start_time = time.time()
        command = self._build_command(local_path, remote_uri)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            duration_s = max(time.time() - start_time, 1e-6)
            size_mib = size_bytes / (1024 * 1024)
            throughput_mib_s = size_mib / duration_s
            success = result.returncode == 0

            self._write_log(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "status": "success" if success else "failed",
                    "tool": self.tool,
                    "mode": self.mode,
                    "local_path": local_path,
                    "remote_uri": remote_uri,
                    "size_bytes": size_bytes,
                    "duration_s": duration_s,
                    "throughput_mib_s": throughput_mib_s,
                    "queue_wait_s": max(start_time - job["queued_at"], 0.0),
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
            )

            if success:
                print(
                    "[CloudUpload] Uploaded "
                    f"{os.path.basename(local_path)} "
                    f"({size_mib:.1f} MiB in {duration_s:.2f}s, {throughput_mib_s:.2f} MiB/s)"
                )
                if self.delete_local_after_upload:
                    os.remove(local_path)
                    print(f"[CloudUpload] Deleted local file after upload: {local_path}")
            else:
                print(
                    "[CloudUpload] Upload failed "
                    f"for {local_path} -> {remote_uri} "
                    f"(exit={result.returncode})"
                )
                if result.stderr.strip():
                    print(result.stderr.strip())
        except Exception as exc:
            self._write_log(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "status": "exception",
                    "tool": self.tool,
                    "mode": self.mode,
                    "local_path": local_path,
                    "remote_uri": remote_uri,
                    "size_bytes": size_bytes,
                    "error": str(exc),
                }
            )
            print(f"[CloudUpload] Unexpected error uploading {local_path}: {exc}")

    def _pending_jobs(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.unfinished_tasks

    def close(self) -> None:
        if self.mode != "async" or self._queue is None:
            return

        old_sigint_handler = None
        old_sigterm_handler = None
        try:
            old_sigint_handler = signal.getsignal(signal.SIGINT)
            old_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            while self._pending_jobs() > 0:
                print(
                    "[CloudUpload] Draining upload queue... "
                    f"pending={self._pending_jobs()}"
                )
                try:
                    self._queue.join()
                except KeyboardInterrupt:
                    # Keep draining until uploads are finished.
                    print("[CloudUpload] Ignoring interrupt until queued uploads are finished...")

            self._queue.put(None)
            self._queue.join()
            if self._worker is not None:
                self._worker.join(timeout=30)
                self._worker = None
        finally:
            if old_sigint_handler is not None:
                signal.signal(signal.SIGINT, old_sigint_handler)
            if old_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, old_sigterm_handler)


def collect_trajectory_with_aria_camera(
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
    image_source_ip=None,
    image_source_port=None,
    instruction_csv_path=None,
    instruction_csv_instruction_col="task_instruction",
    instruction_csv_id_col="id",
    max_trajectories=None,
    instruction_random_seed=None,
    cloud_upload_enabled=False,
    cloud_upload_uri="gs://rise-robot-data/hdf",
    cloud_upload_subdir=None,
    cloud_upload_mode="async",
    cloud_upload_tool="auto",
    cloud_delete_local_after_upload=False,
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

    save_filepath, demo_count = check_save_filepath(save_filepath, instruction)
    success_count = 0
    failure_count = 0
    collected_count = 0

    image_source = connect_to_local_image_source(image_source_ip, image_source_port)
    cloud_uploader = None
    if cloud_upload_enabled:
        dataset_subdir = cloud_upload_subdir
        if dataset_subdir is None and save_filepath:
            dataset_subdir = os.path.basename(os.path.normpath(save_filepath))
        cloud_uploader = EpisodeCloudUploader(
            base_uri=cloud_upload_uri,
            dataset_subdir=dataset_subdir,
            mode=cloud_upload_mode,
            tool=cloud_upload_tool,
            delete_local_after_upload=cloud_delete_local_after_upload,
            log_path=os.path.join(save_filepath, "cloud_upload_log.jsonl"),
        )

    try:
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

            if "aria_idx" in env.camera_ids:
                aria_camera = env.cameras["aria_idx"]
                aria_camera.set_traj_ended(False)
                aria_camera.set_traj_record_state(True)

            num_steps = 0
            if reset_robot:
                env.reset(randomize=randomize_reset)

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
                    if "aria_idx" in env.camera_ids:
                        aria_camera = env.cameras["aria_idx"]
                        aria_camera.set_traj_ended(True)
                        aria_camera.set_traj_record_state(False)
                    if recording_folderpath:
                        env.camera_reader.stop_recording()
                    if save_filepath:
                        traj_writer.close(metadata=controller_info)
                        trajectory_path = os.path.join(save_filepath, f"demo{demo_count}.hdf")
                        modify_aria_images_in_hdf(trajectory_path, image_source)
                        print("Aria images modified and saved back to HDF5 file")
                        image_source.save_trimmed_audio(demo_count, num_steps, save_filepath)

                        if cloud_uploader is not None:
                            upload_paths = [trajectory_path]
                            audio_path = os.path.join(save_filepath, f"demo{demo_count}.wav")
                            if os.path.exists(audio_path):
                                upload_paths.append(audio_path)
                            cloud_uploader.enqueue_paths(upload_paths)
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
                # Keep existing behavior for failure; rotate instruction only on success key.
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

                time.sleep(0.2)
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
    finally:
        if cloud_uploader is not None:
            print("[CloudUpload] Waiting for queued uploads to finish...")
            cloud_uploader.close()


def collect_trajectory_with_aria_camera_human_demo(
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
    image_source_ip=None,
    image_source_port=None,
):
    """
    Collects a robot trajectory.
    - If policy is None, actions will come from the controller
    - If a horizon is given, we will step the environment accordingly
    - Otherwise, we will end the trajectory when the controller tells us to
    - If you need a pointer to the current observation, pass a dictionary in for obs_pointer

    Modified behavior:
    - All input actions are forced to zero before stepping the environment.
    - Saved trajectory still includes an action field, but it is always the zero-action result.
    """

    # Check Parameters #
    assert (controller is not None) or (policy is not None)
    assert (controller is not None) or (horizon is not None)
    if wait_for_controller:
        assert controller is not None
    if obs_pointer is not None:
        assert isinstance(obs_pointer, dict)

    save_filepath, demo_count = check_save_filepath(save_filepath, instruction)

    success_count = 0
    failure_count = 0

    image_source = connect_to_local_image_source(image_source_ip, image_source_port)

    while True:
        # Reset States #
        if controller is not None:
            controller.reset_state()

        # Prepare Data Writers If Necessary #
        if save_filepath:
            traj_writer = TrajectoryWriter(
                os.path.join(save_filepath, f"demo{demo_count}.hdf"),
                metadata={"instruction": instruction},
                save_images=save_images,
            )

        # Prepare For Trajectory #
        if 'aria_idx' in env.camera_ids:
            aria_camera = env.cameras['aria_idx']
            aria_camera.set_traj_ended(False)
            aria_camera.set_traj_record_state(True)

        num_steps = 0
        if reset_robot:
            env.reset(randomize=randomize_reset)

        step = 0
        while True:
            step += 1

            # Collect Miscellaneous Info #
            controller_info = {} if (controller is None) else controller.get_info()
            skip_action = wait_for_controller and (not controller_info["movement_enabled"])
            control_timestamps = {"step_start": time_ms()}

            # Get Observation #
            obs = env.get_observation()
            if obs_pointer is not None:
                obs_pointer.update(obs)

            obs["controller_info"] = controller_info
            obs["timestamp"]["skip_action"] = skip_action

            # Get original action, but do not use it
            control_timestamps["policy_start"] = time_ms()
            if policy is None:
                raw_action, controller_action_info = controller.forward(obs, include_info=True)
            else:
                raw_action = policy.forward(obs)
                controller_action_info = {}

            # Force zero action always
            action = {}
            for key in raw_action.keys():
                action[key] = np.zeros_like(raw_action[key])

            # Regularize Control Frequency #
            control_timestamps["sleep_start"] = time_ms()
            comp_time = time_ms() - control_timestamps["step_start"]
            sleep_left = (1 / env.control_hz) - (comp_time / 1000)
            if sleep_left > 0:
                time.sleep(sleep_left)

            # Step Environment with zero action #
            control_timestamps["control_start"] = time_ms()
            action_info = env.step(action)

            # Optional: record that the applied action was zeroed
            action_info.update(controller_action_info)
            action_info["applied_action"] = action
            action_info["raw_action"] = raw_action

            # Save Data #
            control_timestamps["step_end"] = time_ms()
            obs["timestamp"]["control"] = control_timestamps
            timestep = {
                "observation": obs,
                "action": action_info,   # kept for compatibility, but applied action is always zero
            }
            if save_filepath:
                traj_writer.write_timestep(timestep)

            # Check Termination #
            num_steps += 1
            if horizon is not None:
                end_traj = horizon == num_steps
            else:
                end_traj = controller_info["success"] or controller_info["failure"]

            if controller_info["success"]:
                success_count += 1
            if controller_info["failure"]:
                failure_count += 1

            print(f'Current collected data: {success_count} / {success_count + failure_count}')

            if end_traj:
                if 'aria_idx' in env.camera_ids:
                    aria_camera = env.cameras['aria_idx']
                    aria_camera.set_traj_ended(True)
                    aria_camera.set_traj_record_state(False)

                if recording_folderpath:
                    env.camera_reader.stop_recording()

                if save_filepath:
                    traj_writer.close(metadata=controller_info)
                    trajectory_path = os.path.join(save_filepath, f"demo{demo_count}.hdf")
                    T = modify_aria_images_in_hdf(trajectory_path, image_source)
                    print("Aria images modified and saved back to HDF5 file")
                env.reset()
                break

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


def modify_aria_images_in_hdf(filepath, image_source):
    print(f"[modify_aria] Opening: {filepath}")

    with h5py.File(filepath, "a") as f:
        obs_group = f["observation"]
        ts_group = obs_group["timestamp"]
        img_group = ts_group["images"]

        aria_ds = img_group["aria_idx_rgb"]
        T = aria_ds.shape[0]
        print(f"[modify_aria] Found {T} aria frames")

        target_rgb = np.zeros((T, 1408, 1408, 3), dtype=np.uint8)
        target_slam1 = np.zeros((T, 480, 640, 3), dtype=np.uint8)
        target_slam2 = np.zeros((T, 480, 640, 3), dtype=np.uint8)
        target_eyes = np.zeros((T, 240, 640, 3), dtype=np.uint8)
        target_keycode = np.zeros((T,), dtype=np.int8)

        for t in range(T):
            frame_id = int(aria_ds[t][0] if aria_ds[t].shape else aria_ds[t])
            print(f"[modify_aria] t={t} frame_id={frame_id}")
            img = image_source.get(frame_id)

            if img is None:
                print(f"[modify_aria] WARNING: Missing image for frame_id={frame_id}")
                continue

            target_rgb[t] = np.asarray(img["rgb"], dtype=np.uint8)
            target_slam1[t] = np.asarray(img["slam1"], dtype=np.uint8)
            target_slam2[t] = np.asarray(img["slam2"], dtype=np.uint8)
            target_eyes[t] = np.asarray(img["eyes"], dtype=np.uint8)
            target_keycode[t] = np.asarray(img["keycode"], dtype=np.int8)

        img_group["aria_camera_rgb"] = target_rgb
        img_group["aria_camera_slam1"] = target_slam1
        img_group["aria_camera_slam2"] = target_slam2
        img_group["aria_camera_eyes"] = target_eyes
        img_group["aria_camera_keycode"] = target_keycode
        print("[modify_aria] Finished updating aria images and keycode info.")


class LocalAriaImageSourceClient:
    def __init__(self, server_ip="137.68.92.50", port=5001, timeout=10, sample_rate=48000, control_hz=10):
        self.server_ip = server_ip
        self.port = port
        self.sample_rate = sample_rate
        self.control_hz = control_hz

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)

        print(f"[ImageSourceClient] Connecting to {server_ip}:{port}...")
        self.sock.connect((server_ip, port))
        print("[ImageSourceClient] Connected.")

        self.latest_audio_bytes = None

    def _recv_all(self, length):
        data = b""
        while len(data) < length:
            chunk = self.sock.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _recv_image(self):
        len_buf = self._recv_all(4)
        if not len_buf:
            return None
        (jpeg_len,) = struct.unpack("I", len_buf)
        if jpeg_len == 0:
            return None
        jpeg_bytes = self._recv_all(jpeg_len)
        if jpeg_bytes is None:
            return None
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _recv_keycode(self):
        raw = self._recv_all(1)
        if not raw:
            return None
        (keycode,) = struct.unpack("b", raw)
        return keycode

    def _recv_audio(self):
        len_buf = self._recv_all(4)
        if not len_buf:
            return None
        (audio_len,) = struct.unpack("I", len_buf)
        if audio_len == 0:
            return None
        return self._recv_all(audio_len)

    def get(self, frame_id):
        try:
            self.sock.sendall(struct.pack("Q", frame_id))
            rgb = self._recv_image()
            slam1 = self._recv_image()
            slam2 = self._recv_image()
            eyes = self._recv_image()
            keycode = self._recv_keycode()
            audio_bytes = self._recv_audio()

            if audio_bytes is not None:
                self.latest_audio_bytes = audio_bytes

            return {
                "rgb": rgb,
                "slam1": slam1,
                "slam2": slam2,
                "eyes": eyes,
                "keycode": keycode,
            }
        except Exception as e:
            print(f"[ImageSourceClient] Error fetching frame {frame_id}: {e}")
            return None

    def save_trimmed_audio(self, demo_index, num_frames, save_dir):
        if self.latest_audio_bytes is None:
            print("[AudioClient] No audio received.")
            return

        video_duration = num_frames / self.control_hz
        target_samples = int(video_duration * self.sample_rate)
        print(f"[AudioClient] Video duration: {video_duration:.2f}s")
        print(f"[AudioClient] Target samples: {target_samples}")

        with wave.open(io.BytesIO(self.latest_audio_bytes), "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            raw = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]

        if len(audio) > target_samples:
            audio = audio[-target_samples:]

        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"demo{demo_index}.wav")
        with wave.open(output_path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.sample_rate)
            out.writeframes(audio.astype(np.int16).tobytes())

        print(f"[AudioClient] Saved trimmed audio -> {output_path}")


def connect_to_local_image_source(server_ip, port):
    return LocalAriaImageSourceClient(server_ip=server_ip, port=port)
