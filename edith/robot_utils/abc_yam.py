"""ABC/YAM adapter used by the policy runner.

The adapter publishes the same joint-position topics used by GELLO, but it does
not construct or require a leader. Followers are launched in a separate process
and own CAN for the lifetime of this object.
"""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from .base import RobotAdapter, RobotSnapshot
from .abc_config import DEFAULT_ABC_ROOT, validate_followers


class ABCYAMRobot(RobotAdapter):
    def __init__(self, config, abc_root=DEFAULT_ABC_ROOT, profile="abc_station_config"):
        self.config = config
        self.abc_root = Path(abc_root).expanduser().resolve()
        self.profile = profile
        self.process = None
        self.context = None
        self.publishers = {}
        self.subscribers = {}
        self.comms = None

    @property
    def arm_order(self) -> tuple[str, ...]:
        return tuple(self.config.robots)

    @property
    def action_size(self) -> int:
        return sum(len(self.config.robots[name].init_q) for name in self.arm_order)

    def start(self) -> None:
        if self.process is not None:
            return
        validate_followers(self.config)
        import zmq
        from deploy.robot import communication as comms

        self.comms = comms
        self.context = zmq.Context()
        try:
            # Bind command publishers before followers connect. This also fails
            # loudly if another controller already owns the station topics.
            for name in self.arm_order:
                self.publishers[name] = comms.create_publisher(
                    self.context, f"leader_{name}_actions", linger=0
                )
            for name in self.arm_order:
                self.subscribers[name] = comms.create_subscriber(
                    self.context, f"follower_{name}_state", conflate=1
                )
            child_env = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[2])
            child_env["PYTHONPATH"] = os.pathsep.join(
                [source_root, str(self.abc_root), child_env.get("PYTHONPATH", "")]
            )
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "edith.robot_utils.abc_yam_worker",
                    "--abc-root",
                    str(self.abc_root),
                    "--profile",
                    self.profile,
                ],
                env=child_env,
                start_new_session=True,
            )
        except BaseException:
            self.close()
            raise

    def _ensure_alive(self) -> None:
        if self.process is None:
            raise RuntimeError("ABC/YAM robot has not been started")
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(f"ABC/YAM follower process exited with code {code}")

    def read(self, timeout: float) -> RobotSnapshot:
        self._ensure_alive()
        deadline = time.perf_counter() + timeout
        positions = {}
        timestamps = []
        for name in self.arm_order:
            remaining_ms = max(1, round((deadline - time.perf_counter()) * 1000))
            data, extras = self.comms.subscribe(
                self.subscribers[name],
                timeout_ms=remaining_ms,
                topic_label=f"follower_state:{name}",
            )
            data = np.asarray(data, dtype=np.float64).reshape(-1)
            if data.shape != (7,) or not np.all(np.isfinite(data)):
                raise ValueError(f"Invalid {name} YAM state: {data.shape}")
            positions[name] = data
            timestamps.append(round(float(extras["timestamp"]) * 1e9))
        return RobotSnapshot(positions, max(timestamps))

    def command(self, action: np.ndarray) -> None:
        self._ensure_alive()
        action = self.validate_action(action)
        offset = 0
        for name in self.arm_order:
            width = len(self.config.robots[name].init_q)
            self.comms.publish(
                self.publishers[name],
                action[offset : offset + width],
                extras={"source": "policy"},
            )
            offset += width

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
        for socket in (*self.publishers.values(), *self.subscribers.values()):
            socket.close(linger=0)
        self.publishers.clear()
        self.subscribers.clear()
        if self.context is not None:
            self.context.term()
            self.context = None


# Backward-compatible import for existing ABC deployments.
from .noop import ConfiguredNoOpRobot  # noqa: E402, F401
