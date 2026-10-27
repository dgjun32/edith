"""Robot-independent station configuration; loading JSON never imports a robot SDK."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    camera_type: str = "realsense"


@dataclass(frozen=True)
class JointGroupConfig:
    init_q: tuple[float, ...]


@dataclass(frozen=True)
class StationConfig:
    cameras: dict[str, CameraConfig]
    robots: dict[str, JointGroupConfig]
    adapter: str = "none"
    adapter_config: dict = field(default_factory=dict)


def load_station(path):
    value = json.loads(Path(path).expanduser().read_text())
    if value.get("schema") != "edith.station.v1":
        raise ValueError("Station schema must be edith.station.v1")
    cameras = {
        name: CameraConfig(**config)
        for name, config in value.get("cameras", {}).items()
    }
    for name, camera in cameras.items():
        if not camera.serial or min(camera.width, camera.height, camera.fps) <= 0:
            raise ValueError(f"Invalid camera settings: {name}")
    groups = {}
    for name, positions in value.get("initial_state", {}).items():
        q = np.asarray(positions, dtype=float)
        if q.ndim != 1 or not q.size or not np.isfinite(q).all():
            raise ValueError(f"Invalid initial state: {name}")
        groups[name] = JointGroupConfig(tuple(q.tolist()))
    if not groups:
        raise ValueError("Station needs nonempty initial_state joint groups")
    robot = value.get("robot", {})
    if not isinstance(robot, dict) or not isinstance(robot.get("config", {}), dict):
        raise ValueError("Station robot and robot.config must be JSON objects")
    if not isinstance(robot.get("adapter", "none"), str):
        raise ValueError("Station robot.adapter must be a string")
    reserved = {"aria", "eye_gaze", "hand_pose", "audio", "policy_observation"}
    if reserved.intersection(cameras) or any(
        name.startswith(("leader_", "follower_")) for name in cameras
    ):
        raise ValueError(
            "Camera names must not overlap Aria, recording or robot-state streams"
        )
    return StationConfig(
        cameras, groups, robot.get("adapter", "none"), robot.get("config", {})
    )
