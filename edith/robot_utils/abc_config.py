"""ABC/YAM station loading with no GELLO assumptions."""

import importlib
import os
from pathlib import Path
import sys


DEFAULT_ABC_ROOT = Path.home() / "dev/abc-robot-setting/abc"


def load_station(root=DEFAULT_ABC_ROOT, profile="abc_station_config"):
    root = Path(root).expanduser().resolve()
    if not (root / "deploy/robot/config.py").is_file():
        raise FileNotFoundError(f"ABC deployment checkout not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ["ROBOT_PROFILE"] = profile
    module = importlib.import_module("deploy.robot.config")
    return module.get_i2rt_config()


def validate_followers(config) -> None:
    """Validate follower identity without opening CAN or enabling motors."""

    channels = []
    for side, robot in config.robots.items():
        channel = robot.follower.channel
        if not Path("/sys/class/net", channel).exists():
            raise FileNotFoundError(f"{side} follower CAN interface missing: {channel}")
        channels.append(channel)
    if len(set(channels)) != len(channels):
        raise ValueError("Each arm must use a distinct follower CAN interface")
