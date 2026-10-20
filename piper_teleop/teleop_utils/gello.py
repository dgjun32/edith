"""GELLO configuration and validation for the optional teleop path."""

import json
from pathlib import Path

from piper_teleop.robot_utils.abc_config import validate_followers


def apply_gello_overrides(config, leaders):
    if not leaders:
        return config
    overrides = json.loads(Path(leaders).read_text())
    if set(overrides) != set(config.robots):
        raise ValueError("Leader configuration must explicitly include every arm")
    expected = {"device_name", "servo_ids", "joint_signs"}
    for side, values in overrides.items():
        if set(values) != expected:
            raise ValueError(
                "Each leader needs device_name, servo_ids and joint_signs"
            )
        leader = config.robots[side].leader
        for key, value in values.items():
            if key != "device_name":
                value = tuple(value)
                if len(value) != 7:
                    raise ValueError(f"{key} needs seven values")
                if key == "joint_signs" and any(v not in (-1, 1) for v in value):
                    raise ValueError("joint_signs must contain only -1 or 1")
                if key == "servo_ids" and (
                    len(set(value)) != 7
                    or any(type(v) is not int or not 0 <= v <= 252 for v in value)
                ):
                    raise ValueError("servo_ids must be seven distinct Dynamixel IDs")
            setattr(leader, key, value)
    return config


def validate_gello(config, leaders=None) -> None:
    validate_followers(config)
    if not leaders:
        raise ValueError(
            "GELLO configuration is not complete. Supply --leaders with verified "
            "ports, servo IDs, and joint signs."
        )
    devices = []
    for side, robot in config.robots.items():
        device = Path(robot.leader.device_name)
        if not device.exists():
            raise FileNotFoundError(f"{side} GELLO missing: {device}")
        devices.append(str(device.resolve()))
    if len(set(devices)) != len(devices):
        raise ValueError("Each arm must use a distinct GELLO device")
