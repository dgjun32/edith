"""Robot registry: add a robot here without changing policy or Aria code."""

from .abc_yam import ABCYAMRobot, ConfiguredNoOpRobot


def create_robot(name, *, config, execute, abc_root, profile):
    if name != "abc-yam":
        raise ValueError(f"Unknown robot adapter: {name}")
    if execute:
        return ABCYAMRobot(config, abc_root=abc_root, profile=profile)
    return ConfiguredNoOpRobot(config)
