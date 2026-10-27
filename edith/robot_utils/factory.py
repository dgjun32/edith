"""Load optional robot adapters from the robot repository, never from Aria code."""

import importlib
from .base import RobotAdapter
from .noop import ConfiguredNoOpRobot


def create_robot(name, *, config, execute=False, abc_root=None, profile=None):
    if not execute:
        return ConfiguredNoOpRobot(config)
    if name == "abc-yam":
        from .abc_yam import ABCYAMRobot

        options = {}
        if abc_root is not None:
            options["abc_root"] = abc_root
        if profile is not None:
            options["profile"] = profile
        return ABCYAMRobot(config, **options)
    if name in ("none", "", None):
        raise ValueError(
            "Motor execution requires an explicit robot adapter module:factory"
        )
    module, sep, symbol = name.partition(":")
    if not sep or not module or not symbol:
        raise ValueError(
            "Robot adapter must be module:factory or the optional abc-yam adapter"
        )
    factory = getattr(importlib.import_module(module), symbol)
    robot = factory(config=dict(config.adapter_config))
    if not isinstance(robot, RobotAdapter):
        raise TypeError(
            "External robot factory must return an edith.robot_utils.RobotAdapter"
        )
    expected = tuple(config.robots)
    width = sum(len(group.init_q) for group in config.robots.values())
    if tuple(robot.arm_order) != expected or robot.action_size != width:
        robot.close()
        raise ValueError(
            "Robot adapter joint-group order/action size differs from station initial_state"
        )
    return robot
