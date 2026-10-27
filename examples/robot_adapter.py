"""Executable 3-joint example of the interface a robot repository supplies.

This example is in memory. Replace start/read/command/close in YOUR repository
with its controller API; factory construction itself must not enable motors.
"""

import time
import numpy as np
from edith.robot_utils import RobotAdapter, RobotSnapshot


class ExampleRobot(RobotAdapter):
    arm_order = ("arm",)
    action_size = 3

    def __init__(self, config):
        self.q = np.asarray(config.get("initial_state", [0, 0, 0]), dtype=float)
        self.started = False

    def start(self):
        self.started = True

    def read(self, timeout):
        if not self.started:
            raise RuntimeError("Robot is not started")
        return RobotSnapshot({"arm": self.q.copy()}, time.perf_counter_ns())

    def command(self, action):
        if not self.started:
            raise RuntimeError("Robot is not started")
        self.q = self.validate_action(action).copy()

    def close(self):
        self.started = False


def create_robot(*, config):
    return ExampleRobot(config)
