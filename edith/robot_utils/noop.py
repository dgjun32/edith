"""No hardware dependencies or commands; state is explicitly configured, not measured."""

import time
import numpy as np
from .base import RobotAdapter, RobotSnapshot


class ConfiguredNoOpRobot(RobotAdapter):
    """Action-compatible robot for sensor/policy checks; never opens CAN."""

    def __init__(self, config):
        self.config = config
        self._positions = {
            name: np.asarray(robot.init_q, dtype=np.float64).copy()
            for name, robot in config.robots.items()
        }

    @property
    def arm_order(self) -> tuple[str, ...]:
        return tuple(self.config.robots)

    @property
    def action_size(self) -> int:
        return sum(value.size for value in self._positions.values())

    def start(self) -> None:
        pass

    def read(self, timeout: float) -> RobotSnapshot:
        return RobotSnapshot(
            {name: value.copy() for name, value in self._positions.items()},
            time.perf_counter_ns(),
        )

    def command(self, action: np.ndarray) -> None:
        # Validate the full path, but deliberately do not mutate simulated state:
        # this adapter represents a static, non-executing station check.
        self.validate_action(action)

    def close(self) -> None:
        pass
