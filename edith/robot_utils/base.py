"""Small robot contract used by local collection and policy evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RobotSnapshot:
    """One joint-position observation for every configured arm."""

    positions: Mapping[str, np.ndarray]
    timestamp_ns: int

    def flat(self, arm_order) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.positions[name], dtype=np.float64) for name in arm_order]
        )


class RobotAdapter(ABC):
    """The only interface the evaluation loop needs from a robot."""

    @property
    @abstractmethod
    def arm_order(self) -> tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def read(self, timeout: float) -> RobotSnapshot:
        pass

    @abstractmethod
    def command(self, action: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def validate_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_size,):
            raise ValueError(
                f"Expected action shape ({self.action_size},), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Robot action contains NaN or infinity")
        return action
