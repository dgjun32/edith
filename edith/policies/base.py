"""The contract every EDITH policy implements."""

from abc import ABC, abstractmethod
from pathlib import Path


class BasePolicy(ABC):
    """Base class for policies run by ``edith-eval``.

    A policy is a directory holding ``policy.json`` and the Python file it
    names. The runner constructs the class once, calls ``reset`` with the first
    observation, calls ``act`` every step at ``control_hz``, and always calls
    ``close`` at the end.

    ``obs`` is the dictionary from ``RobotEvaluationEnv.get_policy_observation()``
    (human view, gaze, hands, audio, robot cameras and ``robot_state``) plus
    ``instruction`` and, when given, ``transcript`` and ``policy_event``.
    """

    def __init__(self, policy_dir, config):
        """Load models and assets. ``config`` is policy.json's ``config`` plus overrides."""

        self.policy_dir = Path(policy_dir)
        self.config = config

    def reset(self, obs):
        """Called once with the first observation, before the first ``act``."""

    @abstractmethod
    def act(self, obs):
        """Return the next target: one finite 1-D array of absolute joint positions.

        Order and length match ``obs["robot_state"]``. Return within one step
        (``1 / control_hz``): run slow models in a background thread and return
        the latest target. Return a single action; buffer action chunks yourself.
        """

    def close(self):
        """Release models, threads and connections. Called even after errors."""
