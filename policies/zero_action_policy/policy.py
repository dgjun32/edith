"""Diagnostic outputs only. The runner refuses motor execution for this policy."""

import numpy as np


class Policy:
    def __init__(self, policy_dir, config):
        self.status = {"phase": "zero_action_diagnostic"}

    def act(self, observation):
        q = np.asarray(observation["robot_state"])
        if q.ndim != 1 or not q.size:
            raise ValueError("Expected a nonempty flat robot state")
        return np.zeros_like(q, dtype=float)
