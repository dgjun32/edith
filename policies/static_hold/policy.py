"""Hardware smoke policy: repeat the joint state observed at reset."""

import numpy as np


class StaticHoldPolicy:
    def __init__(self, policy_dir, config):
        self.policy_dir = policy_dir
        self.config = config
        self.target = None

    def reset(self, observation):
        state = np.asarray(observation["robot_state"], dtype=np.float64).reshape(-1)
        if not state.size or not np.all(np.isfinite(state)):
            raise ValueError("Static hold requires a finite, nonempty robot_state")
        self.target = state.copy()

    def act(self, observation):
        if self.target is None:
            self.reset(observation)
        return self.target.copy()
