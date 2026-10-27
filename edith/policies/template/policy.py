"""Starting point for a new policy: copy this directory, then edit act().

    cp -r edith/policies/template my_policy
    edith-eval --station configs/station.local.json --policy-dir my_policy --headless

The full contract is documented on edith.policies.BasePolicy.
"""

import numpy as np

from edith.policies import BasePolicy


class Policy(BasePolicy):
    def __init__(self, policy_dir, config):
        super().__init__(policy_dir, config)
        # Load checkpoints from self.policy_dir and settings from config here.

    def reset(self, obs):
        pass  # Optional: called once with the first observation.

    def act(self, obs):
        # Human:  obs["images"]["aria_rgb"], obs["gaze_point"],
        #         obs["hand_keypoints"], obs["audio"]
        # Robot:  obs["images"]["top_camera_rgb"], ..., obs["robot_state"]
        # Return the next absolute joint targets, ordered like robot_state.
        # Holding the current pose is always a safe default.
        return np.asarray(obs["robot_state"], dtype=float).copy()

    def close(self):
        pass  # Optional: release models and threads. Always called at the end.
