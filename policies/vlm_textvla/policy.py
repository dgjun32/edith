"""vlm_textvla adapter for canonical synchronized Gen 2/YAM observations."""

from piper_teleop.policy_utils.keyframes import KeyframePolicy


class Policy(KeyframePolicy):
    use_keyframe = False
    planner_source = "top"
