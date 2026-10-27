"""fam_hri adapter for canonical synchronized Gen 2/YAM observations."""

from edith.policies.common.keyframes import KeyframePolicy


class Policy(KeyframePolicy):
    use_keyframe = True
    planner_source = "aria"
