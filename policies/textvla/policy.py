"""textvla adapter for canonical synchronized Gen 2/YAM observations."""

from piper_teleop.policy_utils.chunked import ChunkedPolicy


class Policy(ChunkedPolicy):
    view = "text"
