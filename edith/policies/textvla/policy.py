"""textvla adapter for canonical synchronized Gen 2/YAM observations."""

from edith.policies.common.chunked import ChunkedPolicy


class Policy(ChunkedPolicy):
    view = "text"
