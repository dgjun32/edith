"""ABC process entrypoint retained for the legacy local collector."""

from piper_teleop.robot_utils.abc_config import (
    DEFAULT_ABC_ROOT,
    load_station as load_robot_station,
)
from piper_teleop.teleop_utils.gello import apply_gello_overrides, validate_gello


def load_station(root=DEFAULT_ABC_ROOT, profile="abc_station_config", leaders=None):
    """Compatibility wrapper for older collection scripts."""

    return apply_gello_overrides(load_robot_station(root, profile), leaders)


def validate_teleop(config, leaders):
    """Compatibility wrapper; new code imports teleop_utils directly."""

    validate_gello(config, leaders)


def run_follower(cfg):
    """ABC YAM control with command timeout and shutdown without a home move."""
    import numpy as np
    from deploy.robot.followers.yam_follower import YAMFollowerNode

    class LocalFollower(YAMFollowerNode):
        def tick(self):
            socket = self._subscribers[self.leader_topic_name]
            if not socket.poll(timeout=500):
                raise TimeoutError(f"{self.leader_name}: no command for 500 ms")
            super().tick()

        def on_shutdown(self):
            if hasattr(self, "robot"):
                # No automatic move-to-zero on camera loss or Ctrl+C.
                self.robot.close()

        def process_command(self, command, extras):
            if command.shape != (7,) or not np.all(np.isfinite(command)):
                raise ValueError(
                    "YAM requires six finite joint radians plus normalized gripper"
                )
            super().process_command(command, extras)

    LocalFollower(
        name=cfg.name,
        control_rate=cfg.control_rate,
        channel=cfg.channel,
        leader_name=cfg.leader_name,
        gripper_type=cfg.gripper_type,
    ).run()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--abc-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--leaders", required=True)
    args = parser.parse_args()
    config = load_station(args.abc_root, args.profile, args.leaders)
    validate_teleop(config, args.leaders)
    from deploy.robot.launch import launch
    from deploy.robot.specs import teleop_specs

    specs = teleop_specs(config, quiet=False)
    for spec in specs:
        if spec.name.startswith("follower_"):
            spec.target = "piper_teleop.local.abc_runtime:run_follower"
    raise SystemExit(launch(specs))


if __name__ == "__main__":
    main()
