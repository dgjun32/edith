"""Follower-only ABC worker. No GELLO process is constructed."""

import argparse

from edith.robot_utils.abc_config import load_station, validate_followers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-root", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)
    config = load_station(args.abc_root, args.profile)
    validate_followers(config)
    from deploy.robot.launch import launch
    from deploy.robot.specs import follower_specs

    specs = follower_specs(config, quiet=False)
    for spec in specs:
        spec.target = "edith.local.abc_runtime:run_follower"
    return launch(specs)


if __name__ == "__main__":
    raise SystemExit(main())
