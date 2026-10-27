"""Record synchronized Aria/camera observations with a zero-output diagnostic policy."""

from pathlib import Path
from edith.scripts.eval_policy_local import main as evaluate


def main(argv=None):
    return evaluate(
        argv,
        default_policy=Path(__file__).resolve().parents[1]
        / "policies/zero_action_policy",
    )


if __name__ == "__main__":
    raise SystemExit(main())
