"""Launch Edith with local USB Aria Gen 2 and synchronized ABC/YAM views.

The policy directory owns inference; eval_policy_local owns USB cameras, CAN,
recording and synchronization. No external Aria publisher is used.
"""

from pathlib import Path
import sys


POLICY_DIR = Path(__file__).resolve().parents[1] / "policies/policy_edith"


def main(argv=None):
    if not __package__:
        # Also support `python /path/to/policy_edith.py` from an uninstalled checkout.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from edith.scripts import eval_policy_local

    return eval_policy_local.main(argv, default_policy=POLICY_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
