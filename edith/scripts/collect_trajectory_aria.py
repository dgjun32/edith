"""Default Aria collector: local Gen 2 USB + ABC/YAM.

For archived Piper experiments see collect_trajectory_aria_piper_legacy.py.
"""

from edith.scripts.collect_trajectory_yam_local import main

if __name__ == "__main__":
    raise SystemExit(main())
