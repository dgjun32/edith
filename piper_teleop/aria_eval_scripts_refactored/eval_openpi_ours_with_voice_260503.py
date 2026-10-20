"""Compatibility entry point for the local Gen 2/YAM ``policy_edith`` runner.

The former remote Aria publisher and Piper loop have been replaced by USB-NCM
capture on the deployment computer and synchronized local YAM observations.
Use the current runner flags (``--help``), not the old experiment CLI.
Voice instructions arrive through ``--transcript-file``; this launcher does
not capture audio or implicitly enable Whisper/Gemini. See
``policies/policy_edith/README.md`` for presets and optional Gemini planning.
"""

from pathlib import Path
import sys


def main(argv=None):
    if not __package__:
        # Preserve direct script execution without depending on the working directory.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from piper_teleop.scripts.policy_edith import main as run_local

    return run_local(argv)


if __name__ == "__main__":
    raise SystemExit(main())
