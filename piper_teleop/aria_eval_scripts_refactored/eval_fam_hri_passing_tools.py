"""Task preset launcher for FAM-HRI passing-tools handoff."""

from eval_fam_hri import main


if __name__ == "__main__":
    main(default_task="passing_tools", allow_task_override=False)
