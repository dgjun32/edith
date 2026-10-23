"""Task preset launcher for FAM-HRI bottle/beverage sorting."""

from eval_fam_hri import main


if __name__ == "__main__":
    main(default_task="bottle", allow_task_override=False)
