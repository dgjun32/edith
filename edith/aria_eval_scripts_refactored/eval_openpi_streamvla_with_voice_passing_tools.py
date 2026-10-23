"""Task preset launcher for StreamVLA passing-tools evaluation."""

from eval_openpi_streamvla_with_voice import main


if __name__ == "__main__":
    main(default_task="passing_tools", allow_task_override=False)
