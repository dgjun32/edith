"""Load task-specific initial and async VLM prompt templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "ours_prompts"


@dataclass(frozen=True)
class PromptTemplates:
    name: str
    path: Path
    initial: str
    async_update: str


def list_prompt_names(prompt_dir: Optional[Path] = None) -> List[str]:
    root = Path(prompt_dir) if prompt_dir is not None else DEFAULT_PROMPT_DIR
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.txt"))


def _marker_re(title: str) -> re.Pattern:
    escaped = re.escape(title)
    return re.compile(rf"^=+\s*\n{escaped}\s*\n=+\s*$", re.MULTILINE)


def _extract_section(text: str, title: str, next_title: Optional[str] = None) -> str:
    marker = _marker_re(title).search(text)
    if marker is None:
        raise ValueError(f"Prompt file is missing section marker: {title}")

    start = marker.end()
    if next_title is None:
        end = len(text)
    else:
        next_marker = _marker_re(next_title).search(text, start)
        if next_marker is None:
            raise ValueError(f"Prompt file is missing section marker: {next_title}")
        end = next_marker.start()

    return text[start:end].strip()


def load_prompt_templates(
    prompt_name: str,
    prompt_dir: Optional[Path] = None,
) -> PromptTemplates:
    root = Path(prompt_dir) if prompt_dir is not None else DEFAULT_PROMPT_DIR
    path = root / f"{prompt_name}.txt"
    if not path.exists():
        available = ", ".join(list_prompt_names(root)) or "(none)"
        raise FileNotFoundError(
            f"Prompt task {prompt_name!r} not found at {path}. "
            f"Available prompt tasks: {available}"
        )

    text = path.read_text(encoding="utf-8")
    initial = _extract_section(
        text, "INITIAL SCRIPT", next_title="ASYNCHRONOUS UPDATE SCRIPT")
    async_update = _extract_section(text, "ASYNCHRONOUS UPDATE SCRIPT")
    return PromptTemplates(
        name=prompt_name,
        path=path,
        initial=initial,
        async_update=async_update,
    )
