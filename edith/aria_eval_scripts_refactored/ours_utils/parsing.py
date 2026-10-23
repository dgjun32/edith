"""Parsing helpers for model responses."""

from __future__ import annotations

import re


def extract_json_text(raw_text: str) -> str:
    """Robustly extract JSON from markdown code fences or raw text."""
    text = (raw_text or "").strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == open_ch:
                depth += 1
            elif text[idx] == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text

