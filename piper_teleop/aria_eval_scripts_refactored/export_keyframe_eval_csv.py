#!/usr/bin/env python3
"""Snapshot temp/keyframes.json into episode-wise CSV files.

This utility preserves the latest keyframe extraction result before the next
evaluation run overwrites it. It can also compare predicted instructions
against a pre-defined answer key.

Examples
--------
Save the current keyframes.json once:

    python aria_eval_scripts_refactored/export_keyframe_eval_csv.py

Watch keyframes.json and auto-save each newly written episode:

    python aria_eval_scripts_refactored/export_keyframe_eval_csv.py --watch

Watch and compare against an answer-key CSV:

    python aria_eval_scripts_refactored/export_keyframe_eval_csv.py \
        --watch \
        --answer-key config/muffin_episode_task_list_24.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_KEYFRAMES_JSON = _REPO_ROOT / "temp" / "keyframes.json"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "temp" / "keyframe_eval_results"


def _normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _join_instructions(values: list[str]) -> str:
    return " || ".join(values)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_keyframes_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object.")

    saved_keyframes = payload.get("saved_keyframes", [])
    if not isinstance(saved_keyframes, list):
        raise ValueError("'saved_keyframes' must be a list.")

    return payload


def _parse_expected_instruction_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Expected instruction list must be a JSON list.")

    values: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def _candidate_episode_keys(episode_id: str) -> list[str]:
    raw = str(episode_id).strip()
    keys = [raw]

    match = re.fullmatch(r"episode_(\d+)", raw)
    if match:
        number = int(match.group(1))
        keys.extend([match.group(1), str(number)])
    elif raw.isdigit():
        number = int(raw)
        keys.extend([f"episode_{number:03d}", f"{number:03d}", str(number)])

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _load_answer_key(answer_key_path: Path) -> dict[str, list[str]]:
    return _load_answer_key_with_options(
        answer_key_path=answer_key_path,
        episode_column=None,
        answer_columns=None,
    )


def _load_answer_key_with_options(
    answer_key_path: Path,
    episode_column: str | None,
    answer_columns: list[str] | None,
) -> dict[str, list[str]]:
    suffix = answer_key_path.suffix.lower()
    if suffix == ".json":
        raw = _read_json(answer_key_path)
        return _load_answer_key_from_json(raw)
    if suffix == ".csv":
        return _load_answer_key_from_csv(
            path=answer_key_path,
            episode_column=episode_column,
            answer_columns=answer_columns,
        )
    raise ValueError(
        f"Unsupported answer-key format: {answer_key_path}. Use .json or .csv."
    )


def _load_answer_key_from_json(raw: Any) -> dict[str, list[str]]:
    if isinstance(raw, dict):
        if "episodes" in raw:
            raw = raw["episodes"]
        else:
            answer_key: dict[str, list[str]] = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    value = (
                        value.get("expected_instructions")
                        or value.get("instructions")
                        or value.get("expected")
                    )
                answer_key[str(key)] = _parse_expected_instruction_list(value)
            return answer_key

    if not isinstance(raw, list):
        raise ValueError("JSON answer key must be a dict or a list of episode rows.")

    answer_key: dict[str, list[str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each JSON answer-key row must be an object.")
        episode_id = str(
            item.get("episode_id")
            or item.get("episode")
            or item.get("id")
            or ""
        ).strip()
        if not episode_id:
            raise ValueError("Every JSON answer-key row needs an episode_id.")
        expected = (
            item.get("expected_instructions")
            or item.get("instructions")
            or item.get("expected")
        )
        answer_key[episode_id] = _parse_expected_instruction_list(expected)
    return answer_key


def _find_field_case_insensitive(
    fieldnames: list[str],
    candidate_names: list[str],
) -> str | None:
    lowered = {field.lower(): field for field in fieldnames}
    for name in candidate_names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _resolve_requested_fields(
    fieldnames: list[str],
    requested_fields: list[str] | None,
) -> list[str]:
    if not requested_fields:
        return []

    resolved: list[str] = []
    for field in requested_fields:
        actual = _find_field_case_insensitive(fieldnames, [field])
        if actual is None:
            raise ValueError(
                f"Answer-key CSV column not found: {field}. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        resolved.append(actual)
    return resolved


def _detect_ordered_answer_fields(fieldnames: list[str]) -> list[str]:
    ordered_candidates: list[tuple[int, str]] = []
    supported_prefixes = (
        "expected",
        "expected_instruction",
        "instruction",
        "task",
        "task_instruction",
        "subtask",
        "subtask_instruction",
        "answer",
        "gt",
        "ground_truth",
        "ground_truth_instruction",
    )

    for field in fieldnames:
        lower = field.lower()
        match = re.fullmatch(r"([a-z_]+)_(\d+)", lower)
        if not match:
            continue
        prefix, number = match.groups()
        if prefix not in supported_prefixes:
            continue
        ordered_candidates.append((int(number), field))

    ordered_candidates.sort(key=lambda item: item[0])
    return [field for _, field in ordered_candidates]


def _split_answer_columns(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    columns = [part.strip() for part in raw_value.split(",") if part.strip()]
    return columns or None


def _load_answer_key_from_csv(
    path: Path,
    episode_column: str | None,
    answer_columns: list[str] | None,
) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header.")

        episode_field = (
            _find_field_case_insensitive(reader.fieldnames, [episode_column])
            if episode_column
            else _find_field_case_insensitive(
                reader.fieldnames,
                ["episode_id", "episode", "id"],
            )
        )
        if episode_field is None:
            raise ValueError(
                "Answer-key CSV needs one of: episode_id, episode, id."
            )

        fixed_instruction_fields = (
            _resolve_requested_fields(reader.fieldnames, answer_columns)
            or _detect_ordered_answer_fields(reader.fieldnames)
        )

        blob_field = None
        if not fixed_instruction_fields:
            blob_field = _find_field_case_insensitive(
                reader.fieldnames,
                ["expected_instructions", "instructions", "expected"],
            )

        answer_key: dict[str, list[str]] = {}
        for row in reader:
            episode_id = str(row.get(episode_field, "")).strip()
            if not episode_id:
                continue

            expected: list[str] = []
            if fixed_instruction_fields:
                for field in fixed_instruction_fields:
                    text = str(row.get(field, "")).strip()
                    if text:
                        expected.append(text)
            elif blob_field:
                blob = str(row.get(blob_field, "")).strip()
                if blob:
                    expected = [part.strip() for part in blob.split("||") if part.strip()]

            answer_key[episode_id] = expected
        return answer_key


def _resolve_expected_instructions(
    episode_id: str,
    cli_expected: list[str],
    answer_key: dict[str, list[str]] | None,
) -> list[str]:
    if cli_expected:
        return cli_expected
    if not answer_key:
        return []

    for key in _candidate_episode_keys(episode_id):
        if key in answer_key:
            return answer_key[key]
    return []


def _existing_episode_numbers(output_dir: Path) -> list[int]:
    numbers: list[int] = []
    for csv_path in output_dir.glob("episode_*.csv"):
        match = re.fullmatch(r"episode_(\d+)\.csv", csv_path.name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def _infer_next_episode_id(output_dir: Path) -> str:
    existing = _existing_episode_numbers(output_dir)
    next_number = (existing[-1] + 1) if existing else 1
    return f"episode_{next_number:03d}"


def _normalize_episode_id(raw_episode_id: str | None, output_dir: Path) -> str:
    if not raw_episode_id:
        return _infer_next_episode_id(output_dir)

    text = str(raw_episode_id).strip()
    if text.isdigit():
        return f"episode_{int(text):03d}"

    match = re.fullmatch(r"episode_(\d+)", text)
    if match:
        return f"episode_{int(match.group(1)):03d}"

    return text


def _make_rows(
    episode_id: str,
    payload: dict[str, Any],
    source_json_path: Path,
    expected_instructions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    saved_keyframes = payload.get("saved_keyframes", [])
    created_at = _utc_now()
    predicted_instructions = [
        str(item.get("instruction", "")).strip()
        for item in saved_keyframes
    ]
    descriptions = [
        str(item.get("description", "")).strip()
        for item in saved_keyframes
    ]

    count_match = len(predicted_instructions) == len(expected_instructions)
    instruction_match_flags: list[bool] = []
    rows: list[dict[str, Any]] = []

    max_len = max(len(saved_keyframes), len(expected_instructions), 1)
    for index in range(max_len):
        item = saved_keyframes[index] if index < len(saved_keyframes) else {}
        predicted = predicted_instructions[index] if index < len(predicted_instructions) else ""
        expected = expected_instructions[index] if index < len(expected_instructions) else ""
        matches = (
            bool(expected)
            and _normalize_instruction(predicted) == _normalize_instruction(expected)
        )
        instruction_match_flags.append(matches)

        rows.append(
            {
                "episode_id": episode_id,
                "created_at_utc": created_at,
                "source_json_path": str(source_json_path),
                "global_instruction": str(payload.get("instruction", "")).strip(),
                "num_saved_keyframes": len(saved_keyframes),
                "order": item.get("order", index + 1),
                "frame_idx": item.get("frame_idx", ""),
                "cropped_keyframe_path": item.get("cropped_keyframe_path", ""),
                "predicted_instruction": predicted,
                "predicted_description": descriptions[index] if index < len(descriptions) else "",
                "expected_instruction": expected,
                "instruction_match": matches if expected else "",
            }
        )

    has_expected = bool(expected_instructions)
    all_instruction_match = (
        has_expected
        and count_match
        and len(predicted_instructions) == len(expected_instructions)
        and all(instruction_match_flags[: len(expected_instructions)])
    )
    episode_pass = all_instruction_match if has_expected else ""

    for row in rows:
        row["expected_count"] = len(expected_instructions)
        row["count_match"] = count_match if has_expected else ""
        row["all_instruction_match"] = all_instruction_match if has_expected else ""
        row["episode_pass"] = episode_pass

    summary = {
        "episode_id": episode_id,
        "created_at_utc": rows[0]["created_at_utc"],
        "source_json_path": str(source_json_path),
        "global_instruction": str(payload.get("instruction", "")).strip(),
        "num_saved_keyframes": len(saved_keyframes),
        "expected_count": len(expected_instructions),
        "count_match": count_match if has_expected else "",
        "all_instruction_match": all_instruction_match if has_expected else "",
        "episode_pass": episode_pass,
        "predicted_instructions": _join_instructions(predicted_instructions),
        "expected_instructions": _join_instructions(expected_instructions),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _upsert_summary(summary_csv_path: Path, summary_row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if summary_csv_path.exists():
        with summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("episode_id") != summary_row["episode_id"]:
                    rows.append(row)

    rows.append(summary_row)
    rows.sort(key=lambda row: row["episode_id"])
    _write_csv(summary_csv_path, rows)


def snapshot_episode(
    keyframes_json_path: Path,
    output_dir: Path,
    episode_id: str | None,
    cli_expected: list[str],
    answer_key: dict[str, list[str]] | None,
) -> dict[str, Path | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)

    normalized_episode_id = _normalize_episode_id(episode_id, output_dir)
    payload = _load_keyframes_payload(keyframes_json_path)
    expected_instructions = _resolve_expected_instructions(
        normalized_episode_id,
        cli_expected,
        answer_key,
    )

    rows, summary = _make_rows(
        episode_id=normalized_episode_id,
        payload=payload,
        source_json_path=keyframes_json_path,
        expected_instructions=expected_instructions,
    )

    csv_path = output_dir / f"{normalized_episode_id}.csv"
    json_snapshot_path = output_dir / "json" / f"{normalized_episode_id}.json"
    summary_csv_path = output_dir / "summary.csv"

    _write_csv(csv_path, rows)
    with json_snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _upsert_summary(summary_csv_path, summary)

    return {
        "episode_id": normalized_episode_id,
        "csv_path": csv_path,
        "json_snapshot_path": json_snapshot_path,
        "summary_csv_path": summary_csv_path,
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def watch_for_updates(
    keyframes_json_path: Path,
    output_dir: Path,
    poll_interval: float,
    max_episodes: int | None,
    cli_expected: list[str],
    answer_key: dict[str, list[str]] | None,
    start_episode: str | None,
) -> int:
    saved_count = 0
    last_digest: str | None = None
    next_episode_hint = start_episode

    print(f"[watch] watching: {keyframes_json_path}")
    print(f"[watch] output dir: {output_dir}")

    while True:
        if max_episodes is not None and saved_count >= max_episodes:
            print(f"[watch] reached max episodes: {max_episodes}")
            return 0

        if not keyframes_json_path.exists():
            time.sleep(poll_interval)
            continue

        try:
            current_digest = _file_digest(keyframes_json_path)
        except OSError:
            time.sleep(poll_interval)
            continue

        if current_digest == last_digest:
            time.sleep(poll_interval)
            continue

        try:
            result = snapshot_episode(
                keyframes_json_path=keyframes_json_path,
                output_dir=output_dir,
                episode_id=next_episode_hint,
                cli_expected=cli_expected,
                answer_key=answer_key,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[watch] waiting for a valid keyframes.json: {exc}")
            time.sleep(poll_interval)
            continue

        last_digest = current_digest
        saved_count += 1

        episode_id = str(result["episode_id"])
        print(f"[watch] saved {episode_id}")
        print(f"        csv : {result['csv_path']}")
        print(f"        json: {result['json_snapshot_path']}")

        episode_match = re.fullmatch(r"episode_(\d+)", episode_id)
        if episode_match:
            next_episode_hint = f"episode_{int(episode_match.group(1)) + 1:03d}"
        else:
            next_episode_hint = None

        time.sleep(poll_interval)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save temp/keyframes.json into per-episode CSV files."
    )
    parser.add_argument(
        "--keyframes-json",
        type=Path,
        default=_DEFAULT_KEYFRAMES_JSON,
        help=f"Path to keyframes.json (default: {_DEFAULT_KEYFRAMES_JSON})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for episode CSV snapshots (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Episode id like 1, 001, or episode_001. Default: next available.",
    )
    parser.add_argument(
        "--expected",
        action="append",
        default=[],
        help="Expected instruction for the current episode. Repeat this flag once per step.",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help=(
            "Optional CSV/JSON answer key. "
            "Examples: config/muffin_episode_task_list_24.csv"
        ),
    )
    parser.add_argument(
        "--answer-episode-column",
        type=str,
        default=None,
        help=(
            "Optional answer-key episode column name. "
            "Default auto-detects episode_id/episode/id."
        ),
    )
    parser.add_argument(
        "--answer-columns",
        type=str,
        default=None,
        help=(
            "Optional comma-separated ordered answer columns, for example "
            "'task_1,task_2,task_3'. Default auto-detects common patterns."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch keyframes.json and snapshot every newly written episode.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between checks in watch mode.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Stop after saving this many episodes in watch mode.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    keyframes_json_path: Path = args.keyframes_json.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    answer_key: dict[str, list[str]] | None = None
    if args.answer_key is not None:
        answer_key = _load_answer_key_with_options(
            answer_key_path=args.answer_key.expanduser().resolve(),
            episode_column=args.answer_episode_column,
            answer_columns=_split_answer_columns(args.answer_columns),
        )

    if args.watch:
        return watch_for_updates(
            keyframes_json_path=keyframes_json_path,
            output_dir=output_dir,
            poll_interval=args.poll_interval,
            max_episodes=args.max_episodes,
            cli_expected=args.expected,
            answer_key=answer_key,
            start_episode=args.episode_id,
        )

    if not keyframes_json_path.exists():
        parser.error(f"keyframes.json not found: {keyframes_json_path}")

    result = snapshot_episode(
        keyframes_json_path=keyframes_json_path,
        output_dir=output_dir,
        episode_id=args.episode_id,
        cli_expected=args.expected,
        answer_key=answer_key,
    )

    print(f"saved episode : {result['episode_id']}")
    print(f"csv          : {result['csv_path']}")
    print(f"json         : {result['json_snapshot_path']}")
    print(f"summary      : {result['summary_csv_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
