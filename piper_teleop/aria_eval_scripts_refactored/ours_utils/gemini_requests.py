"""Gemini request helpers for ours eval scripts."""

from __future__ import annotations

import json
import re
import threading
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from google import genai
from google.genai import types

from .command_queue import (
    CmdNoOp,
    CmdRemove,
    KeyframeQueue,
    RawAppendElement,
    RawAppendMany,
    RawCommand,
)
from .intervals import mid_frame_from_interval
from .parsing import extract_json_text


@dataclass(frozen=True)
class GeminiRequestConfig:
    model: str
    fps: float
    thinking_budget: int
    request_timeout_sec: float
    max_attempts: int
    retry_backoff_sec: float
    voiced_video: bool
    max_intervals: int
    video_record_fps: float
    visual_queue: bool
    visual_queue_image_size: int
    visual_queue_max_images: int
    visual_queue_jpeg_quality: int
    visual_queue_thumb_cache_max: int


_VISUAL_QUEUE_THUMB_CACHE = {}
_PROMPT_FIELD_RE = re.compile(
    r"\{("
    r"duration|fps|max_intervals|transcript_section|segment_duration|"
    r"t_start|t_end|queue_state|visual_queue_section|current_idx|"
    r"current_instruction"
    r")(?::([^{}]+))?\}"
)


def render_prompt_template(template: str, **values) -> str:
    """Render known prompt placeholders while leaving JSON braces intact."""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        fmt = match.group(2)
        value = values.get(name, "")
        if fmt:
            return ("{:" + fmt + "}").format(value)
        return str(value)

    rendered = _PROMPT_FIELD_RE.sub(replace, template)
    return rendered.replace("{{", "{").replace("}}", "}")


def short_error_reason(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    text = re.sub(r"\s+", " ", text)
    return text[:300]


def call_gemini_once(
    video_bytes: bytes,
    prompt: str,
    api_key: str,
    config: GeminiRequestConfig,
    extra_parts: Optional[List] = None,
) -> Tuple[dict, float]:
    """Make one Gemini request and parse its JSON response."""
    client = genai.Client(api_key=api_key)

    if config.voiced_video:
        video_part = types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
        )
    else:
        video_part = types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=config.fps),
        )

    parts = [video_part]
    if extra_parts:
        parts.extend(extra_parts)
    parts.append(types.Part(text=prompt))

    t0 = time.time()
    response = client.models.generate_content(
        model=config.model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_budget=config.thinking_budget),
            response_mime_type="application/json",
        ),
    )
    latency = time.time() - t0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_text = response.text or ""

    try:
        parsed = json.loads(extract_json_text(raw_text))
    except json.JSONDecodeError as exc:
        snippet = re.sub(r"\s+", " ", raw_text).strip()[:240]
        raise ValueError(
            f"JSON parse failed: {exc}; response_snippet={snippet!r}"
        ) from exc

    return parsed, round(latency, 3)


def call_gemini(
    video_bytes: bytes,
    prompt: str,
    api_key: str,
    config: GeminiRequestConfig,
    extra_parts: Optional[List] = None,
) -> Tuple[dict, float]:
    """Call Gemini with timeout/retry, returning parsed JSON and latency."""
    last_error: Optional[BaseException] = None
    last_reason = "unknown error"

    for attempt in range(1, config.max_attempts + 1):
        result: dict = {}
        done = threading.Event()

        def _target():
            try:
                result["value"] = call_gemini_once(
                    video_bytes, prompt, api_key, config, extra_parts=extra_parts)
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=_target,
            daemon=True,
            name=f"GeminiRequestAttempt-{attempt}",
        ).start()

        will_retry = attempt < config.max_attempts
        if not done.wait(timeout=config.request_timeout_sec):
            last_reason = f"timeout after {config.request_timeout_sec:.1f}s"
            suffix = "retrying with a fresh request." if will_retry else "no attempts left."
            print(f"[VLM] Gemini attempt {attempt}/{config.max_attempts} "
                  f"{last_reason}; {suffix}")
        elif "error" in result:
            last_error = result["error"]
            last_reason = short_error_reason(last_error)
            suffix = "retrying." if will_retry else "no attempts left."
            print(f"[VLM] Gemini attempt {attempt}/{config.max_attempts} failed: "
                  f"{last_reason}; {suffix}")
        else:
            return result["value"]

        if will_retry and config.retry_backoff_sec > 0:
            time.sleep(config.retry_backoff_sec)

    raise RuntimeError(
        f"Gemini failed after {config.max_attempts} attempts: {last_reason}"
    ) from last_error


def _select_visual_queue_indices(total: int, current_idx: int,
                                 max_images: int) -> List[int]:
    """Pick a bounded visual subset: current item plus recent queue tail."""
    total = max(0, int(total))
    max_images = max(0, int(max_images))
    if total <= 0 or max_images <= 0:
        return []
    if total <= max_images:
        return list(range(total))

    selected: List[int] = []

    def add(idx: int):
        if 0 <= idx < total and idx not in selected and len(selected) < max_images:
            selected.append(idx)

    add(current_idx)

    tail_budget = max_images - len(selected)
    for idx in range(max(0, total - tail_budget), total):
        add(idx)

    for idx in range(current_idx - 1, -1, -1):
        add(idx)
        if len(selected) >= max_images:
            break

    return sorted(selected)


def _encode_visual_queue_thumb(kf_img: np.ndarray,
                               resize: int,
                               quality: int,
                               cache_max: int) -> Optional[bytes]:
    cache_key = (id(kf_img), int(resize), int(quality))
    cached = _VISUAL_QUEUE_THUMB_CACHE.get(cache_key)
    if cached is not None:
        return cached

    img = cv2.resize(kf_img, (resize, resize), interpolation=cv2.INTER_AREA)
    ok, jpg_bytes = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return None

    if len(_VISUAL_QUEUE_THUMB_CACHE) >= cache_max:
        _VISUAL_QUEUE_THUMB_CACHE.clear()

    data = jpg_bytes.tobytes()
    _VISUAL_QUEUE_THUMB_CACHE[cache_key] = data
    return data


def load_queue_images_as_parts(
    queue: KeyframeQueue,
    config: GeminiRequestConfig,
) -> Tuple[List, int, int]:
    """Load a bounded keyframe-image subset for Gemini visual context."""
    parts = []
    with queue._lock:
        total = len(queue._items)
        indices = _select_visual_queue_indices(
            total, queue._current_idx, config.visual_queue_max_images)
        snapshot = [
            (idx, queue._items[idx][0], queue._items[idx][1])
            for idx in indices
            if queue._items[idx][0] is not None
        ]

    for idx, kf_img, instr in snapshot:
        jpg_data = _encode_visual_queue_thumb(
            kf_img,
            config.visual_queue_image_size,
            config.visual_queue_jpeg_quality,
            config.visual_queue_thumb_cache_max,
        )
        if jpg_data is None:
            continue
        parts.append(types.Part(
            text=f"[Queue index {idx}] {instr}:",
        ))
        parts.append(types.Part(
            inline_data=types.Blob(data=jpg_data, mime_type="image/jpeg"),
        ))

    return parts, len(snapshot), total


def query_gemini_initial(
    video_bytes: bytes,
    transcript_section: str,
    api_key: str,
    duration_sec: float,
    prompt_template: str,
    config: GeminiRequestConfig,
):
    """One-shot call before robot execution to build the initial queue."""
    prompt = render_prompt_template(
        prompt_template,
        duration=duration_sec,
        fps=float(config.fps),
        transcript_section=transcript_section,
        max_intervals=config.max_intervals,
    )
    #breakpoint()
    parsed, latency = call_gemini(video_bytes, prompt, api_key, config)
    return parsed, latency


def query_gemini_async_update(
    video_bytes: bytes,
    queue: KeyframeQueue,
    seg_start_sec: float,
    seg_end_sec: float,
    transcript_section: str,
    api_key: str,
    prompt_template: str,
    config: GeminiRequestConfig,
):
    """Periodic async call during robot execution; returns a queue command."""
    extra_parts = None
    visual_queue_note = ""
    if config.visual_queue and queue.total > 0:
        extra_parts, visual_count, queue_total = load_queue_images_as_parts(
            queue, config)
        visual_queue_note = (
            "\nA bounded subset of keyframe images from the queue is attached "
            "above (labelled by queue index). The text queue_state is the "
            "complete queue; use attached images only as lightweight visual "
            "context for object identity.\n"
        )
        if visual_count < queue_total:
            print(
                f"[AsyncVLM] Visual queue context: attached "
                f"{visual_count}/{queue_total} keyframe image(s)."
            )

    prompt = render_prompt_template(
        prompt_template,
        queue_state=queue.queue_state_str(),
        current_idx=queue.current_idx,
        current_instruction=queue.current_instruction or "(none)",
        t_start=seg_start_sec,
        t_end=seg_end_sec,
        segment_duration=(seg_end_sec - seg_start_sec),
        fps=float(config.fps),
        visual_queue_section=visual_queue_note,
        transcript_section=transcript_section,
    )
    #breakpoint()
    parsed, latency = call_gemini(
        video_bytes, prompt, api_key, config, extra_parts=extra_parts)
    command_name = parsed.get("command", "NoOp")
    target_idx = parsed.get("target_idx")
    if target_idx is not None:
        try:
            target_idx = int(target_idx)
        except (TypeError, ValueError):
            print(f"[AsyncVLM] Invalid target_idx={target_idx!r} - treating as null")
            target_idx = None

    print(f"[AsyncVLM] command={command_name}  latency={latency:.1f}s  "
          f"reasoning={parsed.get('reasoning', '')}")

    if command_name == "NoOp" or command_name not in (
            "Append", "Replace", "Preempt", "Remove"):
        return CmdNoOp(), parsed, latency

    if command_name == "Remove":
        indices = parsed.get("target_indices")
        if indices and isinstance(indices, list):
            parsed_indices = []
            for idx in indices:
                try:
                    parsed_indices.append(int(idx))
                except (TypeError, ValueError):
                    print(f"[AsyncVLM] Remove: skipping invalid index {idx!r}")
            if parsed_indices:
                return CmdRemove(target_indices=parsed_indices), parsed, latency
        if target_idx is not None:
            return CmdRemove(target_indices=[int(target_idx)]), parsed, latency
        print("[AsyncVLM] Remove: missing target_indices/target_idx - NoOp")
        return CmdNoOp(), parsed, latency

    num_seg_frames_orig = int((seg_end_sec - seg_start_sec) * config.video_record_fps)

    new_elements = parsed.get("new_elements", [])
    if not isinstance(new_elements, list):
        new_elements = []

    if not new_elements:
        interval = parsed.get("keyframe_interval")
        instruction = (parsed.get("instruction") or "").strip()
        if interval is None:
            print(f"[AsyncVLM] {command_name}: missing new_elements/keyframe_interval - NoOp")
            return CmdNoOp(), parsed, latency
        local_frame_idx = mid_frame_from_interval(
            interval, num_seg_frames_orig, config.video_record_fps)
        return RawCommand(
            name=command_name,
            target_idx=target_idx,
            local_frame_idx=local_frame_idx,
            instruction=instruction,
        ), parsed, latency

    if command_name == "Append":
        raw_elements: List[RawAppendElement] = []
        for elem in new_elements:
            if not isinstance(elem, dict):
                continue
            interval = elem.get("keyframe_interval")
            instruction = (elem.get("subtask_instruction")
                           or elem.get("instruction") or "").strip()
            description = (elem.get("keyframe_description") or "").strip()
            if not instruction or not isinstance(interval, dict):
                print("[AsyncVLM] Append: skipping malformed new_element "
                      f"{elem}")
                continue
            local_frame_idx = mid_frame_from_interval(
                interval, num_seg_frames_orig, config.video_record_fps)
            raw_elements.append(RawAppendElement(
                local_frame_idx=local_frame_idx,
                instruction=instruction,
                keyframe_description=description,
            ))

        raw_elements.sort(key=lambda elem: elem.local_frame_idx)
        if not raw_elements:
            print("[AsyncVLM] Append: no valid new_elements - NoOp")
            return CmdNoOp(), parsed, latency
        if len(raw_elements) > 1:
            print(f"[AsyncVLM] Append: parsed {len(raw_elements)} keyframes "
                  "from one clip")
            return RawAppendMany(elements=raw_elements), parsed, latency

        elem = raw_elements[0]
        return RawCommand(
            name="Append",
            target_idx=None,
            local_frame_idx=elem.local_frame_idx,
            instruction=elem.instruction,
            keyframe_description=elem.keyframe_description,
        ), parsed, latency

    elem = new_elements[0]
    if not isinstance(elem, dict):
        print(f"[AsyncVLM] {command_name}: malformed new_element - NoOp")
        return CmdNoOp(), parsed, latency
    interval = elem.get("keyframe_interval", {})
    instruction = (elem.get("subtask_instruction")
                   or elem.get("instruction") or "").strip()
    description = elem.get("keyframe_description", "")
    local_frame_idx = mid_frame_from_interval(
        interval, num_seg_frames_orig, config.video_record_fps)

    return RawCommand(
        name=command_name,
        target_idx=target_idx,
        local_frame_idx=local_frame_idx,
        instruction=instruction,
        keyframe_description=description,
    ), parsed, latency
