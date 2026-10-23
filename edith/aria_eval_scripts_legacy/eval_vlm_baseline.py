#!/usr/bin/env python3
"""
Gemini VLM Baseline Evaluation for Keyframe Detection
======================================================
Sends each episode's video + transcript instruction to Gemini,
asks it to predict keyframe time range, and evaluates against
ground-truth labels.

Usage:
  python eval_vlm_baseline.py --model gemini-2.5-flash
  python eval_vlm_baseline.py --model gemini-2.5-flash --video-fps 4 --caption --gaze-annot
  python eval_vlm_baseline.py --model gemini-2.5-pro --resume --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import tempfile

import cv2
import numpy as np
from google import genai
from google.genai import types

# ── constants ─────────────────────────────────────────────────────────────
VIDEO_FPS = 15  # ground-truth frame rate

PRICING = {
    "gemini-2.5-flash": {
        "input": 0.15,
        "output": 0.60,
        "thinking": 3.50,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "thinking": 10.00,
    },
    "gemini-3-flash-preview": {
        "input": 0.30,
        "output": 1.20,
        "thinking": 7.00,
    },
}

PROMPT_TEMPLATE = """\
You are analyzing a first-person (egocentric) video recorded in a store.
The wearer is looking at items on shelves and interacts with a specific object.

A spoken instruction was given:
  "{transcript}"

{config_note}

Your task: identify the **keyframe time range** — the precise time interval
(in seconds) that best captures the moment of interaction with the target
object mentioned in the instruction.

The keyframe is determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object.
  - Hand gesture: when the hand reaches toward, touches, or grasps the object.
  - Gaze fixation: when the wearer's gaze stabilises on the target object
    (if gaze information is available).
The true keyframe is the moment where these signals align — typically the
hand approaches the object that has been mentioned in the utterance and/or
that the wearer is fixating on.

The keyframe segment is typically short (0.2–2.0 s).
You must always identify this moment — even if the interaction is brief
or partially visible.

The video is {duration:.1f} seconds long (recorded at 15 FPS).

Return your answer as JSON:
{{
  "reasoning": "<brief explanation of which cues you used>",
  "keyframe_start": <float, seconds>,
  "keyframe_end": <float, seconds>
}}

Important: always return positive start/end values. Never return -1.
"""

# ── Config-specific prompt notes (2x2: caption × gaze) ───────────────────
PROMPT_CONFIG_NOTES = {
    "no_cap_no_gaze": (
        "You only have the raw video and the spoken instruction text above.\n"
        "Available cues:\n"
        "  - Utterance content: use the transcript text to understand WHAT "
        "the target object is, and reason about WHEN in the video the "
        "wearer would act on it.\n"
        "  - Hand gesture: watch for the hand reaching toward, touching, or "
        "grasping a shelf object that matches the instruction.\n"
        "Since you cannot see gaze directly, focus on the temporal "
        "correlation between the spoken object name and the hand movement "
        "toward a matching item."
    ),
    "cap_only": (
        "The video has real-time subtitle captions burned in at the top, "
        "showing the spoken instruction as it is said.\n"
        "Available cues:\n"
        "  - Utterance timing (visible as captions): note exactly WHEN "
        "the target object name appears in the subtitles — the keyframe "
        "typically occurs during or shortly after this moment.\n"
        "  - Hand gesture: look for the hand reaching toward an object "
        "that matches the caption content.\n"
        "Correlate the two: the keyframe is where the caption mentions "
        "the target AND the hand begins moving toward it."
    ),
    "gaze_only": (
        "The video has a green dot overlay indicating where the wearer is "
        "looking (eye gaze).\n"
        "Available cues:\n"
        "  - Gaze fixation: when the green dot stops moving and stays "
        "fixed on a specific object, the wearer is fixating on it. "
        "This often precedes or coincides with the hand reaching for "
        "that object.\n"
        "  - Hand gesture: watch for the hand approaching the location "
        "where the gaze dot is fixated.\n"
        "  - Utterance content: use the transcript text to identify the "
        "target object.\n"
        "The keyframe is the moment when gaze fixation and hand approach "
        "converge on the same object mentioned in the instruction."
    ),
    "cap_gaze": (
        "The video has both real-time subtitle captions (at the top) and a "
        "green gaze dot (showing where the wearer is looking).\n"
        "Available cues — use ALL of them:\n"
        "  - Utterance timing (captions): note when the target object "
        "name appears in the subtitles.\n"
        "  - Gaze fixation: when the green dot stabilises on an object, "
        "the wearer is fixating on it.\n"
        "  - Hand gesture: when the hand reaches toward the fixated object.\n"
        "The keyframe is the moment where all three signals converge — "
        "the caption mentions the target, the gaze is fixated on it, and "
        "the hand reaches toward it. If the cues don't perfectly align, "
        "prioritise the moment of hand approach to the gaze-fixated object."
    ),
}


def get_config_name(caption: bool, gaze_annot: bool) -> str:
    """Return a short config identifier for the caption×gaze combination."""
    if caption and gaze_annot:
        return "cap_gaze"
    elif caption:
        return "cap_only"
    elif gaze_annot:
        return "gaze_only"
    else:
        return "no_cap_no_gaze"


# ══════════════════════════════════════════════════════════════════════════
#  VIDEO ENCODING HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _ffmpeg_reencode_h264(input_path: str, output_path: str) -> None:
    """Re-encode a video file to H.264 MP4 using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-an", output_path],
        capture_output=True, check=True,
    )


def _save_as_h264(video_bytes: bytes, output_path: str) -> None:
    """Write video bytes and re-encode to H.264 MP4 for player compatibility."""
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(video_bytes)
    tmp_in.close()
    _ffmpeg_reencode_h264(tmp_in.name, output_path)
    os.unlink(tmp_in.name)


# ══════════════════════════════════════════════════════════════════════════
#  VIDEO PREPARATION (resize → overlay → H.264 encode)
# ══════════════════════════════════════════════════════════════════════════
def _get_active_caption(words: list[dict], time_sec: float) -> str:
    """Return the concatenation of words whose [start, end] spans time_sec."""
    active = [w["word"] for w in words if w["start"] <= time_sec <= w["end"]]
    return " ".join(active)


def _draw_caption(frame: np.ndarray, text: str) -> None:
    """Draw centred caption with semi-transparent background at the top.

    Font size scales with frame height for readability at any resolution.
    """
    if not text:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, h / 512)
    thickness = max(1, int(h / 256))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = int(20 * h / 256)  # scale margin with resolution

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (tx - 6, ty - th - 6),
                  (tx + tw + 6, ty + baseline + 6),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (tx, ty),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_gaze(frame: np.ndarray, gx: float, gy: float) -> None:
    """Draw a green gaze marker at (gx, gy). Radius scales with resolution."""
    h, w = frame.shape[:2]
    x = int(np.clip(gx, 0, w - 1))
    y = int(np.clip(gy, 0, h - 1))
    radius = max(3, int(h / 50))
    cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
    cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), 1)


def prepare_video(
    video_path: str,
    caption: bool = False,
    gaze_annot: bool = False,
    transcript: dict | None = None,
    gaze_data: list[dict] | None = None,
    target_resolution: tuple[int, int] | None = None,
) -> bytes:
    """Prepare video for VLM input.

    Pipeline: read frame → resize (if target_resolution) → overlay → encode.
    Overlays are drawn AFTER resizing so that text and markers remain crisp.
    Output is H.264 encoded for broad compatibility.

    If no processing is needed, returns the raw file bytes as-is.
    """
    needs_processing = caption or gaze_annot or (target_resolution is not None)
    if not needs_processing:
        with open(video_path, "rb") as f:
            return f.read()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine output size
    if target_resolution is not None:
        out_w, out_h = target_resolution
    else:
        out_w, out_h = orig_w, orig_h

    words = transcript.get("words", []) if transcript and caption else []

    # Build frame-indexed gaze lookup (scale coords to output resolution)
    gaze_by_frame: dict[int, tuple[float, float]] = {}
    if gaze_annot and gaze_data:
        sx = out_w / orig_w
        sy = out_h / orig_h
        for entry in gaze_data:
            idx = entry["frame_idx"]
            gaze_by_frame[idx] = (entry["gaze_x"] * sx, entry["gaze_y"] * sy)

    tmp_raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_raw.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw.name, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Resize FIRST
        if target_resolution is not None and (orig_w != out_w or orig_h != out_h):
            frame = cv2.resize(frame, (out_w, out_h),
                               interpolation=cv2.INTER_LANCZOS4)

        time_sec = frame_idx / fps

        # 2. Then overlay on the resized frame (crisp text & markers)
        if caption and words:
            text = _get_active_caption(words, time_sec)
            _draw_caption(frame, text)

        if gaze_annot and frame_idx in gaze_by_frame:
            gx, gy = gaze_by_frame[frame_idx]
            _draw_gaze(frame, gx, gy)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to H.264 for broad compatibility (VSCode, browsers, etc.)
    tmp_h264 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_h264.close()
    _ffmpeg_reencode_h264(tmp_raw.name, tmp_h264.name)

    with open(tmp_h264.name, "rb") as f:
        data = f.read()
    os.unlink(tmp_raw.name)
    os.unlink(tmp_h264.name)
    return data


# ══════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════
def load_ground_truth(episode_dir: str) -> dict | None:
    """Load labels.npy → extract keyframe segment and per-frame labels."""
    labels_path = os.path.join(episode_dir, "labels.npy")
    if not os.path.exists(labels_path):
        return None

    labels = np.load(labels_path).astype(np.int8)
    T = len(labels)

    # Find contiguous segment boundaries
    diffs = np.diff(labels)
    starts = list(np.where(diffs == 1)[0] + 1)
    ends = list(np.where(diffs == -1)[0] + 1)
    if labels[0] == 1:
        starts = [0] + starts
    if labels[-1] == 1:
        ends = ends + [T]

    if not starts:
        return None

    kf_start_frame = int(starts[0])
    kf_end_frame = int(ends[0] - 1)  # inclusive

    kf_start_sec = kf_start_frame / VIDEO_FPS
    kf_end_sec = kf_end_frame / VIDEO_FPS

    # Set of all GT keyframe indices (for frame-level hit check)
    gt_kf_set = set(np.where(labels == 1)[0].tolist())

    return {
        "total_frames": T,
        "duration_sec": T / VIDEO_FPS,
        "kf_start_frame": kf_start_frame,
        "kf_end_frame": kf_end_frame,
        "kf_start_sec": round(kf_start_sec, 4),
        "kf_end_sec": round(kf_end_sec, 4),
        "kf_mid_sec": round((kf_start_sec + kf_end_sec) / 2, 4),
        "kf_duration_sec": round(kf_end_sec - kf_start_sec, 4),
        "labels": labels,          # raw per-frame binary array
        "gt_kf_set": gt_kf_set,    # set of frame indices where label==1
    }


# ══════════════════════════════════════════════════════════════════════════
#  GEMINI API
# ══════════════════════════════════════════════════════════════════════════
def call_gemini(
    client: genai.Client,
    model_name: str,
    video_bytes: bytes,
    prompt: str,
    video_fps: int = 2,
):
    """Send video + prompt to Gemini and return the raw response object."""
    parts = [
        types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=video_fps),
        ),
        types.Part(text=prompt),
    ]

    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            response_mime_type="application/json",
        ),
    )
    return response


def parse_response(response) -> dict | None:
    """Extract keyframe_start / keyframe_end from Gemini JSON response."""
    try:
        text = response.text
        result = json.loads(text)
        if "keyframe_start" in result and "keyframe_end" in result:
            start = float(result["keyframe_start"])
            end = float(result["keyframe_end"])
            # Treat negative values as "model couldn't find it" → failure
            if start < 0 or end < 0:
                return None
            # Swap if needed
            if start > end:
                start, end = end, start
            return {
                "keyframe_start": start,
                "keyframe_end": end,
                "reasoning": result.get("reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    return None


def extract_cost(response, model_name: str) -> dict:
    """Extract token usage and compute cost in USD."""
    usage = response.usage_metadata
    input_tok = getattr(usage, "prompt_token_count", 0) or 0
    output_tok = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tok = getattr(usage, "thoughts_token_count", 0) or 0

    p = PRICING.get(model_name, PRICING["gemini-2.5-flash"])
    cost = (
        input_tok * p["input"] / 1_000_000
        + output_tok * p["output"] / 1_000_000
        + thinking_tok * p["thinking"] / 1_000_000
    )
    return {
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "thinking_tokens": thinking_tok,
        "cost_usd": round(cost, 6),
    }


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════
def evaluate_episode(
    pred: dict, gt: dict, tolerances: list[float],
    frame_tolerances: list[int] | None = None,
) -> dict:
    """Compute all metrics for one episode.

    Core metric: predicted midpoint frame falls within GT keyframe labels.
    The VLM predicts [start, end] in seconds → midpoint timestamp →
    convert to frame index → check if labels[frame] == 1.
    """
    if frame_tolerances is None:
        frame_tolerances = [0, 1, 3, 5]

    ps, pe = pred["keyframe_start"], pred["keyframe_end"]
    gs, ge = gt["kf_start_sec"], gt["kf_end_sec"]
    T = gt["total_frames"]
    gt_kf_set = gt["gt_kf_set"]

    # Clamp predictions to valid range
    dur = gt["duration_sec"]
    ps = max(0.0, min(ps, dur))
    pe = max(0.0, min(pe, dur))
    if ps > pe:
        ps, pe = pe, ps

    # ── Primary metric: midpoint frame → GT label hit ─────────────────
    pred_mid_sec = (ps + pe) / 2
    pred_mid_frame = int(round(pred_mid_sec * VIDEO_FPS))
    pred_mid_frame = max(0, min(T - 1, pred_mid_frame))

    # Exact hit: labels[pred_mid_frame] == 1
    mid_frame_hit = pred_mid_frame in gt_kf_set

    # Tolerance-based frame hits: any frame within ±k of pred_mid_frame
    # is a GT keyframe?
    frame_hits = {}
    for k in frame_tolerances:
        lo = max(0, pred_mid_frame - k)
        hi = min(T - 1, pred_mid_frame + k)
        hit = any(f in gt_kf_set for f in range(lo, hi + 1))
        frame_hits[f"frame_hit@{k}"] = hit

    # Distance to nearest GT keyframe (in frames)
    if gt_kf_set:
        nearest_dist = min(abs(pred_mid_frame - gf) for gf in gt_kf_set)
    else:
        nearest_dist = T

    # ── Segment-level metrics (kept for reference) ────────────────────
    # IoU
    inter_start = max(ps, gs)
    inter_end = min(pe, ge)
    intersection = max(0.0, inter_end - inter_start)
    union = (pe - ps) + (ge - gs) - intersection
    iou = intersection / union if union > 0 else 0.0

    # Coverage / precision
    gt_len = ge - gs
    pred_len = pe - ps
    gt_coverage = intersection / gt_len if gt_len > 0 else 0.0
    pred_precision = intersection / pred_len if pred_len > 0 else 0.0

    # Midpoint error (seconds)
    gt_mid = gt["kf_mid_sec"]
    midpoint_error = abs(pred_mid_sec - gt_mid)

    # Boundary errors
    start_error = abs(ps - gs)
    end_error = abs(pe - ge)

    # Time-based hit rates
    time_hits = {f"hit@{tau}s": midpoint_error <= tau for tau in tolerances}

    return {
        # ─ Frame-level (primary) ─
        "pred_mid_frame": pred_mid_frame,
        "mid_frame_hit": mid_frame_hit,
        "nearest_gt_dist": nearest_dist,
        **frame_hits,
        # ─ Segment-level (secondary) ─
        "iou": round(iou, 4),
        "gt_coverage": round(gt_coverage, 4),
        "pred_precision": round(pred_precision, 4),
        "midpoint_error": round(midpoint_error, 4),
        "start_error": round(start_error, 4),
        "end_error": round(end_error, 4),
        "boundary_error": round((start_error + end_error) / 2, 4),
        **time_hits,
    }


def aggregate_metrics(
    results: list[dict], tolerances: list[float],
    frame_tolerances: list[int] | None = None,
) -> dict:
    """Compute aggregate statistics across all successful episodes."""
    if frame_tolerances is None:
        frame_tolerances = [0, 1, 3, 5]

    successful = [r for r in results if r.get("metrics") is not None]
    n = len(successful)
    if n == 0:
        return {"num_success": 0}

    def _agg(key):
        vals = [r["metrics"][key] for r in successful]
        return {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "std": round(statistics.stdev(vals), 4) if n > 1 else 0.0,
        }

    # ── Frame-level (primary) ─────────────────────────────────────────
    mid_hits = sum(1 for r in successful if r["metrics"]["mid_frame_hit"])
    agg = {
        "num_episodes": len(results),
        "num_success": n,
        "num_failures": len(results) - n,
        # Exact midpoint-frame accuracy
        "mid_frame_accuracy": {
            "hits": mid_hits,
            "total": n,
            "rate": round(mid_hits / n, 4),
        },
        "nearest_gt_dist": _agg("nearest_gt_dist"),
    }

    # Frame-tolerance hit rates: frame_hit@0, @1, @3, @5
    for k in frame_tolerances:
        key = f"frame_hit@{k}"
        hits = sum(1 for r in successful if r["metrics"][key])
        agg[key] = {"hits": hits, "total": n, "rate": round(hits / n, 4)}

    # ── Segment-level (secondary) ─────────────────────────────────────
    agg["iou"] = _agg("iou")
    agg["midpoint_error"] = _agg("midpoint_error")
    agg["gt_coverage"] = _agg("gt_coverage")
    agg["pred_precision"] = _agg("pred_precision")
    agg["start_error"] = _agg("start_error")
    agg["end_error"] = _agg("end_error")
    agg["boundary_error"] = _agg("boundary_error")

    for tau in tolerances:
        key = f"hit@{tau}s"
        hits = sum(1 for r in successful if r["metrics"][key])
        agg[key] = {"hits": hits, "total": n, "rate": round(hits / n, 4)}

    # Total cost
    total_cost = sum(r.get("cost", {}).get("cost_usd", 0) for r in results)
    total_input = sum(r.get("cost", {}).get("input_tokens", 0) for r in results)
    total_output = sum(r.get("cost", {}).get("output_tokens", 0) for r in results)
    total_thinking = sum(r.get("cost", {}).get("thinking_tokens", 0) for r in results)
    agg["total_cost_usd"] = round(total_cost, 4)
    agg["total_input_tokens"] = total_input
    agg["total_output_tokens"] = total_output
    agg["total_thinking_tokens"] = total_thinking

    return agg


# ══════════════════════════════════════════════════════════════════════════
#  RESUME SUPPORT
# ══════════════════════════════════════════════════════════════════════════
def load_existing_results(path: str) -> dict[str, dict]:
    """Load already-completed episode results from JSONL file."""
    completed = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                completed[r["episode"]] = r
    return completed


def append_jsonl(path: str, record: dict):
    """Append a single JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════
def print_summary(
    agg: dict, model: str, tolerances: list[float],
    frame_tolerances: list[int] | None = None,
):
    if frame_tolerances is None:
        frame_tolerances = [0, 1, 3, 5]

    n = agg["num_success"]
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  Gemini VLM Baseline — Evaluation Summary")
    print(f"  Model: {model}  |  Episodes: {agg['num_episodes']}"
          f"  ({n} success, {agg['num_failures']} failures)")
    print(sep)

    if n == 0:
        print("  No successful episodes to report.")
        print(sep)
        return

    # ── Frame-level accuracy (PRIMARY) ────────────────────────────────
    mfa = agg["mid_frame_accuracy"]
    print(f"\n  Frame-Level Accuracy (pred midpoint → GT label):")
    print(f"    Exact match (±0):      {mfa['rate']*100:5.1f}%"
          f"  ({mfa['hits']}/{mfa['total']})")
    for k in frame_tolerances:
        key = f"frame_hit@{k}"
        if key in agg:
            h = agg[key]
            print(f"    frame_hit@±{k:<2d} frames:  {h['rate']*100:5.1f}%"
                  f"  ({h['hits']}/{h['total']})")
    nd = agg["nearest_gt_dist"]
    print(f"    Nearest GT dist:       {nd['mean']:.1f} frames (mean)"
          f"  / {nd['median']:.1f} (median)")

    # ── Segment-level (secondary) ─────────────────────────────────────
    print(f"\n  Segment Overlap:")
    print(f"    Mean IoU:              {agg['iou']['mean']:.3f}"
          f"  (+/- {agg['iou']['std']:.3f})")
    print(f"    Median IoU:            {agg['iou']['median']:.3f}")
    print(f"    Mean GT Coverage:      {agg['gt_coverage']['mean']:.3f}")
    print(f"    Mean Pred Precision:   {agg['pred_precision']['mean']:.3f}")

    print(f"\n  Temporal Accuracy:")
    print(f"    Mean Midpoint Error:   {agg['midpoint_error']['mean']:.2f}s"
          f"  (+/- {agg['midpoint_error']['std']:.2f}s)")
    print(f"    Median Midpoint Error: {agg['midpoint_error']['median']:.2f}s")
    print(f"    Mean Start Error:      {agg['start_error']['mean']:.2f}s")
    print(f"    Mean End Error:        {agg['end_error']['mean']:.2f}s")

    print(f"\n  Time-Based Hit Rate (midpoint within tolerance):")
    for tau in tolerances:
        key = f"hit@{tau}s"
        h = agg[key]
        print(f"    @{tau}s:  {h['rate']*100:5.1f}%  ({h['hits']}/{h['total']})")

    print(f"\n  Cost:")
    print(f"    Total: ${agg['total_cost_usd']:.4f}")
    if n > 0:
        print(f"    Per episode: ${agg['total_cost_usd']/agg['num_episodes']:.4f}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Gemini VLM Baseline Evaluation for Keyframe Detection"
    )
    parser.add_argument("--dataset-dir", default="./dataset",
                        help="Path to dataset/ directory")
    parser.add_argument("--model", default="gemini-2.5-pro",
                        choices=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash", "gemini-3-flash-preview"])
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes that already have results")
    parser.add_argument("--episodes", default=None,
                        help="Comma-separated episode names (default: all)")
    parser.add_argument("--tolerance", default="0.5,1.0,1.5",
                        help="Comma-separated tolerance values in seconds")
    parser.add_argument("--frame-tolerance", default="0,1,3,5",
                        help="Comma-separated frame tolerance values for hit check")
    parser.add_argument("--video-fps", type=int, default=2,
                        help="FPS hint sent to Gemini for video sampling (default: 2)")
    parser.add_argument("--caption", action="store_true",
                        help="Overlay real-time transcript captions on the video")
    parser.add_argument("--gaze-annot", action="store_true",
                        help="Overlay gaze point (green marker) on the video")
    parser.add_argument("--target-resolution", default="256",
                        help="Resize video before overlay, e.g. '512' for "
                             "512x512 or '640x480'. Default 256 (=256x256). "
                             "Set 'none' to keep original resolution. "
                             "Resize happens BEFORE overlays so text stays crisp.")
    parser.add_argument("--api-key",
                        default="")
    parser.add_argument("--debug", action="store_true",
                        help="Save VLM input videos to debug/{config_name}/")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tolerances = [float(t) for t in args.tolerance.split(",")]
    frame_tolerances = [int(t) for t in args.frame_tolerance.split(",")]

    # ── parse target resolution ──
    target_resolution = None
    if args.target_resolution and args.target_resolution.lower() != "none":
        if "x" in args.target_resolution:
            w_str, h_str = args.target_resolution.split("x")
            target_resolution = (int(w_str), int(h_str))
        else:
            n = int(args.target_resolution)
            target_resolution = (n, n)

    # ── discover episodes ──
    dataset_dir = args.dataset_dir
    if args.episodes:
        episode_names = [e.strip() for e in args.episodes.split(",")]
    else:
        episode_names = sorted(
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        )

    # ── experiment tag (encodes config into filenames) ──
    tag_parts = [args.model, f"fps{args.video_fps}"]
    if args.caption:
        tag_parts.append("cap")
    if args.gaze_annot:
        tag_parts.append("gaze")
    exp_tag = "_".join(tag_parts)

    # ── config name for prompt selection & debug dirs ──
    config_name = get_config_name(args.caption, args.gaze_annot)

    print(f"[eval] {len(episode_names)} episodes, model={args.model}, "
          f"video_fps={args.video_fps}, config={config_name}"
          + (f", resolution={target_resolution}" if target_resolution else ""))

    # ── output setup ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── debug dir ──
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join("debug", config_name)
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[debug] saving input videos to {debug_dir}/")
    results_path = os.path.join(
        args.output_dir, f"{exp_tag}_results.jsonl"
    )

    # ── resume ──
    completed = {}
    if args.resume:
        completed = load_existing_results(results_path)
        print(f"[resume] {len(completed)} episodes already done")

    # ── Gemini client ──
    client = genai.Client(api_key=args.api_key)

    all_results: list[dict] = []

    for i, ep_name in enumerate(episode_names):
        # Resume: reuse previous result
        if ep_name in completed:
            all_results.append(completed[ep_name])
            continue

        ep_dir = os.path.join(dataset_dir, ep_name)

        # Load ground truth
        gt = load_ground_truth(ep_dir)
        if gt is None:
            print(f"  [{i+1}/{len(episode_names)}] {ep_name}: SKIP (no labels)")
            continue

        # Load transcript
        transcript_path = os.path.join(ep_dir, "transcript.json")
        if not os.path.exists(transcript_path):
            print(f"  [{i+1}/{len(episode_names)}] {ep_name}: SKIP (no transcript)")
            continue
        with open(transcript_path) as f:
            transcript = json.load(f)

        # Load gaze data (if needed)
        gaze_frames = None
        if args.gaze_annot:
            gaze_path = os.path.join(ep_dir, "gaze.json")
            if os.path.exists(gaze_path):
                with open(gaze_path) as f:
                    gaze_frames = json.load(f).get("frames", [])

        # Load & prepare video (resize → overlay → H.264 encode)
        video_path = os.path.join(ep_dir, "video.mp4")
        if not os.path.exists(video_path):
            print(f"  [{i+1}/{len(episode_names)}] {ep_name}: SKIP (no video)")
            continue
        video_bytes = prepare_video(
            video_path,
            caption=args.caption,
            gaze_annot=args.gaze_annot,
            transcript=transcript,
            gaze_data=gaze_frames,
            target_resolution=target_resolution,
        )

        # Save debug video (H.264 for VSCode compatibility)
        if debug_dir is not None:
            debug_video_path = os.path.join(
                debug_dir, f"input_video_{i}.mp4"
            )
            needs_overlay = args.caption or args.gaze_annot or (target_resolution is not None)
            if needs_overlay:
                # Already H.264 from prepare_video
                with open(debug_video_path, "wb") as fv:
                    fv.write(video_bytes)
            else:
                # Raw bytes from original → re-encode for compatibility
                _save_as_h264(video_bytes, debug_video_path)

        # Build prompt (config-specific notes)
        prompt = PROMPT_TEMPLATE.format(
            transcript=transcript["text"],
            duration=gt["duration_sec"],
            config_note=PROMPT_CONFIG_NOTES[config_name],
        )

        # Call Gemini with retry + timing
        response = None
        prediction = None
        cost_info = {"input_tokens": 0, "output_tokens": 0,
                     "thinking_tokens": 0, "cost_usd": 0}
        raw_text = ""
        inference_time = 0.0

        t0 = time.time()
        for attempt in range(3):
            try:
                response = call_gemini(
                    client, args.model, video_bytes, prompt,
                    video_fps=args.video_fps,
                )
                raw_text = response.text
                cost_info = extract_cost(response, args.model)
                prediction = parse_response(response)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"  [{i+1}] {ep_name}: attempt {attempt+1} failed: {e}"
                      f"  (retry in {wait}s)")
                time.sleep(wait)
        inference_time = round(time.time() - t0, 3)

        # Evaluate
        metrics = None
        if prediction is not None:
            metrics = evaluate_episode(
                prediction, gt, tolerances, frame_tolerances
            )

        # Serializable GT (exclude numpy array and set)
        gt_serializable = {
            k: v for k, v in gt.items()
            if k not in ("labels", "gt_kf_set")
        }

        result = {
            "episode": ep_name,
            "gt": gt_serializable,
            "prediction": prediction,
            "metrics": metrics,
            "cost": cost_info,
            "inference_time_sec": inference_time,
            "raw_response": raw_text if prediction is None else "",
        }
        all_results.append(result)
        append_jsonl(results_path, result)

        # Log
        if metrics:
            hit_mark = "HIT" if metrics["mid_frame_hit"] else "MISS"
            # Predicted frame range (convert seconds → frames at 15fps)
            pred_start_f = int(round(prediction["keyframe_start"] * VIDEO_FPS))
            pred_end_f = int(round(prediction["keyframe_end"] * VIDEO_FPS))
            pred_start_f = max(0, min(pred_start_f, gt["total_frames"] - 1))
            pred_end_f = max(0, min(pred_end_f, gt["total_frames"] - 1))

            tag = (f"{hit_mark}  "
                   f"pred=[{pred_start_f}–{pred_end_f}] "
                   f"mid={metrics['pred_mid_frame']}  "
                   f"gt=[{gt['kf_start_frame']}–{gt['kf_end_frame']}]  "
                   f"nearest={metrics['nearest_gt_dist']}  "
                   f"cost=${cost_info['cost_usd']:.4f}  "
                   f"time={inference_time:.1f}s")
        else:
            tag = "PARSE_FAILURE"
        print(f"  [{i+1}/{len(episode_names)}] {ep_name}: {tag}")

        if args.verbose and prediction and metrics:
            print(f"    Pred: [{prediction['keyframe_start']:.2f}"
                  f"–{prediction['keyframe_end']:.2f}]s  "
                  f"→ frames [{pred_start_f}–{pred_end_f}]  "
                  f"mid={metrics['pred_mid_frame']}")
            print(f"    GT:   [{gt['kf_start_sec']:.2f}"
                  f"–{gt['kf_end_sec']:.2f}]s  "
                  f"→ frames [{gt['kf_start_frame']}–{gt['kf_end_frame']}]")
            if prediction.get("reasoning"):
                reasoning = prediction["reasoning"][:120]
                print(f"    Reason: {reasoning}")

    # ── aggregate ──
    agg = aggregate_metrics(all_results, tolerances, frame_tolerances)
    print_summary(agg, args.model, tolerances, frame_tolerances)

    # Save JSONL summary
    summary_path = os.path.join(
        args.output_dir, f"{exp_tag}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "video_fps": args.video_fps,
            "caption": args.caption,
            "gaze_annot": args.gaze_annot,
            **agg,
        }, f, indent=2)
    print(f"\n[saved] {results_path}")
    print(f"[saved] {summary_path}")

    # ── Save structured results JSON ──────────────────────────────────
    gaze_tag = "true" if args.gaze_annot else "false"
    cap_tag = "true" if args.caption else "false"
    results_json_name = (f"result_{args.model}_gaze_{gaze_tag}"
                         f"_cap_{cap_tag}_fps{args.video_fps}.json")
    results_json_dir = "results"
    os.makedirs(results_json_dir, exist_ok=True)
    results_json_path = os.path.join(results_json_dir, results_json_name)

    episodes_data = []
    for r in all_results:
        ep_entry: dict = {"episode": r["episode"]}
        if r.get("prediction"):
            ps = r["prediction"]["keyframe_start"]
            pe = r["prediction"]["keyframe_end"]
            ep_entry["predicted_interval_sec"] = [round(ps, 4), round(pe, 4)]
            ep_entry["predicted_interval_frames"] = [
                int(round(ps * VIDEO_FPS)),
                int(round(pe * VIDEO_FPS)),
            ]
        if r.get("gt"):
            ep_entry["gt_interval_frames"] = [
                r["gt"]["kf_start_frame"],
                r["gt"]["kf_end_frame"],
            ]
        if r.get("metrics"):
            ep_entry["iou"] = r["metrics"]["iou"]
            ep_entry["mid_frame_hit"] = r["metrics"]["mid_frame_hit"]
            ep_entry["midpoint_error_sec"] = r["metrics"]["midpoint_error"]
            ep_entry["pred_mid_frame"] = r["metrics"]["pred_mid_frame"]
        ep_entry["inference_time_sec"] = r.get("inference_time_sec")
        episodes_data.append(ep_entry)

    results_json = {
        "config": {
            "model": args.model,
            "gaze_annot": args.gaze_annot,
            "caption": args.caption,
            "video_fps": args.video_fps,
            "config_name": config_name,
        },
        "episodes": episodes_data,
        "aggregate": agg,
    }
    with open(results_json_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"[saved] {results_json_path}")


if __name__ == "__main__":
    main()
