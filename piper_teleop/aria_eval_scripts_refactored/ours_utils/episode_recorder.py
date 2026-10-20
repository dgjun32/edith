"""Per-episode recorder: writes every camera view + preview + logs to disk.

One ``EpisodeRecorder`` covers a single episode, from the moment the
initial keyframe selector opens (before 'p'/'s') until the eval loop exits
('e' / 'q' / horizon).  Frames are streamed to disk through cv2.VideoWriter
from a background thread, so RAM usage stays at a handful of frames no
matter how long the episode runs.

Output layout (inside ``episode_dir``):

    egoview.mp4            aria full view (gaze crosshair drawn)
    egoview_gaze_crop.mp4  gaze-centred crop (what the VLM / policy keyframe sees)
    top_view.mp4           robot top camera (arms + grippers visible)
    preview.mp4            the CV2 window as shown to the operator
    keyframes/             png of the policy keyframe every time it changes
    frames.jsonl           one row per recorded frame (t, step, phase, queue, ...)
    events.jsonl           keyframe changes, VLM commands, phase changes
    episode.json           summary written at close (call_log, comp-prob log, ...)
    audio.wav              PCM audio held in the VideoBuffer (initial clip + eval)

All four videos are written from the same ``write()`` call at a fixed FPS
paced by wall-clock time, so they stay time-aligned with each other and
with the ``t`` column in frames.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


_STREAMS = ("egoview", "egoview_gaze_crop", "top_view", "preview")


def _fit(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize *img* to (w, h) preserving aspect ratio, padding with black."""
    w, h = size
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _draw_gaze(img_bgr: np.ndarray, gaze_point) -> np.ndarray:
    out = img_bgr.copy()
    gp = np.asarray(gaze_point, dtype=np.float32).reshape(-1)
    if gp.size >= 2 and np.all(np.isfinite(gp[:2])):
        x = int(np.clip(gp[0], 0, out.shape[1] - 1))
        y = int(np.clip(gp[1], 0, out.shape[0] - 1))
        cv2.circle(out, (x, y), 6, (0, 255, 0), 2)
        cv2.line(out, (x - 12, y), (x + 12, y), (0, 255, 0), 2)
        cv2.line(out, (x, y - 12), (x, y + 12), (0, 255, 0), 2)
    return out


class _StreamWriter:
    """Lazy cv2.VideoWriter that locks its size to the first frame."""

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self.frames = 0

    def write(self, img: np.ndarray):
        if img is None:
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self._writer is None:
            h, w = img.shape[:2]
            self._size = (w, h)
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self._writer.write(_fit(img, self._size))
        self.frames += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class EpisodeRecorder:
    def __init__(self, episode_dir: str, fps: float = 15.0,
                 max_pending: int = 90):
        self.dir = os.path.normpath(episode_dir)
        self.fps = float(fps)
        self.keyframe_dir = os.path.join(self.dir, "keyframes")
        os.makedirs(self.keyframe_dir, exist_ok=True)

        self._writers = {
            name: _StreamWriter(os.path.join(self.dir, f"{name}.mp4"), self.fps)
            for name in _STREAMS
        }
        self._frames_f = open(os.path.join(self.dir, "frames.jsonl"), "w")
        self._events_f = open(os.path.join(self.dir, "events.jsonl"), "w")

        self._t0 = time.monotonic()
        self._next_frame_t = 0.0          # next slot (episode seconds)
        self._last_pack: Optional[dict] = None
        self._frame_idx = 0
        self._dropped = 0
        self._phase = "init"
        self._last_kf_id: Optional[int] = None
        self._kf_count = 0
        self._latest_comp_prob: Optional[float] = None
        self.summary: Dict[str, Any] = {
            "episode_dir": self.dir,
            "fps": self.fps,
            "started_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(
            target=self._drain, daemon=True, name="EpisodeRecorder")
        self._thread.start()
        self._closed = False
        self.log_event("phase", phase="init")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def set_phase(self, phase: str, **extra):
        if phase != self._phase:
            self._phase = phase
            self.log_event("phase", phase=phase, **extra)

    def set_completion_prob(self, prob: Optional[float]):
        self._latest_comp_prob = prob

    def log_event(self, kind: str, **fields):
        row = {"t": round(self.elapsed, 4), "step": fields.pop("step", None),
               "kind": kind, **fields}
        try:
            self._events_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._events_f.flush()
        except ValueError:
            pass  # file closed

    def write(self, obs: dict, step: int, queue_obj: Any = None,
              preview: Optional[np.ndarray] = None,
              instruction: Optional[str] = None):
        """Record one observation.  Cheap: copies the frames and enqueues.

        Paces itself to ``fps`` by wall-clock; calls arriving faster than
        the slot rate are skipped, and gaps are filled by repeating the
        previous frame so every stream stays aligned with real time.
        """
        if self._closed:
            return
        now = self.elapsed
        if now < self._next_frame_t and self._last_pack is not None:
            return

        images = obs["images"]
        keyframe = None
        queue_idx = None
        queue_total = None
        if queue_obj is not None:
            try:
                keyframe = queue_obj.current_keyframe
                queue_idx = queue_obj.current_idx
                queue_total = queue_obj.total
                instruction = instruction or queue_obj.current_instruction
            except Exception:
                pass
        self._track_keyframe(keyframe, step, instruction)

        pack = {
            "egoview": _draw_gaze(images["aria_rgb"], obs.get("gaze_point", ())),
            "egoview_gaze_crop": images.get("aria_cropped_rgb"),
            "top_view": images.get("top_camera_rgb"),
            "preview": preview,
            "meta": {
                "step": int(step),
                "phase": self._phase,
                "queue_idx": queue_idx,
                "queue_total": queue_total,
                "instruction": instruction,
                "keyframe_id": self._kf_count if keyframe is not None else None,
                "completion_prob": self._latest_comp_prob,
                "gaze": [float(v) for v in
                         np.asarray(obs.get("gaze_point", ()), dtype=np.float64)
                         .reshape(-1)[:2]],
            },
        }
        for k in ("egoview", "egoview_gaze_crop", "top_view", "preview"):
            if pack[k] is not None:
                pack[k] = np.ascontiguousarray(pack[k]).copy()

        # fill any gap since the last slot with the previous frame
        if self._last_pack is not None:
            while self._next_frame_t + (1.0 / self.fps) <= now:
                self._enqueue(self._last_pack, self._next_frame_t, filler=True)
                self._next_frame_t += 1.0 / self.fps
        else:
            self._next_frame_t = now

        self._enqueue(pack, self._next_frame_t)
        self._next_frame_t += 1.0 / self.fps
        self._last_pack = pack

    def close(self, extra_summary: Optional[dict] = None,
              audio_records: Optional[List[dict]] = None):
        if self._closed:
            return
        self._closed = True
        self.log_event("phase", phase="closed")
        self._q.put(None)
        self._thread.join(timeout=60.0)
        for w in self._writers.values():
            w.close()

        if audio_records:
            try:
                from .audio_processing import collect_audio_pcm16, write_pcm16_wav
                pcm, sr = collect_audio_pcm16(audio_records)
                if pcm.size > 0 and sr > 0:
                    write_pcm16_wav(pcm, sr, self.path("audio.wav"))
            except Exception as exc:
                print(f"[Recorder] audio.wav failed: {exc}")

        self.summary.update({
            "ended_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(self.elapsed, 3),
            "frames_written": self._frame_idx,
            "frames_dropped_queue_full": self._dropped,
            "keyframes_saved": self._kf_count,
            "streams": {n: {"path": w.path, "frames": w.frames}
                        for n, w in self._writers.items()},
        })
        if extra_summary:
            self.summary.update(_jsonable(extra_summary))
        with open(self.path("episode.json"), "w") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        for fh in (self._frames_f, self._events_f):
            try:
                fh.close()
            except Exception:
                pass
        print(f"[Recorder] Episode saved → {self.dir} "
              f"({self._frame_idx} frames, {self.elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _track_keyframe(self, keyframe, step: int, instruction):
        if keyframe is None:
            return
        kf_id = id(keyframe)
        if kf_id == self._last_kf_id:
            return
        self._last_kf_id = kf_id
        self._kf_count += 1
        digest = hashlib.sha1(np.ascontiguousarray(keyframe).tobytes()).hexdigest()[:10]
        path = os.path.join(
            self.keyframe_dir, f"kf_{self._kf_count:03d}_step{step:06d}_{digest}.png")
        cv2.imwrite(path, keyframe)
        self.log_event("keyframe", step=step, keyframe_id=self._kf_count,
                       instruction=instruction, path=path, digest=digest)

    def _enqueue(self, pack: dict, t: float, filler: bool = False):
        item = {"pack": pack, "t": t, "filler": filler, "idx": self._frame_idx}
        self._frame_idx += 1
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    def _drain(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            pack = item["pack"]
            try:
                for name in _STREAMS:
                    self._writers[name].write(pack.get(name))
                row = {"frame": item["idx"], "t": round(item["t"], 4),
                       "filler": item["filler"], **pack["meta"]}
                self._frames_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"[Recorder] write failed: {exc}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray {obj.shape}>"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def new_episode_dir(root: str, run_name: str, episode_idx: int) -> str:
    d = os.path.join(root, run_name, f"episode_{episode_idx:03d}")
    os.makedirs(d, exist_ok=True)
    return d
