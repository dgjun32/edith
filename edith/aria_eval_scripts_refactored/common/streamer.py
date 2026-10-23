"""Shared helpers for Plan-A surgical fixes across eval_openpi_* scripts.

This module provides:

1. ``BackgroundObservationStreamer`` — a daemon thread that continuously
   polls ``env.get_observation()`` at a steady target FPS and pushes frames
   into the shared ``VideoBuffer``.  This replaces the previous behaviour
   where the buffer was only fed during ``env.step()`` (which does NOT
   run at 30 Hz), causing the periodic VLM calls to see undersampled /
   distorted video segments.

2. ``wait_for_queue_items`` — a helper that holds the robot in place
   (no reset, no step) while the ``KeyframeQueue`` is empty and the last
   subtask has been completed.  The function blocks until either
   (a) the async VLM worker appends a new subtask, or (b) the user
   presses 'q' / ESC to abort the episode.

Usage pattern in each eval() function:

    from common.streamer import BackgroundObservationStreamer, wait_for_queue_items

    streamer = BackgroundObservationStreamer(
        env, video_buffer,
        target_fps=30.0,
        frame_extractor=<task-specific callable or None>,
    )
    streamer.start()
    try:
        ...  # main eval loop; remove manual video_buffer.append() calls
    finally:
        streamer.stop()

For the hold-pose behaviour, replace::

    env.reset()
    client = websocket_client_policy.WebsocketClientPolicy(...)
    return

…with::

    aborted = wait_for_queue_items(queue, window_name=window_name,
                                   preview_builder=lambda: ..., )
    if aborted:
        raise KeyboardInterrupt
    # queue has new items → loop continues
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Observation streamer
# ---------------------------------------------------------------------------

def _default_frame_extractor(obs: dict):
    """Default extractor for files that buffer ``aria_rgb`` + ``aria_cropped_rgb``.

    Returns a kwargs dict for ``video_buffer.append(**kwargs)``, or ``None``
    if the observation is incomplete and should be skipped.
    """
    images = obs.get("images", {}) or {}
    aria = images.get("aria_rgb")
    cropped = images.get("aria_cropped_rgb")
    if aria is None or cropped is None:
        return None

    gaze_pt = np.asarray(obs.get("gaze_point", [0.0, 0.0]),
                         dtype=np.float64).reshape(-1)
    gx = float(gaze_pt[0]) if gaze_pt.size >= 1 else None
    gy = float(gaze_pt[1]) if gaze_pt.size >= 2 else None

    audio_obs = obs.get("audio", {}) or {}
    audio_chunk = np.asarray(
        audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
        dtype=np.int16,
    )
    audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
    audio_sr = int(audio_obs.get("sample_rate_hz", 0) or 0)
    audio_ns = int(audio_obs.get("num_samples", audio_chunk.size) or 0)

    return {
        "aria_bgr": aria,
        "cropped_bgr": cropped,
        "gaze_x": gx,
        "gaze_y": gy,
        "audio_chunk_pcm16": audio_chunk,
        "audio_ts_ns": audio_ts_ns,
        "audio_sample_rate_hz": audio_sr,
        "audio_num_samples": audio_ns,
    }


def _top_view_frame_extractor(obs: dict):
    """Extractor for files where the VLM source is ``top_camera_rgb`` only.

    Stores ``top_camera_rgb`` into both the ``aria`` and ``cropped`` slots
    so ``_select_gemini_frames`` picks it up regardless of
    ``_GEMINI_VIDEO_SOURCE``. Gaze is unused (None).
    """
    images = obs.get("images", {}) or {}
    top = images.get("top_camera_rgb")
    if top is None:
        return None
    # top_camera_rgb is returned in RGB by PiperRobotEnv. The eval pipeline
    # (cv2.VideoWriter + cv2.imshow + _write_video) expects BGR, so convert.
    top_bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)

    audio_obs = obs.get("audio", {}) or {}
    audio_chunk = np.asarray(
        audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
        dtype=np.int16,
    )
    audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
    audio_sr = int(audio_obs.get("sample_rate_hz", 0) or 0)
    audio_ns = int(audio_obs.get("num_samples", audio_chunk.size) or 0)

    return {
        "aria_bgr": top_bgr,
        "cropped_bgr": top_bgr,
        "gaze_x": None,
        "gaze_y": None,
        "audio_chunk_pcm16": audio_chunk,
        "audio_ts_ns": audio_ts_ns,
        "audio_sample_rate_hz": audio_sr,
        "audio_num_samples": audio_ns,
    }


class BackgroundObservationStreamer:
    """Daemon thread that feeds ``video_buffer`` at a steady target FPS.

    Runs independently of ``env.step()`` so the VLM always sees a
    continuously-filled, uniformly-sampled video buffer.

    Parameters
    ----------
    env : PiperRobotEnv
        Must expose ``get_observation(mode=...)``.
    video_buffer : object
        Must expose ``append(**kwargs)`` where ``kwargs`` matches the dict
        returned by ``frame_extractor``.
    target_fps : float
        Polling rate.  The thread skips frames if it falls behind.
    mode : str
        Argument passed to ``env.get_observation``.
    frame_extractor : callable or None
        ``obs -> kwargs | None``.  Defaults to ``_default_frame_extractor``
        (aria + cropped).  Pass ``_top_view_frame_extractor`` for top-view
        only scripts.
    """

    def __init__(
        self,
        env,
        video_buffer,
        target_fps: float = 30.0,
        mode: str = "human_view_streamed_and_keyframe_cropped",
        frame_extractor: Optional[Callable[[dict], Optional[dict]]] = None,
    ):
        self._env = env
        self._buf = video_buffer
        self._fps = float(target_fps)
        self._mode = mode
        self._extract = frame_extractor or _default_frame_extractor
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "BackgroundObservationStreamer":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="BgObservationStreamer",
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def pause(self):
        """Temporarily stop feeding the buffer (frames are dropped)."""
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_latest(self, obs):
        # env.get_observation may return a dict keyed by timestamp.  Pick
        # the newest key; if ``obs`` is already a single-frame dict, return
        # it unchanged.
        if isinstance(obs, dict) and obs:
            first_val = next(iter(obs.values()))
            if isinstance(first_val, dict) and "images" in first_val:
                return obs[next(reversed(obs))]
        return obs

    def _run(self):
        period = 1.0 / self._fps if self._fps > 0 else 1.0 / 30.0
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                try:
                    try:
                        raw = self._env.get_observation(
                            mode=self._mode,
                            audio_drain=True,
                        )
                    except TypeError:
                        raw = self._env.get_observation(mode=self._mode)
                    obs = self._extract_latest(raw)
                    kwargs = self._extract(obs)
                    if kwargs is not None:
                        self._buf.append(**kwargs)
                except Exception as exc:
                    # Don't let a transient env hiccup kill the thread.
                    print(f"[BgObsStreamer] observation error: {exc}")

            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)
            else:
                # We're running behind schedule — reset pacing so we
                # don't spin in a tight loop.
                next_t = time.monotonic()


# ---------------------------------------------------------------------------
# Hold-pose loop (replaces env.reset() on empty queue)
# ---------------------------------------------------------------------------

def wait_for_queue_items(
    queue,
    window_name: str = "Eval",
    preview_builder: Optional[Callable[[], Optional[np.ndarray]]] = None,
    key_poll: Optional[Callable[[], int]] = None,
    poll_interval_sec: float = 0.05,
) -> bool:
    """Block until ``queue`` has an unfinished subtask or the user aborts.

    The robot is NOT stepped or reset while this function runs — it simply
    holds its current pose.  The shared ``BackgroundObservationStreamer``
    keeps feeding the video buffer, so the async VLM worker can still
    observe the scene and Append a new subtask.

    Parameters
    ----------
    queue : KeyframeQueue
        Queue with an ``is_done`` property.
    window_name : str
        CV2 window name used for the preview / keypress surface.
    preview_builder : callable or None
        Optional ``() -> np.ndarray | None`` callable that builds a preview
        frame each tick.  Use this to show the live camera feed while the
        robot is holding pose.
    key_poll : callable or None
        Optional ``() -> int`` keyboard poller.  When provided, callers can
        merge CV2 window input with terminal input for SSH/X11 sessions.
    poll_interval_sec : float
        How often to check the queue / poll the keyboard.

    Returns
    -------
    bool
        ``True`` if the user requested abort ('q' / ESC) before the queue
        got a new item, ``False`` if a new item arrived.
    """
    print("[Eval] Queue exhausted — holding pose. Waiting for VLM to append "
          "a new subtask (press 'q' / ESC to abort)...")
    last_log_t = time.monotonic()
    while queue.is_done:
        frame = None
        if preview_builder is not None:
            try:
                frame = preview_builder()
            except Exception as exc:
                print(f"[Eval hold-pose] preview builder error: {exc}")
                frame = None
        if frame is not None:
            overlay = frame.copy()
            cv2.putText(overlay, "HOLDING POSE — waiting for VLM command",
                        (10, overlay.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, overlay)
        if key_poll is None:
            key = cv2.waitKey(max(1, int(poll_interval_sec * 1000))) & 0xFF
        else:
            key = key_poll()
            time.sleep(poll_interval_sec)
        if key in (ord('q'), ord('Q'), ord('s'), ord('S'), 27):
            print("[Eval hold-pose] User abort.")
            return True

        now = time.monotonic()
        if now - last_log_t > 5.0:
            last_log_t = now
            print("[Eval hold-pose] Still waiting for new subtask... "
                  f"(queue.total={queue.total}, is_done={queue.is_done})")

    print("[Eval hold-pose] New subtask available — resuming.")
    return False


__all__ = [
    "BackgroundObservationStreamer",
    "wait_for_queue_items",
    "_default_frame_extractor",
    "_top_view_frame_extractor",
]
