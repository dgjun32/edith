"""eval_openpi_streaming_human_view_no_vlm.py

Streaming human-view robot evaluation with a fixed language instruction.

Architecture
------------
  Main thread only: robot eval loop + CV2 display.
  No VLM, no keyframe capture, no task queue management.
  The policy receives live human-view input plus the user-provided prompt.
"""

import contextlib
import os
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import wave
from typing import Optional

import cv2
import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer

from openai import OpenAI

from openpi_client import websocket_client_policy
from common.config import SERVER_IP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS = 30
_TEMP_DIR = os.path.normpath("/home/alin/projects/piper_project/temp")


# ---------------------------------------------------------------------------
# Keyboard input helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _terminal_keypress_mode():
    """Allow single-key reads from the terminal during interactive loops."""
    if not sys.stdin.isatty():
        yield False
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _poll_keypress() -> int:
    """Read a key from either the OpenCV window or the terminal."""
    key = cv2.waitKey(1)
    if key != -1:
        return key & 0xFF

    if not sys.stdin.isatty():
        return -1

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return -1

    ch = sys.stdin.read(1)
    return ord(ch) if ch else -1


def _drain_pending_keypresses(duration_s: float = 0.35):
    """Clear stale terminal/OpenCV keys before switching interaction modes."""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        while time.monotonic() < deadline and cv2.waitKey(1) != -1:
            pass

        if sys.stdin.isatty():
            while time.monotonic() < deadline and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)

        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Thread-safe video buffer (for recording)
# ---------------------------------------------------------------------------

class VideoBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._aria = []
        self._cropped = []
        self._gaze = []
        self._audio = []

    def append(
        self,
        aria_bgr: np.ndarray,
        cropped_bgr: np.ndarray,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        audio_chunk_pcm16: Optional[np.ndarray] = None,
        audio_ts_ns: int = 0,
        audio_sample_rate_hz: int = 0,
        audio_num_samples: Optional[int] = None,
    ):
        with self._lock:
            idx = len(self._aria)
            self._aria.append(aria_bgr.copy())
            self._cropped.append(cropped_bgr.copy())
            self._gaze.append({"frame_idx": idx, "gaze_x": gaze_x, "gaze_y": gaze_y})
            if audio_chunk_pcm16 is None:
                audio_chunk_pcm16 = np.zeros((0,), dtype=np.int16)
            else:
                audio_chunk_pcm16 = np.asarray(audio_chunk_pcm16, dtype=np.int16).copy()
            if audio_num_samples is None:
                audio_num_samples = int(audio_chunk_pcm16.size)
            self._audio.append(
                {
                    "frame_idx": idx,
                    "audio_ts_ns": int(audio_ts_ns),
                    "sample_rate_hz": int(audio_sample_rate_hz),
                    "num_samples": int(audio_num_samples),
                    "chunk_pcm16": audio_chunk_pcm16,
                }
            )

    def get_audio_records(self, start: int = 0, end: Optional[int] = None):
        with self._lock:
            if end is None:
                end = len(self._audio)
            records = []
            for record in self._audio[start:end]:
                copied = dict(record)
                copied["chunk_pcm16"] = record["chunk_pcm16"].copy()
                records.append(copied)
            return records


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _write_video(frames_bgr, output_path, fps=_VIDEO_RECORD_FPS):
    if not frames_bgr:
        return
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    imageio.mimwrite(output_path, frames_rgb, fps=fps)


def _collect_audio_pcm16(audio_records):
    chunks = []
    sample_rate_hz = 0
    for record in audio_records:
        chunk = np.asarray(
            record.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
            dtype=np.int16,
        ).reshape(-1)
        if chunk.size == 0:
            continue
        chunk_sample_rate_hz = int(record.get("sample_rate_hz", 0) or 0)
        if chunk_sample_rate_hz <= 0:
            continue
        if sample_rate_hz not in (0, chunk_sample_rate_hz):
            raise ValueError(
                "Audio sample rate changed while recording: "
                f"{sample_rate_hz} -> {chunk_sample_rate_hz}"
            )
        sample_rate_hz = chunk_sample_rate_hz
        chunks.append(chunk.copy())
    if not chunks or sample_rate_hz <= 0:
        return np.zeros((0,), dtype=np.int16), 0
    return np.concatenate(chunks, axis=0), sample_rate_hz


def _write_pcm16_wav(audio_pcm16: np.ndarray, sample_rate_hz: int, output_path: str):
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(np.asarray(audio_pcm16, dtype=np.int16).tobytes())


def _write_video_with_audio(frames_bgr, audio_records, output_path, fps=_VIDEO_RECORD_FPS):
    if not frames_bgr:
        return
    audio_pcm16, sample_rate_hz = _collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate_hz <= 0:
        _write_video(frames_bgr, output_path, fps=fps)
        return

    silent_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    silent_tmp_path = silent_tmp.name
    wav_tmp_path = wav_tmp.name
    silent_tmp.close()
    wav_tmp.close()
    try:
        _write_video(frames_bgr, silent_tmp_path, fps=fps)
        _write_pcm16_wav(audio_pcm16, sample_rate_hz, wav_tmp_path)
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("[video] imageio_ffmpeg unavailable — saving silent video only.")
            os.replace(silent_tmp_path, output_path)
            silent_tmp_path = None
            return
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    silent_tmp_path,
                    "-i",
                    wav_tmp_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_text = (exc.stderr or "").strip()
            print(
                "[video] audio mux failed — saving silent video only."
                + (f" {stderr_text}" if stderr_text else "")
            )
            os.replace(silent_tmp_path, output_path)
            silent_tmp_path = None
    finally:
        if silent_tmp_path and os.path.exists(silent_tmp_path):
            os.unlink(silent_tmp_path)
        if os.path.exists(wav_tmp_path):
            os.unlink(wav_tmp_path)


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


# ---------------------------------------------------------------------------
# Background voice monitor — updates language instruction mid-episode
# ---------------------------------------------------------------------------

class VoiceMonitor:
    """Press-to-talk voice monitor. Only records audio between explicit
    start_recording() / stop_recording() calls. On stop, audio is transcribed
    by Whisper in a background thread and the instruction is updated."""

    def __init__(self, initial_instruction: str):
        self._lock = threading.Lock()
        self._instruction = initial_instruction
        self._audio_records: list = []
        self._recording = False
        self._transcribing = False
        self._updated_at = 0.0

    def start(self):
        # Kept for API compatibility; nothing to start in press-to-talk mode.
        pass

    def stop(self):
        # Kept for API compatibility.
        pass

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def is_transcribing(self) -> bool:
        with self._lock:
            return self._transcribing

    def is_busy(self) -> bool:
        with self._lock:
            return self._recording or self._transcribing

    def status_suffix(self) -> str:
        with self._lock:
            if self._recording:
                return " [REC]"
            if self._transcribing:
                return " [WHISPER]"
            return ""

    def start_recording(self):
        with self._lock:
            if self._transcribing:
                print("\n[VoiceMonitor] Whisper is still running. Please wait.")
                return False
            self._audio_records.clear()
            self._recording = True
        print("\n[VoiceMonitor] ● Recording started. Press 'p' again to stop.")
        return True

    def stop_recording(self):
        with self._lock:
            if not self._recording:
                return False
            self._recording = False
            self._transcribing = True
            records = list(self._audio_records)
            self._audio_records.clear()
        print("[VoiceMonitor] ■ Recording stopped. Transcribing in background...")
        threading.Thread(
            target=self._transcribe_and_update,
            args=(records,),
            daemon=True,
            name="VoiceMonitor-Transcribe",
        ).start()
        return True

    def feed_audio(self, chunk_pcm16: np.ndarray, sample_rate_hz: int,
                   audio_ts_ns: int = 0, num_samples: Optional[int] = None):
        if chunk_pcm16 is None:
            return
        chunk = np.asarray(chunk_pcm16, dtype=np.int16).reshape(-1)
        if chunk.size == 0 or sample_rate_hz <= 0:
            return
        with self._lock:
            if not self._recording:
                return
            self._audio_records.append({
                "chunk_pcm16": chunk.copy(),
                "sample_rate_hz": int(sample_rate_hz),
                "audio_ts_ns": int(audio_ts_ns),
                "num_samples": int(num_samples if num_samples is not None else chunk.size),
            })

    @property
    def instruction(self) -> str:
        with self._lock:
            return self._instruction

    def set_instruction(self, instruction: str):
        with self._lock:
            self._instruction = instruction

    def _transcribe_and_update(self, records: list):
        try:
            try:
                audio_pcm16, sample_rate = _collect_audio_pcm16(records)
            except ValueError as exc:
                print(f"[VoiceMonitor] audio collection error: {exc}")
                return
            if audio_pcm16.size == 0 or sample_rate <= 0:
                print("[VoiceMonitor] No audio captured.")
                return

            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_tmp.close()
            _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
            try:
                result = _whisper_transcribe_wav(wav_tmp.name)
                text = (result.get("text") or "").strip()
                if not text:
                    print("[VoiceMonitor] Empty transcription — instruction unchanged.")
                    return
                with self._lock:
                    old = self._instruction
                    self._instruction = text
                    self._updated_at = time.monotonic()
                print(f"\n[VoiceMonitor] Instruction replaced:\n"
                      f"  old: '{old}'\n  new: '{text}'\n")
            finally:
                if os.path.exists(wav_tmp.name):
                    os.unlink(wav_tmp.name)
        finally:
            with self._lock:
                self._transcribing = False


def _whisper_transcribe_wav(wav_path: str) -> dict:
    """Transcribe a WAV file using OpenAI Whisper API."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        return {"text": response.text, "words": []}
    except Exception as exc:
        print(f"  [WARN] Whisper transcription failed: {exc}")
        return {"text": "", "words": []}


def capture_voice_instruction(env, window_name="Voice Capture"):
    """Stream observations. Press 'p' to start recording and 'p' again to stop."""
    audio_records = []
    recording = False
    transcribing = False
    transcription_state = {"done": False, "text": "", "error": ""}
    timer = Timer(_VIDEO_RECORD_FPS)

    print("\n--- Voice instruction capture ---")
    print("  Press 'p' to start recording, then press 'p' again to stop.")
    print("  Robot control starts only after Whisper returns an instruction.")
    print("  Press 'q' to quit.\n")

    def _transcribe_initial_instruction(records):
        wav_tmp = None
        try:
            try:
                audio_pcm16, sample_rate = _collect_audio_pcm16(records)
            except ValueError as exc:
                transcription_state["error"] = f"audio collection error: {exc}"
                return
            if audio_pcm16.size == 0 or sample_rate <= 0:
                transcription_state["error"] = "No audio captured."
                return

            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_tmp.close()
            _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
            print("[Voice] Transcribing with Whisper...")
            result = _whisper_transcribe_wav(wav_tmp.name)
            transcription_state["text"] = result.get("text", "").strip()
            if not transcription_state["text"]:
                transcription_state["error"] = "Empty transcription."
        finally:
            if wav_tmp is not None and os.path.exists(wav_tmp.name):
                os.unlink(wav_tmp.name)
            transcription_state["done"] = True

    while True:
        obs = _extract_latest_observation(
            env.get_observation(mode='human_view_streamed_and_keyframe_cropped'))

        audio_obs = obs.get("audio", {})
        audio_chunk = np.asarray(
            audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
            dtype=np.int16,
        )
        audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
        audio_sr = int(audio_obs.get("sample_rate_hz", 0) or 0)
        audio_ns = int(audio_obs.get("num_samples", audio_chunk.size) or 0)
        if recording:
            audio_records.append({
                "audio_ts_ns": audio_ts_ns,
                "sample_rate_hz": audio_sr,
                "num_samples": audio_ns,
                "chunk_pcm16": audio_chunk.copy(),
            })

        if recording:
            label = "VOICE: recording... p=stop q=quit"
        elif transcribing:
            label = "VOICE: transcribing with Whisper... robot is paused"
        else:
            label = "VOICE: p=start recording q=quit"
        preview = _build_eval_preview(obs, label, 0)
        cv2.imshow(window_name, preview)
        key = _poll_keypress()
        timer.sleep()

        if transcribing and transcription_state["done"]:
            transcribing = False
            text = transcription_state["text"]
            if text:
                print(f"[Voice] Transcribed: \"{text}\"")
                cv2.destroyWindow(window_name)
                return text
            print(f"[Voice] {transcription_state['error']} Try recording again.")
            audio_records.clear()
            transcription_state = {"done": False, "text": "", "error": ""}

        if key in (ord('q'), ord('Q'), 27) and not transcribing:
            cv2.destroyWindow(window_name)
            return None

        if key in (ord('p'), ord('P')) and not transcribing:
            if not recording:
                audio_records.clear()
                transcription_state = {"done": False, "text": "", "error": ""}
                recording = True
                print("[Voice] Recording started. Press 'p' again to stop.")
                continue

            recording = False
            transcribing = True
            threading.Thread(
                target=_transcribe_initial_instruction,
                args=(list(audio_records),),
                daemon=True,
                name="InitialVoiceInstruction-Transcribe",
            ).start()


# ---------------------------------------------------------------------------
# CV2 preview helpers
# ---------------------------------------------------------------------------

def _build_eval_preview(obs, language_instruction: str, global_step: int):
    images = obs["images"]
    human_view = images["aria_rgb"].copy()
    top_view = images["top_camera_rgb"].copy()

    gaze_point = np.asarray(obs.get("gaze_point", []), dtype=np.float32).reshape(-1)
    if gaze_point.size >= 2 and np.all(np.isfinite(gaze_point[:2])):
        x = int(np.clip(gaze_point[0], 0, human_view.shape[1] - 1))
        y = int(np.clip(gaze_point[1], 0, human_view.shape[0] - 1))
        cv2.circle(human_view, (x, y), 6, (0, 255, 0), 2)
        cv2.line(human_view, (x - 12, y), (x + 12, y), (0, 255, 0), 2)
        cv2.line(human_view, (x, y - 12), (x, y + 12), (0, 255, 0), 2)

    if top_view.shape[:2] != human_view.shape[:2]:
        top_view = cv2.resize(
            top_view,
            (human_view.shape[1], human_view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    preview = np.hstack((human_view, top_view))
    cv2.putText(
        preview,
        f"LIVE | step={global_step}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "p=voice update | e=end episode | q=quit",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"prompt: {language_instruction[:80]}",
        (10, 79),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def _append_episode_record(
    obs,
    video_buffer: VideoBuffer,
    aria_frames,
    audio_records,
    top_view_frames_list=None,
    gaze=True,
    target_shape=None,
):
    images = obs["images"]
    aria_bgr = images["aria_rgb"]
    cropped_bgr = images["aria_cropped_rgb"]

    if target_shape is not None and aria_bgr.shape[:2] != target_shape:
        target_h, target_w = target_shape
        aria_bgr = cv2.resize(
            aria_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

    gaze_pt = np.asarray(obs.get("gaze_point", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    gx = float(gaze_pt[0]) if gaze and gaze_pt.size >= 1 else None
    gy = float(gaze_pt[1]) if gaze and gaze_pt.size >= 2 else None

    audio_obs = obs.get("audio", {})
    audio_chunk = np.asarray(
        audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
        dtype=np.int16,
    )
    audio_ts_ns = int(audio_obs.get("audio_ts_ns", 0) or 0)
    audio_sample_rate_hz = int(audio_obs.get("sample_rate_hz", 0) or 0)
    audio_num_samples = int(audio_obs.get("num_samples", audio_chunk.size) or 0)

    video_buffer.append(
        aria_bgr,
        cropped_bgr,
        gx,
        gy,
        audio_chunk_pcm16=audio_chunk,
        audio_ts_ns=audio_ts_ns,
        audio_sample_rate_hz=audio_sample_rate_hz,
        audio_num_samples=audio_num_samples,
    )
    aria_frames.append(aria_bgr.copy())
    audio_records.append(
        {
            "frame_idx": len(audio_records),
            "audio_ts_ns": audio_ts_ns,
            "sample_rate_hz": audio_sample_rate_hz,
            "num_samples": audio_num_samples,
            "chunk_pcm16": audio_chunk.copy(),
        }
    )
    if top_view_frames_list is not None:
        top_view_frames_list.append(images["top_camera_rgb"].copy())

    return audio_chunk, audio_sample_rate_hz, audio_ts_ns, audio_num_samples


def _handle_eval_key(key: int, voice_monitor: VoiceMonitor, allow_voice: bool = True) -> str:
    if key in (ord("q"), ord("Q"), 27):
        return "quit"
    if key in (ord("e"), ord("E")):
        return "end"
    if key in (ord("p"), ord("P")):
        if not allow_voice:
            return ""
        if voice_monitor.is_recording():
            voice_monitor.stop_recording()
        elif not voice_monitor.is_transcribing():
            voice_monitor.start_recording()
        return "voice"
    return ""


def _pause_for_voice_update(
    env,
    voice_monitor: VoiceMonitor,
    video_buffer: VideoBuffer,
    aria_frames,
    audio_records,
    top_view_frames_list,
    global_step: int,
    window_name: str,
    target_shape,
) -> str:
    timer = Timer(env.control_hz)
    print("[Eval] Robot control paused for voice recording/Whisper.")
    while voice_monitor.is_busy():
        obs = _extract_latest_observation(
            env.get_observation(mode="human_view_streamed_and_keyframe_cropped")
        )
        audio_chunk, audio_sr, audio_ts_ns, audio_ns = _append_episode_record(
            obs,
            video_buffer,
            aria_frames,
            audio_records,
            top_view_frames_list=top_view_frames_list,
            gaze=False,
            target_shape=target_shape,
        )
        voice_monitor.feed_audio(
            audio_chunk,
            audio_sr,
            audio_ts_ns=audio_ts_ns,
            num_samples=audio_ns,
        )

        preview = _build_eval_preview(
            obs,
            voice_monitor.instruction + voice_monitor.status_suffix(),
            global_step,
        )
        cv2.imshow(window_name, preview)
        command = _handle_eval_key(_poll_keypress(), voice_monitor)
        if command in ("quit", "end"):
            return command
        timer.sleep()
    print("[Eval] Voice update finished. Robot control resumed.")
    return ""


def _wait_for_next_episode_start(env, window_name="Next Episode"):
    timer = Timer(_VIDEO_RECORD_FPS)
    print("\nEpisode reset. Press 's' to start the next episode instruction capture, or 'q' to quit.")
    while True:
        obs = _extract_latest_observation(
            env.get_observation(mode="human_view_streamed_and_keyframe_cropped")
        )
        preview = _build_eval_preview(
            obs,
            "RESET DONE: s=start next episode instruction q=quit",
            0,
        )
        cv2.imshow(window_name, preview)
        key = _poll_keypress()
        if key in (ord("s"), ord("S")):
            cv2.destroyWindow(window_name)
            return True
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window_name)
            return False
        timer.sleep()


# ---------------------------------------------------------------------------
# Rollout video saver
# ---------------------------------------------------------------------------

def _save_rollout_video(top_frames, comp_prob_log, output_path, fps=_VIDEO_RECORD_FPS):
    if not top_frames:
        print("[rollout] No frames to save.")
        return

    fw, fh = top_frames[0].shape[1], top_frames[0].shape[0]
    dpi = 100

    comp_prob_log = sorted(comp_prob_log, key=lambda x: x[0])
    max_step = max((s for s, _ in comp_prob_log), default=len(top_frames))
    max_step = max(max_step, len(top_frames))

    fig, ax = plt.subplots(figsize=(fw / dpi, fh / dpi), dpi=dpi)

    def _render(visible_log):
        ax.clear()
        xs, ys = zip(*visible_log) if visible_log else ([], [])
        ax.plot(xs, ys, color="steelblue", linewidth=1.5)
        ax.scatter(xs, ys, color="steelblue", s=12, zorder=3)
        ax.axhline(
            y=0.8,
            color="tomato",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="threshold = 0.8",
        )
        ax.set_xlim(0, max_step)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Global Step", fontsize=8)
        ax.set_ylabel("Completion Prob", fontsize=8)
        ax.set_title("Completion Probability over Time", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), (fw, fh))

    breakpoints = [-1] + [s for s, _ in comp_prob_log]
    rendered = {
        bp: _render([(s, p) for s, p in comp_prob_log if s <= bp]) for bp in breakpoints
    }
    plt.close(fig)

    def _get_plot(step):
        best = max((bp for bp in breakpoints if bp <= step), default=-1)
        return rendered[best]

    print(f"[rollout] Compositing {len(top_frames)} frames...")
    combined_frames = []
    for i, frame in enumerate(top_frames):
        row1 = frame.copy()
        row2 = _get_plot(i)
        combined_frames.append(
            cv2.cvtColor(np.vstack([row1, row2]), cv2.COLOR_BGR2RGB)
        )
    imageio.mimwrite(output_path, combined_frames, fps=fps)
    print(f"[rollout] Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main eval loop (streaming human view only)
# ---------------------------------------------------------------------------

def eval(
    env,
    client,
    voice_monitor: "VoiceMonitor",
    horizon: int,
    video_buffer: VideoBuffer,
    chunksize: int = 8,
    voice_key_grace_s: float = 0.75,
):
    """
    Robot execution loop driven by the live instruction from VoiceMonitor.
    The instruction can be replaced mid-episode whenever the human speaks.

    Termination:
      1. horizon reached
      2. 'e' pressed
      3. 'q' / ESC pressed
    """
    top_view_frames_list = []
    comp_prob_log = []
    aria_frames = []
    audio_records = []
    global_step = 0
    timer = Timer(env.control_hz)
    window_name = "Eval  |  q=quit"
    rollout_video_path = os.path.join(_TEMP_DIR, "rollout.mp4")
    human_video_path = os.path.join(_TEMP_DIR, "human_view_episode.mp4")
    episode_finalized = False
    voice_key_enabled_at = time.monotonic() + voice_key_grace_s

    print(f"[Eval] Robot control starting with instruction: '{voice_monitor.instruction}'")

    def _finalize_episode_recordings():
        nonlocal episode_finalized
        if episode_finalized:
            return
        episode_finalized = True
        cv2.destroyAllWindows()
        _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path)
        _write_video_with_audio(
            aria_frames,
            audio_records,
            human_video_path,
            fps=_VIDEO_RECORD_FPS,
        )

    for step in range(horizon):
        try:
            obs = _extract_latest_observation(
                env.get_observation(mode="human_view_streamed_and_keyframe_cropped")
            )
            images = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for value in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, value])

            target_shape = images["aria_rgb"].shape[:2]
            audio_chunk, audio_sr, audio_ts_ns, audio_ns = _append_episode_record(
                obs,
                video_buffer,
                aria_frames,
                audio_records,
                gaze=True,
            )
            voice_monitor.feed_audio(
                audio_chunk,
                audio_sr,
                audio_ts_ns=audio_ts_ns,
                num_samples=audio_ns,
            )

            # Always read the latest (possibly just-updated) instruction
            language_instruction = voice_monitor.instruction
            preview = _build_eval_preview(
                obs,
                language_instruction + voice_monitor.status_suffix(),
                global_step,
            )
            cv2.imshow(window_name, preview)
            command = _handle_eval_key(
                _poll_keypress(),
                voice_monitor,
                allow_voice=time.monotonic() >= voice_key_enabled_at,
            )
            if command == "quit":
                _finalize_episode_recordings()
                raise KeyboardInterrupt
            if command == "end":
                _finalize_episode_recordings()
                return "ended"
            if voice_monitor.is_busy():
                command = _pause_for_voice_update(
                    env,
                    voice_monitor,
                    video_buffer,
                    aria_frames,
                    audio_records,
                    top_view_frames_list,
                    global_step,
                    window_name,
                    target_shape,
                )
                if command == "quit":
                    _finalize_episode_recordings()
                    raise KeyboardInterrupt
                if command == "end":
                    _finalize_episode_recordings()
                    return "ended"
                continue

            language_instruction = voice_monitor.instruction

            human_view_gaze_cropped_rgb = cv2.cvtColor(images["aria_cropped_rgb"], cv2.COLOR_BGR2RGB)
            observation = {
                "observation.images.top_view": images["top_camera_rgb"],
                "observation.images.human_view_gaze_cropped": human_view_gaze_cropped_rgb,
                "observation.images.left_wrist_view": images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view": images["right_wrist_camera_rgb"],
                "prompt": "Give me this muffin, this muffin, and this muffin.", #language_instruction,
                "observation.state": concatenated_state,
            }

            if global_step == 0:
                plt.imsave(
                    os.path.join(_TEMP_DIR, "test_human_view_gaze_cropped.png"),
                    human_view_gaze_cropped_rgb,
                )

            out = client.infer(observation)
            action_chunk = out["actions"][:chunksize, :]
            completion_prob = float(out["completion_prob"]) if "completion_prob" in out else None
            print(
                "Completion prob (ignored for control):",
                f"{completion_prob:.3f}" if completion_prob is not None else "N/A",
            )
            if completion_prob is not None:
                comp_prob_log.append((global_step, completion_prob))

            for raw_action in action_chunk:
                if voice_monitor.is_busy():
                    command = _pause_for_voice_update(
                        env,
                        voice_monitor,
                        video_buffer,
                        aria_frames,
                        audio_records,
                        top_view_frames_list,
                        global_step,
                        window_name,
                        target_shape,
                    )
                    if command == "quit":
                        _finalize_episode_recordings()
                        raise KeyboardInterrupt
                    if command == "end":
                        _finalize_episode_recordings()
                        return "ended"
                    break

                action = {
                    "left_arm_positions": raw_action[0:6],
                    "left_gripper_position": raw_action[6:7],
                    "right_arm_positions": raw_action[7:13],
                    "right_gripper_position": raw_action[13:14],
                }
                env.step(action)
                global_step += 1

                live_obs = _extract_latest_observation(
                    env.get_observation(mode="human_view_streamed_and_keyframe_cropped")
                )
                live_audio_chunk, live_audio_sr, live_audio_ts_ns, live_audio_ns = (
                    _append_episode_record(
                        live_obs,
                        video_buffer,
                        aria_frames,
                        audio_records,
                        top_view_frames_list=top_view_frames_list,
                        gaze=False,
                        target_shape=target_shape,
                    )
                )
                voice_monitor.feed_audio(
                    live_audio_chunk, live_audio_sr,
                    audio_ts_ns=live_audio_ts_ns,
                    num_samples=live_audio_ns,
                )

                live_instruction = voice_monitor.instruction
                preview = _build_eval_preview(
                    live_obs,
                    live_instruction + voice_monitor.status_suffix(),
                    global_step,
                )
                cv2.imshow(window_name, preview)
                command = _handle_eval_key(
                    _poll_keypress(),
                    voice_monitor,
                    allow_voice=time.monotonic() >= voice_key_enabled_at,
                )
                if command == "quit":
                    _finalize_episode_recordings()
                    raise KeyboardInterrupt
                if command == "end":
                    _finalize_episode_recordings()
                    return "ended"
                if voice_monitor.is_busy():
                    command = _pause_for_voice_update(
                        env,
                        voice_monitor,
                        video_buffer,
                        aria_frames,
                        audio_records,
                        top_view_frames_list,
                        global_step,
                        window_name,
                        target_shape,
                    )
                    if command == "quit":
                        _finalize_episode_recordings()
                        raise KeyboardInterrupt
                    if command == "end":
                        _finalize_episode_recordings()
                        return "ended"
                    break

                timer.sleep()

        except KeyboardInterrupt:
            _finalize_episode_recordings()
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    _finalize_episode_recordings()
    return "horizon"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        "top_camera": "242322070028",
        "left_wrist_camera": "244222076964",
        "right_wrist_camera": "244222071848",
        "aria": f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "Pick up the muffin."
    horizon = 10000

    print("Starting stream-VLA evaluation with voice input.")
    print("Initial capture: press 'p' to start recording, 'p' again to transcribe.")
    print("During eval: press 'p' to pause/update voice instruction, 'e' to end/reset episode.")

    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    voice_monitor: Optional[VoiceMonitor] = None
    wait_for_next_start = False
    try:
        while True:
            try:
                if wait_for_next_start:
                    with _terminal_keypress_mode():
                        if not _wait_for_next_episode_start(env):
                            print("Exiting.")
                            break
                    wait_for_next_start = False

                # Voice capture phase (pre-episode)
                with _terminal_keypress_mode():
                    result = capture_voice_instruction(env)
                    _drain_pending_keypresses()
                if result is None:
                    print("Exiting.")
                    break
                language_instruction = result
                print(f"Instruction: '{language_instruction}'")

                # Start (or update) the background voice monitor
                if voice_monitor is None:
                    voice_monitor = VoiceMonitor(language_instruction)
                    voice_monitor.start()
                else:
                    voice_monitor.set_instruction(language_instruction)

                # Eval phase
                video_buffer = VideoBuffer()
                with _terminal_keypress_mode():
                    episode_status = eval(env, client, voice_monitor, horizon, video_buffer)
                print(f"Episode finished ({episode_status}).")

            except KeyboardInterrupt:
                print("\nEpisode interrupted (Ctrl+C).")

            # Reset for next episode (keep voice monitor alive)
            cv2.destroyAllWindows()
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(
                host="localhost", port=8000)
            if voice_monitor is not None:
                language_instruction = voice_monitor.instruction
            wait_for_next_start = True

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        if voice_monitor is not None:
            voice_monitor.stop()
        if os.path.isdir(_TEMP_DIR):
            import shutil
            shutil.rmtree(_TEMP_DIR)
            os.makedirs(_TEMP_DIR, exist_ok=True)
            print(f"Cleared temp directory: {_TEMP_DIR}")
        env.safe_end()
