"""eval_openpi_streaming_human_view_no_vlm.py

Streaming human-view robot evaluation with a fixed language instruction.

Architecture
------------
  Main thread only: robot eval loop + CV2 display.
  No VLM, no keyframe capture, no task queue management.
  The policy receives live human-view input plus the user-provided prompt.
"""

import argparse
import contextlib
import os
import select
import subprocess
import sys
import tempfile
import termios
import threading
import tty
import wave
from typing import Optional

import cv2
import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edith.envs.piper_env_gaze import PiperRobotEnv
from edith.utils.time_utils import Timer

from openai import OpenAI

from openpi_client import websocket_client_policy
from common.config import SERVER_IP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS = 30
_TEMP_DIR = os.path.normpath("/home/alin/projects/piper_project/temp")
DEFAULT_TASK = "muffin"
DEFAULT_ARIA_PORT = 9999
DEFAULT_POLICY_HOST = "localhost"
DEFAULT_POLICY_PORT = 8000
DEFAULT_HORIZON = 10000
DEFAULT_CHUNKSIZE = 8

TASK_PROFILES = {
    "muffin": {
        "description": "streamed human-view muffin serving",
        "default_instruction": "Give me this muffin, this muffin, and this muffin.",
    },
    "bottle": {
        "description": "streamed human-view beverage/bottle sorting",
        "default_instruction": "Put all the beverages to the yellow basket.",
    },
    "passing_tools": {
        "description": "streamed human-view passing tools",
        "default_instruction": "Give me the hex screwdriver and 12 cm metal support.",
    },
}


class EpisodeEnd(Exception):
    """Signal that the current episode should end and reset."""


class ProgramQuit(Exception):
    """Signal that the evaluation program should exit."""


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
    """Stream observations, collect audio. Press 'p' to transcribe, 's' to skip."""
    audio_records = []
    timer = Timer(_VIDEO_RECORD_FPS)

    print("\n--- Voice instruction capture ---")
    print("  Speak your instruction, then press 'p' to transcribe.")
    print("  Press 's' to skip (use default/previous instruction).")
    print("  Press 'q' to quit.\n")

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
        audio_records.append({
            "audio_ts_ns": audio_ts_ns,
            "sample_rate_hz": audio_sr,
            "num_samples": audio_ns,
            "chunk_pcm16": audio_chunk.copy(),
        })

        preview = _build_eval_preview(
            obs,
            "VOICE: awaiting instruction",
            0,
            controls_text="p=transcribe  s=skip  q=quit",
        )
        cv2.imshow(window_name, preview)
        key = _poll_keypress()
        timer.sleep()

        if key in (ord('q'), ord('Q'), 27):
            return None

        if key in (ord('s'), ord('S')):
            return ""

        if key in (ord('p'), ord('P')):
            audio_pcm16, sample_rate = _collect_audio_pcm16(audio_records)
            if audio_pcm16.size == 0 or sample_rate <= 0:
                print("[Voice] No audio captured. Try again or press 's' to skip.")
                audio_records.clear()
                continue

            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_tmp.close()
            _write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
            try:
                print("[Voice] Transcribing with Whisper...")
                result = _whisper_transcribe_wav(wav_tmp.name)
                text = result.get("text", "").strip()
                if text:
                    print(f"[Voice] Transcribed: \"{text}\"")
                    return text
                else:
                    print("[Voice] Empty transcription. Try again.")
                    audio_records.clear()
            finally:
                os.unlink(wav_tmp.name)


# ---------------------------------------------------------------------------
# CV2 preview helpers
# ---------------------------------------------------------------------------

def _build_eval_preview(
    obs,
    language_instruction: str,
    global_step: int,
    controls_text: str = "q=quit episode",
):
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
        controls_text,
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


def _wait_for_preset_episode_start(env, language_instruction: str) -> bool:
    """Return True when the operator presses 's', False when they quit."""
    window_name = "Preset Language  |  s=start  q=quit"
    timer = Timer(_VIDEO_RECORD_FPS)

    print("\n--- Preset language episode launcher ---")
    print("  Press 's' to start an episode.")
    print("  Press 'q' to quit the program.\n")

    while True:
        obs = _extract_latest_observation(
            env.get_observation(mode="human_view_streamed_and_keyframe_cropped")
        )
        preview = _build_eval_preview(
            obs,
            language_instruction,
            0,
            controls_text="READY: s=start episode  q=quit program",
        )
        cv2.imshow(window_name, preview)
        key = _poll_keypress()

        if key in (ord("s"), ord("S")):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                cv2.destroyAllWindows()
            return True
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
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
    language_instruction: str,
    horizon: int,
    video_buffer: VideoBuffer,
    chunksize: int = 8,
    preset_language: bool = False,
):
    """
    Robot execution loop driven only by the fixed language instruction.

    Termination:
      1. horizon reached
      2. preset mode: 'e' ends episode, 'q' / ESC quits program
      3. voice mode: 'q' / ESC interrupts episode
    """
    top_view_frames_list = []
    comp_prob_log = []
    aria_frames = []
    audio_records = []
    global_step = 0
    timer = Timer(env.control_hz)
    controls_text = (
        "e=end episode  q=quit program"
        if preset_language
        else "q=quit episode"
    )
    window_name = f"Eval  |  {controls_text}"
    rollout_video_path = os.path.join(_TEMP_DIR, "rollout.mp4")
    human_video_path = os.path.join(_TEMP_DIR, "human_view_episode.mp4")

    for step in range(horizon):
        try:
            obs = env.get_observation(mode="human_view_streamed_and_keyframe_cropped")["timestamp"]
            images = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for value in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, value])

            aria_bgr = images["aria_rgb"]
            cropped_bgr = images["aria_cropped_rgb"]
            gaze_pt = np.asarray(obs.get("gaze_point", [0.0, 0.0]), dtype=np.float64).reshape(-1)
            gx = float(gaze_pt[0]) if gaze_pt.size >= 1 else None
            gy = float(gaze_pt[1]) if gaze_pt.size >= 2 else None

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

            human_view_gaze_cropped_rgb = cv2.cvtColor(images["aria_cropped_rgb"], cv2.COLOR_BGR2RGB)
            observation = {
                "observation.images.top_view": images["top_camera_rgb"],
                "observation.images.human_view_gaze_cropped": human_view_gaze_cropped_rgb,
                "observation.images.left_wrist_view": images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view": images["right_wrist_camera_rgb"],
                "prompt": language_instruction,
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
                top_view_frames_list.append(live_obs["images"]["top_camera_rgb"].copy())

                live_aria_bgr = live_obs["images"]["aria_rgb"]
                live_cropped_bgr = live_obs["images"]["aria_cropped_rgb"]
                live_audio_obs = live_obs.get("audio", {})
                live_audio_chunk = np.asarray(
                    live_audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
                    dtype=np.int16,
                )
                live_audio_ts_ns = int(live_audio_obs.get("audio_ts_ns", 0) or 0)
                live_audio_sample_rate_hz = int(live_audio_obs.get("sample_rate_hz", 0) or 0)
                live_audio_num_samples = int(
                    live_audio_obs.get("num_samples", live_audio_chunk.size) or 0
                )

                target_h, target_w = aria_bgr.shape[:2]
                if live_aria_bgr.shape[:2] != (target_h, target_w):
                    live_aria_bgr = cv2.resize(
                        live_aria_bgr,
                        (target_w, target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                video_buffer.append(
                    live_aria_bgr,
                    live_cropped_bgr,
                    None,
                    None,
                    audio_chunk_pcm16=live_audio_chunk,
                    audio_ts_ns=live_audio_ts_ns,
                    audio_sample_rate_hz=live_audio_sample_rate_hz,
                    audio_num_samples=live_audio_num_samples,
                )
                aria_frames.append(live_aria_bgr.copy())
                audio_records.append(
                    {
                        "frame_idx": len(audio_records),
                        "audio_ts_ns": live_audio_ts_ns,
                        "sample_rate_hz": live_audio_sample_rate_hz,
                        "num_samples": live_audio_num_samples,
                        "chunk_pcm16": live_audio_chunk.copy(),
                    }
                )

                preview = _build_eval_preview(
                    live_obs,
                    language_instruction,
                    global_step,
                    controls_text=controls_text,
                )
                cv2.imshow(window_name, preview)
                key = _poll_keypress()
                if preset_language and key in (ord("e"), ord("E")):
                    cv2.destroyAllWindows()
                    raise EpisodeEnd
                if key in (ord("q"), ord("Q"), 27):
                    cv2.destroyAllWindows()
                    if preset_language:
                        raise ProgramQuit
                    raise KeyboardInterrupt

                timer.sleep()

        except (EpisodeEnd, ProgramQuit, KeyboardInterrupt):
            _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path)
            _write_video_with_audio(aria_frames, audio_records, human_video_path, fps=_VIDEO_RECORD_FPS)
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()
    _save_rollout_video(top_view_frames_list, comp_prob_log, rollout_video_path)
    _write_video_with_audio(aria_frames, audio_records, human_video_path, fps=_VIDEO_RECORD_FPS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(default_task: str = DEFAULT_TASK, allow_task_override: bool = True):
    if default_task not in TASK_PROFILES:
        raise ValueError(f"Unknown task preset: {default_task}")

    parser = argparse.ArgumentParser(
        description="StreamVLA eval runner with voice input and task presets."
    )
    if allow_task_override:
        parser.add_argument(
            "--task",
            choices=sorted(TASK_PROFILES),
            default=default_task,
            help="Task preset to use for the initial instruction.",
        )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the task preset instruction.",
    )
    parser.add_argument(
        "--preset-language",
        "--no-voice",
        action="store_true",
        dest="preset_language",
        help="Skip voice capture and use only the preset/--instruction text.",
    )
    parser.add_argument(
        "--aria-host",
        default=SERVER_IP,
        help="TCP host/IP for the Aria camera publisher.",
    )
    parser.add_argument(
        "--aria-port",
        type=int,
        default=DEFAULT_ARIA_PORT,
        help="TCP port for the Aria camera publisher.",
    )
    parser.add_argument(
        "--policy-host",
        default=DEFAULT_POLICY_HOST,
        help="OpenPI websocket policy host.",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=DEFAULT_POLICY_PORT,
        help="OpenPI websocket policy port.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Maximum control steps per episode.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="Number of policy actions to execute from each inferred chunk.",
    )
    args = parser.parse_args()
    if not allow_task_override:
        args.task = default_task
    return args


def _build_camera_ids(aria_host: str, aria_port: int):
    camera_ids = {
        "top_camera": "242322070028",
        "left_wrist_camera": "244222076964",
        "right_wrist_camera": "244222071848",
        "aria": f"{aria_host}_{aria_port}",
    }
    return camera_ids


def main(default_task: str = DEFAULT_TASK, allow_task_override: bool = True):
    args = _parse_args(default_task, allow_task_override=allow_task_override)
    profile = TASK_PROFILES[args.task]

    env = PiperRobotEnv(camera_ids=_build_camera_ids(args.aria_host, args.aria_port))
    language_instruction = args.instruction or profile["default_instruction"]
    mode_name = "preset language" if args.preset_language else "voice input"

    print(f"Starting stream-VLA evaluation with {mode_name} ({args.task}).")
    print(f"Task profile: {profile['description']}")
    print(f"Aria TCP: {args.aria_host}:{args.aria_port}")
    print(f"Policy websocket: {args.policy_host}:{args.policy_port}")
    if args.preset_language:
        print("Preset language mode: voice capture is disabled.")
        print("Press 's' to start, 'e' to end one episode, 'q' to quit.")
    else:
        print("Press 'p' to capture voice instruction, 's' to skip, 'q' to quit.")

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )

    try:
        while True:
            try:
                if args.preset_language:
                    with _terminal_keypress_mode():
                        should_start = _wait_for_preset_episode_start(
                            env,
                            language_instruction,
                        )
                    if not should_start:
                        print("Exiting.")
                        break
                    cv2.destroyAllWindows()
                else:
                    # Voice capture phase
                    result = capture_voice_instruction(env)
                    if result is None:
                        print("Exiting.")
                        break
                    if result:  # non-empty string = new instruction
                        language_instruction = result
                print(f"Instruction: '{language_instruction}'")

                # Eval phase
                video_buffer = VideoBuffer()
                with _terminal_keypress_mode():
                    eval(
                        env,
                        client,
                        language_instruction,
                        args.horizon,
                        video_buffer,
                        chunksize=args.chunksize,
                        preset_language=args.preset_language,
                    )
                print("Episode finished.")

            except EpisodeEnd:
                print("\nEpisode ended.")
            except ProgramQuit:
                print("\nExiting evaluation...")
                break
            except KeyboardInterrupt:
                if args.preset_language:
                    print("\nExiting evaluation...")
                    break
                print("\nEpisode interrupted.")

            # Reset for next episode
            cv2.destroyAllWindows()
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(
                host=args.policy_host,
                port=args.policy_port,
            )

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        if os.path.isdir(_TEMP_DIR):
            import shutil
            shutil.rmtree(_TEMP_DIR)
            os.makedirs(_TEMP_DIR, exist_ok=True)
            print(f"Cleared temp directory: {_TEMP_DIR}")
        env.safe_end()


if __name__ == "__main__":
    main()
