"""eval_openpi_textvla.py

Text-only VLA evaluation with voice-based instruction input.

Press 'p' during the pre-roll observation stream to capture audio,
transcribe it via Whisper, and use the result as the language instruction.
Press 's' to start executing with the current instruction.
Press 'q' to quit.
"""

import contextlib
import os
import select
import sys
import tempfile
import termios
import threading
import time
import tty
import wave
from typing import Optional

import cv2
import numpy as np

from openai import OpenAI

from edith.envs.piper_env_gaze import PiperRobotEnv
from edith.utils.time_utils import Timer

from openpi_client import websocket_client_policy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_RECORD_FPS = 30
_TEMP_DIR = os.path.normpath("/home/alin/projects/piper_project/temp")

SERVER_IP = "137.68.92.53"
PUBLISHER_SERVER_PORT = 9999


# ---------------------------------------------------------------------------
# Keyboard input helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _terminal_keypress_mode():
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
# Audio / Whisper helpers
# ---------------------------------------------------------------------------

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
        chunk_sr = int(record.get("sample_rate_hz", 0) or 0)
        if chunk_sr <= 0:
            continue
        if sample_rate_hz not in (0, chunk_sr):
            raise ValueError(
                f"Audio sample rate changed: {sample_rate_hz} -> {chunk_sr}"
            )
        sample_rate_hz = chunk_sr
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

    def start_recording(self):
        with self._lock:
            self._audio_records.clear()
            self._recording = True
        print("\n[VoiceMonitor] ● Recording started. Press 'p' again to stop.")

    def stop_recording(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            records = list(self._audio_records)
            self._audio_records.clear()
        print("[VoiceMonitor] ■ Recording stopped. Transcribing in background...")
        threading.Thread(
            target=self._transcribe_and_update,
            args=(records,),
            daemon=True,
            name="VoiceMonitor-Transcribe",
        ).start()

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


# ---------------------------------------------------------------------------
# CV2 preview
# ---------------------------------------------------------------------------

def _build_preview(obs, label: str):
    images = obs["images"]
    human_view = images["aria_rgb"].copy()
    top_view = images["top_camera_rgb"].copy()

    gaze_point = np.asarray(obs.get("gaze_point", []), dtype=np.float32).reshape(-1)
    if gaze_point.size >= 2 and np.all(np.isfinite(gaze_point[:2])):
        x = int(np.clip(gaze_point[0], 0, human_view.shape[1] - 1))
        y = int(np.clip(gaze_point[1], 0, human_view.shape[0] - 1))
        cv2.circle(human_view, (x, y), 6, (0, 255, 0), 2)

    if top_view.shape[:2] != human_view.shape[:2]:
        top_view = cv2.resize(
            top_view, (human_view.shape[1], human_view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    preview = np.hstack((human_view, top_view))
    cv2.putText(preview, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Voice instruction capture
# ---------------------------------------------------------------------------

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

        # Collect audio
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

        preview = _build_preview(obs, "VOICE CAPTURE: p=transcribe  s=skip  q=quit")
        cv2.imshow(window_name, preview)
        key = _poll_keypress()
        timer.sleep()

        if key in (ord('q'), ord('Q'), 27):
            return None  # signal quit

        if key in (ord('s'), ord('S')):
            return ""  # skip, keep current instruction

        if key in (ord('p'), ord('P')):
            # Transcribe collected audio
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
# Main eval loop
# ---------------------------------------------------------------------------

def _feed_voice_monitor_from_obs(voice_monitor: Optional[VoiceMonitor], obs: dict):
    if voice_monitor is None:
        return
    audio_obs = obs.get("audio", {})
    chunk = np.asarray(
        audio_obs.get("chunk_pcm16", np.zeros((0,), dtype=np.int16)),
        dtype=np.int16,
    )
    sr = int(audio_obs.get("sample_rate_hz", 0) or 0)
    ts = int(audio_obs.get("audio_ts_ns", 0) or 0)
    ns = int(audio_obs.get("num_samples", chunk.size) or 0)
    voice_monitor.feed_audio(chunk, sr, audio_ts_ns=ts, num_samples=ns)


def eval(env, client, voice_monitor: VoiceMonitor, horizon: int, chunksize: int = 16):
    timer = Timer(env.control_hz)
    window_name = "Eval  |  q=quit"
    global_step = 0

    for step in range(horizon):
        try:
            obs = _extract_latest_observation(
                env.get_observation(mode='human_view_streamed_and_keyframe_cropped'))
            images = obs["images"]
            robot_state = obs["robot_state"]

            # Feed audio to the background voice monitor
            _feed_voice_monitor_from_obs(voice_monitor, obs)

            concatenated_state = np.array([])
            for v in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, v])

            # Always read the latest instruction (may have been replaced by voice)
            current_instruction = voice_monitor.instruction

            observation = {
                "observation.images.top_view": images["top_camera_rgb"],
                "observation.images.left_wrist_view": images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view": images["right_wrist_camera_rgb"],
                "prompt": current_instruction,
                "observation.state": concatenated_state,
            }

            action_chunk = client.infer(observation)["actions"][:chunksize, :]
            for i in range(len(action_chunk)):
                action = {
                    'left_arm_positions': action_chunk[i][0:6],
                    'left_gripper_position': action_chunk[i][6:7],
                    'right_arm_positions': action_chunk[i][7:13],
                    'right_gripper_position': action_chunk[i][13:14],
                }
                env.step(action)
                global_step += 1

                live_obs = _extract_latest_observation(
                    env.get_observation(mode='human_view_streamed_and_keyframe_cropped'))
                _feed_voice_monitor_from_obs(voice_monitor, live_obs)
                live_instruction = voice_monitor.instruction
                rec_label = " [REC ●]" if voice_monitor.is_recording() else ""
                preview = _build_preview(
                    live_obs,
                    f"EVAL step={global_step}{rec_label} | prompt: {live_instruction[:60]}")
                cv2.imshow(window_name, preview)
                key = _poll_keypress()
                if key in (ord('q'), ord('Q'), 27):
                    raise KeyboardInterrupt
                if key in (ord('p'), ord('P')):
                    if voice_monitor.is_recording():
                        voice_monitor.stop_recording()
                    else:
                        voice_monitor.start_recording()

                timer.sleep()

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
        'aria': f'{SERVER_IP}_{PUBLISHER_SERVER_PORT}',
    }
    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "Pick up the muffin."
    horizon = 10000

    print("Starting text-VLA evaluation with voice input.")
    print("Press 'p' to capture voice instruction, 's' to skip, 'q' to quit.")

    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    voice_monitor: Optional[VoiceMonitor] = None
    try:
        while True:
            try:
                # Voice capture phase (pre-episode)
                result = capture_voice_instruction(env)
                if result is None:
                    print("Exiting.")
                    break
                if result:  # non-empty string = new instruction
                    language_instruction = result
                print(f"Instruction: '{language_instruction}'")

                # Start (or update) the background voice monitor for this episode
                if voice_monitor is None:
                    voice_monitor = VoiceMonitor(language_instruction)
                    voice_monitor.start()
                else:
                    voice_monitor.set_instruction(language_instruction)

                # Eval phase
                with _terminal_keypress_mode():
                    eval(env, client, voice_monitor, horizon)
                print("Episode finished.")

            except KeyboardInterrupt:
                print("\nEpisode interrupted (Ctrl+C).")

            # Reset for next episode (keep voice monitor alive across episodes)
            cv2.destroyAllWindows()
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(
                host="localhost", port=8000)
            if voice_monitor is not None:
                language_instruction = voice_monitor.instruction

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        if voice_monitor is not None:
            voice_monitor.stop()
        env.safe_end()
