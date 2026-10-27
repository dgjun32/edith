"""Hardware-free tests for local speech in policies (fake Whisper, VLA and VAD)."""

import importlib.util
import json
from pathlib import Path
import time
import unittest

import numpy as np

from edith.policies.common import speech
from edith.policies.common.tasks import Frame
from edith.policies.loader import _entrypoint

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "edith/policies/policy_edith"
STREAMVLA_DIR = ROOT / "edith/policies/streamvla"

RATE = 16_000
STEP_NS = 62_500  # one 16 kHz frame


def audio_obs(start_ns, count, value=1000, channels=2, muted=None):
    samples = np.zeros((count, channels), np.int32)
    samples[:, 0] = value
    samples[:, 1] = -value
    return {
        "samples": samples,
        "capture_timestamps_ns": start_ns + np.arange(count, dtype=np.int64) * STEP_NS,
        "muted": np.zeros(count, np.uint8) if muted is None else muted,
        "max_amplitude": 10_000.0,
        "sample_rate_hz": RATE,
        "end_timestamp_ns": start_ns + (count - 1) * STEP_NS,
    }


class AudioWindowTests(unittest.TestCase):
    def test_clip_places_samples_by_capture_time_and_zero_fills_gaps(self):
        window = speech.AudioWindow(channel=0)
        window.add(audio_obs(0, RATE // 10))  # 0.0-0.1 s
        window.add(audio_obs(200_000_000, RATE // 10))  # 0.2-0.3 s
        clip = window.clip(0, 300_000_000)
        self.assertEqual(clip.dtype, np.float32)
        self.assertEqual(len(clip), int(0.3 * RATE))
        np.testing.assert_allclose(clip[: RATE // 10 - 1], 0.1)  # 1000 / max_amplitude
        np.testing.assert_array_equal(clip[RATE // 10 + 10 : RATE // 5 - 10], 0)
        np.testing.assert_allclose(clip[RATE // 5 + 1 :], 0.1)

    def test_channel_choice_and_mute_flags(self):
        muted = np.zeros(100, np.uint8)
        muted[:50] = 1
        window = speech.AudioWindow(channel="mean")
        window.add(audio_obs(0, 100))
        np.testing.assert_allclose(window.clip(0, 99 * STEP_NS)[:90], 0)
        window = speech.AudioWindow(channel=1)
        window.add(audio_obs(0, 100, muted=muted))
        clip = window.clip(0, 99 * STEP_NS)
        np.testing.assert_array_equal(clip[:49], 0)
        np.testing.assert_allclose(clip[51:90], -0.1)


class CaptionTests(unittest.TestCase):
    words = [
        {"word": "give", "start": 0.0, "end": 0.3},
        {"word": "me", "start": 0.35, "end": 0.5},
        {"word": "that", "start": 1.0, "end": 1.2},
    ]

    def test_caption_follows_word_timestamps(self):
        self.assertEqual(speech.caption_at(self.words, 0.25, 0.25), "give me")
        self.assertEqual(speech.caption_at(self.words, 0.75, 0.25), "")
        self.assertEqual(speech.caption_at(self.words, 1.1, 0.25), "that")
        self.assertEqual(speech.caption_at([], 1.1, 0.25, "full text"), "full text")

    def test_useful_transcript_counts_hangul_and_rejects_boilerplate(self):
        self.assertTrue(speech.useful_transcript("저거 줘"))
        self.assertFalse(speech.useful_transcript("네"))
        self.assertFalse(speech.useful_transcript("  "))
        self.assertFalse(speech.useful_transcript("Thanks for watching!"))

    def test_draw_caption_returns_marked_copy(self):
        image = np.zeros((224, 224, 3), np.uint8)
        captioned = speech.draw_caption(image, "give me that")
        self.assertEqual(captioned.shape, image.shape)
        self.assertGreater(int(captioned.max()), 200)
        self.assertEqual(int(image.max()), 0)


class FakePlanner:
    def __init__(self):
        self.calls = []
        self.closed = False

    def plan(self, frames, queue_state, transcript, initial):
        self.calls.append((frames, transcript))
        return {"command": "NoOp"}

    def close(self):
        self.closed = True


class FakeTranscriber:
    def __init__(self, text, words):
        self.text, self.words, self.lengths = text, words, []

    def __call__(self, audio):
        self.lengths.append(len(audio))
        return self.text, self.words


def frames(start_ns, count, period_ns=250_000_000):
    return [
        Frame(
            start_ns + i * period_ns,
            np.zeros((224, 224, 3), np.uint8),
            np.zeros((224, 224, 3), np.uint8),
        )
        for i in range(count)
    ]


class SpeechCaptionPlannerTests(unittest.TestCase):
    def test_captions_planner_frames_only_and_passes_transcript(self):
        inner = FakePlanner()
        words = [{"word": "screwdriver", "start": 0.4, "end": 0.9}]
        transcribe = FakeTranscriber("give me that screwdriver", words)
        planner = speech.SpeechCaptionPlanner(
            inner, transcribe, speech.AudioWindow()
        )
        clip = frames(1_000_000_000, 9)  # 0-2 s at 4 fps
        planner.plan(clip, [], "old file transcript", True)
        sent, transcript = inner.calls[0]
        self.assertEqual(transcript, "give me that screwdriver")
        captioned = [i for i, f in enumerate(sent) if f.image.max() > 0]
        self.assertEqual(captioned, [2, 3, 4])  # frames at 0.5-1.0 s overlap 0.4-0.9 s
        for original, planned in zip(clip, sent):
            self.assertIs(planned.keyframe, original.keyframe)
            self.assertEqual(int(original.image.max()), 0)

    def test_audio_windows_are_contiguous_and_silence_keeps_transcript(self):
        inner = FakePlanner()
        transcribe = FakeTranscriber("", [])
        planner = speech.SpeechCaptionPlanner(
            inner, transcribe, speech.AudioWindow()
        )
        planner.plan(frames(0, 33), [], "file text", True)  # 0-8 s
        planner.plan(frames(9_000_000_000, 33), [], "file text", False)  # 9-17 s
        self.assertEqual(transcribe.lengths, [8 * RATE, 9 * RATE])  # 8-17 s, no gap
        self.assertEqual(inner.calls[1][1], "file text")
        self.assertTrue(all(f.image.max() == 0 for f in inner.calls[1][0]))
        planner.close()
        self.assertTrue(inner.closed)


class FakeClient:
    def __init__(self):
        self.closed = False

    def infer(self, request):
        actions = np.zeros((8, 14))
        actions[:, [6, 13]] = 0.5
        return {"actions": actions, "completion_prob": 0.0}

    def close(self):
        self.closed = True


class PolicyIntegrationTests(unittest.TestCase):
    def config(self, **overrides):
        manifest = json.loads((POLICY_DIR / "policy.json").read_text())
        return dict(manifest["config"], **overrides)

    def observation(self, t_ns):
        q = np.zeros(14)
        q[[6, 13]] = 0.5
        return {
            "images": {
                name: np.zeros((48, 64, 3), np.uint8)
                for name in (
                    "aria_rgb",
                    "top_camera_rgb",
                    "left_wrist_camera_rgb",
                    "right_wrist_camera_rgb",
                )
            },
            "gaze_point": np.array([32.0, 24.0]),
            "robot_state": q,
            "arm_order": ("left", "right"),
            "capture_timestamps_ns": {"aria": t_ns, "top": t_ns},
            "audio": audio_obs(t_ns - 1000 * STEP_NS, 1000),
        }

    def test_gemini_planner_is_wrapped_and_audio_is_collected(self):
        Policy = _entrypoint(POLICY_DIR, "policy.py:Policy")
        policy = Policy(
            POLICY_DIR,
            self.config(planner="gemini"),
            client=FakeClient(),
            planner=FakePlanner(),
            transcriber=FakeTranscriber("", []),
        )
        self.assertEqual(type(policy.planner).__name__, "SpeechCaptionPlanner")
        observation = self.observation(5_000_000_000)
        policy.reset(observation)
        policy.act(observation)
        self.assertEqual(len(policy.audio.chunks), 1)
        policy.close()

    def test_manual_planner_or_disabled_speech_never_loads_whisper(self):
        Policy = _entrypoint(POLICY_DIR, "policy.py:Policy")
        for config in (self.config(), self.config(planner="gemini", speech_to_text=False)):
            planner = FakePlanner() if config["planner"] == "gemini" else None
            policy = Policy(POLICY_DIR, config, client=FakeClient(), planner=planner)
            self.assertIsNone(policy.audio)
            policy.close()

    def test_missing_faster_whisper_fails_at_load_and_releases_clients(self):
        if importlib.util.find_spec("faster_whisper") is not None:
            self.skipTest("faster-whisper is installed")
        Policy = _entrypoint(POLICY_DIR, "policy.py:Policy")
        client, planner = FakeClient(), FakePlanner()
        with self.assertRaisesRegex(ImportError, "faster-whisper"):
            Policy(POLICY_DIR, self.config(planner="gemini"), client=client, planner=planner)
        self.assertTrue(client.closed)
        self.assertTrue(planner.closed)


def energy_vad(audio, rate=RATE):
    """Stand-in for Silero VAD: contiguous runs of |x| > 0.05 as (start_s, end_s)."""

    loud = np.abs(audio) > 0.05
    edges = np.flatnonzero(np.diff(np.r_[0, loud.astype(int), 0]))
    return [(a / rate, b / rate) for a, b in zip(edges[::2], edges[1::2])]


class ScriptedTranscriber:
    def __init__(self, *texts):
        self.texts, self.lengths = list(texts), []

    def __call__(self, audio):
        self.lengths.append(len(audio) / RATE)
        return self.texts.pop(0), []


def speak(listener_or_policy, speaking, seconds, observation=None, on_step=None):
    """Feed 15 Hz steps of audio that is loud while ``speaking(t)``."""

    per_step = RATE // 15
    for step in range(int(seconds * 15)):
        t = step * per_step / RATE
        audio = audio_obs(step * per_step * STEP_NS, per_step, 1000 if speaking(t) else 0)
        obs = dict(observation or {}, audio=audio)
        result = listener_or_policy.update(obs) if observation is None else listener_or_policy.act(obs)
        if on_step:
            on_step(t, result)
        time.sleep(0.003)  # let background detection/transcription jobs finish


class UtteranceListenerTests(unittest.TestCase):
    def listener(self, transcriber, **options):
        return speech.UtteranceListener(
            transcriber, speech.AudioWindow(), energy_vad, **options
        )

    def test_each_finished_utterance_becomes_the_instruction(self):
        transcriber = ScriptedTranscriber("give me that", "now the screwdriver")
        listener = self.listener(transcriber)
        seen = {}
        speak(
            listener,
            lambda t: 1.0 <= t < 2.0 or 4.0 <= t < 5.0,
            7.0,
            on_step=lambda t, text: seen.setdefault(round(t, 1), text),
        )
        self.assertIsNone(seen[2.5])  # 0.5 s after speaking: utterance not ended yet
        self.assertEqual(seen[3.9], "give me that")
        self.assertEqual(listener.instruction, "now the screwdriver")
        self.assertEqual(len(transcriber.lengths), 2)
        for length in transcriber.lengths:  # 1 s of speech plus padding only
            self.assertTrue(1.0 <= length <= 1.6, length)

    def test_first_mode_keeps_the_first_instruction(self):
        transcriber = ScriptedTranscriber("give me that", "unused")
        listener = self.listener(transcriber, mode="first")
        speak(listener, lambda t: 1.0 <= t < 2.0 or 4.0 <= t < 5.0, 7.0)
        self.assertEqual(listener.instruction, "give me that")
        self.assertEqual(len(transcriber.lengths), 1)

    def test_long_speech_is_cut_at_max_utterance(self):
        transcriber = ScriptedTranscriber("part one", "part two", "part three")
        listener = self.listener(transcriber, max_utterance_s=1.0)
        speak(listener, lambda t: 0.5 <= t < 3.5, 3.0)  # still speaking at the end
        self.assertGreaterEqual(len(transcriber.lengths), 1)
        self.assertTrue(all(length <= 2.0 for length in transcriber.lengths))
        self.assertTrue(listener.instruction.startswith("part"))


class RecordingVLA:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        actions = np.zeros((8, 14))
        actions[:, [6, 13]] = 0.5
        return {"actions": actions, "completion_prob": 0.0}

    def close(self):
        pass


class StreamVLATests(unittest.TestCase):
    def policy(self, transcriber, **overrides):
        config = json.loads((STREAMVLA_DIR / "policy.json").read_text())["config"]
        Policy = _entrypoint(STREAMVLA_DIR, "policy.py:Policy")
        client = RecordingVLA()
        return Policy(
            STREAMVLA_DIR, dict(config, **overrides),
            client=client, transcriber=transcriber, detect=energy_vad,
        ), client

    def observation(self):
        q = np.zeros(14)
        q[[6, 13]] = 0.5
        return {
            "images": {
                name: np.zeros((48, 64, 3), np.uint8)
                for name in (
                    "aria_rgb",
                    "top_camera_rgb",
                    "left_wrist_camera_rgb",
                    "right_wrist_camera_rgb",
                )
            },
            "gaze_point": np.array([32.0, 24.0]),
            "robot_state": q,
            "arm_order": ("left", "right"),
            "instruction": "",
        }

    def test_spoken_instruction_reaches_vla_with_live_ego_view(self):
        policy, client = self.policy(ScriptedTranscriber("give me that screwdriver"))
        obs = self.observation()
        policy.reset(obs)
        speak(policy, lambda t: 0.5 <= t < 1.5, 4.0, observation=obs)
        prompts = [r["prompt"] for r in client.requests]
        self.assertTrue(prompts)
        self.assertEqual(set(prompts), {"give me that screwdriver"})
        self.assertIn("observation.images.human_view_gaze_cropped", client.requests[0])
        policy.close()

    def test_preset_instruction_is_used_until_someone_speaks(self):
        policy, client = self.policy(
            ScriptedTranscriber("give me that screwdriver"), instruction="preset task"
        )
        obs = self.observation()
        policy.reset(obs)
        speak(policy, lambda t: 1.5 <= t < 2.5, 5.0, observation=obs)
        prompts = [r["prompt"] for r in client.requests]
        self.assertEqual(prompts[0], "preset task")
        self.assertEqual(prompts[-1], "give me that screwdriver")
        policy.close()

    def test_default_config_loads_without_whisper(self):
        from edith.policies import load_policy

        policy = load_policy("streamvla")
        self.assertIsNone(policy.instance.listener.transcribe.model)
        self.assertIsNone(policy.instance.listener.job)
        policy.close()


if __name__ == "__main__":
    unittest.main()
