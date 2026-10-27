"""Local speech-to-text from ``obs["audio"]`` for policies.

- ``SpeechCaptionPlanner`` (policy_edith): captions each planner clip word by
  word, as in the original experiments, before it is sent to Gemini. Captions
  are drawn only on the frames the planner sees; queued keyframes stay clean.
- ``UtteranceListener`` (streamvla): transcribes each utterance when the
  speaker pauses and keeps the latest one as the instruction.

Whisper and voice detection always run in background jobs, never in ``act()``.
"""

from collections import deque
from dataclasses import replace
from functools import lru_cache
import importlib.util
import re
import threading

import cv2
import numpy as np

from .chunked import Job

WHISPER_RATE_HZ = 16_000
BOILERPLATE = (
    "thank you for watching",
    "thanks for watching",
    "have a nice day",
    "subscribe",
)
CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)


class AudioWindow:
    """Recent mono microphone audio from ``obs["audio"]``, read by the planner thread."""

    def __init__(self, channel=0, max_seconds=30.0):
        if channel != "mean" and (type(channel) is not int or channel < 0):
            raise ValueError("speech_channel must be a channel index or 'mean'")
        self.channel = channel
        self.max_ns = int(max_seconds * 1e9)
        self.lock = threading.Lock()
        self.chunks = deque()

    def clear(self):
        with self.lock:
            self.chunks.clear()

    def add(self, audio):
        samples = np.asarray(audio["samples"])
        if samples.ndim != 2 or not len(samples):
            return
        times = np.asarray(audio.get("capture_timestamps_ns", ()), dtype=np.int64)
        if times.shape != (len(samples),):
            step = 1e9 / float(audio["sample_rate_hz"])
            offsets = np.round(np.arange(len(samples))[::-1] * step).astype(np.int64)
            times = int(audio["end_timestamp_ns"]) - offsets
        if self.channel == "mean":
            mono = samples.mean(axis=1)
        else:
            mono = samples[:, min(self.channel, samples.shape[1] - 1)]
        scale = float(audio.get("max_amplitude") or np.iinfo(np.int32).max)
        mono = mono.astype(np.float32) / scale
        muted = np.asarray(audio.get("muted", ()), dtype=bool)
        if muted.shape == mono.shape:
            mono[muted] = 0
        with self.lock:
            self.chunks.append((times, mono))
            while self.chunks and times[-1] - self.chunks[0][0][-1] > self.max_ns:
                self.chunks.popleft()

    def span(self):
        """Return ``(first, last)`` buffered capture times, or None when empty."""

        with self.lock:
            if not self.chunks:
                return None
            return int(self.chunks[0][0][0]), int(self.chunks[-1][0][-1])

    def clip(self, start_ns, end_ns, rate=WHISPER_RATE_HZ):
        """Return float32 audio on a uniform ``rate`` grid over [start_ns, end_ns].

        Samples are placed by capture time; spans with no captured audio are zero.
        """

        with self.lock:
            chunks = list(self.chunks)
        count = max(0, int((end_ns - start_ns) * rate / 1e9))
        if not chunks or not count:
            return np.zeros(count, np.float32)
        times = np.concatenate([chunk[0] for chunk in chunks])
        mono = np.concatenate([chunk[1] for chunk in chunks])
        grid = start_ns + np.arange(count) * (1e9 / rate)
        out = np.interp(grid, times, mono, left=0.0, right=0.0).astype(np.float32)
        if len(times) > 1:
            steps = np.diff(times)
            for i in np.flatnonzero(steps > 2 * np.median(steps)):
                out[(grid > times[i]) & (grid < times[i + 1])] = 0
        return out


def require_faster_whisper():
    if importlib.util.find_spec("faster_whisper") is None:
        raise ImportError(
            "speech_to_text needs faster-whisper: uv pip install -e '.[speech]'"
        )


class WhisperTranscriber:
    """faster-whisper; the model loads on ``load()`` or the first transcription."""

    def __init__(self, config):
        self.config = config
        self.language = config.get("whisper_language") or None
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            if self.model is None:
                require_faster_whisper()
                from faster_whisper import WhisperModel

                self.model = WhisperModel(
                    self.config.get("whisper_model", "small"),
                    device=self.config.get("whisper_device", "auto"),
                    compute_type=self.config.get("whisper_compute_type", "default"),
                )
        return self.model

    def __call__(self, audio):
        """Return ``(text, words)``; word times are seconds from the clip start."""

        if not audio.size:
            return "", []
        segments, _ = self.load().transcribe(
            audio,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments = list(segments)
        words = [
            {"word": word.word.strip(), "start": word.start, "end": word.end}
            for segment in segments
            for word in segment.words or ()
            if word.word.strip()
        ]
        return " ".join(segment.text.strip() for segment in segments).strip(), words


def useful_transcript(text, min_chars=2):
    """Reject empty clips and common Whisper hallucinations on silence."""

    text = " ".join(text.split())
    letters = re.sub(r"[\W_]+", "", text)  # counts Hangul as well as ASCII
    lower = text.lower()
    return len(letters) >= min_chars and not any(p in lower for p in BOILERPLATE)


def caption_at(words, time_s, frame_period_s, transcript=""):
    """Words spoken around ``time_s``, briefly holding the last word afterwards."""

    if not words:
        return transcript.strip()
    half = max(0.03, frame_period_s * 0.55)
    visible = [
        w["word"]
        for w in words
        if max(w["end"], w["start"]) >= time_s - half and w["start"] <= time_s + half
    ]
    if visible:
        return " ".join(visible)
    hold = min(0.18, max(0.04, frame_period_s * 0.75))
    for w in reversed(words):
        end = max(w["end"], w["start"])
        if end <= time_s and time_s - end <= hold:
            return w["word"]
        if end < time_s - hold:
            break
    return ""


@lru_cache(maxsize=8)
def _font(path, size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for candidate in (path, *CJK_FONTS) if path else CJK_FONTS:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return None


def draw_caption(image, text, font_path=None):
    """Return a copy of an RGB frame with ``text`` in a dark box, top center."""

    frame = np.ascontiguousarray(image).copy()
    if not text:
        return frame
    h, w = frame.shape[:2]
    size = max(14, int(h * 0.06))
    font = _font(font_path, size)
    if font is None:
        # OpenCV renders ASCII only; install pillow and a CJK font for Hangul.
        scale, thickness = max(0.4, h / 512), max(1, h // 256)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = (w - tw) // 2, th + 8
        box = frame.copy()
        cv2.rectangle(box, (x - 6, y - th - 6), (x + tw + 6, y + base + 6), (0, 0, 0), -1)
        cv2.addWeighted(box, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        return frame
    from PIL import Image, ImageDraw

    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil, "RGBA")
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    pad = max(4, size // 4)
    x, y = (w - (x1 - x0)) // 2, pad + 2
    draw.rectangle((x - pad, y - pad, x + x1 - x0 + pad, y + y1 - y0 + pad),
                   fill=(0, 0, 0, 153))
    draw.text((x - x0, y - y0), text, font=font, fill=(255, 255, 255, 255))
    return np.asarray(pil)


class SpeechCaptionPlanner:
    """Planner wrapper: transcribe the clip's speech, caption its frames, then plan.

    It runs inside the planner's background job, so Whisper never blocks act().
    Audio windows are contiguous across calls; no speech falls between clips.
    """

    def __init__(self, planner, transcribe, audio, min_chars=2, font_path=None):
        self.planner = planner
        self.transcribe = transcribe
        self.audio = audio
        self.min_chars = min_chars
        self.font_path = font_path
        self.audio_start_ns = None

    def restart(self):
        self.audio_start_ns = None

    def plan(self, frames, queue_state, transcript, initial):
        first, end = frames[0].timestamp_ns, frames[-1].timestamp_ns
        start = first if self.audio_start_ns is None else self.audio_start_ns
        self.audio_start_ns = end
        text, words = self.transcribe(self.audio.clip(start, end))
        if useful_transcript(text, self.min_chars):
            period = (end - first) / 1e9 / max(1, len(frames) - 1)
            frames = [
                replace(
                    frame,
                    image=draw_caption(
                        frame.image,
                        caption_at(words, (frame.timestamp_ns - start) / 1e9, period, text),
                        self.font_path,
                    ),
                )
                for frame in frames
            ]
            transcript = text
        return self.planner.plan(frames, queue_state, transcript, initial)

    def close(self):
        self.planner.close()


def detect_speech(audio, rate=WHISPER_RATE_HZ):
    """Silero VAD bundled with faster-whisper: ``[(start_s, end_s), ...]``."""

    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=500, speech_pad_ms=100)
    return [
        (span["start"] / rate, span["end"] / rate)
        for span in get_speech_timestamps(audio, options, sampling_rate=rate)
    ]


class UtteranceListener:
    """Keep the latest spoken instruction, transcribing each utterance as it ends.

    Each step adds the new audio; about every ``check_s`` a background job runs
    voice detection on the audio not yet handled. An utterance ends once
    ``silence_s`` of silence follows it (or it exceeds ``max_utterance_s``),
    and only then is it transcribed. With ``mode="first"`` it stops listening
    after the first instruction.
    """

    def __init__(
        self,
        transcribe,
        audio,
        detect=detect_speech,
        *,
        silence_s=0.8,
        max_utterance_s=15.0,
        check_s=0.5,
        min_chars=2,
        mode="latest",
        timeout_s=60.0,
    ):
        if mode not in ("latest", "first"):
            raise ValueError("speech_mode must be 'latest' or 'first'")
        self.transcribe, self.audio, self.detect = transcribe, audio, detect
        self.silence_s, self.max_s = float(silence_s), float(max_utterance_s)
        self.check_ns, self.pad_s = int(check_s * 1e9), 0.2
        self.min_chars, self.mode, self.timeout = min_chars, mode, float(timeout_s)
        self.reset(warm_up=False)

    def reset(self, warm_up=True):
        self.audio.clear()
        self.cursor_ns = None
        self.instruction = None
        # Load Whisper and the voice detector before the first utterance; a
        # first-time model download may take minutes.
        self.job = Job(self._warm_up) if warm_up else None
        self.job_timeout = max(self.timeout, 600.0)

    def _warm_up(self):
        load = getattr(self.transcribe, "load", None)
        if load is not None:
            load()
        self.detect(np.zeros(WHISPER_RATE_HZ, np.float32))
        return "", None

    def update(self, observation):
        """Call once per step; returns the latest instruction (or None)."""

        if observation.get("audio") is not None:
            self.audio.add(observation["audio"])
        if self.job is not None:
            done, (text, cursor) = self._poll()
            if done:
                self.job = None
                if cursor is not None:
                    self.cursor_ns = max(self.cursor_ns or cursor, cursor)
                if text:
                    self.instruction = text
        span = self.audio.span()
        if span is None or self.job is not None:
            return self.instruction
        if self.mode == "first" and self.instruction:
            return self.instruction
        if self.cursor_ns is None:
            self.cursor_ns = span[0]
        if span[1] - self.cursor_ns >= self.check_ns:
            self.job = Job(self._listen, self.cursor_ns, span[1])
            self.job_timeout = self.timeout
        return self.instruction

    def _poll(self):
        done, result = self.job.poll(self.job_timeout)
        return done, (result if done else ("", None))

    def _listen(self, start_ns, end_ns):
        clip = self.audio.clip(start_ns, end_ns)
        duration = len(clip) / WHISPER_RATE_HZ
        speech = self.detect(clip)
        if not speech:
            # Nothing said: keep only a short tail so a starting word is not cut.
            return "", max(start_ns, end_ns - int(self.pad_s * 1e9))
        first, last = speech[0][0], speech[-1][1]
        if duration - last < self.silence_s and last - first < self.max_s:
            # Still speaking: resume from where the speech started.
            return "", start_ns + int(max(0.0, first - self.pad_s) * 1e9)
        a = max(0.0, first - self.pad_s)
        b = min(duration, last + self.pad_s)
        text, _ = self.transcribe(
            clip[int(a * WHISPER_RATE_HZ) : int(b * WHISPER_RATE_HZ)]
        )
        text = " ".join(text.split())
        return (text if useful_transcript(text, self.min_chars) else ""), start_ns + int(
            b * 1e9
        )
