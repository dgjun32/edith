"""Audio-only and Whisper helpers for the ours eval scripts."""

from __future__ import annotations

import os
import re
import tempfile
import wave
from typing import List, Optional, Tuple

import numpy as np

FAST_AUDIO_GATE_ENABLED = False
FAST_AUDIO_GATE_FRAME_MS = 30.0
FAST_AUDIO_GATE_NOISE_PERCENTILE = 20.0
FAST_AUDIO_GATE_NOISE_MULT = 3.5
FAST_AUDIO_GATE_MIN_RMS = 0.0020
FAST_AUDIO_GATE_MIN_PEAK = 0.012
FAST_AUDIO_GATE_MIN_ACTIVE_SEC = 0.12
FAST_AUDIO_GATE_MIN_ACTIVE_RATIO = 0.04
FAST_AUDIO_GATE_MIN_SNR_DB = 8.0
FAST_AUDIO_GATE_MIN_SPEECH_BAND_RATIO = 0.35
FAST_AUDIO_GATE_MAX_LOW_NOISE_RATIO = 0.85
FAST_AUDIO_GATE_MAX_HIGH_NOISE_RATIO = 0.90
FAST_AUDIO_GATE_EARLY_ACCEPT = True
FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_SEC = 0.18
FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_RATIO = 0.05
FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_SNR_DB = 12.0


def collect_audio_pcm16(audio_records):
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
                "Audio sample rate changed while recording keyframe video: "
                f"{sample_rate_hz} -> {chunk_sample_rate_hz}"
            )

        sample_rate_hz = chunk_sample_rate_hz
        chunks.append(chunk.copy())

    if not chunks or sample_rate_hz <= 0:
        return np.zeros((0,), dtype=np.int16), 0

    return np.concatenate(chunks, axis=0), sample_rate_hz


def write_pcm16_wav(audio_pcm16: np.ndarray, sample_rate_hz: int, output_path: str):
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(np.asarray(audio_pcm16, dtype=np.int16).tobytes())


def _pcm16_audio_stats(audio_pcm16: np.ndarray, sample_rate_hz: int) -> dict:
    audio_pcm16 = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
    if audio_pcm16.size == 0:
        return {
            "input_rms": 0.0,
            "input_peak": 0.0,
            "num_samples": 0,
            "sample_rate_hz": int(sample_rate_hz or 0),
        }

    audio_float = audio_pcm16.astype(np.float32) / 32768.0
    return {
        "input_rms": float(np.sqrt(np.mean(audio_float * audio_float))),
        "input_peak": float(np.max(np.abs(audio_float))),
        "num_samples": int(audio_pcm16.size),
        "sample_rate_hz": int(sample_rate_hz or 0),
    }


def fast_audio_gate(
    audio_pcm16: np.ndarray,
    sample_rate_hz: int,
    context: str = "whisper",
) -> Tuple[bool, dict]:
    """Cheap SNR/frequency sanity gate before Whisper transcription."""
    stats = _pcm16_audio_stats(audio_pcm16, sample_rate_hz)
    stats["fast_gate_enabled"] = bool(FAST_AUDIO_GATE_ENABLED)
    stats["fast_gate_passed"] = True
    stats["fast_gate_reason"] = "disabled"

    if not FAST_AUDIO_GATE_ENABLED:
        return True, stats

    audio_pcm16 = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
    if audio_pcm16.size == 0 or sample_rate_hz <= 0:
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "empty"
        return False, stats

    input_rms = float(stats.get("input_rms", 0.0))
    input_peak = float(stats.get("input_peak", 0.0))
    if input_rms < FAST_AUDIO_GATE_MIN_RMS or input_peak < FAST_AUDIO_GATE_MIN_PEAK:
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "low_energy"
        return False, stats

    x = audio_pcm16.astype(np.float32) / 32768.0
    x = x - float(np.mean(x))
    if x.size >= 2:
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - 0.97 * x[:-1]
    else:
        y = x

    frame_len = max(1, int(round(
        sample_rate_hz * FAST_AUDIO_GATE_FRAME_MS / 1000.0)))
    pad = (-y.size) % frame_len
    padded = np.pad(y, (0, pad), mode="constant") if pad else y
    frames = padded.reshape(-1, frame_len)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_floor = float(np.percentile(
        frame_rms, FAST_AUDIO_GATE_NOISE_PERCENTILE))
    threshold = max(
        FAST_AUDIO_GATE_MIN_RMS,
        noise_floor * FAST_AUDIO_GATE_NOISE_MULT,
    )
    active = frame_rms >= threshold
    active_ratio = float(np.mean(active)) if active.size else 0.0
    active_sec = float(np.sum(active) * frame_len / float(sample_rate_hz))
    active_frame_rms = frame_rms[active]
    active_rms = (
        float(np.sqrt(np.mean(active_frame_rms * active_frame_rms)))
        if active_frame_rms.size else 0.0
    )
    snr_db = float(20.0 * np.log10(
        max(active_rms, 1e-8) / max(noise_floor, 1e-8)))
    rms_p20 = float(np.percentile(frame_rms, 20.0)) if frame_rms.size else 0.0
    rms_p90 = float(np.percentile(frame_rms, 90.0)) if frame_rms.size else 0.0
    modulation_ratio = rms_p90 / max(rms_p20, 1e-6)

    stats.update({
        "fast_gate_noise_floor": noise_floor,
        "fast_gate_threshold": float(threshold),
        "fast_gate_active_ratio": active_ratio,
        "fast_gate_active_sec": active_sec,
        "fast_gate_active_rms": active_rms,
        "fast_gate_snr_db": snr_db,
        "fast_gate_modulation_ratio": modulation_ratio,
    })

    if (
        active_sec < FAST_AUDIO_GATE_MIN_ACTIVE_SEC
        or active_ratio < FAST_AUDIO_GATE_MIN_ACTIVE_RATIO
    ):
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "no_active_region"
        return False, stats

    if snr_db < FAST_AUDIO_GATE_MIN_SNR_DB:
        stats["fast_gate_passed"] = False
        stats["fast_gate_reason"] = "low_snr"
        return False, stats

    if (
        FAST_AUDIO_GATE_EARLY_ACCEPT
        and active_sec >= FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_SEC
        and active_ratio >= FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_ACTIVE_RATIO
        and snr_db >= FAST_AUDIO_GATE_EARLY_ACCEPT_MIN_SNR_DB
    ):
        stats["fast_gate_reason"] = "early_accept_voice"
        return True, stats

    stride = max(1, int(sample_rate_hz // 16000))
    sx = x[::stride]
    spectral_rate = sample_rate_hz / float(stride)
    if sx.size >= 256:
        window = np.hanning(sx.size).astype(np.float32)
        power = np.abs(np.fft.rfft(sx * window)) ** 2
        freqs = np.fft.rfftfreq(sx.size, d=1.0 / spectral_rate)

        def _band_ratio(lo_hz: float, hi_hz: float, total_power: float) -> float:
            mask = (freqs >= lo_hz) & (freqs <= min(hi_hz, spectral_rate / 2.0))
            if not np.any(mask) or total_power <= 1e-12:
                return 0.0
            return float(np.sum(power[mask]) / total_power)

        total_mask = (freqs >= 40.0) & (freqs <= min(7600.0, spectral_rate / 2.0))
        total_power = float(np.sum(power[total_mask])) if np.any(total_mask) else 0.0
        speech_ratio = _band_ratio(120.0, 4200.0, total_power)
        low_noise_ratio = _band_ratio(40.0, 100.0, total_power)
        high_noise_ratio = _band_ratio(6000.0, 7600.0, total_power)
        stats.update({
            "fast_gate_speech_band_ratio": speech_ratio,
            "fast_gate_low_noise_ratio": low_noise_ratio,
            "fast_gate_high_noise_ratio": high_noise_ratio,
        })

        if (
            speech_ratio < FAST_AUDIO_GATE_MIN_SPEECH_BAND_RATIO
            and (
                low_noise_ratio > FAST_AUDIO_GATE_MAX_LOW_NOISE_RATIO
                or high_noise_ratio > FAST_AUDIO_GATE_MAX_HIGH_NOISE_RATIO
            )
        ):
            stats["fast_gate_passed"] = False
            stats["fast_gate_reason"] = "noise_dominated"
            return False, stats

        if (
            active_ratio > 0.80
            and modulation_ratio < 1.35
            and speech_ratio < 0.55
        ):
            stats["fast_gate_passed"] = False
            stats["fast_gate_reason"] = "steady_noise"
            return False, stats

    stats["fast_gate_reason"] = "passed"
    print(
        f"  [AudioGate:{context}] pass "
        f"rms={input_rms:.4f}, active={active_sec:.2f}s/"
        f"{active_ratio:.2f}, snr={snr_db:.1f}dB, noise={noise_floor:.4f}"
    )
    return True, stats


def whisper_transcribe_wav(wav_path: str) -> dict:
    """Transcribe a WAV file using OpenAI Whisper API.

    Returns {"text": "...", "words": [{"word": "...", "start": <float>, "end": <float>}]}
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # uses OPENAI_API_KEY env var
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        result = {"text": response.text, "words": []}
        if hasattr(response, "words") and response.words:
            for w in response.words:
                result["words"].append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })
        return result
    except Exception as exc:
        print(f"  [WARN] Whisper transcription failed: {exc}")
        return {"text": "", "words": []}


def is_useful_whisper_transcript(
    transcript_data: Optional[dict],
    min_chars: int,
) -> bool:
    if not transcript_data:
        return False

    text = re.sub(r"\s+", " ", transcript_data.get("text", "") or "").strip()
    alnum = re.sub(r"[^A-Za-z0-9]+", "", text)
    if len(alnum) < min_chars:
        return False

    lower = text.lower()
    boilerplate = (
        "thank you for watching",
        "thanks for watching",
        "have a nice day",
        "subscribe",
    )
    if any(phrase in lower for phrase in boilerplate):
        return False

    return True


def transcribe_audio_records_for_whisper(
    audio_records: List[dict],
    context: str = "whisper",
) -> dict:
    audio_pcm16, sample_rate = collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate <= 0:
        return {"text": "", "words": []}

    
    gate_passed, gate_stats = fast_audio_gate(
        audio_pcm16, sample_rate, context=context)
    """
    if not gate_passed:
        print(
            f"  [AudioGate:{context}] skip Whisper: "
            f"{gate_stats.get('fast_gate_reason', 'unknown')} "
            f"(rms={gate_stats.get('input_rms', 0.0):.4f}, "
            f"peak={gate_stats.get('input_peak', 0.0):.4f}, "
            f"active={gate_stats.get('fast_gate_active_sec', 0.0):.2f}s, "
            f"snr={gate_stats.get('fast_gate_snr_db', 0.0):.1f}dB, "
            f"speech_band={gate_stats.get('fast_gate_speech_band_ratio', 0.0):.2f})"
        )
        return {"text": "", "words": [], "audio_stats": gate_stats}
    """
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
    try:
        transcript = whisper_transcribe_wav(wav_tmp.name)
        transcript["audio_stats"] = gate_stats
        text = (transcript.get("text", "") or "").strip()
        if text:
            print(f"  [Whisper:{context}] \"{text[:100]}\"")
        return transcript
    finally:
        os.unlink(wav_tmp.name)


def apply_whisper_transcript_to_frames(
    frames_bgr: List[np.ndarray],
    audio_records: List[dict],
    fps: float,
    min_chars: int,
    whisper_enabled: bool,
    transcript_data_override: Optional[dict] = None,
) -> Tuple[List[np.ndarray], str, dict]:
    """Transcribe audio and overlay aligned captions on frames when useful."""
    transcript_section = ""
    transcript_data: dict = {}

    if not whisper_enabled:
        return frames_bgr, transcript_section, transcript_data

    if transcript_data_override is not None:
        transcript_data = transcript_data_override
    else:
        transcript_data = transcribe_audio_records_for_whisper(
            audio_records, context="vlm")

    if not is_useful_whisper_transcript(transcript_data, min_chars):
        return frames_bgr, transcript_section, transcript_data

    transcript_text = transcript_data.get("text", "")
    from .video_processing import overlay_captions_on_frames

    captioned_frames = overlay_captions_on_frames(
        frames_bgr,
        transcript_data.get("words", []),
        transcript_text,
        0.0,
        fps,
    )
    transcript_section = (
        f"\nA speech transcript was detected in this clip:\n"
        f"  \"{transcript_text}\"\n"
        f"Use this transcript to understand the human's spoken instruction.\n"
    )
    return captioned_frames, transcript_section, transcript_data
