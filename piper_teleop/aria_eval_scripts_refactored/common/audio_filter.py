"""Lightweight audio denoising helpers for Whisper input.

The filter is intentionally dependency-free: it uses only NumPy so eval
scripts can run on the robot machine without adding SciPy/librosa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WhisperAudioFilterConfig:
    enabled: bool = True
    high_pass_hz: float = 120.0
    low_pass_hz: float = 7600.0
    noise_percentile: float = 20.0
    noise_gate_multiplier: float = 2.5
    noise_gate_min_gain: float = 0.15
    frame_ms: float = 20.0
    target_rms: float = 0.08
    max_normalize_gain: float = 4.0


class WhisperAudioFilter:
    """Filter PCM16 audio before it is sent to Whisper.

    Processing stages:
    1. DC offset removal
    2. one-pole high-pass to reduce low-frequency hum/handling noise
    3. optional one-pole low-pass to reduce hiss
    4. RMS noise gate with a soft floor
    5. conservative normalization for quieter speech
    """

    def __init__(self, config: Optional[WhisperAudioFilterConfig] = None):
        self._config = config or WhisperAudioFilterConfig()

    def filter_pcm16(
        self,
        audio_pcm16: np.ndarray,
        sample_rate_hz: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        audio = np.asarray(audio_pcm16, dtype=np.int16).reshape(-1)
        stats: Dict[str, float] = {
            "enabled": float(self._config.enabled),
            "sample_rate_hz": float(sample_rate_hz),
            "num_samples": float(audio.size),
        }
        if (
            not self._config.enabled
            or audio.size == 0
            or sample_rate_hz <= 0
        ):
            return audio.copy(), stats

        x = audio.astype(np.float32) / 32768.0
        stats["input_rms"] = self._rms(x)
        stats["input_peak"] = float(np.max(np.abs(x))) if x.size else 0.0

        x = x - float(np.mean(x))
        x = self._high_pass(x, sample_rate_hz, self._config.high_pass_hz)
        x = self._low_pass(x, sample_rate_hz, self._config.low_pass_hz)
        x, gate_stats = self._noise_gate(x, sample_rate_hz)
        x = self._normalize(x)
        x = np.clip(x, -0.98, 0.98)

        stats.update(gate_stats)
        stats["output_rms"] = self._rms(x)
        stats["output_peak"] = float(np.max(np.abs(x))) if x.size else 0.0
        return (x * 32767.0).astype(np.int16), stats

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float32))))

    @staticmethod
    def _high_pass(x: np.ndarray, sample_rate_hz: int, cutoff_hz: float) -> np.ndarray:
        if cutoff_hz <= 0 or x.size < 2:
            return x
        dt = 1.0 / float(sample_rate_hz)
        rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
        alpha = rc / (rc + dt)
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, x.size):
            y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
        return y

    @staticmethod
    def _low_pass(x: np.ndarray, sample_rate_hz: int, cutoff_hz: float) -> np.ndarray:
        nyquist = sample_rate_hz / 2.0
        if cutoff_hz <= 0 or cutoff_hz >= nyquist or x.size < 2:
            return x
        dt = 1.0 / float(sample_rate_hz)
        rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
        alpha = dt / (rc + dt)
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, x.size):
            y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
        return y

    def _noise_gate(self, x: np.ndarray, sample_rate_hz: int):
        frame_len = max(1, int(sample_rate_hz * self._config.frame_ms / 1000.0))
        if x.size < frame_len:
            return x, {"noise_gate_threshold": 0.0, "noise_gate_open_ratio": 1.0}

        pad = (-x.size) % frame_len
        padded = np.pad(x, (0, pad), mode="constant") if pad else x
        frames = padded.reshape(-1, frame_len)
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
        noise_floor = float(np.percentile(frame_rms, self._config.noise_percentile))
        threshold = noise_floor * self._config.noise_gate_multiplier

        if threshold <= 1e-6:
            return x, {
                "noise_floor": noise_floor,
                "noise_gate_threshold": threshold,
                "noise_gate_open_ratio": 1.0,
            }

        # Soft gate: attenuate quiet frames without hard-muting breaths or
        # word endings. Smooth over a few frames to avoid chattering.
        relative = (frame_rms - threshold) / max(threshold, 1e-6)
        open_amount = np.clip(relative + 1.0, 0.0, 1.0)
        if open_amount.size >= 3:
            kernel = np.ones(3, dtype=np.float32) / 3.0
            open_amount = np.convolve(open_amount, kernel, mode="same")
        gains = (
            self._config.noise_gate_min_gain
            + (1.0 - self._config.noise_gate_min_gain) * open_amount
        )
        sample_gains = np.repeat(gains, frame_len)[:x.size].astype(np.float32)
        return x * sample_gains, {
            "noise_floor": noise_floor,
            "noise_gate_threshold": threshold,
            "noise_gate_open_ratio": float(np.mean(open_amount > 0.5)),
        }

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        rms = self._rms(x)
        if rms <= 1e-6:
            return x
        gain = min(self._config.target_rms / rms, self._config.max_normalize_gain)
        return x * gain
