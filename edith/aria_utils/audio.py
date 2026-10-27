"""Continuous Gen 2 PCM intervals; preserve the main-branch samples contract."""

from collections import deque
import threading
import numpy as np
from edith.eval_utils.sync import Sample

AUDIO_SAMPLE_RATE_HZ = 48_000  # Provisional until actual capture timestamps arrive.


class AudioBuffer:
    def __init__(self, max_seconds=5.0):
        self.lock = threading.Lock()
        self.chunks = deque()
        self.max_seconds = max_seconds
        self.sample_rate_hz = AUDIO_SAMPLE_RATE_HZ
        self.channels = 0
        self.buffered = self.delivered = self.dropped = 0
        self.last_source_ns = -1
        self.max_amplitude = float(np.iinfo(np.int32).max)

    def add(
        self, host_ns, source_ns, received_ns, samples, muted=False, max_amplitude=None
    ):
        source_ns = np.asarray(source_ns, dtype=np.int64)
        host_ns = np.asarray(host_ns, dtype=np.int64)
        samples = np.asarray(samples, dtype=np.int32)
        if (
            samples.ndim != 2
            or not samples.shape[1]
            or source_ns.shape != (len(samples),)
            or host_ns.shape != source_ns.shape
        ):
            raise ValueError("Audio frame/timestamp dimensions do not match")
        if len(source_ns) > 1 and (
            np.any(np.diff(source_ns) <= 0) or np.any(np.diff(host_ns) <= 0)
        ):
            raise ValueError("Audio timestamps must increase")
        if not len(samples):
            return
        with self.lock:
            if self.channels and samples.shape[1] != self.channels:
                raise ValueError("Aria audio channel count changed")
            self.channels = samples.shape[1]
            if len(source_ns) > 1:
                rate = int(round(1e9 / float(np.median(np.diff(source_ns))), -2))
                if self.delivered and rate != self.sample_rate_hz:
                    raise ValueError("Aria audio sample rate changed")
                self.sample_rate_hz = rate
            if max_amplitude is not None:
                if not np.isfinite(max_amplitude) or max_amplitude <= 0:
                    raise ValueError("Invalid audio amplitude scale")
                self.max_amplitude = float(max_amplitude)
            keep = source_ns > self.last_source_ns
            if not np.any(keep):
                return
            self.last_source_ns = int(source_ns[-1])
            samples, host_ns, source_ns = samples[keep], host_ns[keep], source_ns[keep]
            flags = np.full(len(samples), bool(muted), np.uint8)
            self.chunks.append((host_ns, source_ns, int(received_ns), samples, flags))
            self.buffered += len(samples)
            while (
                self.buffered > round(self.max_seconds * self.sample_rate_hz)
                and len(self.chunks) > 1
            ):
                count = len(self.chunks.popleft()[3])
                self.buffered -= count
                self.dropped += count

    def take(self, until_ns):
        parts = []
        hosts = []
        sources = []
        flags = []
        received = int(until_ns)
        with self.lock:
            while self.chunks:
                host, source, chunk_received, samples, muted = self.chunks[0]
                count = int(np.searchsorted(host, until_ns, side="right"))
                if count == 0:
                    break
                received = chunk_received
                parts.append(samples[:count])
                hosts.append(host[:count])
                sources.append(source[:count])
                flags.append(muted[:count])
                self.chunks.popleft()
                if count < len(samples):
                    self.chunks.appendleft(
                        (
                            host[count:],
                            source[count:],
                            received,
                            samples[count:],
                            muted[count:],
                        )
                    )
                    break
            taken = sum(len(p) for p in parts)
            self.buffered -= taken
            self.delivered += taken
            data = (
                np.concatenate(parts)
                if parts
                else np.empty((0, self.channels or 1), np.int32)
            )
            host = np.concatenate(hosts) if hosts else np.empty(0, np.int64)
            source = np.concatenate(sources) if sources else np.empty(0, np.int64)
            muted = np.concatenate(flags) if flags else np.empty(0, np.uint8)
            metadata = {
                "capture_timestamps_ns": host,
                "source_timestamps_ns": source,
                "muted": muted,
                "sample_rate_hz": self.sample_rate_hz,
                "num_channels": self.channels,
                "max_amplitude": self.max_amplitude,
                "dropped_samples": self.dropped,
            }
            return Sample(
                int(host[-1]) if len(host) else int(until_ns),
                received,
                int(source[-1]) if len(source) else 0,
                self.delivered + self.dropped,
                data,
                "aria_ticsync_capture",
                metadata=metadata,
            )
