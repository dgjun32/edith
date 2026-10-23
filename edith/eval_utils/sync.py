"""Bounded, thread-safe synchronization in the host perf_counter clock."""

from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class Sample:
    timestamp_ns: int
    received_ns: int
    source_timestamp_ns: int
    sequence: int
    data: object
    clock: str
    fields: tuple[str, ...] = ()


class Synchronizer:
    def __init__(
        self,
        names,
        tolerance_ms=40,
        max_age_ms=500,
        capacity=32,
        anchor=None,
        anchor_stride=1,
    ):
        if not names or len(set(names)) != len(names):
            raise ValueError("Stream names must be nonempty and unique")
        anchor = anchor or names[0]
        if anchor not in names:
            raise ValueError(f"Anchor {anchor!r} is not a configured stream")
        if (
            not 0 < tolerance_ms <= max_age_ms
            or capacity < 2
            or anchor_stride < 1
        ):
            raise ValueError("Require 0 < tolerance <= max age and capacity >= 2")
        self.buffers = {name: deque(maxlen=capacity) for name in names}
        self.anchor = anchor
        self.anchor_stride = int(anchor_stride)
        self.tolerance_ns = int(tolerance_ms * 1e6)
        self.max_age_ns = int(max_age_ms * 1e6)
        self.condition = threading.Condition()
        self.error = None
        self.last_anchor = -1
        self.last_anchor_sequence = None
        self.dropped = {name: 0 for name in names}

    def fail(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def add(self, name, sample):
        with self.condition:
            buf = self.buffers[name]
            if buf and sample.timestamp_ns <= buf[-1].timestamp_ns:
                self.dropped[name] += 1
                return
            if len(buf) == buf.maxlen:
                self.dropped[name] += 1
            buf.append(sample)
            self.condition.notify_all()

    def _match(self, now):
        for buf in self.buffers.values():
            while buf and now - buf[0].timestamp_ns > self.max_age_ns:
                buf.popleft()
        if not all(self.buffers.values()):
            return None
        # Preserve the master-camera timeline. Starting from the oldest pending
        # anchor prevents a slower non-anchor stream from making us jump to the
        # newest frame and silently collapse the requested output rate.
        for anchor in self.buffers[self.anchor]:
            if anchor.timestamp_ns <= self.last_anchor:
                continue
            if (
                self.last_anchor_sequence is not None
                and anchor.sequence
                < self.last_anchor_sequence + self.anchor_stride
            ):
                continue
            selected = {self.anchor: anchor}
            for name, buf in self.buffers.items():
                if name != self.anchor:
                    selected[name] = min(
                        buf,
                        key=lambda sample: abs(
                            sample.timestamp_ns - anchor.timestamp_ns
                        ),
                    )
            times = [sample.timestamp_ns for sample in selected.values()]
            if max(times) - min(times) <= self.tolerance_ns and max(times) <= now:
                self.last_anchor = anchor.timestamp_ns
                self.last_anchor_sequence = anchor.sequence
                for name, sample in selected.items():
                    buf = self.buffers[name]
                    # The master frame is consumed once. Other streams retain
                    # the selected sample so slower sensors can be resampled
                    # onto multiple master-camera ticks without falsifying its
                    # original capture timestamp.
                    while buf and (
                        buf[0].timestamp_ns <= sample.timestamp_ns
                        if name == self.anchor
                        else buf[0].timestamp_ns < sample.timestamp_ns
                    ):
                        buf.popleft()
                return selected
        return None

    def get(self, timeout=5):
        deadline = time.perf_counter() + timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError("Local sensor stream failed") from self.error
                result = self._match(time.perf_counter_ns())
                if result is not None:
                    return result
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    ages = {
                        name: None
                        if not buf
                        else round(
                            (time.perf_counter_ns() - buf[-1].timestamp_ns) / 1e6,
                            1,
                        )
                        for name, buf in self.buffers.items()
                    }
                    raise TimeoutError(f"No synchronized bundle; latest ages ms={ages}")
                self.condition.wait(min(remaining, 0.1))
