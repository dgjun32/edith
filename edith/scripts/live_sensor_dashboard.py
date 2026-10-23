"""Four-view live preview for Aria Gen 2 and the configured robot cameras.

This is an inspection UI, not a recorder: frames are displayed as they arrive
and are not passed through the bounded synchronization/consumption policy used
for datasets. It never opens robot CAN or sends actions.
"""

import argparse
from pathlib import Path
import threading
import time

import numpy as np

from edith.aria_utils import AriaGen2Camera, draw_perception_bgr
from edith.local.sources import RealSenseSource
from edith.robot_utils.abc_config import DEFAULT_ABC_ROOT, load_station


class LatestSamples:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = {}
        self.error = ""

    def add(self, name, sample):
        with self.lock:
            self.samples[name] = sample

    def fail(self, error):
        with self.lock:
            self.error = f"{type(error).__name__}: {error}"

    def snapshot(self):
        with self.lock:
            return dict(self.samples), self.error


class RetryingAria:
    def __init__(self, sink, serial, ip, profile, retry_seconds):
        self.sink = sink
        self.serial = serial
        self.ip = ip
        self.profile = profile
        self.retry_seconds = retry_seconds
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.status = "discovering USB-NCM device..."
        self.camera = None
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="aria-hotplug-preview"
        )
        self.thread.start()

    def _set_status(self, value):
        with self.lock:
            self.status = value

    def get_status(self):
        with self.lock:
            return self.status

    def project(self, eye_gaze, hand_pose):
        with self.lock:
            camera = self.camera
        if camera is None:
            return {"gaze": None, "left": [None] * 21, "right": [None] * 21}
        return camera.project_perception_to_rgb(eye_gaze, hand_pose)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self._set_status("connecting...")
                self.camera = AriaGen2Camera(
                    self.sink,
                    serial=self.serial,
                    profile=self.profile,
                    ip=self.ip,
                )
                self._set_status("connected; waiting for RGB frames...")
                self.stop_event.wait()
            except Exception as error:
                self._set_status(f"disconnected: {type(error).__name__}: {error}")
                self.stop_event.wait(self.retry_seconds)
            finally:
                if self.camera is not None:
                    try:
                        self.camera.close()
                    except Exception as error:
                        self._set_status(f"cleanup error: {error}")
                    self.camera = None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


def _fit_rgb(rgb, width=640, height=400):
    import cv2

    canvas = np.zeros((height, width, 3), np.uint8)
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size)
    y = (height - bgr.shape[0]) // 2
    x = (width - bgr.shape[1]) // 2
    canvas[y : y + bgr.shape[0], x : x + bgr.shape[1]] = bgr
    return canvas


def _text_lines(image, text, start_y, color, max_chars=72):
    import cv2

    words = str(text).split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > max_chars and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, line in enumerate(lines[:4]):
        cv2.putText(
            image,
            line,
            (12, start_y + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def _tile(name, sample, status="", projection=None):
    import cv2

    if sample is None:
        tile = np.zeros((450, 640, 3), np.uint8)
        _text_lines(tile, status or "waiting for frames...", 210, (0, 180, 255))
        age = None
    else:
        tile = np.zeros((450, 640, 3), np.uint8)
        image = cv2.cvtColor(sample.data, cv2.COLOR_RGB2BGR)
        if projection is not None:
            image = draw_perception_bgr(image, projection)
        tile[:400] = _fit_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        age = (time.perf_counter_ns() - sample.timestamp_ns) / 1e6
    cv2.rectangle(tile, (0, 400), (639, 449), (30, 30, 30), -1)
    cv2.putText(
        tile,
        name,
        (12, 432),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if age is not None:
        cv2.putText(
            tile,
            f"age {age:.1f} ms | seq {sample.sequence}",
            (300, 432),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 255, 100) if age < 500 else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return tile


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    p.add_argument("--profile", default="abc_station_config")
    p.add_argument("--aria-serial", default="")
    p.add_argument(
        "--aria-ip",
        default="",
        help="Glasses USB-NCM IP to bypass USB discovery (not an external camera server)",
    )
    p.add_argument("--aria-profile", default="mp_streaming_demo")
    p.add_argument("--aria-retry-seconds", type=float, default=2)
    p.add_argument("--duration", type=float, default=0, help="0 runs until q/Esc")
    return p


def main(argv=None):
    import cv2

    args = parser().parse_args(argv)
    if args.aria_retry_seconds <= 0 or args.duration < 0:
        raise ValueError("Retry interval must be positive and duration nonnegative")
    config = load_station(args.abc_root, args.profile)
    sink = LatestSamples()
    cameras = []
    aria = None
    try:
        for name, camera_config in config.cameras.items():
            cameras.append(RealSenseSource(name, camera_config, sink))
        aria = RetryingAria(
            sink,
            args.aria_serial,
            args.aria_ip,
            args.aria_profile,
            args.aria_retry_seconds,
        )
        cv2.namedWindow("Aria Gen 2 + YAM cameras", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Aria Gen 2 + YAM cameras", 1280, 900)
        started = time.perf_counter()
        order = ("aria", *config.cameras)
        while not args.duration or time.perf_counter() - started < args.duration:
            samples, source_error = sink.snapshot()
            tiles = []
            projection = None
            if "eye_gaze" in samples and "hand_pose" in samples:
                projection = aria.project(
                    samples["eye_gaze"].data,
                    samples["hand_pose"].data,
                )
            for name in order:
                status = aria.get_status() if name == "aria" else source_error
                tiles.append(
                    _tile(
                        name,
                        samples.get(name),
                        status,
                        projection if name == "aria" else None,
                    )
                )
            dashboard = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4])))
            cv2.putText(
                dashboard,
                "LIVE PREVIEW - NO MOTOR COMMANDS - q/Esc to close",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Aria Gen 2 + YAM cameras", dashboard)
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if aria is not None:
            aria.close()
        for camera in reversed(cameras):
            camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
