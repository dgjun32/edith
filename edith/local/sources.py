"""Local RealSense capture and robot-independent state acquisition."""

import threading
import time
import numpy as np
from .sync import Sample


class RealSenseSource:
    def __init__(self, name, config, sink):
        import pyrealsense2 as rs

        self.rs, self.name, self.sink = rs, name, sink
        self.stop_event = threading.Event()
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(config.serial)
        cfg.enable_stream(
            rs.stream.color, config.width, config.height, rs.format.rgb8, config.fps
        )
        # Set before start so the first usable frames have a common host clock.
        device = cfg.resolve(rs.pipeline_wrapper(self.pipeline)).get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
        try:
            self.pipeline.start(cfg)
        except RuntimeError as error:
            raise RuntimeError(
                f"{name} RealSense {config.serial} could not start; check USB and "
                "whether another process owns the camera (fuser /dev/video*)"
            ) from error
        self.thread = threading.Thread(
            target=self._run, daemon=True, name=f"capture-{name}"
        )
        self.thread.start()

    def _run(self):
        rs = self.rs
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                frame = frames.get_color_frame()
                if not frame:
                    continue
                received = time.perf_counter_ns()
                if (
                    frame.get_frame_timestamp_domain()
                    != rs.timestamp_domain.global_time
                ):
                    continue  # Warm up until hardware->host mapping is available.
                source = round(frame.get_timestamp() * 1e6)
                # RealSense global_time is host wall time; convert to monotonic.
                timestamp = source + received - time.time_ns()
                self.sink.add(
                    self.name,
                    Sample(
                        timestamp,
                        received,
                        source,
                        frame.get_frame_number(),
                        np.array(frame.get_data(), copy=True),
                        "realsense_global_capture",
                    ),
                )
        except Exception as error:
            if not self.stop_event.is_set():
                self.sink.fail(error)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.pipeline.stop()


# Compatibility for explicitly selected legacy ABC collectors.
def __getattr__(name):
    if name == "ABCSubscriber":
        from edith.robot_utils.abc_streams import ABCSubscriber

        return ABCSubscriber
    raise AttributeError(name)


class RobotStateSource:
    """Publish a repository adapter's snapshots into the shared host-clock buffers."""

    def __init__(self, robot, sink, poll_hz=30):
        self.robot, self.sink = robot, sink
        self.stop_event = threading.Event()
        self.period = 1 / poll_hz
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="robot-observations"
        )
        self.thread.start()

    def _run(self):
        sequence = 0
        try:
            while not self.stop_event.is_set():
                snapshot = self.robot.read(timeout=1)
                received = time.perf_counter_ns()
                sequence += 1
                if snapshot.flat(self.robot.arm_order).shape != (
                    self.robot.action_size,
                ):
                    raise ValueError(
                        "Robot snapshot dimension does not match its action contract"
                    )
                for group in self.robot.arm_order:
                    values = np.asarray(snapshot.positions[group], dtype=float)
                    if values.ndim != 1 or not np.isfinite(values).all():
                        raise ValueError(f"Invalid robot state for {group}")
                    self.sink.add(
                        f"follower_{group}_state",
                        Sample(
                            snapshot.timestamp_ns,
                            received,
                            snapshot.timestamp_ns,
                            sequence,
                            values.copy(),
                            "robot_adapter_host_capture",
                        ),
                    )
                self.stop_event.wait(self.period)
        except Exception as error:
            if not self.stop_event.is_set():
                self.sink.fail(error)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Robot read did not honor its timeout during shutdown")
