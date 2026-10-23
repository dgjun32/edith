"""Local RealSense capture and ABC IPC state/action subscribers."""

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


class ABCSubscriber:
    def __init__(self, names, sink):
        self.names, self.sink = names, sink
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="abc-local-state"
        )
        self.thread.start()

    def _run(self):
        import zmq
        from deploy.robot import communication as comms

        context = zmq.Context()
        sockets = {}
        try:
            poller = zmq.Poller()
            for name in self.names:
                sock = comms.create_subscriber(context, name)
                sock.setsockopt(zmq.RCVHWM, 32)
                sockets[sock] = name
                poller.register(sock, zmq.POLLIN)
            sequence = dict.fromkeys(self.names, 0)
            while not self.stop_event.is_set():
                for sock in dict(poller.poll(100)):
                    data, metadata = comms.subscribe(sock)
                    timestamp = round(float(metadata["timestamp"]) * 1e9)
                    name = sockets[sock]
                    if data.shape != (7,) or not np.all(np.isfinite(data)):
                        raise ValueError(
                            f"Invalid YAM joint vector on {name}: {data.shape}"
                        )
                    sequence[name] += 1
                    self.sink.add(
                        name,
                        Sample(
                            timestamp,
                            time.perf_counter_ns(),
                            timestamp,
                            sequence[name],
                            data,
                            "abc_host_publish",
                        ),
                    )
        except Exception as error:
            self.sink.fail(error)
        finally:
            for sock in sockets:
                sock.close(linger=0)
            context.term()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
