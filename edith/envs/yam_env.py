"""Edith observations from local Gen 2 + robot cameras and ABC YAM streams.

ABC owns CAN and leader calibration. This environment never constructs Piper
hardware or opens an external image socket. Actions come from ABC GELLO nodes.
"""

from contextlib import ExitStack
from edith.eval_utils.sync import Synchronizer
from edith.local.sources import ABCSubscriber, RealSenseSource
from edith.aria_utils import AriaGen2Camera


class YAMLocalEnv:
    def __init__(
        self,
        config,
        aria_serial="",
        aria_profile="mp_streaming_demo",
        tolerance_ms=100,
        max_age_ms=500,
        record_vrs="",
        aria_ip="",
        sync_anchor="top",
        output_fps=15,
    ):
        self.config = config
        self.resources = ExitStack()
        self.aria = None
        if sync_anchor not in config.cameras:
            raise ValueError(
                f"Sync anchor {sync_anchor!r} must be one of {tuple(config.cameras)}"
            )
        if output_fps <= 0:
            raise ValueError("Output FPS must be positive")
        source_fps = config.cameras[sync_anchor].fps
        anchor_stride = round(source_fps / output_fps)
        actual_fps = source_fps / anchor_stride if anchor_stride > 0 else 0
        if anchor_stride < 1 or abs(actual_fps - output_fps) > 1e-6:
            raise ValueError(
                f"Requested {output_fps} FPS cannot be produced exactly from "
                f"{sync_anchor} at {source_fps} FPS"
            )
        self.sync_anchor = sync_anchor
        self.output_fps = actual_fps
        self.sync = Synchronizer(
            ["aria", "eye_gaze", "hand_pose", *config.cameras],
            tolerance_ms,
            max_age_ms,
            anchor=sync_anchor,
            anchor_stride=anchor_stride,
        )
        self.robot_topics = []
        try:
            for name, camera in config.cameras.items():
                if camera.camera_type != "realsense":
                    raise ValueError(
                        "Local synchronized capture currently requires RealSense cameras"
                    )
                source = RealSenseSource(name, camera, self.sync)
                self.resources.callback(source.close)
            aria = AriaGen2Camera(
                self.sync,
                aria_serial,
                aria_profile,
                record_vrs,
                ip=aria_ip,
            )
            self.aria = aria
            self.resources.callback(aria.close)
        except BaseException:
            self.resources.close()
            raise

    def attach_robot_streams(self):
        from collections import deque

        if self.robot_topics:
            return
        self.robot_topics = [
            topic
            for side in self.config.robots
            for topic in (f"follower_{side}_state", f"leader_{side}_actions")
        ]
        with self.sync.condition:
            for name in self.robot_topics:
                self.sync.buffers[name] = deque(maxlen=64)
                self.sync.dropped[name] = 0
        subscriber = ABCSubscriber(self.robot_topics, self.sync)
        self.resources.callback(subscriber.close)

    def get_bundle(self, timeout=5):
        bundle = self.sync.get(timeout)
        if self.aria is not None:
            bundle["audio"] = self.aria.audio.take(
                bundle[self.sync_anchor].timestamp_ns
            )
        return bundle

    def get_observation(self, timeout=5):
        bundle = self.get_bundle(timeout)
        images = {
            f"{name}_rgb": sample.data
            for name, sample in bundle.items()
            if sample.data.ndim == 3
        }
        state = {}
        for side in self.config.robots:
            key = f"follower_{side}_state"
            if key in bundle:
                q = bundle[key].data
                state[f"{side}_arm_positions"] = q[:6].tolist()
                state[f"{side}_gripper_position"] = q[6:].tolist()
        return {
            "timestamp": {"images": images, "robot_state": state},
            "capture_timestamps_ns": {
                name: sample.timestamp_ns for name, sample in bundle.items()
            },
        }

    def safe_end(self):
        self.resources.close()
