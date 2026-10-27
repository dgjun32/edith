"""Optional ABC/YAM sensor compatibility adapter. Core capture uses LocalSensorEnv."""

from edith.envs.local_env import LocalSensorEnv


class YAMLocalEnv(LocalSensorEnv):
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

    def attach_robot_streams(self):
        from collections import deque
        from edith.robot_utils.abc_streams import ABCSubscriber

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
