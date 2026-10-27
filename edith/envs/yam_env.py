"""ABC/YAM teleop sensors: LocalSensorEnv plus ABC leader/follower IPC streams."""

from edith.envs.local_env import LocalSensorEnv


class YAMLocalEnv(LocalSensorEnv):
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
