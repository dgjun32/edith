"""Hardware-free tests for policy/robot/Aria/eval separation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from edith.eval_utils.environment import RobotEvaluationEnv
from edith.eval_utils.observation import build_observation
from edith.local.sync import Sample
from edith.policy_utils import load_policy
from edith.robot_utils.abc_yam import ConfiguredNoOpRobot


SOURCE_ROOT = Path(__file__).resolve().parents[1]
STATIC_POLICY = SOURCE_ROOT / "policies/static_hold"


class PolicyDirectoryTests(unittest.TestCase):
    def test_static_policy_is_complete_single_directory(self):
        policy = load_policy(STATIC_POLICY)
        observation = {"robot_state": np.arange(14, dtype=float)}
        policy.reset(observation)
        observation["robot_state"][:] = -1
        np.testing.assert_array_equal(policy.act(observation), np.arange(14))
        self.assertEqual(policy.control_hz, 30)

    def test_entrypoint_cannot_escape_policy_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.json").write_text(
                '{"schema":"edith.policy.v1",'
                '"entrypoint":"../outside.py:Policy","action_space":"joint_position"}'
            )
            with self.assertRaisesRegex(ValueError, "inside the policy directory"):
                load_policy(root)


class AdapterContractTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            robots={
                "left": SimpleNamespace(init_q=[0.0] * 7),
                "right": SimpleNamespace(init_q=[1.0] * 7),
            }
        )

    def test_noop_adapter_validates_without_changing_state(self):
        robot = ConfiguredNoOpRobot(self.config())
        before = robot.read(0).flat(robot.arm_order)
        robot.command(np.arange(14, dtype=float))
        after = robot.read(0).flat(robot.arm_order)
        np.testing.assert_array_equal(before, after)
        with self.assertRaisesRegex(ValueError, "Expected action shape"):
            robot.command(np.zeros(13))

    def test_observation_contract_uses_synced_state(self):
        fallback = SimpleNamespace(
            positions={"left": np.zeros(7), "right": np.zeros(7)}
        )
        image = np.zeros((2, 3, 3), np.uint8)
        bundle = {
            "aria": Sample(100, 100, 100, 1, image, "test"),
            "eye_gaze": Sample(100, 100, 100, 1, np.arange(3), "test", ("a", "b", "c")),
            "hand_pose": Sample(
                100, 100, 100, 1, np.arange(4), "test", ("a", "b", "c", "d")
            ),
            "follower_left_state": Sample(101, 101, 101, 1, np.ones(7), "test"),
            "follower_right_state": Sample(102, 102, 102, 1, np.full(7, 2), "test"),
        }
        obs = build_observation(
            bundle,
            arm_order=("left", "right"),
            fallback_state=fallback,
            instruction="test",
        )
        np.testing.assert_array_equal(obs["robot_state"][:7], np.ones(7))
        np.testing.assert_array_equal(obs["robot_state"][7:], np.full(7, 2))
        self.assertEqual(set(obs["images"]), {"aria_rgb", "aria_cropped_rgb"})
        self.assertEqual(set(obs["aria_perception"]), {"eye_gaze", "hand_pose"})
        self.assertEqual(obs["aria_perception"]["eye_gaze"]["fields"], ("a", "b", "c"))

    def test_evaluation_env_inserts_aria_gaze_hands_and_robot_views(self):
        robot = ConfiguredNoOpRobot(self.config())
        fallback = robot.read(0)
        image = np.zeros((2, 3, 3), np.uint8)
        bundle = {
            name: Sample(100, 101, 99, index, image, "test")
            for index, name in enumerate(("aria", "top", "left", "right"), 1)
        }
        bundle["eye_gaze"] = Sample(
            100, 101, 99, 5, np.arange(3), "test", ("a", "b", "c")
        )
        bundle["hand_pose"] = Sample(
            100, 101, 99, 6, np.arange(4), "test", ("a", "b", "c", "d")
        )
        pcm = np.ones((800, 2), np.int32)
        bundle["audio"] = Sample(100, 101, 99, 800, pcm, "test")

        class Aria:
            audio = SimpleNamespace(sample_rate_hz=48_000)

            @staticmethod
            def project_perception_to_rgb(eye_gaze, hand_pose):
                del eye_gaze, hand_pose
                return {
                    "gaze": (1.0, 2.0),
                    "left": [(3.0, 4.0)] * 21,
                    "right": [None] * 21,
                }

        class Sensors:
            aria = Aria()

            @staticmethod
            def get_bundle(timeout):
                self.assertEqual(timeout, 2.0)
                return bundle

            @staticmethod
            def safe_end():
                pass

        env = RobotEvaluationEnv(Sensors(), robot, fallback, instruction="test")
        wrapped = env.get_observation(timeout=2.0)
        observation = wrapped["timestamp"]
        self.assertEqual(
            set(observation["images"]),
            {
                "aria_rgb",
                "aria_cropped_rgb",
                "top_camera_rgb",
                "left_wrist_camera_rgb",
                "right_wrist_camera_rgb",
            },
        )
        np.testing.assert_array_equal(observation["gaze_point"], [1.0, 2.0])
        self.assertEqual(observation["hand_keypoints"]["left"].shape, (21, 2))
        self.assertTrue(np.isnan(observation["hand_keypoints"]["right"]).all())
        np.testing.assert_array_equal(observation["audio"]["samples"], pcm)
        self.assertEqual(observation["audio"]["sample_rate_hz"], 48_000)
        self.assertEqual(observation["audio"]["end_timestamp_ns"], 100)
        self.assertEqual(
            set(observation["robot_state"]),
            {
                "left_arm_positions",
                "left_gripper_position",
                "right_arm_positions",
                "right_gripper_position",
            },
        )
        result = env.step(
            {
                "left_arm_positions": np.zeros(6),
                "left_gripper_position": np.zeros(1),
                "right_arm_positions": np.ones(6),
                "right_gripper_position": np.ones(1),
            }
        )
        self.assertEqual(result["commanded_joint_position"].shape, (14,))


class StartupSafetyTests(unittest.TestCase):
    def test_sensor_failure_prevents_robot_construction(self):
        from edith.scripts.eval_policy_local import main

        with patch(
            "edith.envs.yam_env.YAMLocalEnv",
            side_effect=RuntimeError("aria unavailable"),
        ):
            with patch(
                "edith.scripts.eval_policy_local.create_robot"
            ) as create_robot:
                with self.assertRaisesRegex(RuntimeError, "aria unavailable"):
                    main(
                        [
                            "--policy-dir",
                            str(STATIC_POLICY),
                            "--execute-actions",
                            "--headless",
                        ]
                    )
                create_robot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
