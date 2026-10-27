"""Core capture, external robot adapters, audio recording and no-motor diagnostics."""

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import h5py
from edith.config import StationConfig, JointGroupConfig
from edith.robot_utils.factory import create_robot
from edith.policy_utils import load_policy
from edith.aria_utils.audio import AudioBuffer
from edith.eval_utils.recording import BundleWriter

ROOT = Path(__file__).resolve().parents[1]


class IndependentRobotTests(unittest.TestCase):
    def test_default_cli_and_config_do_not_import_robot_dependencies(self):
        code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('i2rt','deploy','piper_sdk'):
            raise AssertionError('Robot dependency imported: '+name)
sys.meta_path.insert(0, Guard())
from edith.config import load_station
from edith.envs.local_env import LocalSensorEnv
from edith.scripts.live_sensor_dashboard import parser
from edith.scripts.eval_policy_local import main
config=load_station('configs/station.example.json')
assert config.adapter=='none'
main(['--policy-dir','policies/zero_action_policy','--check-policy-only'])
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["policy"]["name"], "zero_action_policy"
        )

    def test_external_robot_plugin_with_different_joint_count(self):
        config = StationConfig(
            {},
            {"arm": JointGroupConfig((0, 0, 0))},
            "examples.robot_adapter:create_robot",
            {},
        )
        robot = create_robot(config.adapter, config=config, execute=True)
        try:
            robot.start()
            robot.command([1, 2, 3])
            np.testing.assert_equal(robot.read(0.1).flat(robot.arm_order), [1, 2, 3])
        finally:
            robot.close()

    def test_no_execution_never_imports_external_robot_factory(self):
        config = StationConfig(
            {}, {"arm": JointGroupConfig((1, 2, 3))}, "missing_driver:create_robot", {}
        )
        with patch(
            "edith.robot_utils.factory.importlib.import_module",
            side_effect=AssertionError("Must not import driver"),
        ):
            robot = create_robot(config.adapter, config=config, execute=False)
            robot.start()
            robot.command([0, 0, 0])
            np.testing.assert_equal(robot.read(0.1).flat(robot.arm_order), [1, 2, 3])
            robot.close()

    def test_zero_action_policy_is_generic_and_cannot_execute_motors(self):
        from edith.scripts.eval_policy_local import main

        policy = load_policy(ROOT / "policies/zero_action_policy")
        np.testing.assert_equal(policy.act({"robot_state": np.ones(3)}), np.zeros(3))
        with patch(
            "edith.scripts.eval_policy_local.load_station",
            side_effect=AssertionError("devices"),
        ):
            with self.assertRaisesRegex(ValueError, "observe_only"):
                main(
                    [
                        "--policy-dir",
                        str(ROOT / "policies/zero_action_policy"),
                        "--execute-actions",
                    ]
                )


class AudioRecordingTests(unittest.TestCase):
    def test_runner_records_first_and_subsequent_policy_audio_inputs(self):
        from types import SimpleNamespace
        from edith.eval_utils.sync import Sample
        from edith.scripts.eval_policy_local import parser, _run

        inputs = []

        class Sensors:
            index = 0

            def __init__(self, *args):
                pass

            def get_bundle(self, timeout):
                self.index += 1
                audio = AudioBuffer()
                stamps = np.arange(8, dtype=np.int64) * 62500 + self.index * 1_000_000
                audio.add(
                    stamps,
                    stamps,
                    int(stamps[-1]),
                    np.full((8, 2), self.index, np.int32),
                )
                image = np.zeros((2, 2, 3), np.uint8)
                return {
                    "aria": Sample(
                        int(stamps[-1]),
                        int(stamps[-1]),
                        int(stamps[-1]),
                        self.index,
                        image,
                        "test",
                    ),
                    "audio": audio.take(int(stamps[-1])),
                }

            def safe_end(self):
                pass

        def act(observation):
            inputs.append(observation["audio"]["samples"].copy())
            return np.zeros(3)

        policy = SimpleNamespace(
            control_hz=15,
            root=ROOT / "policies/zero_action_policy",
            manifest={"name": "zero_action_policy", "observe_only": True},
            reset=lambda observation: None,
            act=act,
        )
        config = StationConfig({}, {"arm": JointGroupConfig((1, 2, 3))})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.h5"
            args = parser().parse_args(
                ["--duration", "0.001", "--headless", "--output", str(path)]
            )
            with (
                patch("edith.envs.local_env.LocalSensorEnv", Sensors),
                patch(
                    "edith.scripts.eval_policy_local.load_station", return_value=config
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(_run(args, policy), 0)
            self.assertGreaterEqual(len(inputs), 2)
            with h5py.File(path) as file:
                self.assertEqual(file.attrs["complete_rows"], len(inputs))
                for row, expected in enumerate(inputs):
                    np.testing.assert_equal(
                        file["audio/pcm"][row].reshape(-1, 2), expected
                    )
                np.testing.assert_equal(
                    file["policy_observation/action"][:], np.zeros((len(inputs), 3))
                )

    def test_per_sample_clock_and_pcm_survive_hdf5_roundtrip(self):
        from edith.eval_utils.observation import build_observation
        from types import SimpleNamespace

        buffer = AudioBuffer()
        src = np.arange(8, dtype=np.int64) * 62500 + 1_000_000
        pcm = np.arange(24, dtype=np.int32).reshape(8, 3)
        buffer.add(src + 10000, src, 2_000_000, pcm, muted=True)
        parts = [buffer.take(int(src[3] + 10000)), buffer.take(int(src[-1] + 10000))]
        self.assertEqual(buffer.take(9999999).data.size, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.h5"
            writer = BundleWriter(path, {"execute_actions": False})
            for part in parts:
                bundle = {"audio": part}
                obs = build_observation(
                    bundle,
                    arm_order=("arm",),
                    fallback_state=SimpleNamespace(positions={"arm": np.zeros(3)}),
                )
                self.assertEqual(obs["audio"]["sample_rate_hz"], 16000)
                writer.append(bundle, obs, np.zeros(3))
            writer.close()
            with h5py.File(path) as file:
                np.testing.assert_equal(
                    np.concatenate([v.reshape(-1, 3) for v in file["audio/pcm"][:]]),
                    pcm,
                )
                np.testing.assert_equal(
                    np.concatenate(file["audio/frame_source_timestamps_ns"][:]), src
                )
                np.testing.assert_equal(
                    np.concatenate(file["audio/frame_capture_timestamps_ns"][:]),
                    src + 10000,
                )
                self.assertTrue(np.concatenate(file["audio/frame_muted"][:]).all())
                self.assertTrue((file["policy_observation/action"][:] == 0).all())
                self.assertEqual(file["audio"].attrs["sample_rate_hz"], 16000)


if __name__ == "__main__":
    unittest.main()
