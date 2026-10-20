"""Contract/regression tests for the polished source policies; no robot/cloud."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

from piper_teleop.policy_utils import load_policy
from piper_teleop.policy_utils.chunked import ChunkedPolicy
from piper_teleop.policy_utils.keyframes import KeyframePolicy
from piper_teleop.policy_utils.openpi import OpenPIClient, pack, unpack
from piper_teleop.policy_utils.processing import gaze_crop, packet
from piper_teleop.policy_utils.tasks import Frame, Task, TaskQueue

ROOT = Path(__file__).resolve().parents[1]


def observation(t=1_000_000_000, instruction="pick up the cup"):
    red = np.zeros((16, 24, 3), np.uint8)
    red[:, :, 0] = 255
    return {
        "robot_state": np.zeros(14),
        "arm_order": ("left", "right"),
        "images": {
            k: red.copy()
            for k in (
                "top_camera_rgb",
                "left_wrist_camera_rgb",
                "right_wrist_camera_rgb",
                "aria_rgb",
            )
        },
        "gaze_point": np.array([12, 8]),
        "instruction": instruction,
        "capture_timestamps_ns": {"aria": t, "top": t},
        "sequence_numbers": {"aria": t, "top": t},
    }


class FakeClient:
    def __init__(self, result=None, gate=None):
        self.result = result or {"actions": np.full((2, 14), 0.01)}
        self.requests = []
        self.closed = False
        self.gate = gate

    def infer(self, request):
        self.requests.append(request)
        if self.gate:
            self.gate.wait(2)
        return self.result

    def close(self):
        self.closed = True
        if self.gate:
            self.gate.set()


def next_command(policy, obs):
    for _ in range(100):
        result = policy.act(obs)
        if policy.status["phase"] == "executing":
            return result
        time.sleep(0.001)
    raise AssertionError("No action produced")


class DirectoryTests(unittest.TestCase):
    def test_all_policy_directories_load_without_network_or_hardware(self):
        with patch(
            "piper_teleop.policy_utils.openpi.OpenPIClient._connect",
            side_effect=AssertionError("network"),
        ):
            for path in sorted((ROOT / "policies").glob("*/policy.json")):
                with self.subTest(path=path):
                    policy = load_policy(path.parent)
                    self.assertEqual(policy.manifest["action_space"], "joint_position")
                    policy.close()

    def test_config_overrides_do_not_change_manifest_file(self):
        path = ROOT / "policies/textvla"
        before = (path / "policy.json").read_bytes()
        policy = load_policy(path, {"instruction": "test"})
        self.assertEqual(policy.manifest["config"]["instruction"], "test")
        self.assertEqual(before, (path / "policy.json").read_bytes())
        policy.close()

    def test_loader_rejects_action_chunks(self):
        from piper_teleop.policy_utils.loader import LoadedPolicy
        from types import SimpleNamespace

        policy = LoadedPolicy(
            ROOT, {}, SimpleNamespace(act=lambda obs: np.zeros((2, 14)))
        )
        with self.assertRaisesRegex(ValueError, "one finite 1-D"):
            policy.act({})

    def test_isolated_policy_relative_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            policies = []
            for i in (1, 2):
                root = Path(tmp) / str(i)
                root.mkdir()
                (root / "helper.py").write_text(f"VALUE = {i}\n")
                (root / "policy.py").write_text(
                    "from .helper import VALUE\nclass Policy:\n def __init__(self, **kw): pass\n def act(self, obs): return [VALUE]\n"
                )
                (root / "policy.json").write_text(
                    json.dumps(
                        {
                            "schema": "edith.policy.v1",
                            "entrypoint": "policy.py:Policy",
                            "action_space": "joint_position",
                        }
                    )
                )
                policies.append(load_policy(root))
            self.assertEqual(policies[0].act({})[0], 1)
            self.assertEqual(policies[1].act({})[0], 2)


class ImageTests(unittest.TestCase):
    def test_rgb_is_not_swapped_and_state_order_is_explicit(self):
        obs = observation()
        p = packet(obs, "test", 24)
        image = p["observation.images.top_view"]
        self.assertEqual(int(image[12, 12, 0]), 255)
        self.assertEqual(int(image[12, 12, 2]), 0)
        obs["arm_order"] = ("right", "left")
        with self.assertRaises(ValueError):
            packet(obs, "test")

    def test_crop_boundaries_variable_resolution_and_nan(self):
        image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
        self.assertEqual(gaze_crop(image, [19, 9], 8).shape, (8, 8, 3))
        np.testing.assert_array_equal(gaze_crop(image, [19, 9], 8), image[2:10, 12:20])
        np.testing.assert_array_equal(gaze_crop(image, [np.nan, np.nan]), image)
        with self.assertRaises(ValueError):
            gaze_crop(image, [np.nan, np.nan], invalid="error")
        self.assertEqual(gaze_crop(image, [19, 9], 512).shape, image.shape)

    def test_missing_image_and_malformed_state_fail(self):
        obs = observation()
        del obs["images"]["top_camera_rgb"]
        with self.assertRaises(KeyError):
            packet(obs, "test")
        obs = observation()
        obs["robot_state"] = np.zeros(13)
        with self.assertRaises(ValueError):
            packet(obs, "test")


class ChunkTests(unittest.TestCase):
    def test_chunk_one_action_per_call_and_one_inference(self):
        client = FakeClient(
            {"actions": np.stack([np.full(14, 0.01), np.full(14, 0.02)])}
        )
        policy = ChunkedPolicy(ROOT, {}, client=client)
        obs = observation()
        policy.reset(obs)
        np.testing.assert_array_equal(next_command(policy, obs), np.full(14, 0.01))
        np.testing.assert_array_equal(policy.act(obs), np.full(14, 0.02))
        self.assertEqual(len(client.requests), 1)
        policy.close()

    def test_slow_inference_is_nonblocking_and_holds_last_target(self):
        gate = threading.Event()
        policy = ChunkedPolicy(ROOT, {}, client=FakeClient(gate=gate))
        start = time.monotonic()
        np.testing.assert_array_equal(policy.act(observation()), np.zeros(14))
        self.assertLess(time.monotonic() - start, 0.2)
        policy.close()

    def test_prompt_change_discards_old_response(self):
        gate = threading.Event()
        client = FakeClient(gate=gate)
        policy = ChunkedPolicy(ROOT, {}, client=client)
        obs = observation()
        policy.act(obs)
        obs["instruction"] = "new task"
        policy.act(obs)
        gate.set()
        for _ in range(100):
            policy.act(obs)
            if len(client.requests) > 1:
                break
            time.sleep(0.001)
        self.assertEqual(client.requests[-1]["prompt"], "new task")
        self.assertEqual(policy.last_action.tolist(), np.zeros(14).tolist())
        policy.close()

    def test_invalid_actions_grippers_step_and_completion(self):
        cases = [
            {"actions": np.zeros((2, 13))},
            {"actions": np.full((2, 14), np.nan)},
            {"actions": np.full((2, 14), 2)},
            {"actions": np.full((2, 14), 0.3)},
            {"actions": np.zeros((2, 14)), "completion_prob": float("nan")},
        ]
        for result in cases:
            policy = ChunkedPolicy(ROOT, {}, client=FakeClient(result))
            with self.assertRaises(ValueError):
                next_command(policy, observation())
            policy.close()

    def test_stale_result_is_rejected(self):
        policy = ChunkedPolicy(ROOT, {"max_action_age_s": 0.005}, client=FakeClient())
        policy.act(observation())
        time.sleep(0.02)
        with self.assertRaises(TimeoutError):
            policy.act(observation())
        policy.close()

    def test_reset_invalidates_prior_inference(self):
        gate = threading.Event()
        client = FakeClient(gate=gate)
        policy = ChunkedPolicy(ROOT, {}, client=client)
        policy.act(observation())
        policy.reset(observation(instruction="new"))
        gate.set()
        obs = observation(instruction="new")
        for _ in range(100):
            policy.act(obs)
            if len(client.requests) > 1:
                break
            time.sleep(0.001)
        np.testing.assert_array_equal(policy.last_action, np.zeros(14))
        policy.close()


class QueueTests(unittest.TestCase):
    def frames(self):
        return [
            Frame(
                1_000_000_000 + i * 400_000_000,
                np.zeros((2, 2, 3), np.uint8),
                np.full((2, 2, 3), i, np.uint8),
            )
            for i in range(3)
        ]

    def item(self, start=0.7, end=0.8):
        return {
            "subtask_instruction": "pick up cup",
            "keyframe_interval": {"start": start, "end": end},
        }

    def test_capture_timestamp_selects_keyframe(self):
        q = TaskQueue()
        q.apply({"action_queue": [self.item()]}, self.frames())
        self.assertEqual(int(q.current.keyframe[0, 0, 0]), 2)

    def test_atomic_invalid_queue_response(self):
        q = TaskQueue([Task("existing", None)])
        with self.assertRaises(ValueError):
            q.apply(
                {"command": "Append", "new_elements": [self.item(), self.item(5, 6)]},
                self.frames(),
            )
        self.assertEqual(len(q.items), 1)

    def test_append_replace_preempt_remove_and_revision(self):
        q = TaskQueue([Task("existing", None)])
        q.apply({"command": "Append", "new_elements": [self.item()]}, self.frames())
        self.assertEqual(q.revision, 0)
        q.apply(
            {"command": "Replace", "target_index": 0, "new_elements": [self.item()]},
            self.frames(),
        )
        self.assertEqual(q.revision, 1)
        q.apply({"command": "Preempt", "new_elements": [self.item()]}, self.frames())
        self.assertEqual(q.revision, 2)
        q.apply({"command": "Remove", "target_indices": [2]}, self.frames())
        self.assertEqual(len(q.items), 2)
        with self.assertRaises(ValueError):
            q.apply({"command": "Remove", "target_indices": [0]}, self.frames())

    def test_queue_exhaustion_holds_without_robot_reset(self):
        client = FakeClient(
            {"actions": np.full((1, 14), 0.01), "completion_prob": 0.99}
        )
        policy = KeyframePolicy(ROOT, {"min_subtask_steps": 1}, client=client)
        obs = observation()
        next_command(policy, obs)
        self.assertIsNone(policy.task_queue.current)
        np.testing.assert_array_equal(policy.act(obs), np.full(14, 0.01))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(policy.status["phase"], "waiting_for_subtasks")
        policy.close()

    def test_no_keyframe_ablation_and_operator_event_idempotence(self):
        client = FakeClient()
        policy = KeyframePolicy(ROOT, {"use_keyframe": False}, client=client)
        obs = observation()
        next_command(policy, obs)
        self.assertNotIn(
            "observation.images.human_view_keyframe_cropped", client.requests[0]
        )
        obs["policy_event"] = {
            "id": "a",
            "command": "Append",
            "new_elements": [self.item(0, 0)],
        }
        policy.act(obs)
        policy.act(obs)
        self.assertEqual(len(policy.task_queue.items), 2)
        policy.close()

    def test_async_planner_preserves_frame_times_and_no_duplicate_frames(self):
        class Planner:
            def __init__(self):
                self.calls = []

            def plan(self, frames, queue, text, initial):
                self.calls.append((frames, text, initial))
                return {
                    "action_queue": [
                        {
                            "subtask_instruction": "cup",
                            "keyframe_interval": {"start": 0, "end": 0},
                        }
                    ]
                }

            def close(self):
                pass

        planner = Planner()
        policy = KeyframePolicy(
            ROOT,
            {"planner_interval_s": 0.5, "planner_fps": 4},
            client=FakeClient(),
            planner=planner,
        )
        obs = observation()
        policy.act(obs)
        policy.act(obs)
        self.assertEqual(len(policy.frames), 1)
        policy.act(observation(t=1_300_000_000))
        policy.act(observation(t=1_600_000_000))
        for _ in range(100):
            policy.act(observation(t=1_600_000_000))
            if policy.task_queue.current:
                break
            time.sleep(0.001)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(planner.calls[0][0]), 3)
        self.assertEqual(policy.task_queue.current.instruction, "cup")
        policy.close()


class WireTests(unittest.TestCase):
    def test_numpy_protocol_roundtrip_and_object_rejection(self):
        value = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        np.testing.assert_array_equal(unpack(pack({"image": value}))["image"], value)
        with self.assertRaises(TypeError):
            pack({"bad": np.array([object()])})

    def test_real_loopback_websocket_and_robot_metadata_guard(self):
        from websockets.sync.server import serve

        for robot in ("yam", "piper"):

            def handler(connection):
                connection.send(
                    pack(
                        {
                            "robot_type": robot,
                            "action_space": "joint_position",
                            "arm_order": ["left", "right"],
                        }
                    )
                )
                if robot == "yam":
                    request = unpack(connection.recv(timeout=2))
                    connection.send(
                        pack({"actions": np.asarray([request["observation.state"]])})
                    )

            with serve(handler, "127.0.0.1", 0) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                client = OpenPIClient(
                    {
                        "server_url": f"ws://127.0.0.1:{server.socket.getsockname()[1]}",
                        "timeout_s": 1,
                    }
                )
                try:
                    if robot == "yam":
                        result = client.infer(packet(observation(), "test"))
                        self.assertEqual(result["actions"].shape, (1, 14))
                    else:
                        with self.assertRaisesRegex(ValueError, "YAM-trained"):
                            client.infer(packet(observation(), "test"))
                finally:
                    client.close()
                    server.shutdown()
                    thread.join(timeout=2)


class PlannerBackendTests(unittest.TestCase):
    def test_invalid_prompt_fails_before_creating_cloud_client(self):
        from piper_teleop.policy_utils.planning import GeminiPlanner

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-only-placeholder"}):
            with patch("google.genai.Client") as client:
                with self.assertRaises(FileNotFoundError):
                    GeminiPlanner(
                        ROOT / "policies/edith",
                        {"gemini_model": "test-model", "prompt_file": "missing.txt"},
                    )
                client.assert_not_called()

    def test_gemini_payload_uses_real_frame_timestamps_and_source_prompt(self):
        from types import SimpleNamespace
        from piper_teleop.policy_utils.planning import GeminiPlanner

        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text='{"command":"NoOp"}')

        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate), close=lambda: None
        )
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-only-placeholder"}):
            with patch("google.genai.Client", return_value=client):
                planner = GeminiPlanner(
                    ROOT / "policies/edith",
                    {
                        "gemini_model": "test-model",
                        "prompt_file": "prompts/passing_tools.txt",
                    },
                )
                frames = [
                    Frame(
                        1_000_000_000,
                        np.zeros((4, 4, 3), np.uint8),
                        np.zeros((4, 4, 3), np.uint8),
                    ),
                    Frame(
                        1_700_000_000,
                        np.zeros((4, 4, 3), np.uint8),
                        np.zeros((4, 4, 3), np.uint8),
                    ),
                ]
                self.assertEqual(
                    planner.plan(frames, [], "pick up cup", True), {"command": "NoOp"}
                )
                texts = "\n".join(part.text or "" for part in calls[0]["contents"])
                self.assertIn("t=0.700000s", texts)
                self.assertIn("pick up cup", texts)
                self.assertNotIn("{duration", texts)
                self.assertEqual(
                    sum(p.inline_data is not None for p in calls[0]["contents"]), 2
                )
                planner.close()

    def test_fam_hri_external_api_receives_rgb_gaze_and_dwell(self):
        from types import SimpleNamespace
        from piper_teleop.policy_utils.fam_hri import FamHRIPlanner

        received = []

        def process(text, frames, *args, **kwargs):
            received.extend(frames)
            return {
                "subtasks": ["pick up cup"],
                "actions": [{"object_label": "cup"}],
                "keyframe_labels": [{"gaze_object_label": "cup"}],
            }, None

        pipeline = SimpleNamespace(
            KFInput=lambda **kw: SimpleNamespace(**kw),
            PipelineOpts=lambda **kw: kw,
            build_labeler=lambda *a: object(),
            build_planner=lambda *a: object(),
            process_utterance=process,
        )
        with patch(
            "piper_teleop.policy_utils.fam_hri.importlib.import_module",
            return_value=pipeline,
        ):
            planner = FamHRIPlanner({"llm_model": "test", "min_dwell_s": 0.5})
        frames = []
        for i in range(4):
            image = np.zeros((8, 8, 3), np.uint8)
            image[:, :, 0] = 255
            frames.append(
                Frame(1_000_000_000 + i * 300_000_000, image, image, (2.0, 2.0), image)
            )
        result = planner.plan(frames, [], "pick up cup", True)
        self.assertEqual(result["command"], "Append")
        self.assertEqual(result["new_elements"][0]["keyframe_interval"]["start"], 0.6)
        self.assertEqual(received[0].dwell_s, 0.6)
        self.assertEqual(int(received[0].image_rgb[0, 0, 0]), 255)
        planner.close()

    def test_queue_version_changes_even_for_future_edits(self):
        q = TaskQueue([Task("a", None), Task("b", None), Task("c", None)])
        revision = q.revision
        q.apply({"command": "Remove", "target_indices": [1]}, [])
        self.assertEqual(q.revision, revision)
        self.assertEqual(q.version, 1)


class RunnerTests(unittest.TestCase):
    def test_station_preflight_failure_closes_policy(self):
        from piper_teleop.scripts.eval_policy_local import main

        policy = load_policy(ROOT / "policies/textvla")
        with patch(
            "piper_teleop.scripts.eval_policy_local.load_policy", return_value=policy
        ):
            with patch(
                "piper_teleop.scripts.eval_policy_local.load_station",
                side_effect=FileNotFoundError("station configuration"),
            ):
                with self.assertRaises(FileNotFoundError):
                    main([])
        self.assertTrue(policy.instance.closed)
        self.assertTrue(policy.instance.client.closed)

    def test_check_policy_only_never_discovers_devices(self):
        from piper_teleop.scripts.eval_policy_local import main

        with patch(
            "piper_teleop.scripts.eval_policy_local.load_station",
            side_effect=AssertionError("station"),
        ):
            with patch("builtins.print"):
                self.assertEqual(
                    main(
                        [
                            "--policy-dir",
                            str(ROOT / "policies/textvla"),
                            "--check-policy-only",
                        ]
                    ),
                    0,
                )

    def test_mismatched_output_rate_fails_before_devices(self):
        from piper_teleop.scripts.eval_policy_local import main

        with patch(
            "piper_teleop.scripts.eval_policy_local.load_station",
            side_effect=AssertionError("station"),
        ):
            with self.assertRaisesRegex(ValueError, "match policy control_hz"):
                main(
                    [
                        "--policy-dir",
                        str(ROOT / "policies/textvla"),
                        "--output-fps",
                        "30",
                    ]
                )

    def test_transcript_and_events_are_observation_inputs(self):
        from types import SimpleNamespace
        from piper_teleop.scripts.eval_policy_local import _add_policy_inputs

        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp) / "utterance.txt"
            text.write_text("new instruction")
            event = Path(tmp) / "event.json"
            event.write_text('{"id":"n1","command":"next"}')
            obs = {}
            _add_policy_inputs(
                obs, SimpleNamespace(transcript_file=text, policy_events_file=event)
            )
            self.assertEqual(obs["transcript"], "new instruction")
            self.assertEqual(obs["policy_event"]["command"], "next")


if __name__ == "__main__":
    unittest.main()
