"""Hardware-free synchronization, SDK callback, IPC and recording contracts."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from edith.local.sync import Sample, Synchronizer
from edith.local.aria import AriaGen2Camera
from edith.local.recording import BundleWriter


def sample(t, sequence=1, data=None):
    return Sample(
        t,
        t,
        t,
        sequence,
        np.zeros((2, 3, 3), dtype=np.uint8) if data is None else data,
        "test",
    )


class SyncTests(unittest.TestCase):
    def test_nearest_match_consumed_once(self):
        sync = Synchronizer(["aria", "top"], 10, 100)
        sync.add("aria", sample(100_000_000))
        sync.add("top", sample(80_000_000))
        sync.add("top", sample(105_000_000, 2))
        result = sync._match(110_000_000)
        self.assertEqual(result["top"].sequence, 2)
        self.assertIsNone(sync._match(110_000_000))

    def test_total_spread_not_just_anchor_distance(self):
        sync = Synchronizer(["aria", "left", "right"], 10, 100)
        for name, t in [("aria", 100), ("left", 91), ("right", 109)]:
            sync.add(name, sample(t * 1_000_000))
        self.assertIsNone(sync._match(110_000_000))

    def test_stale_missing_future_and_out_of_order(self):
        sync = Synchronizer(["aria", "top"], 10, 100, capacity=2)
        sync.add("aria", sample(100_000_000))
        self.assertIsNone(sync._match(110_000_000))
        sync.add("top", sample(100_000_000))
        sync.add("top", sample(90_000_000))
        self.assertEqual(sync.dropped["top"], 1)
        self.assertIsNone(sync._match(99_000_000))
        self.assertIsNone(sync._match(300_000_000))

    def test_capacity_timeout_and_source_failure(self):
        sync = Synchronizer(["aria", "top"], capacity=2)
        for i in range(3):
            sync.add("aria", sample(time.perf_counter_ns(), i))
        self.assertEqual(len(sync.buffers["aria"]), 2)
        self.assertEqual(sync.dropped["aria"], 1)
        with self.assertRaises(TimeoutError):
            sync.get(0.01)
        sync.fail(ValueError("USB disconnected"))
        with self.assertRaises(RuntimeError):
            sync.get(0.01)

    def test_get_complete_bundle_and_state(self):
        names = ["aria", "top", "follower_left_state", "leader_left_actions"]
        sync = Synchronizer(names)
        t = time.perf_counter_ns() - 1_000_000
        for name in names:
            sync.add(name, sample(t, data=np.zeros(7) if "left" in name else None))
        self.assertEqual(set(sync.get(0.01)), set(names))

    def test_robot_camera_anchor_subsamples_30_to_15_and_reuses_slow_sample(self):
        sync = Synchronizer(
            ["aria", "top"],
            tolerance_ms=60,
            max_age_ms=500,
            anchor="top",
            anchor_stride=2,
        )
        base = 1_000_000_000
        sync.add("aria", sample(base, sequence=1))
        sync.add("top", sample(base, sequence=10))
        first = sync._match(base + 1_000_000)
        self.assertEqual(first["top"].sequence, 10)

        sync.add("top", sample(base + 33_000_000, sequence=11))
        self.assertIsNone(sync._match(base + 34_000_000))
        sync.add("top", sample(base + 66_000_000, sequence=12))
        sync.add("top", sample(base + 100_000_000, sequence=13))
        sync.add("top", sample(base + 133_000_000, sequence=14))
        self.assertIsNone(sync._match(base + 67_000_000))
        sync.add("aria", sample(base + 100_000_000, sequence=2))
        second = sync._match(base + 101_000_000)
        self.assertEqual(second["top"].sequence, 12)
        self.assertEqual(second["aria"].sequence, 2)

        third = sync._match(base + 134_000_000)
        self.assertEqual(third["top"].sequence, 14)
        self.assertEqual(third["aria"].sequence, 2)


class AriaTests(unittest.TestCase):
    def test_direct_ip_inspection_connects_without_streaming(self):
        from edith.aria_utils import inspect_device_connection

        completed = SimpleNamespace(returncode=0, stderr="")
        with patch(
            "edith.aria_utils.gen2.subprocess.run",
            return_value=completed,
        ) as run:
            status = inspect_device_connection("serial", "192.0.2.1")
        self.assertTrue(status.ready)
        self.assertEqual(status.devices, ("serial",))
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def camera(self, valid=True):
        camera = AriaGen2Camera.__new__(AriaGen2Camera)
        camera.sequences = {"aria": 0, "eye_gaze": 0, "hand_pose": 0}
        camera.sink = Synchronizer(["aria", "eye_gaze", "hand_pose"])
        clock = SimpleNamespace(
            is_connected=lambda: True,
            is_valid=lambda: valid,
            compute_latency_ms=lambda t: 25,
        )
        camera.receiver = SimpleNamespace(get_time_sync=lambda: clock)
        return camera

    def test_capture_time_mapping_not_arrival_time(self):
        camera = self.camera()
        image = SimpleNamespace(to_numpy_array=lambda: np.zeros((5, 7, 3), np.uint8))
        with patch(
            "edith.aria_utils.gen2.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            camera._rgb(image, SimpleNamespace(capture_timestamp_ns=500))
        frame = camera.sink.buffers["aria"][0]
        self.assertEqual(frame.timestamp_ns, 975_000_000)
        self.assertEqual(frame.source_timestamp_ns, 500)
        self.assertEqual(frame.data.shape, (5, 7, 3))

    def test_audio_maps_frame_capture_times_and_is_taken_once(self):
        from edith.aria_utils import AudioBuffer

        camera = self.camera()
        camera.audio = AudioBuffer()
        step = 20_833  # one 48 kHz frame in device nanoseconds
        frames = np.arange(8, dtype=np.int32).reshape(4, 2)
        with patch(
            "edith.aria_utils.gen2.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            camera._audio(
                SimpleNamespace(data=frames.reshape(-1).tolist()),
                SimpleNamespace(capture_timestamps_ns=[i * step for i in range(4)]),
                2,
            )
        self.assertEqual(camera.audio.sample_rate_hz, 48_000)
        last = 975_000_000  # receive time minus 25 ms TicSync latency
        first = camera.audio.take(last - 2 * step)
        np.testing.assert_array_equal(first.data, frames[:2])
        self.assertEqual(first.timestamp_ns, last - 2 * step)
        self.assertEqual(first.source_timestamp_ns, step)
        rest = camera.audio.take(last)
        np.testing.assert_array_equal(rest.data, frames[2:])
        self.assertEqual((rest.timestamp_ns, rest.sequence), (last, 4))
        empty = camera.audio.take(last + 1)
        self.assertEqual(empty.data.shape, (0, 2))
        self.assertEqual(empty.timestamp_ns, last + 1)

    def test_audio_bounds_memory_and_reports_drops_in_sequence(self):
        from edith.aria_utils import AudioBuffer

        audio = AudioBuffer(max_seconds=2 / 48_000)
        for start in (0, 2, 4):
            times = np.array([start, start + 1], dtype=np.int64)
            audio.add(times, times, start, np.full((2, 1), start, np.int32))
        taken = audio.take(10)
        np.testing.assert_array_equal(taken.data[:, 0], [4, 4])
        self.assertEqual(taken.sequence, 6)

    def test_bad_audio_block_is_reported(self):
        from edith.aria_utils import AudioBuffer

        camera = self.camera()
        camera.audio = AudioBuffer()
        camera._audio(
            SimpleNamespace(data=[1, 2, 3]),
            SimpleNamespace(capture_timestamps_ns=[1, 2]),
            2,
        )
        self.assertIsInstance(camera.sink.error, ValueError)

    def test_invalid_sync_does_not_emit_frames(self):
        camera = self.camera(False)
        camera._rgb(None, None)
        self.assertFalse(camera.sink.buffers["aria"])

    def test_bad_decoded_frame_is_reported(self):
        camera = self.camera()
        camera._rgb(
            SimpleNamespace(to_numpy_array=lambda: np.zeros(10)),
            SimpleNamespace(capture_timestamp_ns=5),
        )
        self.assertIsInstance(camera.sink.error, ValueError)

    def test_eye_gaze_uses_tracking_time_and_named_schema(self):
        from edith.aria_utils import EYE_GAZE_FIELDS

        camera = self.camera()
        vergence = SimpleNamespace(
            left_gaze_valid=True,
            right_gaze_valid=True,
            left_blink_valid=True,
            right_blink_valid=True,
            left_blink=False,
            right_blink=True,
            left_yaw=0.1,
            right_yaw=0.2,
            left_pitch=0.3,
            right_pitch=0.4,
            left_yaw_low=0.01,
            right_yaw_low=0.02,
            left_yaw_high=0.11,
            right_yaw_high=0.22,
            tx_left_eye=1.0,
            ty_left_eye=2.0,
            tz_left_eye=3.0,
            tx_right_eye=4.0,
            ty_right_eye=5.0,
            tz_right_eye=6.0,
        )
        gaze = SimpleNamespace(
            tracking_timestamp=SimpleNamespace(total_seconds=lambda: 0.5),
            yaw=0.25,
            pitch=-0.5,
            depth=1.2,
            yaw_low=0.2,
            yaw_high=0.3,
            pitch_low=-0.6,
            pitch_high=-0.4,
            combined_gaze_valid=True,
            spatial_gaze_point_valid=True,
            combined_gaze_origin_in_cpf=np.arange(3),
            spatial_gaze_point_in_cpf=np.arange(3, 6),
            vergence=vergence,
        )
        with patch(
            "edith.aria_utils.gen2.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            camera._eye_gaze(gaze)
        result = camera.sink.buffers["eye_gaze"][0]
        self.assertEqual(result.source_timestamp_ns, 500_000_000)
        self.assertEqual(result.timestamp_ns, 975_000_000)
        self.assertEqual(result.fields, EYE_GAZE_FIELDS)
        self.assertEqual(result.data.shape, (len(EYE_GAZE_FIELDS),))

    def test_hand_pose_records_two_fixed_size_hands(self):
        from edith.aria_utils import HAND_POSE_FIELDS

        camera = self.camera()
        side = SimpleNamespace(
            confidence=0.75,
            landmark_positions_device=np.arange(63).reshape(21, 3),
            transform_device_wrist=SimpleNamespace(
                to_quat_and_translation=lambda: np.arange(7).reshape(1, 7)
            ),
            wrist_and_palm_normal_device=SimpleNamespace(
                wrist_normal_device=np.array([1, 0, 0]),
                palm_normal_device=np.array([0, 1, 0]),
            ),
        )
        pose = SimpleNamespace(
            tracking_timestamp=SimpleNamespace(total_seconds=lambda: 0.75),
            left_hand=side,
            right_hand=None,
        )
        with patch(
            "edith.aria_utils.gen2.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            camera._hand_pose(pose)
        result = camera.sink.buffers["hand_pose"][0]
        self.assertEqual(result.fields, HAND_POSE_FIELDS)
        self.assertEqual(result.data.shape, (156,))
        self.assertEqual(result.data[0], 1)
        self.assertEqual(result.data[78], 0)
        self.assertTrue(np.isnan(result.data[79:]).all())

    def test_calibrated_gaze_and_hand_projection(self):
        from edith.aria_utils import EYE_GAZE_FIELDS, HAND_POSE_FIELDS

        class IdentityTransform:
            def inverse(self):
                return self

            def __matmul__(self, point):
                return point

        camera_calibration = SimpleNamespace(
            get_transform_device_camera=lambda: IdentityTransform(),
            project=lambda point: np.asarray(point[:2]) * 100,
        )
        calibration = SimpleNamespace(
            get_camera_calib=lambda label: camera_calibration,
            get_transform_device_cpf=lambda: IdentityTransform(),
        )
        camera = self.camera()
        camera.calibration_lock = threading.Lock()
        camera.device_calibration = calibration
        gaze = np.zeros(len(EYE_GAZE_FIELDS))
        gi = {name: index for index, name in enumerate(EYE_GAZE_FIELDS)}
        gaze[gi["spatial_gaze_point_valid"]] = 1
        gaze[gi["spatial_point_x_m"]] = 1
        gaze[gi["spatial_point_y_m"]] = 2
        gaze[gi["spatial_point_z_m"]] = 3
        hands = np.full(len(HAND_POSE_FIELDS), np.nan)
        hi = {name: index for index, name in enumerate(HAND_POSE_FIELDS)}
        hands[hi["left_present"]] = 1
        for landmark in (
            name
            for name in HAND_POSE_FIELDS
            if name.startswith("left_") and name.endswith(("_x_m", "_y_m", "_z_m"))
        ):
            hands[hi[landmark]] = 0.5
        projected = camera.project_perception_to_rgb(gaze, hands)
        self.assertEqual(projected["gaze"], (100.0, 200.0))
        self.assertEqual(len(projected["left"]), 21)
        self.assertEqual(projected["left"][0], (50.0, 50.0))
        self.assertTrue(all(point is None for point in projected["right"]))

    def test_cleanup_disconnects_even_when_device_stop_fails(self):
        calls = []
        camera = self.camera()
        camera.started = True

        def fail():
            calls.append("device")
            raise RuntimeError("disconnected")

        camera.device = SimpleNamespace(stop_streaming=fail)
        camera.receiver = SimpleNamespace(stop_server=lambda: calls.append("receiver"))
        camera.client = SimpleNamespace(disconnect=lambda d: calls.append("disconnect"))
        with self.assertRaises(RuntimeError):
            camera.close()
        self.assertEqual(calls, ["device", "receiver", "disconnect"])


class RecordTests(unittest.TestCase):
    def test_roundtrip_rgb_states_and_timestamps(self):
        import cv2
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.h5"
            writer = BundleWriter(path, {"instruction": "test"})
            red = np.zeros((24, 24, 3), np.uint8)
            red[:, :, 0] = 255
            writer.append(
                {
                    "aria": sample(100, data=red),
                    "follower_left_state": Sample(
                        101,
                        101,
                        101,
                        1,
                        np.arange(7),
                        "test",
                        tuple(f"joint_{i}" for i in range(7)),
                    ),
                }
            )
            writer.close()
            with h5py.File(path) as f:
                self.assertEqual(f.attrs["complete_rows"], 1)
                self.assertEqual(f["aria/timestamp_ns"][0], 100)
                np.testing.assert_array_equal(
                    f["follower_left_state/values"][0], np.arange(7)
                )
                self.assertIn("joint_0", f["follower_left_state"].attrs["value_fields"])
                bgr = cv2.imdecode(f["aria/rgb_jpeg"][0], cv2.IMREAD_COLOR)
                self.assertGreater(int(bgr[10, 10, 2]), 250)
            with self.assertRaises(FileExistsError):
                BundleWriter(path, {})

    def test_roundtrip_variable_length_audio(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.h5"
            writer = BundleWriter(path, {})
            pcm = np.arange(6, dtype=np.int32).reshape(3, 2)
            writer.append({"audio": sample(100, data=pcm)})
            writer.append({"audio": sample(200, data=np.zeros((0, 2), np.int32))})
            writer.close()
            with h5py.File(path) as f:
                self.assertEqual(f["audio"].attrs["num_channels"], 2)
                np.testing.assert_array_equal(
                    f["audio/pcm"][0].reshape(-1, 2), pcm
                )
                self.assertEqual(f["audio/pcm"][1].size, 0)


class IntegrationTests(unittest.TestCase):
    def test_actual_abc_ipc_wire_format(self):
        import uuid
        import zmq
        from edith.local.abc_runtime import load_station
        from edith.local.sources import ABCSubscriber

        load_station()
        from deploy.robot import communication as comms

        name = "edith_test_" + uuid.uuid4().hex
        sync = Synchronizer(["aria", name], tolerance_ms=100)
        context = zmq.Context()
        publisher = comms.create_publisher(context, name)
        subscriber = ABCSubscriber([name], sync)
        try:
            deadline = time.perf_counter() + 2
            while not sync.buffers[name] and time.perf_counter() < deadline:
                comms.publish(publisher, np.arange(7, dtype=float))
                time.sleep(0.02)
            self.assertTrue(sync.buffers[name])
            received = sync.buffers[name][-1]
            sync.add("aria", sample(received.timestamp_ns))
            bundle = sync.get(0.1)
            np.testing.assert_array_equal(bundle[name].data, np.arange(7))
            self.assertEqual(bundle[name].clock, "abc_host_publish")
        finally:
            subscriber.close()
            publisher.close(linger=0)
            context.term()

    def test_camera_failure_never_launches_robot(self):
        from edith.scripts.collect_trajectory_yam_local import main

        with patch(
            "edith.envs.yam_env.YAMLocalEnv",
            side_effect=RuntimeError("camera unavailable"),
        ):
            with patch(
                "edith.scripts.collect_trajectory_yam_local.subprocess.Popen"
            ) as launch:
                with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
                    main(["--headless"])
                launch.assert_not_called()

    def test_incomplete_leader_overrides_rejected(self):
        import json
        from edith.local.abc_runtime import load_station

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaders.json"
            path.write_text(json.dumps({"left": {"device_name": "/dev/fake"}}))
            with self.assertRaises(ValueError):
                load_station(leaders=path)


if __name__ == "__main__":
    unittest.main()
