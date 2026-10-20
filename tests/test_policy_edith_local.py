"""USB-only capture and compatibility launch paths, without starting devices."""

from contextlib import redirect_stdout
import builtins
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from piper_teleop.aria_utils import AriaGen2Camera
from piper_teleop.policy_utils import load_policy


ROOT = Path(__file__).resolve().parents[1]


class USBTransportTests(unittest.TestCase):
    def test_auto_and_direct_ip_capture_explicitly_use_usb(self):
        import aria.sdk_gen2 as sdk

        for ip in ("", "192.0.2.1"):
            with self.subTest(ip=ip):
                target = SimpleNamespace(serial="usb-glasses", ip="192.0.2.1")
                device = MagicMock()
                device.is_streaming.return_value = False
                client = MagicMock()
                client.usb_network_devices.return_value = [target]
                client.connect.return_value = device
                receiver = MagicMock()
                with (
                    patch("aria.sdk_gen2.DeviceClient", return_value=client),
                    patch(
                        "aria.sdk_gen2.HttpStreamingConfig", side_effect=SimpleNamespace
                    ),
                    patch(
                        "aria.sdk_gen2.HttpServerConfig", side_effect=SimpleNamespace
                    ),
                    patch("aria.stream_receiver.StreamReceiver", return_value=receiver),
                    patch("piper_teleop.aria_utils.gen2.time.sleep"),
                ):
                    camera = AriaGen2Camera(MagicMock(), serial="usb-glasses", ip=ip)
                    try:
                        if ip:
                            client.usb_network_devices.assert_not_called()
                            self.assertEqual(client.connect.call_args.args[0].ip, ip)
                        else:
                            client.usb_network_devices.assert_called_once_with()
                            client.connect.assert_called_once_with(target)
                        streaming = device.set_streaming_config.call_args.args[0]
                        self.assertEqual(
                            streaming.streaming_interface,
                            sdk.StreamingInterface.USB_NCM,
                        )
                        self.assertEqual(streaming.profile_name, "mp_streaming_demo")
                        receiver.register_rgb_callback.assert_called_once()
                        receiver.register_eye_gaze_callback.assert_called_once()
                        receiver.register_hand_pose_callback.assert_called_once()
                        receiver.start_server.assert_called_once_with()
                    finally:
                        camera.close()
                    device.stop_streaming.assert_called_once_with()
                    client.disconnect.assert_called_once_with(device)

    def test_missing_usb_device_never_falls_back_to_other_transports(self):
        client = MagicMock()
        client.usb_network_devices.return_value = []
        with (
            patch("aria.sdk_gen2.DeviceClient", return_value=client),
            patch("aria.stream_receiver.StreamReceiver") as receiver,
        ):
            with self.assertRaisesRegex(RuntimeError, "connect the glasses over USB"):
                AriaGen2Camera(MagicMock())
        client.connect.assert_not_called()
        receiver.assert_not_called()
        self.assertEqual(
            [entry[0] for entry in client.method_calls], ["usb_network_devices"]
        )


class EdithEntrypointTests(unittest.TestCase):
    def test_direct_scripts_work_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            for relative in (
                "piper_teleop/scripts/policy_edith.py",
                "piper_teleop/aria_eval_scripts_refactored/eval_openpi_ours_with_voice_260503.py",
            ):
                with self.subTest(script=relative):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / relative), "--check-policy-only"],
                        cwd=cwd,
                        text=True,
                        capture_output=True,
                        timeout=15,
                        check=True,
                    )
                    report = json.loads(result.stdout)
                    self.assertEqual(report["policy"]["name"], "policy_edith")

    def test_new_and_original_launchers_load_named_policy_without_legacy_imports(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(
                (
                    "piper_teleop.envs.piper_env",
                    "piper_teleop.robots.piper_robot",
                    "piper_teleop.sensors.aria_camera",
                    "common",
                    "ours_utils",
                    "openpi_client",
                )
            ):
                raise AssertionError(f"Legacy remote/hardware dependency: {name}")
            return original_import(name, *args, **kwargs)

        paths = [
            ROOT / "piper_teleop/scripts/policy_edith.py",
            ROOT
            / "piper_teleop/aria_eval_scripts_refactored"
            / "eval_openpi_ours_with_voice_260503.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                output = io.StringIO()
                with (
                    patch("builtins.__import__", side_effect=guarded_import),
                    patch.object(sys, "argv", [str(path), "--check-policy-only"]),
                    patch(
                        "piper_teleop.scripts.eval_policy_local.load_station",
                        side_effect=AssertionError("Must not access station"),
                    ),
                    patch(
                        "piper_teleop.policy_utils.openpi.OpenPIClient._connect",
                        side_effect=AssertionError("Must not connect to model"),
                    ),
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as stopped,
                ):
                    runpy.run_path(str(path), run_name="__main__")
                self.assertEqual(stopped.exception.code, 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report["policy"]["name"], "policy_edith")
                self.assertEqual(
                    Path(report["directory"]), ROOT / "policies/policy_edith"
                )
                self.assertEqual(
                    report["policy"]["config"]["server_url"], "ws://127.0.0.1:8000"
                )

    def test_previous_policy_path_resolves_to_same_assets(self):
        policy = load_policy(ROOT / "policies/edith")
        try:
            self.assertEqual(policy.root, ROOT / "policies/policy_edith")
            self.assertEqual(policy.manifest["name"], "policy_edith")
            self.assertTrue((policy.root / "prompts/passing_tools.txt").is_file())
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
