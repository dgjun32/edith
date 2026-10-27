"""Local USB Aria Gen 2 + ABC YAM teleoperation and synchronized recording."""

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from edith.aria_utils import (
    EYE_GAZE_FIELDS,
    HAND_POSE_FIELDS,
    draw_perception_bgr,
)
from edith.robot_utils.abc_config import DEFAULT_ABC_ROOT
from edith.robot_utils.abc_runtime import load_abc_teleop_station
from edith.teleop_utils.gello import validate_gello


EYE_GAZE_INDEX = {name: index for index, name in enumerate(EYE_GAZE_FIELDS)}
HAND_POSE_INDEX = {name: index for index, name in enumerate(HAND_POSE_FIELDS)}
def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    p.add_argument("--profile", default="abc_station_config")
    p.add_argument(
        "--leaders",
        type=Path,
        help="JSON mapping arm to GELLO device_name, servo_ids, joint_signs",
    )
    p.add_argument(
        "--mode",
        choices=("preview", "teleop", "attach"),
        default="preview",
        help="preview: cameras only; teleop: launch ABC GELLO/YAM; attach: observe already-running ABC teleop",
    )
    p.add_argument("--aria-serial", default="")
    p.add_argument(
        "--aria-ip",
        default="",
        help="Glasses USB-NCM IP to bypass USB discovery (not an external camera server)",
    )
    p.add_argument("--aria-profile", default="mp_streaming_demo")
    p.add_argument(
        "--sync-camera",
        default="top",
        help="Robot camera used as the master capture timeline",
    )
    p.add_argument(
        "--output-fps",
        type=float,
        default=15,
        help="Exact bundle rate obtained by integer subsampling of the master camera",
    )
    p.add_argument("--tolerance-ms", type=float, default=100)
    p.add_argument("--max-age-ms", type=float, default=500)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--output", type=Path, help="New HDF5 file; never overwrites")
    p.add_argument(
        "--record-vrs",
        default="",
        help="Optional raw Gen 2 VRS output (includes profile sensor streams)",
    )
    p.add_argument("--instruction", default="")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Seconds to run; 0 means until Ctrl+C/q",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Inspect config/dependencies without opening cameras or motors",
    )
    return p


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        not 0 < args.tolerance_ms <= args.max_age_ms
        or args.timeout <= 0
        or args.duration < 0
    ):
        raise ValueError("Invalid timing configuration")
    config = load_abc_teleop_station(args.abc_root, args.profile, args.leaders)
    missing = []
    for module in (
        "numpy",
        "cv2",
        "h5py",
        "zmq",
        "pyrealsense2",
        "aria",
        "i2rt",
        "tyro",
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if args.check:
        print(
            json.dumps(
                {
                    "station": asdict(config),
                    "missing_modules": missing,
                    "mode": args.mode,
                    "aria_transport": "USB_NCM",
                },
                indent=2,
            )
        )
        if args.mode == "teleop":
            validate_gello(config, args.leaders)
        return 1 if missing else 0
    if missing:
        raise RuntimeError(f"Missing dependencies: {missing}. See README_LOCAL_YAM.md")
    if args.mode == "teleop":
        validate_gello(config, args.leaders)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.record_vrs and Path(args.record_vrs).exists():
        raise FileExistsError(args.record_vrs)
    import cv2
    import numpy as np
    from edith.envs.yam_env import YAMLocalEnv
    from edith.eval_utils.recording import BundleWriter

    env = writer = process = None
    try:
        env = YAMLocalEnv(
            config,
            args.aria_serial,
            args.aria_profile,
            args.tolerance_ms,
            args.max_age_ms,
            args.record_vrs,
            args.aria_ip,
            args.sync_camera,
            args.output_fps,
        )
        # Verify all camera frames and clock mappings before launching motors.
        env.get_bundle(args.timeout)
        if args.mode != "preview":
            env.attach_robot_streams()
        if args.mode == "teleop":
            child_env = os.environ.copy()
            root = str(Path(__file__).resolve().parents[2])
            child_env["PYTHONPATH"] = os.pathsep.join(
                [root, str(args.abc_root.resolve()), child_env.get("PYTHONPATH", "")]
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "edith.robot_utils.abc_runtime",
                    "--abc-root",
                    str(args.abc_root.resolve()),
                    "--profile",
                    args.profile,
                    "--leaders",
                    str(args.leaders.resolve()),
                ],
                env=child_env,
                start_new_session=True,
            )
        if args.output:
            writer = BundleWriter(
                args.output,
                {
                    "station": asdict(config),
                    "instruction": args.instruction,
                    "mode": args.mode,
                    "tolerance_ms": args.tolerance_ms,
                    "max_age_ms": args.max_age_ms,
                    "joint_units": "radians",
                    "gripper_units": "i2rt normalized",
                    "sync": "software capture-time alignment; not hardware trigger sync",
                    "sync_anchor": args.sync_camera,
                    "output_fps": args.output_fps,
                    "resampling": "nearest capture time; slower non-anchor samples may repeat",
                },
            )
        start = time.perf_counter()
        count = 0
        print(
            "Local synchronization active. q/Esc or Ctrl+C stops; --output records every accepted bundle.",
            flush=True,
        )
        while not args.duration or time.perf_counter() - start < args.duration:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"ABC teleoperation exited: {process.returncode}")
            bundle = env.get_bundle(args.timeout)
            if writer:
                writer.append(bundle)
            count += 1
            spread = (
                max(s.timestamp_ns for s in bundle.values())
                - min(s.timestamp_ns for s in bundle.values())
            ) / 1e6
            if not args.headless:
                tiles = []
                for name, sample in bundle.items():
                    if sample.data.ndim == 3:
                        rgb = sample.data
                        # Preserve aspect ratio when displaying the Gen 2 view.
                        scale = min(480 / rgb.shape[1], 360 / rgb.shape[0])
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if name == "aria":
                            projection = env.aria.project_perception_to_rgb(
                                bundle["eye_gaze"].data,
                                bundle["hand_pose"].data,
                            )
                            bgr = draw_perception_bgr(bgr, projection)
                        small = cv2.resize(
                            bgr,
                            None,
                            fx=scale,
                            fy=scale,
                        )
                        tile = np.zeros((400, 480, 3), np.uint8)
                        tile[: small.shape[0], : small.shape[1]] = small
                        cv2.putText(
                            tile,
                            f"{name} | spread {spread:.1f} ms",
                            (8, 383),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                        )
                        tiles.append(tile)
                dashboard = np.hstack(tiles)
                gaze = bundle["eye_gaze"].data
                hands = bundle["hand_pose"].data
                left = bool(hands[HAND_POSE_INDEX["left_present"]])
                right = bool(hands[HAND_POSE_INDEX["right_present"]])
                gaze_valid = bool(gaze[EYE_GAZE_INDEX["combined_gaze_valid"]])
                status = (
                    f"{env.output_fps:g} FPS {env.sync_anchor} master | "
                    f"gaze {'valid' if gaze_valid else 'invalid'} "
                    f"yaw {gaze[EYE_GAZE_INDEX['yaw_rad']]:+.3f} "
                    f"pitch {gaze[EYE_GAZE_INDEX['pitch_rad']]:+.3f} | "
                    f"hands L:{'yes' if left else 'no'} "
                    f"R:{'yes' if right else 'no'} | spread {spread:.1f} ms"
                )
                cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1], 32), (0, 0, 0), -1)
                cv2.putText(
                    dashboard,
                    status,
                    (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 255, 80),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Edith local Aria Gen 2 + YAM", dashboard)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if count % 100 == 0:
                print(
                    f"{count} synchronized bundles @ {env.output_fps:g} FPS; "
                    f"anchor {env.sync_anchor}; spread {spread:.1f} ms; "
                    f"drops {env.sync.dropped}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_process(process)
        finally:
            try:
                if env:
                    env.safe_end()
            finally:
                if writer:
                    writer.close()
                if not args.headless:
                    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
