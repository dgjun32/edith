"""Run one policy directory with synchronized Aria Gen 2 and robot views.

Evaluation does not open CAN by default. Pass ``--execute-actions`` to launch
the selected robot adapter; the first action is bounded against the live
startup state before the command refresh thread starts.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from edith.aria_utils import inspect_device_connection
from edith.eval_utils.action_lease import ActionLease
from edith.eval_utils.environment import RobotEvaluationEnv
from edith.eval_utils.sync import Sample
from edith.policy_utils import load_policy
from edith.config import load_station
from edith.robot_utils.factory import create_robot


DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "policies/static_hold"


def parser(*, default_policy=DEFAULT_POLICY):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy-dir",
        type=Path,
        default=default_policy,
        help="Directory containing policy.json (default: %(default)s)",
    )
    p.add_argument(
        "--policy-config",
        type=Path,
        action="append",
        default=[],
        help="JSON config overrides; repeat to layer task/model settings",
    )
    p.add_argument(
        "--check-policy-only",
        action="store_true",
        help="Validate policy without cameras, CAN or inference",
    )
    p.add_argument(
        "--transcript-file",
        type=Path,
        help="Local UTF-8 transcript, refreshed on each observation",
    )
    p.add_argument(
        "--policy-events-file",
        type=Path,
        help="Local JSON operator queue edit with a unique id",
    )
    p.add_argument(
        "--station",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/station.example.json",
        help="Robot-independent camera/joint-group JSON",
    )
    p.add_argument(
        "--robot",
        default=None,
        help="Robot repository module:factory; defaults to station robot.adapter",
    )
    p.add_argument(
        "--abc-root",
        type=Path,
        default=Path.home() / "dev/abc-robot-setting/abc",
        help="Only used with --robot abc-yam",
    )
    p.add_argument("--profile", default="abc_station_config")
    p.add_argument("--aria-serial", default="")
    p.add_argument(
        "--aria-ip",
        default="",
        help="Glasses USB-NCM IP to bypass USB discovery (not an external camera server)",
    )
    p.add_argument("--aria-profile", default="mp_streaming_demo")
    p.add_argument("--sync-camera", default="top")
    p.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Defaults to policy control_hz; must match it",
    )
    p.add_argument("--tolerance-ms", type=float, default=100)
    p.add_argument("--max-age-ms", type=float, default=500)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--instruction", default="")
    p.add_argument("--output", type=Path)
    p.add_argument("--record-vrs", default="")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--execute-actions",
        action="store_true",
        help="Start the selected robot adapter and execute policy actions",
    )
    p.add_argument(
        "--max-first-action-delta",
        type=float,
        default=0.05,
        help="Maximum absolute first joint-position delta from the live state",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Read-only policy, dependency, and device discovery check",
    )
    return p


def _realsense_serials():
    try:
        import pyrealsense2 as rs

        return sorted(
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ), ""
    except Exception as error:
        return [], f"{type(error).__name__}: {error}"


def _check_report(args, config, policy):
    aria = inspect_device_connection(args.aria_serial, args.aria_ip)
    realsense, realsense_error = _realsense_serials()
    expected = {camera.serial for camera in config.cameras.values()}
    found = set(realsense)
    modules = (
        "numpy",
        "cv2",
        "h5py",
        "pyrealsense2",
        "aria",
    )
    missing_modules = [
        name for name in modules if importlib.util.find_spec(name) is None
    ]
    can = {}
    if args.robot == "abc-yam" and args.execute_actions:
        can = {
            name: {
                "interface": group.follower.channel,
                "present": Path("/sys/class/net", group.follower.channel).exists(),
            }
            for name, group in config.robots.items()
        }
    report = {
        "ready": (
            aria.ready
            and not missing_modules
            and not realsense_error
            and not (expected - found)
            and all(item["present"] for item in can.values())
        ),
        "policy": {
            "directory": str(policy.root),
            "name": policy.manifest.get("name", ""),
            "control_hz": policy.control_hz,
            "action_space": policy.manifest.get("action_space", ""),
        },
        "aria": {
            "transport": "USB_NCM",
            "target": args.aria_ip or "USB discovery",
            "devices": aria.devices,
            "error": aria.error,
        },
        "realsense": {
            "expected": sorted(expected),
            "found": realsense,
            "missing": sorted(expected - found),
            "error": realsense_error,
        },
        "robot": {
            "adapter": args.robot,
            "execute_actions": args.execute_actions,
            "can": can,
            "gello_required": False,
        },
        "missing_modules": missing_modules,
    }
    return report


def _synthetic_robot_samples(bundle, snapshot, action, arm_order):
    anchor = bundle["aria"].timestamp_ns
    received = time.perf_counter_ns()
    offset = 0
    for name in arm_order:
        state = np.asarray(snapshot.positions[name], dtype=np.float64)
        width = state.size
        bundle[f"follower_{name}_state"] = Sample(
            anchor, received, anchor, 0, state, "dry_run_config"
        )
        bundle[f"leader_{name}_actions"] = Sample(
            anchor,
            received,
            anchor,
            0,
            np.asarray(action[offset : offset + width], dtype=np.float64),
            "dry_run_policy",
        )
        offset += width


def _display(bundle, spread_ms):
    import cv2

    tiles = []
    for name, sample in bundle.items():
        if not isinstance(sample.data, np.ndarray) or sample.data.ndim != 3:
            continue
        rgb = sample.data
        scale = min(480 / rgb.shape[1], 360 / rgb.shape[0])
        small = cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale
        )
        tile = np.zeros((400, 480, 3), np.uint8)
        tile[: small.shape[0], : small.shape[1]] = small
        cv2.putText(
            tile,
            f"{name} | spread {spread_ms:.1f} ms",
            (8, 383),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    if not tiles:
        return False
    cv2.imshow("Edith synchronized policy evaluation", np.hstack(tiles))
    return cv2.waitKey(1) & 0xFF in (ord("q"), 27)


def _add_policy_inputs(observation, args):
    # STT/UI processes can atomically replace these local files. No new camera
    # consumer and no cloud transcription is implicitly started by a policy.
    if args.transcript_file:
        observation["transcript"] = args.transcript_file.read_text(
            encoding="utf-8"
        ).strip()
    if args.policy_events_file:
        event = json.loads(args.policy_events_file.read_text(encoding="utf-8"))
        if not isinstance(event, dict):
            raise ValueError("Policy event must be a JSON object")
        observation["policy_event"] = event


def main(argv=None, *, default_policy=DEFAULT_POLICY):
    args = parser(default_policy=default_policy).parse_args(argv)
    if (
        not 0 < args.tolerance_ms <= args.max_age_ms
        or args.timeout <= 0
        or args.duration <= 0
        or args.max_first_action_delta < 0
    ):
        raise ValueError("Invalid timing or first-action safety configuration")
    overrides = {}
    for path in args.policy_config:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("--policy-config must contain a JSON object")
        overrides.update(value)
    policy = load_policy(args.policy_dir, overrides)
    try:
        return _run(args, policy)
    finally:
        policy.close()


def _run(args, policy):
    if args.execute_actions and policy.manifest.get("observe_only"):
        raise ValueError("This policy is observe_only; motor execution is forbidden")
    if args.check_policy_only:
        print(
            json.dumps(
                {"policy": policy.manifest, "directory": str(policy.root)}, indent=2
            )
        )
        return 0
    if args.output_fps is None:
        args.output_fps = policy.control_hz
    if args.output_fps != policy.control_hz:
        raise ValueError(
            "--output-fps must match policy control_hz so action chunks keep their intended rate"
        )
    if args.robot == "abc-yam":
        from edith.robot_utils.abc_config import load_station as load_abc_station

        config = load_abc_station(args.abc_root, args.profile)
    else:
        config = load_station(args.station)
        args.robot = args.robot or config.adapter
    if args.check:
        report = _check_report(args, config, policy)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.record_vrs and Path(args.record_vrs).exists():
        raise FileExistsError(args.record_vrs)

    from edith.envs.local_env import LocalSensorEnv
    from edith.eval_utils.recording import BundleWriter

    sensors = evaluation = robot = lease = writer = None
    try:
        sensors = LocalSensorEnv(
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
        # Sensor health is a hard prerequisite for any motor initialization.
        first_bundle = sensors.get_bundle(args.timeout)
        robot = create_robot(
            args.robot,
            config=config,
            execute=args.execute_actions,
            abc_root=args.abc_root,
            profile=args.profile,
        )
        robot.start()
        startup = robot.read(60 if args.execute_actions else 0.1)
        if args.execute_actions:
            sensors.attach_robot(robot)
            # Startup can take seconds. Infer only from a fresh bundle that
            # includes the selected adapter's synchronized live state.
            first_bundle = sensors.get_bundle(args.timeout)
        else:
            _synthetic_robot_samples(
                first_bundle, startup, startup.flat(robot.arm_order), robot.arm_order
            )
        evaluation = RobotEvaluationEnv(
            sensors,
            robot,
            startup,
            instruction=args.instruction,
            control_hz=policy.control_hz,
        )
        first_observation = evaluation.observation_from_bundle(first_bundle)
        _add_policy_inputs(first_observation, args)
        policy.reset(first_observation)
        first_action = robot.validate_action(policy.act(first_observation))
        if not args.execute_actions:
            _synthetic_robot_samples(
                first_bundle, startup, first_action, robot.arm_order
            )
        startup_flat = startup.flat(robot.arm_order)
        first_delta = float(np.max(np.abs(first_action - startup_flat)))
        if args.execute_actions and first_delta > args.max_first_action_delta:
            raise RuntimeError(
                f"First action delta {first_delta:.6f} exceeds "
                f"--max-first-action-delta {args.max_first_action_delta:.6f}"
            )
        lease = ActionLease(robot, first_action, policy.control_hz)
        lease.start()
        if args.output:
            writer = BundleWriter(
                args.output,
                {
                    "station": asdict(config),
                    "policy_manifest": policy.manifest,
                    "policy_directory": str(policy.root),
                    "robot_adapter": args.robot,
                    "robot_execution_backend": type(robot).__module__
                    + "."
                    + type(robot).__name__,
                    "robot_state_source": "adapter_live"
                    if args.execute_actions
                    else "configured_initial_state",
                    "execute_actions": args.execute_actions,
                    "instruction": args.instruction,
                    "first_action_max_abs_delta": first_delta,
                    "tolerance_ms": args.tolerance_ms,
                    "max_age_ms": args.max_age_ms,
                    "sync_anchor": args.sync_camera,
                    "sync": "software capture-time alignment; not hardware trigger sync",
                },
            )
            # Record the reset/first-act input as well as subsequent inputs.
            writer.append(
                first_bundle, observation=first_observation, action=first_action
            )
        print(
            f"Policy active: {policy.manifest.get('name', policy.root.name)}; "
            f"execute_actions={args.execute_actions}; first_delta={first_delta:.6f}",
            flush=True,
        )
        started = time.perf_counter()
        count = 1
        action = first_action
        while time.perf_counter() - started < args.duration:
            lease.check()
            observation = evaluation.get_policy_observation(args.timeout)
            bundle = evaluation.last_bundle
            if not args.execute_actions:
                _synthetic_robot_samples(bundle, startup, action, robot.arm_order)
                observation = evaluation.observation_from_bundle(bundle)
            _add_policy_inputs(observation, args)
            action = robot.validate_action(policy.act(observation))
            if not args.execute_actions:
                _synthetic_robot_samples(bundle, startup, action, robot.arm_order)
            lease.update(action)
            if writer:
                writer.append(bundle, observation=observation, action=action)
            count += 1
            times = [sample.timestamp_ns for sample in bundle.values()]
            spread_ms = (max(times) - min(times)) / 1e6
            if not args.headless and _display(bundle, spread_ms):
                break
            if count % 100 == 0:
                print(
                    f"{count} bundles; spread={spread_ms:.1f} ms; "
                    f"drops={sensors.sync.dropped}",
                    flush=True,
                )
        print(f"Completed {count} synchronized policy bundles.", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        # ExitStack attempts every cleanup even if a device shutdown fails.
        with ExitStack() as cleanup:
            if not args.headless:
                import cv2

                cleanup.callback(cv2.destroyAllWindows)
            if writer is not None:
                cleanup.callback(writer.close)
            if evaluation is not None:
                cleanup.callback(evaluation.safe_end)
            else:
                if sensors is not None:
                    cleanup.callback(sensors.safe_end)
                if robot is not None:
                    cleanup.callback(robot.close)
            if lease is not None:
                cleanup.callback(lease.close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
