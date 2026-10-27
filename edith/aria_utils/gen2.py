"""Aria Gen 2 USB-NCM capture and read-only discovery helpers."""

from dataclasses import dataclass
import math
import subprocess
import sys
import threading
import time

import numpy as np

from edith.eval_utils.sync import Sample
from .audio import AUDIO_SAMPLE_RATE_HZ, AudioBuffer  # noqa: F401


EYE_GAZE_FIELDS = (
    "yaw_rad",
    "pitch_rad",
    "depth_m",
    "yaw_low_rad",
    "yaw_high_rad",
    "pitch_low_rad",
    "pitch_high_rad",
    "combined_gaze_valid",
    "spatial_gaze_point_valid",
    "combined_origin_x_m",
    "combined_origin_y_m",
    "combined_origin_z_m",
    "spatial_point_x_m",
    "spatial_point_y_m",
    "spatial_point_z_m",
    "left_gaze_valid",
    "right_gaze_valid",
    "left_blink_valid",
    "right_blink_valid",
    "left_blink",
    "right_blink",
    "left_yaw_rad",
    "right_yaw_rad",
    "left_pitch_rad",
    "right_pitch_rad",
    "left_yaw_low_rad",
    "right_yaw_low_rad",
    "left_yaw_high_rad",
    "right_yaw_high_rad",
    "left_eye_x_m",
    "left_eye_y_m",
    "left_eye_z_m",
    "right_eye_x_m",
    "right_eye_y_m",
    "right_eye_z_m",
)

HAND_LANDMARK_NAMES = (
    "thumb_fingertip",
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "pinky_fingertip",
    "wrist",
    "thumb_intermediate",
    "thumb_distal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_distal",
    "palm_center",
)


def _hand_fields():
    fields = []
    for side in ("left", "right"):
        fields.extend((f"{side}_present", f"{side}_confidence"))
        for landmark in HAND_LANDMARK_NAMES:
            fields.extend(f"{side}_{landmark}_{axis}_m" for axis in ("x", "y", "z"))
        fields.extend(
            (
                f"{side}_wrist_qw",
                f"{side}_wrist_qx",
                f"{side}_wrist_qy",
                f"{side}_wrist_qz",
                f"{side}_wrist_x_m",
                f"{side}_wrist_y_m",
                f"{side}_wrist_z_m",
                f"{side}_wrist_normal_x",
                f"{side}_wrist_normal_y",
                f"{side}_wrist_normal_z",
                f"{side}_palm_normal_x",
                f"{side}_palm_normal_y",
                f"{side}_palm_normal_z",
            )
        )
    return tuple(fields)


HAND_POSE_FIELDS = _hand_fields()
EYE_GAZE_INDEX = {name: index for index, name in enumerate(EYE_GAZE_FIELDS)}
HAND_POSE_INDEX = {name: index for index, name in enumerate(HAND_POSE_FIELDS)}


@dataclass(frozen=True)
class AriaUSBStatus:
    devices: tuple[str, ...]
    error: str = ""

    @property
    def ready(self) -> bool:
        return len(self.devices) == 1 and not self.error


class AriaGen2Camera:
    """Own one local Gen 2 stream and emit RGB samples in the host clock."""

    def __init__(
        self,
        sink,
        serial="",
        profile="mp_streaming_demo",
        record_vrs="",
        ip="",
    ):
        self.sink = sink
        self.client = self.device = self.receiver = None
        self.started = False
        self.capture_lock = threading.Lock()
        self.sequences = {"aria": 0, "eye_gaze": 0, "hand_pose": 0}
        self.audio = AudioBuffer()
        self.calibration_lock = threading.Lock()
        self.device_calibration = None
        try:
            import aria.sdk_gen2 as sdk
            from aria.stream_receiver import StreamReceiver

            self.client = sdk.DeviceClient()
            if ip:
                target = sdk.DeviceTarget()
                target.ip = ip
                target.serial = serial
            else:
                # Discover only the USB cable's network interface. There is no
                # fallback to Wi-Fi or to the former external Aria publisher.
                targets = self.client.usb_network_devices()
                targets = [
                    target
                    for target in targets
                    if not serial or target.serial == serial
                ]
                if len(targets) != 1:
                    raise RuntimeError(
                        f"Expected one USB Aria Gen 2 device, found {len(targets)}; "
                        "connect the glasses over USB and use --aria-serial or "
                        "--aria-ip with the glasses' USB-NCM address"
                    )
                target = targets[0]
            self.device = self.client.connect(target)
            if self.device.is_streaming():
                raise RuntimeError(
                    "Aria already streaming; stop the existing session first"
                )
            config = sdk.HttpStreamingConfig()
            config.profile_name = profile
            config.streaming_interface = sdk.StreamingInterface.USB_NCM
            config.batch_period_ms = 0
            self.device.set_streaming_config(config)
            self.device.start_streaming()
            self.started = True
            self.receiver = StreamReceiver(enable_image_decoding=True)
            server = sdk.HttpServerConfig()
            # The glasses connect over USB-NCM, so the receiver must be
            # reachable on that USB interface, not only on host loopback.
            server.address = "0.0.0.0"
            server.port = 6768
            self.receiver.set_server_config(server)
            self.receiver.set_rgb_queue_size(4)
            self.receiver.set_eye_gaze_queue_size(16)
            self.receiver.set_hand_pose_queue_size(16)
            self.receiver.register_rgb_callback(self._rgb)
            self.receiver.register_eye_gaze_callback(self._eye_gaze)
            self.receiver.register_hand_pose_callback(self._hand_pose)
            self.receiver.register_audio_callback(self._audio)
            self.receiver.register_device_calib_callback(self._device_calib)
            if record_vrs:
                self.receiver.record_to_vrs(record_vrs)
            self.receiver.start_server()
        except BaseException:
            self.close()
            raise

    def _capture_to_host(self, source_ns):
        """Return ``(host capture ns, host receive ns)``, or None until TicSync is valid."""

        # Native TicSync must not be read concurrently with device shutdown.
        with self.capture_lock:
            if not self.started or self.receiver is None:
                return None
            sync = self.receiver.get_time_sync()
            if sync is None or not sync.is_connected() or not sync.is_valid():
                return None
            latency_ms = sync.compute_latency_ms(source_ns)
            received_ns = time.perf_counter_ns()
        if not math.isfinite(latency_ms) or latency_ms < 0:
            return None
        return received_ns - round(latency_ms * 1e6), received_ns

    def _mapped_sample(self, name, source_ns, data, fields=()):
        mapped = self._capture_to_host(source_ns)
        if mapped is None:
            return
        timestamp_ns, received_ns = mapped
        self.sequences[name] += 1
        self.sink.add(
            name,
            Sample(
                timestamp_ns,
                received_ns,
                source_ns,
                self.sequences[name],
                data,
                "aria_ticsync_capture",
                fields,
            ),
        )

    def _rgb(self, image, record):
        try:
            source_ns = int(record.capture_timestamp_ns)
            pixels = np.array(image.to_numpy_array(), copy=True)
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError(
                    f"Expected decoded Gen 2 RGB image, got {pixels.shape}"
                )
            self._mapped_sample("aria", source_ns, pixels)
        except Exception as error:
            self.sink.fail(error)

    def _eye_gaze(self, gaze):
        try:
            source_ns = round(gaze.tracking_timestamp.total_seconds() * 1e9)
            vergence = gaze.vergence
            values = np.asarray(
                (
                    gaze.yaw,
                    gaze.pitch,
                    gaze.depth,
                    gaze.yaw_low,
                    gaze.yaw_high,
                    gaze.pitch_low,
                    gaze.pitch_high,
                    gaze.combined_gaze_valid,
                    gaze.spatial_gaze_point_valid,
                    *np.asarray(gaze.combined_gaze_origin_in_cpf).reshape(3),
                    *np.asarray(gaze.spatial_gaze_point_in_cpf).reshape(3),
                    vergence.left_gaze_valid,
                    vergence.right_gaze_valid,
                    vergence.left_blink_valid,
                    vergence.right_blink_valid,
                    vergence.left_blink,
                    vergence.right_blink,
                    vergence.left_yaw,
                    vergence.right_yaw,
                    vergence.left_pitch,
                    vergence.right_pitch,
                    vergence.left_yaw_low,
                    vergence.right_yaw_low,
                    vergence.left_yaw_high,
                    vergence.right_yaw_high,
                    vergence.tx_left_eye,
                    vergence.ty_left_eye,
                    vergence.tz_left_eye,
                    vergence.tx_right_eye,
                    vergence.ty_right_eye,
                    vergence.tz_right_eye,
                ),
                dtype=np.float64,
            )
            if values.shape != (len(EYE_GAZE_FIELDS),):
                raise ValueError(f"Unexpected eye gaze shape {values.shape}")
            self._mapped_sample("eye_gaze", source_ns, values, EYE_GAZE_FIELDS)
        except Exception as error:
            self.sink.fail(error)

    @staticmethod
    def _one_hand(side):
        # Presence + confidence + 21 xyz landmarks + wrist SE3 + two normals.
        values = np.full(78, np.nan, dtype=np.float64)
        values[0] = 0.0
        if side is None:
            return values
        values[0] = 1.0
        values[1] = side.confidence
        landmarks = np.asarray(side.landmark_positions_device, dtype=np.float64)
        if landmarks.shape != (21, 3):
            raise ValueError(f"Unexpected hand landmark shape {landmarks.shape}")
        values[2:65] = landmarks.reshape(-1)
        wrist_pose = np.asarray(
            side.transform_device_wrist.to_quat_and_translation(), dtype=np.float64
        ).reshape(-1)
        if wrist_pose.shape != (7,):
            raise ValueError(f"Unexpected wrist pose shape {wrist_pose.shape}")
        values[65:72] = wrist_pose
        normals = side.wrist_and_palm_normal_device
        if normals is not None:
            values[72:75] = np.asarray(normals.wrist_normal_device).reshape(3)
            values[75:78] = np.asarray(normals.palm_normal_device).reshape(3)
        return values

    def _hand_pose(self, pose):
        try:
            source_ns = round(pose.tracking_timestamp.total_seconds() * 1e9)
            values = np.concatenate(
                (self._one_hand(pose.left_hand), self._one_hand(pose.right_hand))
            )
            if values.shape != (len(HAND_POSE_FIELDS),):
                raise ValueError(f"Unexpected hand pose shape {values.shape}")
            self._mapped_sample("hand_pose", source_ns, values, HAND_POSE_FIELDS)
        except Exception as error:
            self.sink.fail(error)

    def _audio(self, audio_data, record, num_channels):
        try:
            channels = int(num_channels)
            samples = np.asarray(audio_data.data, dtype=np.int32)
            # One device capture timestamp per frame (all channels at one instant).
            source_ns = np.asarray(record.capture_timestamps_ns, dtype=np.int64)
            if not samples.size and not source_ns.size:
                return
            if channels < 1 or samples.size != source_ns.size * channels:
                raise ValueError(
                    f"Unexpected Gen 2 audio block: {samples.size} samples, "
                    f"{source_ns.size} timestamps, {channels} channels"
                )
            mapped = self._capture_to_host(int(source_ns[-1]))
            if mapped is None:
                return
            host_end_ns, received_ns = mapped
            self.audio.add(
                source_ns - source_ns[-1] + host_end_ns,
                source_ns,
                received_ns,
                # SDK delivers contiguous channel blocks, as in its audio viewer.
                samples.reshape(channels, -1).T.copy(),
                muted=bool(getattr(record, "audio_muted", False)),
                max_amplitude=float(
                    getattr(audio_data, "max_amplitude", np.iinfo(np.int32).max)
                ),
            )
        except Exception as error:
            self.sink.fail(error)

    def _device_calib(self, calibration):
        with self.calibration_lock:
            self.device_calibration = calibration

    @staticmethod
    def _pixel(camera_calibration, transform_camera_device, point_device):
        point = camera_calibration.project(transform_camera_device @ point_device)
        if point is None:
            return None
        point = np.asarray(point, dtype=np.float64).reshape(2)
        return (float(point[0]), float(point[1]))

    def project_perception_to_rgb(self, eye_gaze, hand_pose):
        """Project a synchronized numeric gaze/hand sample into RGB pixels."""

        empty = {"gaze": None, "left": [None] * 21, "right": [None] * 21}
        with self.calibration_lock:
            calibration = self.device_calibration
        if calibration is None:
            return empty
        try:
            camera = calibration.get_camera_calib("camera-rgb")
            transform_camera_device = camera.get_transform_device_camera().inverse()
            result = dict(empty)
            if eye_gaze[EYE_GAZE_INDEX["spatial_gaze_point_valid"]]:
                point_cpf = np.asarray(
                    [
                        eye_gaze[EYE_GAZE_INDEX["spatial_point_x_m"]],
                        eye_gaze[EYE_GAZE_INDEX["spatial_point_y_m"]],
                        eye_gaze[EYE_GAZE_INDEX["spatial_point_z_m"]],
                    ],
                    dtype=np.float64,
                )
                point_device = calibration.get_transform_device_cpf() @ point_cpf
                result["gaze"] = self._pixel(
                    camera, transform_camera_device, point_device
                )
            for side in ("left", "right"):
                if not hand_pose[HAND_POSE_INDEX[f"{side}_present"]]:
                    continue
                points = []
                for landmark in HAND_LANDMARK_NAMES:
                    point_device = np.asarray(
                        [
                            hand_pose[HAND_POSE_INDEX[f"{side}_{landmark}_{axis}_m"]]
                            for axis in ("x", "y", "z")
                        ],
                        dtype=np.float64,
                    )
                    points.append(
                        self._pixel(camera, transform_camera_device, point_device)
                        if np.isfinite(point_device).all()
                        else None
                    )
                result[side] = points
            return result
        except Exception:
            # Projection is a visualization aid. Capture and recording must
            # continue even if a calibration update is temporarily unavailable.
            return empty

    def close(self):
        try:
            with self.capture_lock:
                was_started = self.started
                self.started = False
            if was_started:
                self.device.stop_streaming()
                # Gen 2 can still deliver queued MP/RGB callbacks briefly after
                # stop_streaming(). Draining them before destroying the HTTP
                # receiver avoids a native SDK shutdown race.
                time.sleep(2)
        finally:
            try:
                if self.receiver is not None:
                    self.receiver.stop_server()
                    self.receiver = None
            finally:
                if self.device is not None:
                    self.client.disconnect(self.device)
                    self.device = None


def inspect_usb_devices(serial="") -> AriaUSBStatus:
    """Discover Gen 2 USB devices without starting a stream."""

    try:
        import aria.sdk_gen2 as sdk

        client = sdk.DeviceClient()
        devices = client.usb_network_devices()
        serials = tuple(str(device.serial) for device in devices)
        if serial:
            serials = tuple(item for item in serials if item == serial)
        return AriaUSBStatus(serials)
    except Exception as error:
        return AriaUSBStatus((), f"{type(error).__name__}: {error}")


def inspect_device_connection(serial="", ip="", timeout=5) -> AriaUSBStatus:
    """Verify SDK authentication without starting an Aria sensor stream."""

    if not ip:
        return inspect_usb_devices(serial)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    # DeviceClient.connect can wait in native SDK code when a remembered IP is
    # no longer present. Probe in a disposable child so preflight always has a
    # hard deadline and cannot strand the evaluator before sensor startup.
    probe = """
import sys
import aria.sdk_gen2 as sdk

client = sdk.DeviceClient()
target = sdk.DeviceTarget()
target.ip = sys.argv[1]
target.serial = sys.argv[2]
device = client.connect(target)
client.disconnect(device)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, ip, serial],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return AriaUSBStatus((serial or ip,))
        detail = result.stderr.strip().splitlines()
        error = detail[-1] if detail else f"SDK probe exited {result.returncode}"
        return AriaUSBStatus((), error)
    except subprocess.TimeoutExpired:
        return AriaUSBStatus((), f"Timed out connecting to {ip} after {timeout:g}s")
    except Exception as error:
        return AriaUSBStatus((), f"{type(error).__name__}: {error}")
