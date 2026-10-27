"""Append synchronized bundles to one HDF5 file with RGB JPEG payloads."""

import json
import time

import numpy as np


class BundleWriter:
    def __init__(self, path, metadata):
        import h5py

        self.h5py = h5py
        self.file = h5py.File(path, "x")
        self.file.attrs["schema"] = "edith.observations.v3"
        self.file.attrs["metadata"] = json.dumps(metadata)
        self.file.attrs["created_unix_ns"] = time.time_ns()
        self.file.attrs["complete_rows"] = 0
        self.count = 0

    def append(self, bundle, observation=None, action=None):
        import cv2

        for name, sample in bundle.items():
            group = self.file.require_group(name)
            for key in (
                "timestamp_ns",
                "received_ns",
                "source_timestamp_ns",
                "sequence",
            ):
                if key not in group:
                    group.create_dataset(
                        key, (0,), maxshape=(None,), dtype="i8", chunks=True
                    )
                dataset = group[key]
                dataset.resize(self.count + 1, axis=0)
                dataset[self.count] = getattr(sample, key)
            group.attrs["clock"] = sample.clock
            data = sample.data
            if sample.fields:
                if data.ndim != 1 or len(sample.fields) != len(data):
                    raise ValueError(f"Field schema does not match {name} data")
                encoded_fields = json.dumps(sample.fields)
                existing_fields = group.attrs.get("value_fields", encoded_fields)
                if existing_fields != encoded_fields:
                    raise ValueError(f"Field schema changed for {name}")
                group.attrs["value_fields"] = encoded_fields
            if data.ndim == 3:
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                )
                if not ok:
                    raise RuntimeError(f"JPEG encode failed: {name}")
                if "rgb_jpeg" not in group:
                    group.create_dataset(
                        "rgb_jpeg",
                        (0,),
                        maxshape=(None,),
                        dtype=self.h5py.vlen_dtype(np.dtype("uint8")),
                    )
                    group.attrs["image_shape"] = data.shape
                dataset = group["rgb_jpeg"]
                dataset.resize(self.count + 1, axis=0)
                dataset[self.count] = encoded.reshape(-1)
            elif data.ndim == 2:
                self._append_audio_timing(group, sample)
                # Audio rows hold a variable number of (frames, channels) PCM
                # frames, stored flattened in frame-major order.
                if "pcm" not in group:
                    group.create_dataset(
                        "pcm",
                        (0,),
                        maxshape=(None,),
                        dtype=self.h5py.vlen_dtype(np.dtype("int32")),
                    )
                if len(data):
                    if group.attrs.get("num_channels", data.shape[1]) != data.shape[1]:
                        raise ValueError(f"Channel count changed for {name}")
                    group.attrs["num_channels"] = data.shape[1]
                dataset = group["pcm"]
                dataset.resize(self.count + 1, axis=0)
                dataset[self.count] = np.asarray(data, dtype=np.int32).reshape(-1)
            else:
                if "values" not in group:
                    group.create_dataset(
                        "values",
                        (0, len(data)),
                        maxshape=(None, len(data)),
                        dtype="f8",
                        chunks=True,
                    )
                dataset = group["values"]
                dataset.resize(self.count + 1, axis=0)
                dataset[self.count] = data
        if observation is not None:
            self._append_policy_observation(observation, action)
        self.count += 1
        self.file.attrs.modify("complete_rows", self.count)
        self.file.flush()

    def _append_policy_observation(self, observation, action):
        group = self.file.require_group("policy_observation")
        arrays = {
            "robot_state": observation["robot_state"],
            "gaze_point": observation["gaze_point"],
            "left_hand_pixels": observation["hand_keypoints"]["left"].reshape(-1),
            "right_hand_pixels": observation["hand_keypoints"]["right"].reshape(-1),
            "action": action,
        }
        for name, value in arrays.items():
            value = np.asarray(value, dtype=np.float64)
            if name not in group:
                group.create_dataset(
                    name,
                    (0, value.size),
                    maxshape=(None, value.size),
                    dtype="f8",
                    chunks=True,
                )
            dataset = group[name]
            dataset.resize(self.count + 1, axis=0)
            dataset[self.count] = value.reshape(-1)
        text = {
            key: observation[key]
            for key in (
                "instruction",
                "transcript",
                "policy_event",
                "arm_order",
                "capture_timestamps_ns",
                "source_timestamps_ns",
                "sequence_numbers",
            )
            if key in observation
        }
        text["image_keys"] = list(observation["images"])
        text["audio_present"] = observation.get("audio") is not None
        if "metadata_json" not in group:
            group.create_dataset(
                "metadata_json", (0,), maxshape=(None,), dtype=self.h5py.string_dtype()
            )
        dataset = group["metadata_json"]
        dataset.resize(self.count + 1, axis=0)
        dataset[self.count] = json.dumps(text)

    def _append_audio_timing(self, group, sample):
        metadata = sample.metadata or {}
        for field, name, dtype in (
            ("capture_timestamps_ns", "frame_capture_timestamps_ns", "int64"),
            ("source_timestamps_ns", "frame_source_timestamps_ns", "int64"),
            ("muted", "frame_muted", "uint8"),
        ):
            if name not in group:
                group.create_dataset(
                    name,
                    (0,),
                    maxshape=(None,),
                    dtype=self.h5py.vlen_dtype(np.dtype(dtype)),
                )
            dataset = group[name]
            dataset.resize(self.count + 1, axis=0)
            values = np.asarray(metadata.get(field, []), dtype=dtype)
            if metadata and len(values) != len(sample.data):
                raise ValueError("Audio metadata length does not match PCM")
            dataset[self.count] = values
        for key in ("sample_rate_hz", "max_amplitude", "dropped_samples"):
            if key in metadata:
                group.attrs[key] = metadata[key]
        group.attrs["pcm_layout"] = "frames_by_channels"

    def close(self):
        self.file.close()
