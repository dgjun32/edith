"""Append synchronized bundles to one HDF5 file with RGB JPEG payloads."""

import json
import time

import numpy as np


class BundleWriter:
    def __init__(self, path, metadata):
        import h5py

        self.h5py = h5py
        self.file = h5py.File(path, "x")
        self.file.attrs["schema"] = "edith.yam.aria_gen2.v2"
        self.file.attrs["metadata"] = json.dumps(metadata)
        self.file.attrs["created_unix_ns"] = time.time_ns()
        self.file.attrs["complete_rows"] = 0
        self.count = 0

    def append(self, bundle):
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
        self.count += 1
        self.file.attrs.modify("complete_rows", self.count)
        self.file.flush()

    def close(self):
        self.file.close()
