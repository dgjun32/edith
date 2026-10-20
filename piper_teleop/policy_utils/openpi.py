"""Finite-timeout OpenPI wire client; no dependency on a robot or camera driver.

Wire keys follow Physical-Intelligence/openpi's msgpack_numpy protocol.
The upstream openpi-client 0.1.2 pins NumPy <2, whereas this station's i2rt
requires NumPy 2.2.6, so only the protocol is implemented here.
"""

from collections.abc import Mapping
import os
import threading
from urllib.parse import urlsplit

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError(f"Unsupported inference dtype: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"dtype": value.dtype.str,
            b"shape": value.shape,
            b"data": value.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"dtype": value.dtype.str,
            b"data": value.item(),
        }
    raise TypeError(f"Cannot pack {type(value).__name__}")


def _decode(value):
    if value.get(b"__ndarray__") is True:
        dtype = np.dtype(value[b"dtype"])
        shape = tuple(value[b"shape"])
        if dtype.kind not in "biuf" or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid ndarray in policy response")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(shape).copy()
    if value.get(b"__npgeneric__") is True:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Invalid scalar in policy response")
        return dtype.type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(value):
    if not isinstance(value, bytes):
        raise RuntimeError(f"Inference server returned an error: {str(value)[:300]}")
    result = msgpack.unpackb(value, object_hook=_decode, raw=False)
    if not isinstance(result, Mapping):
        raise ValueError("Inference server must return a mapping")
    return result


class OpenPIClient:
    def __init__(self, config):
        self.url = config.get("server_url", "ws://127.0.0.1:8000")
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError("server_url must be a ws:// or wss:// URL")
        self.timeout = float(config.get("timeout_s", 10))
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.expected = config.get(
            "expected_metadata",
            {
                "robot_type": "yam",
                "action_space": "joint_position",
                "arm_order": ["left", "right"],
            },
        )
        if self.expected != {
            "robot_type": "yam",
            "action_space": "joint_position",
            "arm_order": ["left", "right"],
        }:
            raise ValueError(
                "These policies require YAM joint_position metadata in left/right order"
            )
        self.api_key_env = config.get("api_key_env", "OPENPI_API_KEY")
        self.connection = None
        self.lock = threading.Lock()
        self.closed = False

    def _connect(self):
        from websockets.sync.client import connect

        token = os.environ.get(self.api_key_env)
        headers = {"Authorization": f"Api-Key {token}"} if token else None
        conn = connect(
            self.url,
            open_timeout=self.timeout,
            close_timeout=1,
            compression=None,
            max_size=64 * 1024 * 1024,
            additional_headers=headers,
        )
        with self.lock:
            if self.closed:
                conn.close()
                raise RuntimeError("Policy client is closed")
            self.connection = conn
        try:
            metadata = unpack(conn.recv(timeout=self.timeout))
            for key, expected in self.expected.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"Checkpoint metadata {key!r} must be {expected!r}; got {metadata.get(key)!r}. "
                        "Use a YAM-trained checkpoint and matching server metadata."
                    )
            return conn
        except BaseException:
            conn.close()
            self.connection = None
            raise

    def infer(self, observation):
        if self.closed:
            raise RuntimeError("Policy client is closed")
        conn = self.connection if self.connection is not None else self._connect()
        try:
            conn.send(pack(observation))
            return unpack(conn.recv(timeout=self.timeout))
        except BaseException:
            conn.close()
            self.connection = None
            raise

    def close(self):
        with self.lock:
            self.closed = True
            conn, self.connection = self.connection, None
        if conn is not None:
            conn.close()
