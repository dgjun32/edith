"""Policy-directory loader.

A policy deployment consists of one directory containing ``policy.json``, its
entrypoint, checkpoints, normalizers, and any policy-specific Python modules.
Nothing in the evaluation runner knows the policy class or checkpoint layout.
"""

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


@dataclass
class LoadedPolicy:
    root: Path
    manifest: dict
    instance: object

    @property
    def control_hz(self) -> float:
        return float(self.manifest.get("control_hz", 30.0))

    def reset(self, observation: dict) -> None:
        reset = getattr(self.instance, "reset", None)
        if reset is not None:
            reset(observation)

    def act(self, observation: dict) -> np.ndarray:
        action = np.asarray(self.instance.act(observation), dtype=np.float64)
        return action.reshape(-1)

    def close(self) -> None:
        close = getattr(self.instance, "close", None)
        if close is not None:
            close()


def _entrypoint(root: Path, value: str):
    filename, separator, symbol = value.partition(":")
    if not separator or not filename or not symbol:
        raise ValueError("Policy entrypoint must have the form 'file.py:ClassName'")
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Policy entrypoint must stay inside the policy directory") from error
    if not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(f"Policy entrypoint not found: {path}")
    module_name = "edith_policy_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import policy entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Allow policy-local helper imports. The complete deployment still lives in
    # one directory, while the runner remains policy-agnostic.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec.loader.exec_module(module)
    try:
        return getattr(module, symbol)
    except AttributeError as error:
        raise ImportError(f"Policy symbol {symbol!r} not found in {path}") from error


def load_policy(policy_dir) -> LoadedPolicy:
    root = Path(policy_dir).expanduser().resolve()
    manifest_path = root / "policy.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing policy manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "edith.policy.v1":
        raise ValueError("policy.json schema must be 'edith.policy.v1'")
    policy_type = _entrypoint(root, manifest.get("entrypoint", ""))
    instance = policy_type(policy_dir=root, config=manifest.get("config", {}))
    if not callable(getattr(instance, "act", None)):
        raise TypeError("Policy entrypoint instance must define act(observation)")
    control_hz = float(manifest.get("control_hz", 30.0))
    if not 0 < control_hz <= 500:
        raise ValueError("Policy control_hz must be in (0, 500]")
    return LoadedPolicy(root, manifest, instance)
