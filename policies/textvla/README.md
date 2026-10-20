# textvla

See the [policy migration and usage guide](../README.md) for the observation,
action, model-metadata and dependency requirements.

```bash
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --policy-dir policies/textvla --check-policy-only
.venv/bin/python -m piper_teleop.scripts.eval_policy_local \
  --policy-dir policies/textvla \
  --policy-config policies/textvla/presets/passing_tools.json \
  --duration 60 --headless
```

Default inference endpoint: `ws://127.0.0.1:8000`. This requires a YAM-compatible
checkpoint server even in no-CAN mode. `--check-policy-only` does not connect to
it. Override settings with `--policy-config` without changing the manifest.
