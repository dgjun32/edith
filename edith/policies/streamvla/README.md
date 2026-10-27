# streamvla

π<sub>0.5</sub> with the human's live egocentric view. Every step the VLA
receives the robot cameras and state, the gaze-cropped first-person view, and
the current instruction.

The instruction comes from speech: when the speaker pauses, the utterance is
transcribed locally with Whisper and becomes the instruction. Until someone
speaks, the preset `instruction`, `--instruction` or `--transcript-file` text
is used; with none of these, the robot holds still.

```bash
uv pip install --python .venv/bin/python -e '.[policies,speech]'
.venv/bin/python -m edith.scripts.eval_policy_local \
  --policy-dir streamvla --check-policy-only
.venv/bin/python -m edith.scripts.eval_policy_local \
  --station configs/station.local.json --policy-dir streamvla --duration 60 --headless
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `speech_to_text` | `true` | `false`: text instructions only |
| `speech_mode` | `latest` | `latest`: every utterance replaces the instruction; `first`: keep the first |
| `utterance_silence_s` | `0.8` | Pause that ends an utterance |
| `max_utterance_s` | `15` | Longer speech is transcribed in pieces |
| `whisper_model`, `whisper_device`, `whisper_language` | `small`, `auto`, `en` | faster-whisper settings |

Voice detection (Silero VAD) and Whisper run in the background and never block
`act()`. Whisper loads when the episode starts; the first run downloads the
model. With `latest`, any utterance (even an aside) replaces the instruction;
use `first` when one instruction should hold for the whole episode.

Default inference endpoint: `ws://127.0.0.1:8000`. This requires a YAM-compatible
checkpoint server even in no-CAN mode; `--check-policy-only` does not connect to
it. Override settings with `--policy-config` without changing the manifest. See
the [policy migration and usage guide](../README.md) for the observation, action,
model-metadata and dependency requirements.
