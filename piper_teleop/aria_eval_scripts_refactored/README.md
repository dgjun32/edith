> **Gen 2 / YAM migration:** supported directory-policy adapters are in
> [`policies/`](../../policies/README.md). The scripts below are historical
> references with their original hardware-owning evaluation loops. Use
> `eval_policy_local --policy-dir ...` for the synchronized local runtime.

# aria_eval_scripts_final — Plan-A patches & usage

This directory hosts the real-time robot-evaluation scripts that run the
Piper dual-arm robot with Aria glasses as the human-side input. The
scripts come in several variants that combine a VLM (Gemini) front-end
with a VLA (openpi) back-end. Below is a summary of:

1. **What changed in the most recent “Plan A” patch round**
2. **How to use each entry-point script**
3. **Where the VLM debug videos land** (so you can verify gaze / Whisper
   captions are actually being overlaid)

All work is against the 8 “fat” evaluation scripts plus the two
`*_while_episode.py` variants that live in the same directory. A small
`common/` package now holds shared helpers so the behavioural fixes can
be imported rather than copy-pasted.

---

## 1. What changed in this round

### 1.1 Background observation streamer

**Why**: Previously the `VideoBuffer` was only written inside `eval()`
when a) `env.step()` executed or b) the pre-episode keyframe loop
captured a frame. `env.step()` does NOT run at a fixed 30 Hz during
inference cycles, so the *async* VLM segments (call #2 onwards) were
undersampled and sometimes shorter than the advertised
`VLM_INTERVAL_SEC`. That showed up as <8 s videos in `temp/` and
generally bad temporal sampling going into Gemini.

**Fix**: A new daemon thread — `BackgroundObservationStreamer` — polls
`env.get_observation()` at a steady `_VIDEO_RECORD_FPS` (default 30) and
pushes frames into the shared `VideoBuffer`. It runs completely
independently of `env.step()` so the buffer fills uniformly regardless
of policy-inference timing.

The streamer lives in `common/streamer.py` and is instantiated at the
start of every `eval()`. All in-loop `video_buffer.append(...)` calls
have been removed from `eval()`; the streamer is now the sole writer
(see inline `# NOTE: streamer feeds video_buffer in background` comments).

The streamer takes a pluggable `frame_extractor(obs) -> dict`:

| Script | Extractor |
| --- | --- |
| `eval_openpi_ours_with_voice.py` | `_default_frame_extractor` (aria + cropped) |
| `eval_openpi_ours_without_keyframe_with_voice_*.py` | `_default_frame_extractor` (aria + cropped) |
| `eval_openpi_ours_without_egoview_with_voice.py` | `_egoview_plus_top_frame_extractor` (aria + cropped + top-view) |
| `eval_openpi_vlm_textvla_with_voice_*.py` | `_top_view_frame_extractor` (top-view only) |

### 1.2 Hold-pose instead of `env.reset()` on empty queue

**Why**: Previously, when the `KeyframeQueue` was exhausted (either via
manual `'n'` advance, automatic `completion_prob ≥ 0.8` advance, or the
top-of-loop check), the eval loop called `env.reset()` + reconnected the
policy + returned. That killed the episode every time the VLA finished
its current subtask and no new one had been appended yet, even though
the async VLM worker was about to append one.

**Fix**: On every code path that used to reset + end, we now call
`wait_for_queue_items(queue, …)` from `common/streamer.py`. That helper:

- Does **not** step or reset the robot (arm holds its last commanded
  pose).
- Keeps the shared preview window alive so the operator can see the
  scene.
- Polls `queue.is_done` until either a new subtask appears (`return
  False` → outer loop resumes) or the user presses `q` / ESC
  (`return True` → outer loop raises `KeyboardInterrupt`).
- Logs a reminder every 5 s that it is still waiting.

This is applied in three places in each `eval()`:

1. The top-of-loop `if queue.is_done:` check.
2. The manual `'n'` advance branch when `queue.advance()` fails.
3. The auto-advance branch when `queue.advance()` fails after a
   high-completion-probability trigger.

### 1.3 VLM-debug video overlays

**Why**: The saved `temp/*.mp4` files were ambiguous — it was hard to
tell whether gaze crosshairs or Whisper captions had actually been
drawn on the frames that went to Gemini.

**Fix** (in `_prepare_vlm_video`):

- `_draw_gaze_crosshair` now draws a **thick green crosshair with a red
  outer ring and a yellow “GAZE” label** directly beneath the marker.
  On any reasonable frame size this is immediately visible.
- Every frame that leaves `_prepare_vlm_video` also gets a thin black
  banner across the top:
  ```
  VLM INPUT | src=<source> | cropped=<bool> | clip_start=<sec> | frame=<idx>
  ```
  so you can tell at a glance which video source was used, whether the
  frames were gaze-cropped, and which frame of the clip you are
  looking at.
- Whisper captions (when speech is detected) continue to be burned in
  by `_burn_captions_on_video_bytes`, unchanged from before.

### 1.4 Per-file prompt fixes (previous patch round)

- `eval_openpi_vlm_textvla_with_voice_*.py`: all “first-person
  (egocentric)” / “wearer’s viewpoint” language in the initial and
  async-update prompts has been rewritten to reflect that the video is
  a **robot top-view (overhead)** camera feed.
- `eval_openpi_ours_without_egoview_with_voice.py`: the return-JSON
  schema now starts with `human_intent`, matching
  `eval_openpi_ours_with_voice.py`.

### 1.5 `*_while_episode.py`: press-to-talk Whisper

- `eval_openpi_streamvla_with_voice_while_episode.py`,
  `eval_openpi_textvla_with_voice_while_episode.py`: the 3-second
  timer-driven `VoiceMonitor` has been replaced with an explicit
  **press-`p`-to-start / press-`p`-to-stop** recorder. Audio is only
  buffered while recording, and transcription runs in a background
  thread on stop so the eval loop never blocks. A `[REC ●]` indicator
  in the preview makes the recording state obvious.

---

## 2. File map & usage

All commands assume you are in the `piper_310` conda env with
`GEMINI_API_KEY` and `OPENAI_API_KEY` exported. The policy server must
already be running on `localhost:8000`.

### 2.1 `eval_openpi_ours_with_voice.py` — canonical “Ours”

- **VLM input**: `aria_cropped_rgb` (gaze-centred crop of the Aria
  egocentric feed)
- **VLA input**: top-view + wrist cameras + `human_view_keyframe_cropped`
  + text prompt from queue
- **Run**:

      python eval_openpi_ours_with_voice.py

  Extra: `--no-vlm-draw-gaze-marker` disables the centre-gaze marker.

### 2.2 `eval_openpi_ours_without_egoview_with_voice.py`

- Same as #2.1 but the VLM consumes the **robot top-view** camera
  instead of the egocentric Aria feed. VLA input is unchanged.
- **Run**:

      python eval_openpi_ours_without_egoview_with_voice.py

### 2.3 `eval_openpi_ours_without_keyframe_with_voice_{passing_tools, serving_muffin, sorting_tumbler}.py`

- VLM consumes the egocentric cropped Aria video like #2.1.
- VLA consumes **only text instructions** from the queue (no keyframe
  image passed).
- Prompt templates use a closed vocabulary specific to each task
  (`passing_tools`, `serving_muffin`, `sorting_tumbler`).
- **Run**:

      python eval_openpi_ours_without_keyframe_with_voice_passing_tools.py
      python eval_openpi_ours_without_keyframe_with_voice_serving_muffin.py
      python eval_openpi_ours_without_keyframe_with_voice_sorting_tumbler.py

### 2.4 `eval_openpi_vlm_textvla_with_voice_{passing_tools, serving_muffin, sorting_tumbler}.py`

- Same prompt structure as #2.3 but the VLM input is the **robot
  top-view** camera (overhead).
- VLA consumes only text instructions (same as #2.3).
- **Run**:

      python eval_openpi_vlm_textvla_with_voice_passing_tools.py
      python eval_openpi_vlm_textvla_with_voice_serving_muffin.py
      python eval_openpi_vlm_textvla_with_voice_sorting_tumbler.py

### 2.5 `eval_openpi_streamvla_with_voice*.py`

- VLM-less StreamVLA baseline. The task-specific launchers lock the task
  preset, and `--instruction` only overrides the language string.
- Default mode captures voice once before the episode starts. Press `p` to
  transcribe, `s` to keep the preset/CLI instruction, or `q` to quit.
- Preset-language mode skips voice entirely. Use `--preset-language`
  or the alias `--no-voice`. After launch, press `s` to start an episode,
  `e` to end only the current episode and return to the start screen, or
  `q` to quit the whole program.

Preset-language commands:

```bash
python eval_openpi_streamvla_with_voice_muffin.py --preset-language
python eval_openpi_streamvla_with_voice_bottle.py --preset-language
python eval_openpi_streamvla_with_voice_passing_tools.py --preset-language
```

Preset-language commands with explicit instruction override:

```bash
python eval_openpi_streamvla_with_voice_muffin.py --preset-language --instruction "Give me this muffin, this muffin, and this muffin."
python eval_openpi_streamvla_with_voice_bottle.py --preset-language --instruction "Put all the beverages to the yellow basket."
python eval_openpi_streamvla_with_voice_passing_tools.py --preset-language --instruction "Give me the hex screwdriver and 12 cm metal support."
```

Voice-enabled task launchers:

```bash
python eval_openpi_streamvla_with_voice_muffin.py
python eval_openpi_streamvla_with_voice_bottle.py
python eval_openpi_streamvla_with_voice_passing_tools.py
```

Generic runner with task selection:

```bash
python eval_openpi_streamvla_with_voice.py --task muffin
python eval_openpi_streamvla_with_voice.py --task bottle --preset-language
python eval_openpi_streamvla_with_voice.py --task passing_tools --preset-language
```

The current preset instructions are:

| Task | Instruction |
| --- | --- |
| `muffin` | `Give me this muffin, this muffin, and this muffin.` |
| `bottle` | `Put all the beverages to the yellow basket.` |
| `passing_tools` | `Give me the hex screwdriver and 12 cm metal support.` |

### 2.5b `eval_openpi_textvla_with_voice.py`

- VLM-less TextVLA baseline. Voice instruction is captured **once** before
  the episode starts.
- **Run**: `python eval_openpi_textvla_with_voice.py`.

### 2.6 `eval_openpi_streamvla_with_voice_while_episode.py` & `eval_openpi_textvla_with_voice_while_episode.py`

- Same architecture as #2.5 but the text instruction can be replaced
  **during the episode** using press-to-talk Whisper:

  1. While `eval()` is running and the preview window is focused,
     press **`p`** to start recording.
  2. Speak the new instruction.
  3. Press **`p`** again to stop; Whisper runs in a background thread
     and the VLA prompt is swapped in as soon as the transcript is
     available.
  4. `q` / ESC aborts the episode.

  The preview shows `[REC ●]` while recording.
- **Run**: `python eval_openpi_streamvla_with_voice_while_episode.py`
  or `python eval_openpi_textvla_with_voice_while_episode.py`.

### 2.7 Interactive keys common to every script

- `p`  – (keyframe selector only) commit the recorded clip to Gemini.
- `n`  – manually advance to the next keyframe in the queue. If the
  queue is empty, the robot now holds its pose.
- `q` / `Esc` – abort the current episode.

---

## 3. Where to look when debugging VLM input

All scripts save their artefacts under `_TEMP_DIR`
(default `/home/alin/projects/piper_project/temp`).

| File | What it is |
| --- | --- |
| `video.mp4` | Raw Aria video captured during **keyframe selection** (pre-episode). Audio muxed. |
| `cropped_video.mp4` | Aria `aria_cropped_rgb` variant of the keyframe-selection recording. |
| `initial_vlm_input_<source>.mp4` | **Exact bytes sent to Gemini** for the initial keyframe extraction. Has VLM-INPUT banner, gaze crosshair (if `frames_are_gaze_cropped`), and burned-in Whisper captions. |
| `async_seg_<NNNN>_<source>.mp4` | **Raw** frames that fed the Nth async VLM call, *before* gaze crosshair / banner / captions are applied. Useful as a "ground truth" reference. |
| `vlm_input_<NNNN>_<source>.mp4` | **Exact bytes sent to Gemini** for the Nth async call. Banner + crosshair + captions as above. Compare against `async_seg_` to see what the overlay added. |
| `keyframe_simulation.mp4` | Side-by-side visualisation of the queue state as the episode progresses. |
| `rollout.mp4` | 3-row composite: current keyframe / top-view with VLM events / completion-probability plot. |
| `keyframes.json` | Queue snapshot after the initial extraction. |
| `gaze.json` | Gaze points recorded during keyframe selection. |
| `test_*.png` | One-off debug stills saved at `global_step == 0` to verify the observation feed. |

### Verifying gaze crosshair is actually drawn

Open any `vlm_input_*.mp4` or `initial_vlm_input_*.mp4` from `temp/`.
With the Plan-A overlays you should now see:

- A **thick green crosshair** with a **red outer ring** dead-centre.
- A small **“GAZE”** text label in yellow just below the crosshair.
- A thin **black banner** across the top of every frame with:
  `VLM INPUT | src=<source> | cropped=True | clip_start=<sec> | frame=<idx>`.

If `cropped=False` in the banner, the gaze crosshair is intentionally
**not drawn** because the frames were not gaze-centred crops.

### Verifying Whisper captions

Captions are burned in by `_burn_captions_on_video_bytes` *only when
Whisper detects non-empty speech* in the clip's audio track. If a
segment has no speech, there will be no caption — that is expected. The
console log prints the transcribed text with a `[Whisper] "..."` prefix
right before captions are burned.

---

## 4. `common/` package

```
common/
├── __init__.py           # re-exports streamer helpers
└── streamer.py           # BackgroundObservationStreamer,
                          # wait_for_queue_items,
                          # _default_frame_extractor,
                          # _top_view_frame_extractor
```

Each eval script now does:

```python
from common.streamer import (
    BackgroundObservationStreamer,
    wait_for_queue_items,
)
```

…and, inside `eval()`, replaces the removed in-loop appends with one
`BackgroundObservationStreamer(...).start()` + a matching `stop()` in
the exception / clean-exit paths.

---

## 5. Known limitations / follow-ups

- **`_prepare_vlm_video` still subsamples frames to `VLM_FPS`**
  (default 4 FPS) before encoding. If you want the saved mp4 to match
  the *raw* 30 FPS capture, change `VLM_FPS` to `_VIDEO_RECORD_FPS`.
- **`extract_keyframes` still feeds the buffer synchronously** during
  the pre-episode `'p'` capture. That is intentional — the streamer
  only runs inside `eval()`. The pre-episode buffer is also discarded
  by `video_buffer.skip_to_current()` right before the streamer takes
  over, so there is no overlap.
- The per-task scripts (`ours_without_keyframe_*`, `vlm_textvla_*`)
  still duplicate ~90% of the code from `ours_with_voice.py`. A proper
  refactor into a `common/core.py` was deferred (“Plan B”). The current
  structure is stable; bug fixes only need to be replayed across the
  eight files.
