import numpy as np
import cv2
import os
import json

import imageio
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

from edith.legacy.envs.piper_env import PiperRobotEnv
from edith.legacy.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from eval_vlm_baseline import PROMPT_CONFIG_NOTES


_VIDEO_RECORD_FPS = 30
_MAX_INTERVALS = 8
_PROMPT_CONFIG_NAME = "no_cap_no_gaze"

PROMPT_TEMPLATE_MULTI = """\
You are analyzing a top-view (overhead) video of a table recorded by a
stationary camera mounted above the workspace. The video shows objects
resting on the table surface.

A spoken instruction was given:
    "{transcript}"

{config_note}

Your task has two steps:

**Step 1: Determine the number of keyframes.**
Analyze the spoken instruction and count the number of distinct object
references. Each time the instruction refers to a target object (by
pointing words like "this", "that", or by naming it), it corresponds to
one keyframe — the frame that most clearly shows that object on the table.

Examples:
    - "I want this cupcake" → 1 keyframe
        subgoal_instructions: ["pick up this cupcake"]
    - "give me this cupcake and this beverage" → 2 keyframes
        subgoal_instructions: ["pick up this cupcake", "pick up this beverage"]
    - "put that one in this basket" → 2 keyframes
        subgoal_instructions: ["pick up that one", "place it in this basket"]
    - "put that one in this basket and the coca cola in this basket"
        → 4 keyframes
        subgoal_instructions: ["pick up that one", "place it in this basket",
            "pick up the coca cola", "place it in this basket"]
    - "Give me that soda" → 1 keyframe
        subgoal_instructions: ["pick up that soda"]

IMPORTANT: Do NOT use any other verbs (e.g., "point to", "grab", "put", "get",
    "move", "take"). Always rephrase to "pick up" or "place it".

**Step 2: Identify each keyframe's time range.**
For each keyframe, identify the short time interval where a person's hand
(or finger) is clearly pointing at or hovering over the target object, as
seen from the overhead camera.

Selection criteria (in order of priority):
    1. Hand pointing: the person's hand/finger is visibly directed at or
       positioned over the target object.
    2. Object visibility: the target object is clearly visible and unoccluded
       beneath or near the pointing hand.
    3. Gesture stability: the hand has settled on the target (not mid-motion
       between two locations).

Each interval should be short and precise, typically 0.2–1.5 seconds.
Do not merge two separate pointing gestures into one interval.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
    "reasoning": "<brief explanation of how you identified each object and selected the interval>",
    "num_keyframes": <int, number of distinct object references>,
    "object_references": ["<object 1>", "<object 2>", ...],
    "subgoal_instructions": ["pick up <object>", "place it <destination>", ...],
    "keyframe_intervals": [
        {{"start": <float, seconds>, "end": <float, seconds>}}
    ]
}}

Important:
    - All lists (object_references, subgoal_instructions, keyframe_intervals)
        must have the same length, equal to num_keyframes.
    - Maximum {max_intervals} intervals allowed.
    - Intervals must be sorted chronologically.
    - Intervals should be non-overlapping.
    - Always return positive start/end values.
    - Never return -1.
"""


def _write_video(frames_bgr, output_path, fps=_VIDEO_RECORD_FPS):
        if not frames_bgr:
                return
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        imageio.mimwrite(output_path, frames_rgb, fps=fps)


def _extract_json_text(raw_text: str) -> str:
        text = (raw_text or "").strip()
        if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                        text = text[4:].strip()
        return text


def _query_gemini_multi_keyframes(video_path: str, instruction: str, api_key: str, num_frames: int):
        duration_sec = num_frames / _VIDEO_RECORD_FPS
        prompt = PROMPT_TEMPLATE_MULTI.format(
                transcript=instruction,
                duration=duration_sec,
                fps=float(_VIDEO_RECORD_FPS),
        config_note=PROMPT_CONFIG_NOTES[_PROMPT_CONFIG_NAME],
                max_intervals=_MAX_INTERVALS,
        )

        with open(video_path, "rb") as file:
                video_bytes = file.read()

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=types.Content(parts=[
                        types.Part(
                                inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                                video_metadata=types.VideoMetadata(fps=4),
                        ),
                        types.Part(text=prompt),
                ]),
                config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                ),
        )

        raw_text = response.text
        parsed = json.loads(_extract_json_text(raw_text))
        intervals = parsed.get("keyframe_intervals", [])
        frame_indices = []
        for interval in intervals[:_MAX_INTERVALS]:
                start = float(interval.get("start", 0.0))
                end = float(interval.get("end", start))
                if start > end:
                        start, end = end, start
                mid_sec = (start + end) / 2.0
                frame_idx = int(round(mid_sec * _VIDEO_RECORD_FPS))
                frame_idx = max(0, min(frame_idx, num_frames - 1))
                frame_indices.append(frame_idx)

        frame_indices = sorted(set(frame_indices))
        if not frame_indices:
                frame_indices = [num_frames // 2]
        return frame_indices, parsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _build_preview_frame(obs, status_text, status_color=(0, 255, 0)):
    images = obs["images"]
    preview = images["top_camera_rgb"].copy()

    cv2.putText(preview, "top view", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, status_text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    return preview


def _build_eval_preview(obs, keyframe_img, keyframe_idx, total_keyframes, current_instruction=None):
    """Live view + current keyframe thumbnail shown during rollout."""
    human_view = obs["images"]["aria_rgb"].copy()
    kf_display = cv2.resize(keyframe_img,
                            (human_view.shape[1], human_view.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    preview = np.hstack((human_view, kf_display))
    cv2.putText(preview,
                f"LIVE | KEYFRAME [{keyframe_idx + 1}/{total_keyframes}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "n=next keyframe  q=quit",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if current_instruction:
        cv2.putText(preview, f"subtask: {current_instruction[:80]}",
                    (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Multiple keyframe selection
# ---------------------------------------------------------------------------

def extract_keyframes(env, language_instruction: str, api_key: str):
    """
    1. Show live camera feed (aria | top side-by-side); press 'p' to capture all frames up to that point.
    2. Save video.mp4, cropped_video.mp4, gaze.json to temp/.
    3. Query Gemini VLM (multi-keyframe prompt) with video.mp4.
    4. Extract keyframes from aria_frames and aria_cropped_frames.
    5. Save keyframe_{i}.png and cropped_keyframe_{i}.png to temp/.
    6. Return (list_of_cropped_keyframes_bgr, list_of_subtask_instructions).
    """
    window_name = "Keyframe Selector"
    temp_dir = os.path.normpath("/home/alin/projects/piper_project/temp")
    os.makedirs(temp_dir, exist_ok=True)

    video_path         = os.path.join(temp_dir, "video.mp4")
    keyframe_meta_path = os.path.join(temp_dir, "keyframes.json")

    top_frames = []

    print("Keyframe selection: press 'p' to capture & query VLM, 'q' to quit.")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation())
            live_frame = _build_preview_frame(live_obs, "LIVE: p=save&proceed q=quit", (0, 255, 0))

            # --- accumulate top-view frames ---
            top_frames.append(live_obs["images"]["top_camera_rgb"].copy())

            cv2.imshow(window_name, live_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('p'), ord('P')) and len(top_frames) > 0:
                total = len(top_frames)
                print(f"[keyframe] Captured {total} frames @ {_VIDEO_RECORD_FPS} FPS "
                      f"({total / _VIDEO_RECORD_FPS:.2f}s)")

                _write_video(top_frames, video_path)
                print(f"[keyframe] Saved video → {video_path}")

                print("[keyframe] Querying Gemini for multiple keyframes...")
                frame_indices, parsed_response = _query_gemini_multi_keyframes(
                    video_path=video_path,
                    instruction=language_instruction,
                    api_key=api_key,
                    num_frames=total,
                )

                keyframes = []
                saved_items = []
                for idx, frame_idx in enumerate(frame_indices, start=1):
                    # Save top-view keyframe
                    top_bgr = top_frames[frame_idx].copy()
                    kf_path = os.path.join(temp_dir, f"keyframe_{idx:02d}.png")
                    plt.imsave(kf_path, cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB))
                    print(f"[keyframe] Saved keyframe #{idx} (frame {frame_idx}) → {kf_path}")

                    keyframes.append(top_bgr)
                    saved_items.append({
                        "order": idx,
                        "frame_idx": int(frame_idx),
                        "keyframe_path": kf_path,
                    })

                subtask_instructions = parsed_response.get("subgoal_instructions", [])
                if not isinstance(subtask_instructions, list):
                    subtask_instructions = []
                subtask_instructions = [str(item).strip() for item in subtask_instructions if str(item).strip()]
                if len(subtask_instructions) < len(keyframes):
                    subtask_instructions.extend(
                        [language_instruction] * (len(keyframes) - len(subtask_instructions))
                    )
                elif len(subtask_instructions) > len(keyframes):
                    subtask_instructions = subtask_instructions[:len(keyframes)]

                with open(keyframe_meta_path, "w") as file:
                    json.dump(
                        {
                            "instruction": language_instruction,
                            "num_frames": total,
                            "video_fps": _VIDEO_RECORD_FPS,
                            "frame_indices": [int(i) for i in frame_indices],
                            "subtask_instructions": subtask_instructions,
                            "saved_keyframes": saved_items,
                            "gemini_response": parsed_response,
                            "source": "top_camera_rgb",
                        },
                        file,
                        indent=2,
                    )
                print(f"[keyframe] Saved metadata → {keyframe_meta_path}")
                print(f"[keyframe] n_keyframes: {len(keyframes)}")
                print(f"[keyframe] subtask instructions ({len(subtask_instructions)}):")
                for idx, subtask in enumerate(subtask_instructions, start=1):
                    print(f"  [{idx}] {subtask}")

                return keyframes, subtask_instructions

            elif key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def eval(env, client, language_instruction, horizon, keyframes, subtask_instructions):
    """
    Run the policy for up to `horizon` steps.

    keyframes : list of np.ndarray  (aria_rgb crop, BGR)
        The policy receives keyframes[current_keyframe_idx] at every step.

    Keyboard during rollout (OpenCV preview window):
      n  - advance to next keyframe; if none left → reset robot and return
      q  - raise KeyboardInterrupt (stops the episode)

    Returns the (possibly updated) language_instruction.
    """
    if not keyframes:
        print("No keyframes provided.")
        return language_instruction

    current_keyframe_idx = 0
    if not isinstance(subtask_instructions, list):
        subtask_instructions = []
    if len(subtask_instructions) < len(keyframes):
        subtask_instructions = subtask_instructions + [language_instruction] * (len(keyframes) - len(subtask_instructions))
    elif len(subtask_instructions) > len(keyframes):
        subtask_instructions = subtask_instructions[:len(keyframes)]
    gripper_was_closed = False   # tracks closed→open transition for auto keyframe advance
    GRIPPER_WARMUP_STEPS = 30    # ignore gripper state for the first N action steps
    GRIPPER_RESET_DELAY = 30     # steps to wait after open-transition before reset+advance
    gripper_action_step = 0      # counts total action steps executed
    gripper_open_countdown = 0   # >0 means we're counting down to reset+advance
    pending_reset = False        # signals reset+advance should happen after inner loop
    timer = Timer(env.control_hz)
    window_name = "Eval  |  n=next keyframe  q=quit"

    for step in range(horizon):
        try:
            obs = env.get_observation()["timestamp"]
            images = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for v in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, v])

            keyframe_img = keyframes[current_keyframe_idx]
            current_instruction = subtask_instructions[current_keyframe_idx]

            observation = {
                "observation.images.top_view":                    images["top_camera_rgb"],
                "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
                "observation.images.top_view_keyframe":           keyframe_img,
                "prompt":                                         current_instruction,
                "observation.state":                              concatenated_state,
            }

            out = client.infer(observation)
            action_chunk = out["actions"]
            completion_prob = float(out["completion_prob"]) if "completion_prob" in out else None
            if completion_prob is not None:
                print(f"  [BCE] completion_prob: {completion_prob:.3f}")
            
            for raw_action in action_chunk:
                action = {
                    'left_arm_positions':     raw_action[0:6],
                    'left_gripper_position':  raw_action[6:7],
                    'right_arm_positions':    raw_action[7:13],
                    'right_gripper_position': raw_action[13:14],
                }
                env.step(action)

                ### AUTO KEYFRAME ADVANCE (gripper closed → open transition) ###
                gripper_val = float(raw_action[6])
                gripper_action_step += 1
                if gripper_action_step > GRIPPER_WARMUP_STEPS:
                    if gripper_val < 0.5:
                        gripper_was_closed = True
                if gripper_was_closed and gripper_val > 0.9 and gripper_action_step > GRIPPER_WARMUP_STEPS \
                        and gripper_open_countdown == 0:
                    # Transition detected: start countdown to reset+advance
                    gripper_was_closed = False
                    gripper_open_countdown = GRIPPER_RESET_DELAY
                    print(f"  [Auto] Gripper opened — resetting in {GRIPPER_RESET_DELAY} steps...")
                if gripper_open_countdown > 0:
                    gripper_open_countdown -= 1
                    if gripper_open_countdown == 0:
                        pending_reset = True
                        break  # exit action_chunk loop; reset handled below
                ###############################################################

                # Show preview and check keys every action step
                live_obs = _extract_latest_observation(
                    env.get_observation())
                preview = _build_eval_preview(
                    live_obs, keyframe_img, current_keyframe_idx, len(keyframes), current_instruction)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('n'), ord('N')):
                    next_idx = current_keyframe_idx + 1
                    if next_idx >= len(keyframes):
                        print("No more keyframes. Resetting robot.")
                        cv2.destroyAllWindows()
                        env.reset()

                    else:
                        current_keyframe_idx = next_idx
                        keyframe_img = keyframes[current_keyframe_idx]
                        print(f"  → Keyframe [{current_keyframe_idx + 1}/{len(keyframes)}]")
                        print(f"  → Subtask: {subtask_instructions[current_keyframe_idx]}")
                        break  # discard remaining actions in chunk; re-infer with new keyframe

                elif key in (ord('q'), ord('Q'), 27):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt

                timer.sleep()

            # --- post action_chunk: handle pending reset+advance ---
            if pending_reset:
                pending_reset = False
                next_idx = current_keyframe_idx + 1
                if next_idx >= len(keyframes):
                    print("[Auto] No more keyframes. Resetting robot.")
                    cv2.destroyAllWindows()
                    env.reset()
                    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
                    return language_instruction
                else:
                    print("[Auto] Resetting robot...")
                    env.reset()
                    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
                    current_keyframe_idx = next_idx
                    # Reset gripper state machine so warmup applies fresh to new keyframe
                    gripper_was_closed = False
                    gripper_action_step = 0
                    gripper_open_countdown = 0
                    print(f"  [Auto] Reset done → Keyframe [{current_keyframe_idx + 1}/{len(keyframes)}]")
                    print(f"  [Auto] Subtask → {subtask_instructions[current_keyframe_idx]}")

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
            raise
        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()
    return language_instruction


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
    }

    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "I want this muffin, and this muffin."
    horizon = 10000
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please export it before running.")

    print("🚀 Starting. Instruction:", language_instruction)
    print("💡 During eval: n=next keyframe  q=stop episode")
    keyframes, subtask_instructions = extract_keyframes(env, language_instruction, GEMINI_API_KEY)
    # 2. Connect to policy server (after keyframe selection)
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    def _options_menu(lang_instr):
        print(f"\n🔄 Current instruction: '{lang_instr}'")
        print("Options:")
        print("  1. Press Enter  → continue with current instruction")
        print("  2. Type new instruction  → update and continue")
        print("  3. Type 'quit'  → exit")
        user_input = input("Enter your choice: ").strip()
        if user_input.lower() == 'quit':
            return None
        elif user_input == '':
            return lang_instr
        else:
            print(f"✅ New instruction: '{user_input}'")
            return user_input

    try:
        while True:
            # 3. Run rollout
            try:
                language_instruction = eval(
                    env, client, language_instruction, horizon, keyframes, subtask_instructions)
                print("✅ Episode finished.")

            except KeyboardInterrupt:
                pass  # fall through to options menu

            # After episode ends (normal finish or Ctrl+C), show menu
            result = _options_menu(language_instruction)
            if result is None:
                print("Exiting evaluation...")
                break
            language_instruction = result
            # Re-record keyframes for next episode
            keyframes, subtask_instructions = extract_keyframes(env, language_instruction, GEMINI_API_KEY)
            client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("🛑 Cleaning up...")
        temp_dir = os.path.normpath("/home/alin/projects/piper_project/temp")
        if os.path.isdir(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            print(f"🗑️  Cleared temp directory: {temp_dir}")
        env.safe_end()

