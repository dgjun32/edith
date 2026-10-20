import numpy as np

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer
from piper_teleop.utils.inference_client import PiRobotClient

import pandas as pd


from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json
import os
import imageio
from google import genai
from google.genai import types


# FPS used when accumulating and saving temp videos
_VIDEO_RECORD_FPS = 30

_KEYFRAME_PROMPT = """\
You are analyzing a first-person video recorded by a person wearing a head-mounted camera.
The person was given this instruction: "{instruction}"

Watch the video and identify the keyframe moment — the precise time when the person's hand
reaches toward, touches, or is about to grasp the target object mentioned in the instruction.

The video is {duration:.1f} seconds long.

Return your answer as JSON only:
{{
  "reasoning": "<brief explanation>",
  "keyframe_start": <float, seconds>,
  "keyframe_end": <float, seconds>
}}
"""


def _write_video(frames_bgr, output_path, fps=_VIDEO_RECORD_FPS):
    """Write a list of BGR uint8 frames to an mp4 using imageio."""
    if not frames_bgr:
        return
    # imageio expects RGB
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    imageio.mimwrite(output_path, frames_rgb, fps=fps)


def _query_gemini_keyframe(
    video_path: str,
    instruction: str,
    api_key: str,
    num_frames: int,
    model: str = "gemini-3.1-flash-lite-preview",
    video_fps_hint: int = 2,
) -> int:
    """
    Send saved video to Gemini and return the predicted keyframe frame index.
    The video was recorded at _VIDEO_RECORD_FPS; VLM output (seconds) is
    converted to frame index accordingly.
    Returns a clamped frame index in [0, num_frames-1].
    """
    duration_sec = num_frames / _VIDEO_RECORD_FPS
    prompt = _KEYFRAME_PROMPT.format(instruction=instruction, duration=duration_sec)

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    client = genai.Client(api_key=api_key)
    parts = [
        types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=video_fps_hint),
        ),
        types.Part(text=prompt),
    ]
    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    try:
        result = json.loads(response.text)
        start = float(result["keyframe_start"])
        end   = float(result["keyframe_end"])
        if start > end:
            start, end = end, start
        mid_sec = (start + end) / 2.0
        print(f"[gemini] keyframe: {start:.2f}s – {end:.2f}s  (mid={mid_sec:.2f}s)")
        print(f"[gemini] reasoning: {result.get('reasoning', '')[:160]}")
        # Convert VLM seconds → frame index (video recorded at _VIDEO_RECORD_FPS)
        frame_idx = int(round(mid_sec * _VIDEO_RECORD_FPS))
        frame_idx = max(0, min(frame_idx, num_frames - 1))
        print(f"[gemini] → frame index {frame_idx} / {num_frames - 1}")
        return frame_idx
    except Exception as e:
        print(f"[gemini] parse error: {e}  raw: {response.text[:200]}")
        # Fall back to middle frame
        return num_frames // 2


def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _build_preview_frame(obs, paused=False):
    images = obs["images"]
    human_view = images["aria_rgb"].copy()
    top_view = images["top_camera_rgb"]

    gaze_point = np.asarray(obs["gaze_point"], dtype=np.float32).reshape(-1)
    if gaze_point.size >= 2 and np.all(np.isfinite(gaze_point[:2])):
        x, y = gaze_point[:2].astype(np.int32)
        x = 255#int(np.clip(x, 0, human_view.shape[1] - 1))
        y = 255#int(np.clip(y, 0, human_view.shape[0] - 1))
        cv2.circle(human_view, (x, y), 6, (0, 255, 0), 2)
        cv2.line(human_view, (x - 12, y), (x + 12, y), (0, 255, 0), 2)
        cv2.line(human_view, (x, y - 12), (x, y + 12), (0, 255, 0), 2)

    if top_view.shape[:2] != human_view.shape[:2]:
        top_view = cv2.resize(
            top_view,
            (human_view.shape[1], human_view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    preview = np.hstack((human_view, top_view))
    cv2.putText(
        preview,
        "human(gaze) | top",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    status_text = "PAUSED: a=discard d=accept q=quit" if paused else "LIVE: p=save&proceed q=quit"
    status_color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(
        preview,
        status_text,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    return preview


def extract_keyframe(env, language_instruction: str, api_key: str):
    """
    1. Show live camera feed; press 'p' to capture all frames up to that point.
    2. Save video.mp4, cropped_video.mp4, gaze.json to temp/.
    3. Query Gemini VLM with video.mp4 to get the keyframe time range.
    4. Extract the keyframe from aria_cropped_frames (already in memory).
    5. Draw green marker at frame centre, save keyframe.png to temp/.
    6. Return the keyframe as a BGR numpy array.
    """
    window_name = "Keyframe Selector"

    temp_dir = os.path.normpath("/home/alin/projects/piper_project/temp")
    os.makedirs(temp_dir, exist_ok=True)
    video_path         = os.path.join(temp_dir, "video.mp4")
    cropped_video_path = os.path.join(temp_dir, "cropped_video.mp4")
    gaze_path          = os.path.join(temp_dir, "gaze.json")
    keyframe_img_path  = os.path.join(temp_dir, "keyframe.png")

    # Accumulate in memory; write all at once when 'p' is pressed
    aria_frames         = []
    aria_cropped_frames = []
    gaze_records        = []

    print("Keyframe selection: press 'p' to capture & query VLM, 'q' to quit.")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation(mode='human_view_streamed_and_keyframe_cropped'))
            live_frame = _build_preview_frame(live_obs, paused=False)

            # --- accumulate frames & gaze ---
            aria_frames.append(live_obs["images"]["aria_rgb"].copy())
            aria_cropped_frames.append(live_obs["images"]["aria_cropped_rgb"].copy())

            gaze_point = np.asarray(live_obs["gaze_point"], dtype=np.float64).reshape(-1)
            gaze_records.append({
                "frame_idx": len(gaze_records),
                "gaze_x": float(gaze_point[0]) if gaze_point.size >= 1 else None,
                "gaze_y": float(gaze_point[1]) if gaze_point.size >= 2 else None,
            })

            cv2.imshow(window_name, live_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('p'), ord('P')) and len(aria_frames) > 0:
                n = len(aria_frames)
                print(f"[keyframe] Captured {n} frames @ {_VIDEO_RECORD_FPS} FPS "
                      f"({n / _VIDEO_RECORD_FPS:.2f}s)")

                # --- save videos ---
                _write_video(aria_frames, video_path)
                print(f"[keyframe] Saved video → {video_path}")
                _write_video(aria_cropped_frames, cropped_video_path)
                print(f"[keyframe] Saved cropped video → {cropped_video_path}")

                # --- save gaze ---
                with open(gaze_path, 'w') as fp:
                    json.dump({"frames": gaze_records}, fp, indent=2)
                print(f"[keyframe] Saved gaze → {gaze_path}")

                # --- query Gemini to get keyframe frame index ---
                print("[keyframe] Querying Gemini VLM...")
                frame_idx = _query_gemini_keyframe(
                    video_path, language_instruction, api_key, num_frames=n)

                # --- extract keyframe from cropped frames (BGR) ---
                keyframe_bgr = aria_cropped_frames[frame_idx].copy()

                # --- draw green marker at image centre ---
                h, w = keyframe_bgr.shape[:2]
                cx, cy = w // 2, h // 2
                radius = max(8, h // 32)
                cv2.circle(keyframe_bgr, (cx, cy), radius, (0, 255, 0), 2)
                cv2.line(keyframe_bgr, (cx - radius * 2, cy), (cx + radius * 2, cy), (0, 255, 0), 2)
                cv2.line(keyframe_bgr, (cx, cy - radius * 2), (cx, cy + radius * 2), (0, 255, 0), 2)

                # --- save keyframe image (BGR → RGB for matplotlib) ---
                plt.imsave(keyframe_img_path,
                           cv2.cvtColor(keyframe_bgr, cv2.COLOR_BGR2RGB))
                print(f"[keyframe] Saved keyframe (frame {frame_idx}) → {keyframe_img_path}")

                return keyframe_bgr

            elif key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt
    finally:
        cv2.destroyAllWindows()



def eval(env, client, language_instruction, horizon, keyframe_obs=None, api_key=""): 
    timer = Timer(env.control_hz)
    for step in range(horizon):
        try:
            obs = env.get_observation()["timestamp"]
            images = obs["images"] # keys of ("top_camera_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb"), each shape of (240, 424, 3)
            robot_state = obs["robot_state"] # keys of ('left_arm_positions', 'left_gripper_position', 'right_arm_positions', 'right_gripper_position')
            concatenated_state = np.array([])
            for k, v in robot_state.items():
                concatenated_state = np.concatenate([concatenated_state, v])
            # TODO: need to check to environment observations / appropriate observation format
            # Inside the episode loop, construct the observation.
            # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
            # We provide utilities for resizing images + uint8 conversion so you match the training routines.
            # The typical resize_size for pre-trained pi0 models is 224.
            # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
            observation = {
                "observation.images.top_view": images["top_camera_rgb"],
                "observation.images.left_wrist_view": images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view": images["right_wrist_camera_rgb"],
                "observation.images.human_view_keyframe_cropped": keyframe_obs,
                "prompt": language_instruction,
                "observation.state": concatenated_state
            }
            # Call the policy server with the current observation.
            # This returns an action chunk of shape (action_horizon, action_dim).
            # Note that you typically only need to call the policy every N steps and execute steps
            # from the predicted action chunk open-loop in the remaining steps.
            action_chunk = client.infer(observation)["actions"]
            for i in range(len(action_chunk)):
                action = {'left_arm_positions': action_chunk[i][0:6],
                'left_gripper_position': action_chunk[i][6:7],
                'right_arm_positions': action_chunk[i][7:13],
                'right_gripper_position': action_chunk[i][13:14]}
                env.step(action)
                # TODO: need to check the appropriate action chunk format -> should be dictionary with 4 keys: 'left_arm_positions', 'left_gripper_position', 'right_arm_positions', 'right_gripper_position'
                timer.sleep()
        except KeyboardInterrupt:
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
            # Allow user to change instruction during execution
            print(f"\n🔄 Current instruction: '{language_instruction}'")
            print("Options:")
            print("1. Enter new instruction")
            print("2. Press Enter to continue with current instruction")
            print("3. Type 'quit' to exit")
            
            user_input = input("Enter your choice: ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting evaluation...")
                return language_instruction
            elif user_input == '':
                print(f"Continuing with: '{language_instruction}'")
                keyframe_obs = extract_keyframe(env, language_instruction, api_key)
                continue
            else:
                # Update instruction
                language_instruction = user_input
                print(f"✅ New instruction set: '{language_instruction}'")
                keyframe_obs = extract_keyframe(env, language_instruction, api_key)
                continue
        except Exception as e:
            print(f"Error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    SERVER_IP = "137.68.92.21"
    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
        'aria' : f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}"
    }
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # ← insert your Gemini API key here
    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "Give me this one."
    horizon = 10000
    print("🚀 Starting evaluation with instruction:", language_instruction)
    print("💡 Press Ctrl+C anytime to change the instruction during execution!")
    keyframe_obs = extract_keyframe(env, language_instruction, GEMINI_API_KEY)
    
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    
    try:
        while True:  # Outer loop to allow multiple instruction changes
            try:
                final_instruction = eval(env, client, language_instruction, horizon, keyframe_obs, GEMINI_API_KEY)
                print(f"✅ Evaluation completed with final instruction: '{final_instruction}'")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C at the outer level
                print(f"\n🔄 Current instruction: '{language_instruction}'")
                print("Options:")
                print("1. Enter new instruction")
                print("2. Press Enter to continue evaluation")
                print("3. Type 'quit' to exit")
                
                user_input = input("Enter your choice: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting evaluation...")
                    break
                elif user_input == '':
                    print(f"Continuing evaluation with: '{language_instruction}'")
                    continue
                else:
                    # Update instruction and continue
                    language_instruction = user_input
                    print(f"✅ New instruction set: '{language_instruction}'")
                    client.set_lang_instruction(language_instruction)
                    continue
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("🛑 Cleaning up...")
        env.safe_end()
    
