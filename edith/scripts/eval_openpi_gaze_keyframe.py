import numpy as np

from edith.envs.piper_env_gaze import PiperRobotEnv
from edith.utils.time_utils import Timer
from edith.utils.inference_client import PiRobotClient

import pandas as pd


from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).

import pandas as pd
import cv2
import matplotlib.pyplot as plt


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
    status_text = "PAUSED: a=discard d=accept q=quit" if paused else "LIVE: s=stop q=quit"
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


def extract_keyframe(env):
    window_name = "Keyframe Selector"
    paused_obs = None
    paused_frame = None
    live_obs = None
    live_frame = None

    print("Keyframe selection: press 's' to stop, 'a' to discard, 'd' to accept, 'q' to quit.")
    try:
        while True:
            if paused_obs is None:
                live_obs = _extract_latest_observation(env.get_observation(mode='keyframe_cropped'))
                live_frame = _build_preview_frame(live_obs, paused=False)
                frame_to_show = live_frame
            else:
                frame_to_show = paused_frame

            cv2.imshow(window_name, frame_to_show)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('s'), ord('S')) and paused_obs is None and live_obs is not None:
                paused_obs = live_obs
                paused_frame = _build_preview_frame(paused_obs, paused=True)
            elif key in (ord('a'), ord('A')) and paused_obs is not None:
                paused_obs = None
                paused_frame = None
            elif key in (ord('d'), ord('D')) and paused_obs is not None:
                
                paused_obs["images"]["aria_rgb"] = cv2.cvtColor(paused_obs["images"]["aria_rgb"], cv2.COLOR_BGR2RGB)
                plt.imsave("human_view.png", paused_obs["images"]["aria_rgb"])
                return paused_obs
            elif key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt
    finally:
        cv2.destroyAllWindows()



def eval(env, client, language_instruction, horizon, keyframe_obs=None): 
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
                keyframe_obs = extract_keyframe(env)
                keyframe_obs = keyframe_obs["images"]["aria_rgb"]
                continue
            else:
                # Update instruction
                language_instruction = user_input
                print(f"✅ New instruction set: '{language_instruction}'")
                keyframe_obs = extract_keyframe(env)
                keyframe_obs = keyframe_obs["images"]["aria_rgb"]
                continue
        except Exception as e:
            print(f"Error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    SERVER_IP = "137.68.92.53"
    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
        'aria' : f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}"
    }
    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "Give me the oreo cupcake."
    horizon = 10000
    print("🚀 Starting evaluation with instruction:", language_instruction)
    print("💡 Press Ctrl+C anytime to change the instruction during execution!")
    keyframe_obs = extract_keyframe(env)
    keyframe_obs = keyframe_obs["images"]["aria_rgb"]
    
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    
    try:
        while True:  # Outer loop to allow multiple instruction changes
            try:
                final_instruction = eval(env, client, language_instruction, horizon, keyframe_obs)
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
    
