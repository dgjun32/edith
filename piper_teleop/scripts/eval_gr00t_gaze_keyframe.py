import numpy as np

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer
from piper_teleop.utils.inference_client import Gr00tRobotInferenceSyncClient, Gr00tRobotInferenceAsyncClient

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
    human_cropped_view = keyframe_obs
    #top_view = keyframe_obs["images"]["top_camera_rgb"]

    timer = Timer(env.control_hz)

    #TODO (Dongkyu): Extract the keyframe.
    ##Before we start action extraciton, I need to extract the keyframe. I want to loop an visualize the human view with gaze point and top view in real time. When S botton is pressed, the frame stops. If 'a' button is pressed, discard if 'd' is pressedl, accept and select it as the target frame.
    ## For visualization, I want to use the cv2 library to show the human view with gaze point and top view in real time. Please look at view_aria_gaze.py for reference.

    keyframe_idx = 0
    for i in range(horizon):
        try:
            # TODO (Dongkyu): modify the environment observation to include gaze information and human view image
            obs = env.get_observation(mode='downsampled')["timestamp"]
            images = obs["images"]
            robot_state = obs["robot_state"]
            
            gr00t_obs = {
                #### Currently, human view with gaze is a dummy with diagonal image array
                "video.human_view": images["aria_rgb"][np.newaxis, :, :, :],# should be the shape of (1, 256, 256, 3) ranging from 0 to 255
                "video.human_view_keyframe_cropped": human_cropped_view[np.newaxis, :, :, :], # TODO (dongkyu): get keyframe image from the environment
                "gaze_point.keyframe_gaze_point": np.array([255.0, 255.0]).reshape(1, 2), # should be the shape of (1, 2),
                "video.top_view": images["top_camera_rgb"][np.newaxis, :, :, :],
                "video.left_wrist_view": images["left_wrist_camera_rgb"][np.newaxis, :, :, :],
                "video.right_wrist_view": images["right_wrist_camera_rgb"][np.newaxis, :, :, :],
                "state.left_arm_positions": np.expand_dims(robot_state["left_arm_positions"], axis=0).astype(np.float32),
                "state.left_gripper_position": np.expand_dims(robot_state["left_gripper_position"], axis=0).astype(np.float32),
                "state.right_arm_positions": np.expand_dims(robot_state["right_arm_positions"], axis=0).astype(np.float32),
                "state.right_gripper_position": np.expand_dims(robot_state["right_gripper_position"], axis=0).astype(np.float32),
                "annotation.human.task_description": [language_instruction],
            }

            action = client.get_action(gr00t_obs)
            env.step(action)
            #breakpoint()
            timer.sleep()
        except KeyboardInterrupt:
            # Allow user to change instruction during execution
            env.reset()
            client = Gr00tRobotInferenceSyncClient(
                action_horizon=8,
                language_instruction=language_instruction,
            )
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
                human_cropped_view = keyframe_obs["images"]["aria_rgb"]
                continue
            else:
                # Update instruction
                language_instruction = user_input
                print(f"✅ New instruction set: '{language_instruction}'")
                
                # Update client with new instruction
                client.set_lang_instruction(language_instruction)
                keyframe_obs = extract_keyframe(env)
                human_cropped_view = keyframe_obs["images"]["aria_rgb"]
                continue
        except Exception as e:
            print(f"Error occurred: {e}")
            input("Press Enter to continue...")
    
    return language_instruction


if __name__ == "__main__":
    SERVER_IP = "137.68.92.12"
    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
        'aria' : f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}"
    }
    env = PiperRobotEnv(camera_ids=camera_ids)
    env.reset()
    #language_instruction = "Pick up the entire cups and place them on the white box."
    # language_instruction = "pick up the orange and put it in the basket"
    # language_instruction = "Try picking up the right of orange"
    # language_instruction = "Try picking up the banana"
    #language_instruction = "Pick up the plastic cup and put it on the white box. Then pick up the black straw and put it inside the plastic cup."
    #language_instruction = "pick up the red coke can using rigsht arm and put it inside the plastic trash bin."
    #language_instruction = "Pick up skyblue cup using right arm and put it inside the trash bin."
    language_instruction = "Give me the cherry cupcake."
    #language_instruction = "Pick up entire cups on the white table."
    #language_instruction = "Pick up the white paper cup using right arm, and put it in inside the paper trash bin."
    horizon = 10000
    
    print("🚀 Starting evaluation with instruction:", language_instruction)
    print("💡 Press Ctrl+C anytime to change the instruction during execution!")

    keyframe_obs = extract_keyframe(env)
    human_cropped_view = keyframe_obs["images"]["aria_rgb"]

    #client = Gr00tRobotInferenceAsyncClient(
    #    min_action_len=16,
    #    language_instruction=language_instruction,
    # )
    client = Gr00tRobotInferenceSyncClient(
         action_horizon=8,
         language_instruction=language_instruction,
     )
    
    try:
        while True:  # Outer loop to allow multiple instruction changes
            try:
                final_instruction = eval(env, client, language_instruction, horizon, keyframe_obs=human_cropped_view)
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
                    env.reset()
                    client.set_lang_instruction(language_instruction)
                    keyframe_obs = extract_keyframe(env)
                    human_cropped_view = keyframe_obs["images"]["aria_rgb"]
                    continue
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("🛑 Cleaning up...")
        env.safe_end()
    
