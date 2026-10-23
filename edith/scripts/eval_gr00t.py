import numpy as np

from edith.envs.piper_env import PiperRobotEnv
from edith.utils.time_utils import Timer
from edith.utils.inference_client import Gr00tRobotInferenceSyncClient, Gr00tRobotInferenceAsyncClient

import pandas as pd

def eval(env, client, language_instruction, horizon):

    timer = Timer(env.control_hz)
    for i in range(horizon):
        try:
            obs = env.get_observation()["timestamp"]
            images = obs["images"]
            robot_state = obs["robot_state"]
            gr00t_obs = {
                "video.top_view": images["top_camera_rgb"][np.newaxis, :, :, :], # shape of (1, 240, 424, 3)
                "video.left_wrist_view": images["left_wrist_camera_rgb"][np.newaxis, :, :, :], # shape of (1, 240, 424, 3)
                "video.right_wrist_view": images["right_wrist_camera_rgb"][np.newaxis, :, :, :], # shape of (1, 240, 424, 3)
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
                action_horizon=16,
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
                continue
            else:
                # Update instruction
                language_instruction = user_input
                print(f"✅ New instruction set: '{language_instruction}'")
                
                # Update client with new instruction
                client.set_lang_instruction(language_instruction)
                continue
        except Exception as e:
            print(f"Error occurred: {e}")
            input("Press Enter to continue...")
    
    return language_instruction


if __name__ == "__main__":
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
    }
    env = PiperRobotEnv(camera_ids=camera_ids)
    env.reset()
    #language_instruction = "Pick up the entire cups and place them on the white box."
    # language_instruction = "pick up the orange and put it in the basket"
    # language_instruction = "Try picking up the right of orange"
    # language_instruction = "Try picking up the banana"
    #language_instruction = "Pick up the plastic cup and put it on the white box. Then pick up the black straw and put it inside the plastic cup."
    #language_instruction = "pick up the red coke can using right arm and put it inside the plastic trash bin."
    #language_instruction = "Pick up skyblue cup using right arm and put it inside the trash bin."
    language_instruction = "Give me the berry cupcake."
    #language_instruction = "Put the object to the plastic trash bin."
    #language_instruction = "Pick up entire cups on the white table."
    #language_instruction = "Pick up the white paper cup using right arm, and put it in inside the paper trash bin."
    horizon = 10000
    
    print("🚀 Starting evaluation with instruction:", language_instruction)
    print("💡 Press Ctrl+C anytime to change the instruction during execution!")

    
    #client = Gr00tRobotInferenceAsyncClient(
    #    min_action_len=16,
    #    language_instruction=language_instruction,
    # )
    client = Gr00tRobotInferenceSyncClient(
         action_horizon=16,
         language_instruction=language_instruction,
     )
    
    try:
        while True:  # Outer loop to allow multiple instruction changes
            try:
                final_instruction = eval(env, client, language_instruction, horizon)
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
    
