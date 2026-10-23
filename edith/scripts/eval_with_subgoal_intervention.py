import numpy as np
import traceback

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
                "video.top_view": images["top_camera_rgb"][np.newaxis, :, :, :],
                "video.left_wrist_view": images["left_wrist_camera_rgb"][np.newaxis, :, :, :],
                "video.right_wrist_view": images["right_wrist_camera_rgb"][np.newaxis, :, :, :],
                "state.left_arm_positions": np.expand_dims(robot_state["left_arm_positions"], axis=0).astype(np.float32),
                "state.left_gripper_position": np.expand_dims(robot_state["left_gripper_position"], axis=0).astype(np.float32),
                "state.right_arm_positions": np.expand_dims(robot_state["right_arm_positions"], axis=0).astype(np.float32),
                "state.right_gripper_position": np.expand_dims(robot_state["right_gripper_position"], axis=0).astype(np.float32),
                "annotation.human.task_description": [language_instruction],
            }
            
            action = client.get_action_with_subgoal_id(gr00t_obs)
            env.step(action)
            #breakpoint()
            timer.sleep()
        except KeyboardInterrupt:
            # Allow user to change subgoal_id during execution
            print(f"\n🎯 Current subgoal_id: {client.subgoal_id}")
            print(f"🔄 Current instruction: '{language_instruction}'")
            print("=" * 70)
            
            subgoal_input = input("Enter desired subgoal_id (or press Enter to continue, 'quit' to exit): ").strip()
            
            if subgoal_input.lower() == 'quit':
                print("Exiting evaluation...")
                return language_instruction
            elif subgoal_input == '':
                print(f"▶️ Continuing with subgoal_id: {client.subgoal_id}")
                continue
            else:
                # Try to parse subgoal_id as integer
                try:
                    new_subgoal_id = int(subgoal_input)
                    client.subgoal_id = new_subgoal_id
                    print(f"✅ Subgoal_id updated: {client.subgoal_id}")
                    continue
                except ValueError:
                    print(f"⚠️ Invalid subgoal_id '{subgoal_input}'. Must be an integer.")
                    print(f"▶️ Continuing with current subgoal_id: {client.subgoal_id}")
                    continue
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            print("=" * 70)
            print("📋 Full traceback:")
            traceback.print_exc()
            print("=" * 70)
            input("Press Enter to continue...")
    
    return language_instruction


if __name__ == "__main__":
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
    }
    env = PiperRobotEnv(camera_ids=camera_ids)
    #language_instruction = "Pick up the entire cups and place them on the white box."
    # language_instruction = "pick up the orange and put it in the basket"
    # language_instruction = "Try picking up the right of orange"
    # language_instruction = "Try picking up the banana"
    #language_instruction = "Pick up the plastic cup and put it on the white box. Then pick up the black straw and put it inside the plastic cup."
    #language_instruction = "pick up the red coke can using right arm and put it inside the plastic trash bin."
    #language_instruction = "Pick up skyblue cup using right arm and put it inside the trash bin."
    language_instruction = ""
    #language_instruction = "Pick up entire cups on the white table."
    #language_instruction = "Pick up the white paper cup using right arm, and put it in inside the paper trash bin."
    horizon = 10000
    
    print("🚀 Starting evaluation with instruction:", language_instruction)
    print("💡 Press Ctrl+C anytime to change the subgoal_id during execution!")

    
    #client = Gr00tRobotInferenceAsyncClient(
    #    min_action_len=16,
    #    language_instruction=language_instruction,
    # )
    client = Gr00tRobotInferenceSyncClient(
         action_horizon=16,
         language_instruction=language_instruction,
     )
    
    # Query initial subgoal_id before starting evaluation
    print("=" * 70)
    while True:
        initial_subgoal_input = input("🎯 Enter initial subgoal_id (default is 0): ").strip()
        
        if initial_subgoal_input == '':
            # Use default value
            print(f"✅ Using default subgoal_id: {client.subgoal_id}")
            break
        else:
            try:
                initial_subgoal_id = int(initial_subgoal_input)
                client.subgoal_id = initial_subgoal_id
                print(f"✅ Initial subgoal_id set to: {client.subgoal_id}")
                break
            except ValueError:
                print(f"⚠️ Invalid input '{initial_subgoal_input}'. Please enter an integer.")
                continue
    print("=" * 70)
    
    try:
        while True:  # Outer loop to allow multiple instruction changes
            try:
                final_instruction = eval(env, client, language_instruction, horizon)
                print(f"✅ Evaluation completed with final instruction: '{final_instruction}'")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C at the outer level - allow subgoal_id change
                print(f"\n🎯 Current subgoal_id: {client.subgoal_id}")
                print(f"🔄 Current instruction: '{language_instruction}'")
                print("=" * 70)
                
                subgoal_input = input("Enter desired subgoal_id (or press Enter to continue, 'quit' to exit): ").strip()
                
                if subgoal_input.lower() == 'quit':
                    print("Exiting evaluation...")
                    break
                elif subgoal_input == '':
                    print(f"▶️ Continuing evaluation with subgoal_id: {client.subgoal_id}")
                    continue
                else:
                    # Try to parse subgoal_id as integer
                    try:
                        new_subgoal_id = int(subgoal_input)
                        client.subgoal_id = new_subgoal_id
                        print(f"✅ Subgoal_id updated: {client.subgoal_id}")
                        continue
                    except ValueError:
                        print(f"⚠️ Invalid subgoal_id '{subgoal_input}'. Must be an integer.")
                        print(f"▶️ Continuing with current subgoal_id: {client.subgoal_id}")
                        continue
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("=" * 70)
        print("📋 Full traceback:")
        traceback.print_exc()
        print("=" * 70)
    finally:
        print("🛑 Cleaning up...")
        env.safe_end()
    
