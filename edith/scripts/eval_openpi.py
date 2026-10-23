import numpy as np

from edith.envs.piper_env import PiperRobotEnv
from edith.utils.time_utils import Timer
from edith.utils.inference_client import PiRobotClient

import pandas as pd


from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).



def eval(env, client, language_instruction, horizon): 
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
                "prompt": language_instruction,
                "observation.state": concatenated_state
            }

            # Call the policy server with the current observation.
            # This returns an action chunk of shape (action_horizon, action_dim).
            # Note that you typically only need to call the policy every N steps and execute steps
            # from the predicted action chunk open-loop in the remaining steps.
            action_chunk = client.infer(observation)["actions"][:16,:]
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
                continue
            else:
                # Update instruction
                language_instruction = user_input
                print(f"✅ New instruction set: '{language_instruction}'")
                continue
        except Exception as e:
            print(f"Error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
    }
    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "Clean up the cans on the table."
    horizon = 10000
    print("🚀 Starting evaluation with instruction:", language_instruction)
    print("💡 Press Ctrl+C anytime to change the instruction during execution!")

    
    #client = Gr00tRobotInferenceAsyncClient(
    #    min_action_len=16,
    #    language_instruction=language_instruction,
    # )
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    
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
    
