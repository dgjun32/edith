import os
import re
import datetime as dt

from droid.trajectory_utils.misc_aria_with_button import collect_trajectory_with_aria_camera

from piper_teleop.envs.piper_env import PiperRobotEnv
from piper_teleop.teleoperators.joint_teleop_controller import DroidJointTeleopController, JointTeleopControllerConfig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default="pick up paper cup and put it inside the metal trash bin.")
    parser.add_argument("--save_filepath", type=str, default="/home/alin/projects/piper_project/data/hdf/trash_separation_3_objs_unseen_251126_pnp_cup_to_metal")
    parser.add_argument("--server_ip", type=str, default=None)
    parser.add_argument("--publisher_server_port", type=int, default=9999)
    parser.add_argument("--image_server_port", type=int, default=5001)
    args = parser.parse_args()

    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
        'aria_idx' : f"{args.server_ip}_{args.publisher_server_port}",
    }
    print(args.server_ip)
    # Adjust action frequency here - higher Hz = faster/more frequent actions
    control_frequency = 15  # Change this value to adjust action frequency
    # control_frequency = 10   # Slower (10 Hz)
    # control_frequency = 30   # Faster (30 Hz) 
    # control_frequency = 50   # Very fast (50 Hz) - may be limited by hardware
    
    env = PiperRobotEnv(camera_ids=camera_ids)
    instruction = args.instruction

    cfg = JointTeleopControllerConfig()
    controller = DroidJointTeleopController(cfg)
    print(controller)
    save_filepath = os.path.join("/home/alin/projects/piper_project/data/hdf/", args.save_filepath)
    reset_time = 1.0
    try:
        collect_trajectory_with_aria_camera(
            env,
            randomize_reset=False,
            instruction=instruction,
            controller=controller,
            save_filepath=save_filepath,
            reset_time=reset_time,
            image_source_ip=args.server_ip,
            image_source_port=args.image_server_port,
        )
    except KeyboardInterrupt:
        print("Stopping teleop...")
    finally:
        env.safe_end()