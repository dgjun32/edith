import os
import re
import datetime as dt

try:
    from edith.legacy.utils.misc import collect_trajectory
except ModuleNotFoundError:
    # Backward-compatible fallback
    from droid.trajectory_utils.misc import collect_trajectory
import inspect
print(inspect.getfile(collect_trajectory))

from edith.legacy.envs.piper_env import PiperRobotEnv
from edith.legacy.teleoperators.joint_teleop_controller import DroidJointTeleopController, JointTeleopControllerConfig

if __name__ == "__main__":
    import argparse
    import inspect
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default="pick up paper cup and put it inside the metal trash bin.")
    parser.add_argument("--save_filepath", type=str, default="/home/alin/projects/piper_project/data/hdf/trash_separation_3_objs_unseen_251126_pnp_cup_to_metal")
    parser.add_argument("--instruction_csv_path", type=str, default=None)
    parser.add_argument("--instruction_csv_instruction_col", type=str, default="task_instruction")
    parser.add_argument("--instruction_csv_id_col", type=str, default="id")
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--instruction_random_seed", type=int, default=None)
    parser.add_argument("--task_guid_config_path", type=str, default=None)
    args = parser.parse_args()

    camera_ids = {
        'top_camera': '242322070028',
        'left_wrist_camera': '244222076964',
        'right_wrist_camera': '244222071848',
    }


    # Adjust action frequency here - higher Hz = faster/more frequent actions
    control_frequency = 15  # Change this value to adjust action frequency
    # control_frequency = 10   # Slower (10 Hz)
    # control_frequency = 30   # Faster (30 Hz) 
    # control_frequency = 50   # Very fast (50 Hz) - may be limited by hardware
    
    env = PiperRobotEnv(camera_ids=camera_ids) 
    instruction = args.instruction

    cfg = JointTeleopControllerConfig()
    controller = DroidJointTeleopController(cfg)
    save_filepath = os.path.join("/home/alin/projects/piper_project/data/hdf/", args.save_filepath)
    reset_time = 1.0
    try:
        collect_kwargs = {
            "randomize_reset": False,
            "instruction": instruction,
            "controller": controller,
            "save_filepath": save_filepath,
            "reset_time": reset_time,
        }

        sig = inspect.signature(collect_trajectory)
        if "instruction_csv_path" in sig.parameters:
            collect_kwargs.update(
                {
                    "instruction_csv_path": args.instruction_csv_path,
                    "instruction_csv_instruction_col": args.instruction_csv_instruction_col,
                    "instruction_csv_id_col": args.instruction_csv_id_col,
                    "max_trajectories": args.max_trajectories,
                    "instruction_random_seed": args.instruction_random_seed,
                    "task_guid_config_path": args.task_guid_config_path,
                }
            )

        collect_trajectory(env, **collect_kwargs)
    except KeyboardInterrupt:
        print("Stopping teleop...")
    finally:
        env.safe_end()
