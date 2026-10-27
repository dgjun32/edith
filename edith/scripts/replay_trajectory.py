import numpy as np
import time
from droid.trajectory_utils.misc import TrajectoryReader

from edith.legacy.envs.piper_env import PiperRobotEnv
from edith.legacy.utils.time_utils import Timer

def replay_trajectory(
    env, filepath=None,
):
    traj_reader = TrajectoryReader(filepath, read_images=False)
    horizon = traj_reader.length()
    timer = Timer(env.control_hz)
    for i in range(horizon):
        # Get HDF5 Data #
        timestep = traj_reader.read_timestep()
        env.step(timestep["action"])
        timer.sleep()


# trajectory_folderpath = "/home/alin/projects/piper_project/data/hdf/filtered_data"

# Make the robot env
env = PiperRobotEnv()

# Replay Trajectory #
h5_filepath = "/home/alin/projects/piper_project/data/hdf/haeone_test_260430/demo0.hdf"
replay_trajectory(env, filepath=h5_filepath)
