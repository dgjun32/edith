from typing import List, Tuple, Dict, Any

import numpy as np

from piper_teleop.robots.piper_robot import PiperRobot
from piper_teleop.sensors.camera import RealsenseCamera
from piper_teleop.sensors.remote_camera import RemoteCamera

class PiperRobotEnv:
    def __init__(self,
                 piper_names: List[str] = ['left', 'right'],
                 can_ports: List[str] = ['can_left', 'can_right'],
                 camera_ids: Dict[str, str] = {},
                ):
        self.piper_names = piper_names
        self.can_ports = can_ports
        self.camera_ids = camera_ids
        self.control_hz = 15

        self.piper_robots: List[PiperRobot] = []
        for piper_name, can_port in zip(self.piper_names, self.can_ports):
            self.piper_robots.append(PiperRobot(piper_name, can_port))

        self.cameras = {}
        for camera_name, camera_id in self.camera_ids.items():
            if(camera_name == 'aria_idx'):
                host, port = camera_id.split('_')
                self.cameras[camera_name] = RemoteCamera(serial_number=camera_id, visualize=False,host=host, port=int(port))
            else:
                self.cameras[camera_name] = RealsenseCamera(camera_id, visualize=False)

    def reset(self, randomize: bool = False):
        for piper_robot in self.piper_robots:
            piper_robot.reset()

    #def reset(self):
    #    pos_list = (np.array([-0.18839084,  0.09786061, -0.80672609, -0.35845572,  0.53307591, 0.33012903]), 
    #                np.array([0.30475194,  0.08421214, -0.69485048, -0.09417797,  0.41730822, -0.00986111]))
    #    for piper_robot, pos in zip(self.piper_robots, pos_list):
    #        piper_robot.reset(pos)


    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Get robot state
        state_dict = {}
        for piper_robot in self.piper_robots:
            piper_state_dict = piper_robot.get_state()
            for key, value in piper_state_dict.items():
                state_dict[piper_robot.name + "_" + key] = value
        obs_dict["timestamp"]["robot_state"] = state_dict

        # Get camera images
        image_obs = {}
        for camera_name, camera in self.cameras.items():
            color_image = camera.get_image()
            image_obs[camera_name + "_rgb"] = color_image
        obs_dict["timestamp"]["images"] = image_obs

        return obs_dict

    def step(self, action: Dict[str, Any]):
        action_info = {}
        for piper_robot in self.piper_robots:
            piper_action = {}
            for key, value in action.items():
                if piper_robot.name in key:
                    piper_action[key.replace(piper_robot.name + "_", "")] = value
            piper_action_info = piper_robot.step(piper_action)
            for key, value in piper_action_info.items():
                action_info[piper_robot.name + "_" + key] = value
        return action_info


    def get_control_hz(self):
        return self.control_hz

    def safe_end(self):
        for piper_robot in self.piper_robots:
            piper_robot.safe_end()
