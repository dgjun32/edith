from typing import List, Tuple, Dict, Any

import numpy as np

from piper_teleop.robots.piper_robot import PiperRobot
from piper_teleop.sensors.camera import RealsenseCamera
from piper_teleop.sensors.aria_camera import AriaCamera

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
            if(camera_name == 'aria'):
                host, port, *rotation_parts = camera_id.split('_')
                rotation = '_'.join(rotation_parts) if rotation_parts else 'cw_90'
                self.cameras[camera_name] = AriaCamera(
                    serial_number=camera_id,
                    visualize=False,
                    host=host,
                    port=int(port),
                    rotation=rotation,
                )
            else:
                self.cameras[camera_name] = RealsenseCamera(camera_id, visualize=False)

    def reset(self, randomize: bool = False):
        for piper_robot in self.piper_robots:
            piper_robot.reset()

    def set_gaze_offset(self, x_off: float, y_off: float):
        """Forward gaze calibration offsets (full-res px) to the Aria camera."""
        camera = self.cameras.get('aria')
        if camera is not None and hasattr(camera, 'set_gaze_offset'):
            camera.set_gaze_offset(x_off, y_off)

    #def reset(self):
    #    pos_list = (np.array([-0.18839084,  0.09786061, -0.80672609, -0.35845572,  0.53307591, 0.33012903]), 
    #                np.array([0.30475194,  0.08421214, -0.69485048, -0.09417797,  0.41730822, -0.00986111]))
    #    for piper_robot, pos in zip(self.piper_robots, pos_list):
    #        piper_robot.reset(pos)


    def get_observation(self, mode='raw', audio_drain: bool = True):
        #mode: 'raw' or 'keyframe_cropped' or 'downsampled' or 'human_view_streamed_and_keyframe_cropped'
        assert mode in ['raw', 'keyframe_cropped', 'downsampled', 'human_view_streamed_and_keyframe_cropped'], "Invalid mode"

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
        gaze_point = None
        audio_obs = None
        aria_timestamp_ns = None
        for camera_name, camera in self.cameras.items():
            if camera_name == 'aria':
                if mode == 'human_view_streamed_and_keyframe_cropped':
                    cropped_image, streamed_image, gx, gy, gx_rescaled, gy_rescaled = camera.get_observation(mode=mode)
                    image_obs[camera_name + "_rgb"] = streamed_image
                    image_obs[camera_name + "_cropped_rgb"] = cropped_image
                    gaze_point = np.array([gx_rescaled, gy_rescaled])
                else:
                    image, gaze_point = camera.get_observation(mode=mode)
                    image_obs[camera_name + "_rgb"] = image
                    camera.save_gaze(filename=f"/home/alin/projects/piper_project/gaze_images/aria_gaze.png")
                aria_timestamp_ns = camera.get_timestamp()
                audio_obs = camera.get_audio_chunk(drain=audio_drain)
            else:
                color_image = camera.get_image()
                image_obs[camera_name + "_rgb"] = color_image
        obs_dict["timestamp"]["images"] = image_obs
        obs_dict["timestamp"]["gaze_point"] = np.expand_dims(gaze_point, axis=0)
        if aria_timestamp_ns is not None:
            obs_dict["timestamp"]["aria_timestamp_ns"] = int(aria_timestamp_ns)
        if audio_obs is not None:
            obs_dict["timestamp"]["audio"] = audio_obs
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
        for camera_name, camera in self.cameras.items():
            if camera_name == 'aria':
                camera.stop()
