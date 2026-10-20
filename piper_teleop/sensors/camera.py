from copy import deepcopy

import numpy as np

import cv2
import pyrealsense2 as rs


class RealsenseCamera:

    def __init__(
        self,
        camera_id: str,
        width: int = 424,
        height: int = 240,
        fps: int = 30,
        depth: bool = False,
        visualize: bool = False,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.depth = depth
        self.visualize = visualize

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(camera_id)
        self.config.enable_stream(rs.stream.color, width, height,
                                  rs.format.rgb8, fps)
        if depth:
            self.config.enable_stream(rs.stream.depth, width, height,
                                      rs.format.z16, fps)
        self.pipeline.start(self.config)

    def get_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        if self.visualize:
            bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.camera_id + "_rgb", bgr_image)
            cv2.waitKey(1)

        if self.depth:
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            return deepcopy(color_image), deepcopy(depth_image)

        return deepcopy(color_image)
