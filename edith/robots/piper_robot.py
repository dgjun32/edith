import time
import numpy as np

from piper_sdk import C_PiperInterface_V2


class PiperRobot:
    """
    Piper Robot Environment wrapper that provides compatible interface
    with the original collect_trajectory function structure.
    """

    def __init__(self, name: str = 'piper', can_port: str = 'can0'):

        self.name = name
        self.can_port = can_port

        self.use_mit_control = True # MIT control mode provides more compliant motion
        self.motor_max_speed = 1000
        self.motor_max_acc = 300
        self.gripper_effort = 700
        self.kp = np.array([2, 3.5, 3.5, 8, 8, 8])
        self.kd = np.array([0.2, 0.35, 0.35, 0.8, 0.8, 0.8])

        # Initialize Piper Interface
        self.piper = C_PiperInterface_V2(self.can_port)
        self.piper.ConnectPort()
        self.piper.EnableArm(7)

        # check the initial state of the robot
        print(self.piper)
        self.check_enable(self.piper)

        # Initialize robot to home position
        self.reset()

    def check_enable(self, piper: C_PiperInterface_V2):
        '''
        Enable the robotic arm and check enable status, try for 5s, exit program if enable timeout
        '''
        enable_flag = False
        # Set timeout duration (seconds)
        timeout = 5
        # Record time before entering loop
        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("Enable status:", enable_flag)
            piper.EnableArm(7)
            print("--------------------")
            # Check if timeout exceeded
            if elapsed_time > timeout:
                print("Timeout....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)

        if (elapsed_time_flag):
            print("Program auto-enable timeout, exiting program")
            exit(0)

    def reset(self, arm_pos: np.ndarray = np.zeros(6)):
        """Reset robot to home position"""
        print("Resetting robot")

        # for i in range(1,7):
        #     self.piper.MotorMaxSpdSet(i, self.motor_max_speed)
        #     self.piper.JointMaxAccConfig(i, self.motor_max_acc)
        #     time.sleep(0.1)
        # print(self.piper.GetAllMotorAngleLimitMaxSpd())

        self.piper.GripperCtrl(0, self.gripper_effort, 0x01, 0)
        self.safe_joint_control(arm_pos)

        # Wait for movement to complete
        time.sleep(0.5)

    def safe_joint_control(self, joint_pos: np.ndarray, execution_time: float = 1.0):

        start_joint_pos = self.get_joint_positions()
        control_hz = 30.0
        num_steps = int(execution_time * control_hz)

        joint_pos_diff = joint_pos - start_joint_pos
        target_joint_pos = start_joint_pos

        for i in range(num_steps):
            target_joint_pos += joint_pos_diff / num_steps
            if self.use_mit_control:
                self.joint_mit_control(target_joint_pos, self.kp, self.kd)
            else:
                self.joint_control(target_joint_pos)
            time.sleep(1.0 / control_hz)

    def safe_disable(self, execution_time: float = 1.0):
        target_joint_pos = self.get_joint_positions()
        control_hz = 30.0
        num_steps = int(execution_time * control_hz)

        for i in range(num_steps):
            kp = self.kp * (1.0 - i / num_steps)
            kd = self.kd * (1.0 - i / num_steps)
            self.joint_mit_control(target_joint_pos, kp, kd)
            time.sleep(1.0 / control_hz)
        self.piper.DisableArm(7)

    def safe_end(self):
        self.safe_joint_control(np.zeros(6))
        if self.use_mit_control:
            self.safe_disable()
        self.piper.GripperCtrl(0, self.gripper_effort, 0x01, 0)
        time.sleep(0.5)

    def get_state(self):
        """Get current robot state in format compatible with VRPolicy"""
        joint_positions = self.get_joint_positions()
        gripper_position = self.get_gripper_position()

        state_dict = {
            "arm_positions": joint_positions.tolist(),
            "gripper_position": [gripper_position]
        }

        return state_dict

    def get_joint_positions(self):
        joint_feedback = self.piper.GetArmJointMsgs()
        joint_positions = np.array([
            np.radians(joint_feedback.joint_state.joint_1 / 1000.0),
            np.radians(joint_feedback.joint_state.joint_2 / 1000.0),
            np.radians(joint_feedback.joint_state.joint_3 / 1000.0),
            np.radians(joint_feedback.joint_state.joint_4 / 1000.0),
            np.radians(joint_feedback.joint_state.joint_5 / 1000.0),
            np.radians(joint_feedback.joint_state.joint_6 / 1000.0)
        ])
        return joint_positions

    def get_gripper_position(self):
        gripper_feedback = self.piper.GetArmGripperMsgs()
        gripper_position = gripper_feedback.gripper_state.grippers_angle / 100000.0  # Normalize
        return gripper_position

    def joint_mit_control(
            self,
            target_joint_pos: np.ndarray,
            kp: np.ndarray,
            kd: np.ndarray,
            ff_torque: np.ndarray = np.zeros(6),
    ):
        self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD) # MIT control mode
        direction = np.array([-1, -1, 1, -1, 1, -1])
        for i in range(6):
            self.piper.JointMitCtrl(i + 1, direction[i] * target_joint_pos[i],
                                    0, kp[i], kd[i],
                                    direction[i] * ff_torque[i])


    def joint_control(self, target_joint_pos: np.ndarray,):
        joint_action_factor = 1000 * 180 / np.pi
        joint_action = target_joint_pos[:6]*joint_action_factor
        joint_1 = round(joint_action[0])
        joint_2 = round(joint_action[1])
        joint_3 = round(joint_action[2])
        joint_4 = round(joint_action[3])
        joint_5 = round(joint_action[4])
        joint_6 = round(joint_action[5])
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # Joint control mode
        self.piper.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)

    def step(self, action):
        """Execute action on the robot"""
        # Ensure action is numpy array
        arm_positions = np.array(action["arm_positions"])
        gripper_position = np.array(action["gripper_position"])

        if self.use_mit_control:
            self.joint_mit_control(arm_positions, self.kp, self.kd)
        else:
            self.joint_control(arm_positions)

        # Gripper control
        gripper_action_factor = 100000
        gripper_action = gripper_position[0] * gripper_action_factor
        gripper_action = round(gripper_action)
        self.piper.GripperCtrl(gripper_action, self.gripper_effort, 0x01, 0)

        action_info = {
            "arm_positions": arm_positions.tolist(),
            "gripper_position": gripper_position.tolist()
        }

        return action_info
