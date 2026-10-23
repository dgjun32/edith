import numpy as np
from edith.teleoperators.dynamixlel_driver import DynamixelDriver
import tkinter as tk
import threading


class JointTeleopControllerConfig:
    port_name = "/dev/ttyUSB0"
    baud_rate = 57600
    dxl_ids = [20, 21, 22, 23, 24, 25, 26]

    offset = np.array([
        1.0 * np.pi,
        1.5 * np.pi,
        0.5 * np.pi,
        1.0 * np.pi,
        1.0 * np.pi,
        1.0 * np.pi,
        1.0 * np.pi,
    ])
    action_limit = np.zeros((7,2))
    action_limit[0,:] = np.radians([-150, 150])
    action_limit[1,:] = np.radians([0, 180])
    action_limit[2,:] = np.radians([-170, 0])
    action_limit[3,:] = np.radians([-100, 100])
    action_limit[4,:] = np.radians([-70, 70])
    action_limit[5,:] = np.radians([-120, 110])
    action_limit[6,:] = np.array([0.0, 1.0])
    action_gain = np.array([1, -1, -1, 1, -1, 1, -2])

    # for bimanual
    dxl_ids = [20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36]
    offset = np.concatenate([offset, offset])
    action_limit = np.concatenate([action_limit, action_limit])
    action_gain = np.array([1, -1, -1, 1, -1, 1, 2, 1, -1, -1, 1, -1, 1, -2])



class JointTeleopController:
    def __init__(self, cfg: JointTeleopControllerConfig):
        self.cfg = cfg
        # Use the optimized driver instead of raw dynamixel_sdk
        self.driver = DynamixelDriver(self.cfg.dxl_ids, self.cfg.port_name, self.cfg.baud_rate)

        self.is_connected = True

    def disconnect(self):
        """Disconnect from the dynamixels"""
        if hasattr(self, 'driver'):
            self.driver.close()
            self.is_connected = False
            print("Port closed")
        else:
            print("Port was not connected")

    def forward(self):
        """Optimized forward method using driver's cached joint angles"""
        # Get all joint angles directly from driver (already in radians)
        joint_angles_rad = self.driver.get_joints() - self.cfg.offset
        action = self.cfg.action_gain * joint_angles_rad

        action = np.clip(action, self.cfg.action_limit[:,0], self.cfg.action_limit[:,1])

        return action

    def __del__(self):
        self.disconnect()

class BimanualJointTeleopController(JointTeleopController):
    def __init__(self, cfg: JointTeleopControllerConfig):
        super().__init__(cfg)

    def forward(self):
        action = super().forward()
        action_dict = {
            "left_arm_positions": action[:6],
            "left_gripper_position": action[6:7],
            "right_arm_positions": action[7:13],
            "right_gripper_position": action[13:14],
        }
        return action_dict

    def __del__(self):
        self.disconnect()


class DroidJointTeleopController(BimanualJointTeleopController):

    def __init__(self, cfg: JointTeleopControllerConfig):
        super().__init__(cfg)

        self.state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }

        # GUI related variables
        self.gui_window = None
        self.gui_thread = None
        self.gui_running = False

        self.start_gui()

    def reset_state(self):
        self.state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
        }
        self._update_status_label("Ready")

    def get_info(self):
        state = self.state.copy()
        self.reset_state()
        return state

    def start_gui(self):
        """Start GUI window in a separate thread"""
        if not self.gui_running:
            self.gui_running = True
            self.gui_thread = threading.Thread(target=self._create_gui_window,
                                               daemon=True)
            self.gui_thread.start()

    def _create_gui_window(self):
        """Create small GUI window and handle keyboard events"""
        self.gui_window = tk.Tk()
        self.gui_window.title("Teleop Controller")
        self.gui_window.geometry("1000x1000")
        self.gui_window.resizable(False, False)

        # Set focus to receive keyboard input
        self.gui_window.focus_set()

        # Add labels
        info_label = tk.Label(
            self.gui_window, 
            text="Press 'S' for Success\nPress 'F' for Failure",
            font=("Helvetica", 20)
        )
        info_label.pack(pady=10)

        status_label = tk.Label(self.gui_window, text="Status: Ready", font=("Helvetica", 20))
        status_label.pack(pady=5)

        self.status_label = status_label  # For status updates

        # Bind keyboard events
        self.gui_window.bind('<KeyPress>', self._on_key_press)
        self.gui_window.bind('<Key>', self._on_key_press)  # Additional binding

        # Handle window close event
        self.gui_window.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Start GUI loop
        self.gui_window.mainloop()

    def _on_key_press(self, event):
        """Handle keyboard events"""
        key = event.keysym.lower()  # Convert to lowercase

        if key == 's':
            
            self.state["success"] = True
            self.state["failure"] = False  # Set failure to false when success is true
            self._update_status_label("Success: True")
            print("Success state set to True")

        elif key == 'f':
            self.state["failure"] = True
            self.state["success"] = False  # Set success to false when failure is true
            self._update_status_label("Failure: True")
            print("Failure state set to True")

    def _update_status_label(self, text):
        """Update status label"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Status: {text}", font=("Helvetica", 20))

    def _on_window_close(self):
        """Handle window close event"""
        self.gui_running = False
        if self.gui_window:
            self.gui_window.destroy()

    def stop_gui(self):
        """Close GUI window"""
        self.gui_running = False
        if self.gui_window:
            self.gui_window.quit()
            self.gui_window.destroy()

    def forward(self, obs_dict, include_info=False):
        action = super().forward()
        info_dict = self.state

        if include_info:
            return action, info_dict
        else:
            return action

    def __del__(self):
        self.stop_gui()
        self.disconnect()




def main():
    """Example usage of the JointTeleopController with GUI"""

    # Use DroidJointTeleopController (with GUI)
    print("Starting DroidJointTeleopController with GUI...")
    cfg = JointTeleopControllerConfig()
    droid_controller = DroidJointTeleopController(cfg)

    try:
        print("Reading joint positions and state...")
        print("-" * 40)

        # Read positions continuously
        import time
        for i in range(1000):
            action, info = droid_controller.forward(None, include_info=True)

            # Print joint positions
            for j in range(6):
                print(f"left_joint {j}: {np.degrees(action['left_arm_positions'][j]):.1f}°")
            print(f"left_gripper: {action['left_gripper_position']:.2f}")
            for j in range(6):
                print(f"right_joint {j}: {np.degrees(action['right_arm_positions'][j]):.1f}°")
            print(f"right_gripper: {action['right_gripper_position']:.2f}")

            # Print state information
            print(f"Success: {info['success']}, Failure: {info['failure']}")
            print("-" * 40)

            time.sleep(0.2)  # Wait 1 second

            # Exit if GUI window was closed
            if not droid_controller.gui_running:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")

    finally:
        # Always disconnect and close GUI
        droid_controller.stop_gui()
        droid_controller.disconnect()
        print("Controller and GUI closed.")


if __name__ == "__main__":
    main()
