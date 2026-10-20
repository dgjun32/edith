import numpy as np
import cv2

from piper_teleop.envs.piper_env_gaze import PiperRobotEnv
from piper_teleop.utils.time_utils import Timer

from openpi_client import websocket_client_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_latest_observation(observation):
    return observation[next(reversed(observation))]


def _build_preview_frame(obs, status_text, status_color=(0, 255, 0)):
    images = obs["images"]
    human_view = images["aria_rgb"].copy()
    top_view = images["top_camera_rgb"].copy()

    gaze_point = np.asarray(obs["gaze_point"], dtype=np.float32).reshape(-1)
    if gaze_point.size >= 2 and np.all(np.isfinite(gaze_point[:2])):
        x = int(np.clip(gaze_point[0], 0, human_view.shape[1] - 1))
        y = int(np.clip(gaze_point[1], 0, human_view.shape[0] - 1))
        cv2.circle(human_view, (x, y), 6, (0, 255, 0), 2)
        cv2.line(human_view, (x - 12, y), (x + 12, y), (0, 255, 0), 2)
        cv2.line(human_view, (x, y - 12), (x, y + 12), (0, 255, 0), 2)

    if top_view.shape[:2] != human_view.shape[:2]:
        top_view = cv2.resize(
            top_view,
            (human_view.shape[1], human_view.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    preview = np.hstack((human_view, top_view))
    cv2.putText(preview, "human(gaze) | top", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, status_text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    return preview


def _build_eval_preview(obs, keyframe_img, keyframe_idx, total_keyframes):
    """Live view + current keyframe thumbnail shown during rollout."""
    human_view = obs["images"]["aria_rgb"].copy()
    kf_display = cv2.resize(keyframe_img,
                            (human_view.shape[1], human_view.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    preview = np.hstack((human_view, kf_display))
    cv2.putText(preview,
                f"LIVE | KEYFRAME [{keyframe_idx + 1}/{total_keyframes}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, "n=next keyframe  q=quit",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Multiple keyframe selection
# ---------------------------------------------------------------------------

def extract_keyframes(env):
    """
    Interactively select multiple keyframes from the live camera stream.

    Controls:
      d        - add current live frame to the keyframe list
      a        - remove the last added keyframe (undo)
      f/Enter  - finish selection and return the list
      q/ESC    - quit (raises KeyboardInterrupt)

    Returns:
        list of np.ndarray  each (H, W, 3) uint8, aria_rgb crop
    """
    window_name = "Keyframe Selector  |  d=add  a=undo  f=finish  q=quit"
    keyframes = []
    live_obs = None

    print("Keyframe selection: d=add, a=undo, f=finish, q=quit")
    try:
        while True:
            live_obs = _extract_latest_observation(
                env.get_observation(mode='keyframe_cropped'))

            n = len(keyframes)
            status = f"Collected: {n} keyframe(s)  |  d=add  a=undo  f=finish"
            frame_to_show = _build_preview_frame(live_obs, status, (0, 200, 255))

            # Thumbnail strip of collected keyframes at bottom
            if keyframes:
                thumb_h = 64
                thumbs = [cv2.resize(kf, (thumb_h, thumb_h), interpolation=cv2.INTER_AREA)
                          for kf in keyframes[-6:]]
                thumb_row = np.hstack(thumbs)
                w = frame_to_show.shape[1]
                if thumb_row.shape[1] < w:
                    pad = np.zeros((thumb_h, w - thumb_row.shape[1], 3), dtype=np.uint8)
                    thumb_row = np.hstack((thumb_row, pad))
                else:
                    thumb_row = thumb_row[:, :w]
                frame_to_show = np.vstack((frame_to_show, thumb_row))

            cv2.imshow(window_name, frame_to_show)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('d'), ord('D')):
                keyframe_img = live_obs["images"]["aria_rgb"].copy()
                keyframes.append(keyframe_img)
                print(f"  Added keyframe #{len(keyframes)}")

            elif key in (ord('a'), ord('A')):
                if keyframes:
                    keyframes.pop()
                    print(f"  Undone. Remaining: {len(keyframes)}")

            elif key in (ord('f'), ord('F'), 13):   # 13 = Enter
                if not keyframes:
                    print("  No keyframes yet. Add at least one with 'd'.")
                else:
                    print(f"  Finished. {len(keyframes)} keyframe(s) selected.")
                    return keyframes

            elif key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt

    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def eval(env, client, language_instruction, horizon, keyframes):
    """
    Run the policy for up to `horizon` steps.

    keyframes : list of np.ndarray  (aria_rgb crop, BGR)
        The policy receives keyframes[current_keyframe_idx] at every step.

    Keyboard during rollout (OpenCV preview window):
      n  - advance to next keyframe; if none left → reset robot and return
      q  - raise KeyboardInterrupt (stops the episode)

    Returns the (possibly updated) language_instruction.
    """
    if not keyframes:
        print("No keyframes provided.")
        return language_instruction

    current_keyframe_idx = 0
    gripper_was_closed = False   # tracks closed→open transition for auto keyframe advance
    GRIPPER_WARMUP_STEPS = 30    # ignore gripper state for the first N action steps
    GRIPPER_RESET_DELAY = 30     # steps to wait after open-transition before reset+advance
    gripper_action_step = 0      # counts total action steps executed
    gripper_open_countdown = 0   # >0 means we're counting down to reset+advance
    pending_reset = False        # signals reset+advance should happen after inner loop
    timer = Timer(env.control_hz)
    window_name = "Eval  |  n=next keyframe  q=quit"

    for step in range(horizon):
        try:
            obs = env.get_observation(mode="downsampled")["timestamp"]
            images = obs["images"]
            robot_state = obs["robot_state"]

            concatenated_state = np.array([])
            for v in robot_state.values():
                concatenated_state = np.concatenate([concatenated_state, v])

            keyframe_img = keyframes[current_keyframe_idx]

            observation = {
                "observation.images.top_view":                    images["top_camera_rgb"],
                "observation.images.human_view":                  images["aria_rgb"],
                "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
                "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
                "observation.images.human_view_keyframe_cropped": keyframe_img,
                "observation.gaze_point":                         obs["gaze_point"],
                "prompt":                                         language_instruction,
                "observation.state":                              concatenated_state,
            }

            action_chunk = client.infer(observation)["actions"]

            for raw_action in action_chunk:
                print('left gripper action:', raw_action[6])
                action = {
                    'left_arm_positions':     raw_action[0:6],
                    'left_gripper_position':  raw_action[6:7],
                    'right_arm_positions':    raw_action[7:13],
                    'right_gripper_position': raw_action[13:14],
                }
                env.step(action)

                ### AUTO KEYFRAME ADVANCE (gripper closed → open transition) ###
                gripper_val = float(raw_action[6])
                gripper_action_step += 1
                if gripper_action_step > GRIPPER_WARMUP_STEPS:
                    if gripper_val < 0.5:
                        gripper_was_closed = True
                if gripper_was_closed and gripper_val > 0.9 and gripper_action_step > GRIPPER_WARMUP_STEPS \
                        and gripper_open_countdown == 0:
                    # Transition detected: start countdown to reset+advance
                    gripper_was_closed = False
                    gripper_open_countdown = GRIPPER_RESET_DELAY
                    print(f"  [Auto] Gripper opened — resetting in {GRIPPER_RESET_DELAY} steps...")
                if gripper_open_countdown > 0:
                    gripper_open_countdown -= 1
                    if gripper_open_countdown == 0:
                        pending_reset = True
                        break  # exit action_chunk loop; reset handled below
                ###############################################################

                # Show preview and check keys every action step
                live_obs = _extract_latest_observation(
                    env.get_observation(mode='keyframe_cropped'))
                preview = _build_eval_preview(
                    live_obs, keyframe_img, current_keyframe_idx, len(keyframes))
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('n'), ord('N')):
                    next_idx = current_keyframe_idx + 1
                    if next_idx >= len(keyframes):
                        print("No more keyframes. Resetting robot.")
                        cv2.destroyAllWindows()
                        env.reset()
                        return language_instruction
                    else:
                        current_keyframe_idx = next_idx
                        keyframe_img = keyframes[current_keyframe_idx]
                        print(f"  → Keyframe [{current_keyframe_idx + 1}/{len(keyframes)}]")

                elif key in (ord('q'), ord('Q'), 27):
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt

                timer.sleep()

            # --- post action_chunk: handle pending reset+advance ---
            if pending_reset:
                pending_reset = False
                next_idx = current_keyframe_idx + 1
                if next_idx >= len(keyframes):
                    print("[Auto] No more keyframes. Resetting robot.")
                    cv2.destroyAllWindows()
                    env.reset()
                    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
                    return language_instruction
                else:
                    print("[Auto] Resetting robot...")
                    env.reset()
                    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
                    current_keyframe_idx = next_idx
                    print(f"  [Auto] Reset done → Keyframe [{current_keyframe_idx + 1}/{len(keyframes)}]")

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            env.reset()
            client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

            raise

        except Exception as e:
            print(f"Error at step {step}: {e}")
            input("Press Enter to continue...")

    cv2.destroyAllWindows()
    return language_instruction


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SERVER_IP = "137.68.92.42"
    PUBLISHER_SERVER_PORT = 9999
    camera_ids = {
        'top_camera':         '242322070028',
        'left_wrist_camera':  '244222076964',
        'right_wrist_camera': '244222071848',
        'aria':               f"{SERVER_IP}_{PUBLISHER_SERVER_PORT}",
    }

    env = PiperRobotEnv(camera_ids=camera_ids)
    language_instruction = "put this muffin on the plate."
    horizon = 10000

    print("🚀 Starting. Instruction:", language_instruction)
    print("💡 During eval: n=next keyframe  q=stop episode")

    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

    try:
        while True:
            # 1. Select multiple keyframes interactively
            keyframes = extract_keyframes(env)

            # 2. Run rollout
            try:
                language_instruction = eval(
                    env, client, language_instruction, horizon, keyframes)
                print("✅ Episode finished.")

            except KeyboardInterrupt:
                print(f"\n🔄 Interrupted. Instruction: '{language_instruction}'")
                user_input = input(
                    "New instruction (Enter=keep, 'quit'=exit): ").strip()
                if user_input.lower() == 'quit':
                    break
                elif user_input:
                    language_instruction = user_input
                    print(f"✅ Instruction updated: '{language_instruction}'")
                # loop back to keyframe selection

    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Cleaning up...")
        env.safe_end()

