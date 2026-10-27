"""
Interactive gaze Y-offset calibration tool for AriaCamera.

Shows the raw frame with the raw gaze point (green) and the offset-adjusted
gaze point (red) side by side with the keyframe crop result.
Use the trackbar to tune the Y offset, then copy the printed value into
aria_camera.py wherever +40.0 appears.

Usage:
    python aria_gaze_calibrate.py --host 137.68.92.29 --port 9999
"""

import argparse
import time
import cv2
import numpy as np
from aria_camera import AriaCamera, gaze_crop, CROP_SIZE

TRACKBAR_OFFSET = 200   # centre of slider == 0 offset; range -200..+200
SLIDER_MAX = 400


def draw_cross(image, x, y, color, radius=8, thickness=2):
    h, w = image.shape[:2]
    x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    cv2.circle(image, (x, y), radius, color, thickness)
    cv2.line(image, (x - radius * 2, y), (x + radius * 2, y), color, thickness)
    cv2.line(image, (x, y - radius * 2), (x, y + radius * 2), color, thickness)
    return image


def build_display(raw_frame, gaze_point, y_offset, crop_size=CROP_SIZE):
    """Return a single BGR display image."""
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

    gx, gy = gaze_point[0], gaze_point[1]
    gy_adj = gy + y_offset

    # Left panel: full frame with both gaze points
    left = frame.copy()
    draw_cross(left, gx, gy,       color=(0, 255, 0))          # green = raw
    draw_cross(left, gx, gy_adj,   color=(0, 0, 255))          # red   = adjusted
    # scale down to fit screen
    scale = 512 / max(left.shape[:2])
    left = cv2.resize(left, (int(left.shape[1] * scale), int(left.shape[0] * scale)))

    # Right panel: keyframe crop using adjusted point
    crop, _, _ = gaze_crop(raw_frame, gx, gy_adj, crop_size=crop_size)
    right = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    right = cv2.resize(right, (512, 512))
    # draw centre marker to show where gaze landed inside the crop
    cx, cy = right.shape[1] // 2, right.shape[0] // 2
    draw_cross(right, cx, cy, color=(0, 0, 255))

    # Pad left to same height as right if needed
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    if lh < rh:
        pad = np.zeros((rh - lh, lw, 3), dtype=np.uint8)
        left = np.vstack([left, pad])
    elif rh < lh:
        pad = np.zeros((lh - rh, rw, 3), dtype=np.uint8)
        right = np.vstack([right, pad])

    display = np.hstack([left, right])

    # Legend
    cv2.putText(display, "GREEN = raw gaze   RED = adjusted gaze", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(display, f"Y offset: {y_offset:+.1f} px", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(display, "q=quit  UP/DOWN=fine-tune  slider=coarse", (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return display


def main():
    parser = argparse.ArgumentParser(description="Aria gaze Y-offset calibration")
    parser.add_argument("--host", type=str, default="137.68.92.53")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--rotation", type=str, default="cw_90")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--initial-offset", type=float, default=40.0,
                        help="Starting Y offset (default 40.0)")
    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port} ...")
    camera = AriaCamera(host=args.host, port=args.port, timeout=30,
                        visualize=False, rotation=args.rotation)
    print("Connected. Adjust the trackbar or use UP/DOWN arrow keys. Press q to quit.")

    WIN = "Aria Gaze Calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1024, 560)

    # Trackbar value encodes offset as: offset = slider_val - TRACKBAR_OFFSET
    initial_slider = int(args.initial_offset) + TRACKBAR_OFFSET
    initial_slider = int(np.clip(initial_slider, 0, SLIDER_MAX))
    cv2.createTrackbar("Y offset", WIN, initial_slider, SLIDER_MAX, lambda v: None)

    y_offset = args.initial_offset

    try:
        while True:
            raw, gaze = camera.get_observation(mode='raw')
            if raw is None:
                time.sleep(0.01)
                continue

            # Read slider
            slider_val = cv2.getTrackbarPos("Y offset", WIN)
            y_offset = float(slider_val - TRACKBAR_OFFSET)

            display = build_display(raw, gaze, y_offset, crop_size=args.crop_size)
            cv2.imshow(WIN, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == 82:  # UP arrow — fine +1
                slider_val = min(slider_val + 1, SLIDER_MAX)
                cv2.setTrackbarPos("Y offset", WIN, slider_val)
            elif key == 84:  # DOWN arrow — fine -1
                slider_val = max(slider_val - 1, 0)
                cv2.setTrackbarPos("Y offset", WIN, slider_val)

    finally:
        cv2.destroyAllWindows()
        final = float(cv2.getTrackbarPos("Y offset", WIN) - TRACKBAR_OFFSET) if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 0 else y_offset
        print(f"\n=== Calibration result ===")
        print(f"Y offset: {y_offset:+.1f}")
        print(f"\nIn aria_camera.py, replace the +40.0 occurrences with {y_offset:+.1f}:")
        print(f"  gaze_crop(..., self.latest_gaze_point[1]{y_offset:+.1f}, ...)")
        print(f"  downsampled_crop(..., self.latest_gaze_point[1]{y_offset:+.1f}, ...)")


if __name__ == "__main__":
    main()
