"""
Test script for AriaCamera client.

Usage:
    python test_aria_camera.py --host localhost --port 9999
"""

import argparse
import time
import cv2
import numpy as np
from aria_camera import AriaCamera


def main():
    parser = argparse.ArgumentParser(description="Test Aria Camera TCP Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--visualize", action="store_true", help="Show images in camera class")
    args = parser.parse_args()
    
    print("Initializing Aria Camera TCP Client...")
    camera = AriaCamera(
        host=args.host,
        port=args.port,
        visualize=args.visualize,
        timeout=30  # 30 seconds for Aria initialization
    )
    
    print("Starting data reception loop...")
    print("Commands:")
    print("  'q' - Quit")
    print("  'r' - Toggle recording state")
    print("  's' - Save current frame with gaze overlay\n")
    print("Note: For real-time visualization, use aria_camera_vis.py instead\n")
    
    recording = False
    frame_count = 0
    saved_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get latest observation
            rgb_image, gaze_point = camera.get_observation()
            
            if rgb_image is not None:
                frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    timestamp = camera.get_timestamp()
                    print(f"Frame {frame_count}:")
                    print(f"  FPS: {fps:.2f}")
                    print(f"  Timestamp: {timestamp}")
                    print(f"  RGB shape: {rgb_image.shape}")
                    print(f"  Gaze point: {gaze_point}")
                    print(f"  Recording: {recording}")
                    print()
                
                # Simple display if visualize enabled
                if args.visualize:
                    cv2.imshow("Aria RGB", rgb_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('r'):
                        recording = not recording
                        camera.set_traj_record_state(recording)
                        print(f"Recording state: {recording}")
                    elif key == ord('s'):
                        filename = f"aria_snapshot_{saved_count:04d}.png"
                        if camera.save_gaze(filename):
                            saved_count += 1
                else:
                    # No display, just check for quit (requires terminal input)
                    import select
                    import sys
                    if select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline().strip()
                        if line == 'q':
                            print("Quitting...")
                            break
                        elif line == 'r':
                            recording = not recording
                            camera.set_traj_record_state(recording)
                            print(f"Recording state: {recording}")
                        elif line == 's':
                            filename = f"aria_snapshot_{saved_count:04d}.png"
                            if camera.save_gaze(filename):
                                saved_count += 1
            
            time.sleep(0.001)  # Small delay to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nFinal Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        if saved_count > 0:
            print(f"  Snapshots saved: {saved_count}")


if __name__ == "__main__":
    main()

