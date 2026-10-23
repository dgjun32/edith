"""
Aria Camera Visualization Tool

Usage:
    python aria_camera_vis.py --host 192.168.1.100 --port 9999
"""

import argparse
import time
import cv2
import numpy as np
from aria_camera import AriaCamera


def draw_gaze_point(
    image,
    x,
    y,
    color=(0, 255, 0),
    radius=6,
    thickness=2,
    cross_size=15,
):
    """Overlay gaze point on RGB image"""
    image = image.copy()  # Don't modify original
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    # Circle
    cv2.circle(image, (x, y), radius, color, thickness)

    # Crosshair
    cv2.line(image, (x - cross_size, y), (x + cross_size, y), color, thickness)
    cv2.line(image, (x, y - cross_size), (x, y + cross_size), color, thickness)

    return image


def main():
    parser = argparse.ArgumentParser(description="Aria Camera Visualization")
    parser.add_argument("--host", type=str, default="137.68.92.28", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--rotation", type=str, default="cw_90", 
                       help="Gaze rotation mode (cw_90, cw_180, cw_270, none)")
    parser.add_argument("--no-fps", action="store_true", help="Hide FPS counter")
    parser.add_argument("--save-video", type=str, default=None, 
                       help="Save video to file (e.g., output.mp4)")
    args = parser.parse_args()
    
    print(f"[Visualization] Connecting to Aria Camera at {args.host}:{args.port}")
    
    # Initialize Aria Camera
    camera = AriaCamera(
        host=args.host,
        port=args.port,
        timeout=30,  # 30 seconds for Aria initialization
        visualize=False,  # We handle visualization here
        rotation=args.rotation
    )
    
    print("[Visualization] Aria Camera initialized successfully!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Toggle recording state")
    print("  's' - Save current frame\n")
    
    # Video writer setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (256, 256))
        print(f"[Visualization] Recording to {args.save_video}")
    
    frame_count = 0
    start_time = time.time()
    recording = False
    saved_count = 0
    
    try:
        while True:
            # Get latest observation
            rgb_image, gaze_point = camera.get_observation()
            
            if rgb_image is None:
                time.sleep(0.001)
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Create visualization
            vis_image = rgb_image.copy()
            
            # Draw gaze point
            if gaze_point is not None:
                vis_image = draw_gaze_point(vis_image, gaze_point[0], gaze_point[1])
            
            # Add FPS and info text
            if not args.no_fps:
                cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if gaze_point is not None:
                cv2.putText(vis_image, f"Gaze: ({gaze_point[0]:.1f}, {gaze_point[1]:.1f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if recording:
                cv2.putText(vis_image, "REC", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(vis_image, (230, 85), 8, (0, 0, 255), -1)
            
            # Display
            cv2.imshow("Aria RGB + Gaze", vis_image)
            
            # Save to video file if enabled
            if video_writer is not None:
                video_writer.write(vis_image)
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                timestamp = camera.get_timestamp()
                print(f"Frame {frame_count}: FPS={fps:.1f}, "
                      f"Gaze={gaze_point if gaze_point is not None else 'None'}, "
                      f"Timestamp={timestamp}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Visualization] Quitting...")
                break
            elif key == ord('r'):
                recording = not recording
                camera.set_traj_record_state(recording)
                print(f"[Visualization] Recording state: {'ON' if recording else 'OFF'}")
            elif key == ord('s'):
                # Save current frame
                filename = f"aria_frame_{saved_count:04d}.png"
                cv2.imwrite(filename, vis_image)
                saved_count += 1
                print(f"[Visualization] Saved frame to {filename}")
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n[Visualization] Interrupted by user")
    
    finally:
        # Cleanup
        if video_writer is not None:
            video_writer.release()
            print(f"[Visualization] Video saved to {args.save_video}")
        
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n[Visualization] Final Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        if saved_count > 0:
            print(f"  Frames saved: {saved_count}")


if __name__ == "__main__":
    main()

