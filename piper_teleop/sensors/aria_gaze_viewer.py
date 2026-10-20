"""
Simple real-time viewer for aria_gaze.png

Watches the file and automatically refreshes when it changes.

Usage:
    python aria_gaze_viewer.py
    python aria_gaze_viewer.py --file /path/to/image.png
    python aria_gaze_viewer.py --interval 0.1
"""

import cv2
import time
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Real-time Aria Gaze Image Viewer")
    parser.add_argument(
        "--file",
        type=str,
        default="/home/alin/projects/piper_project/aria_gaze.png",
        help="Path to the image file to watch"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Refresh interval in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Aria Gaze Viewer",
        help="Window title"
    )
    args = parser.parse_args()
    
    print(f"[Viewer] Watching file: {args.file}")
    print(f"[Viewer] Refresh interval: {args.interval}s")
    print("Press 'q' to quit\n")
    
    last_modified = 0
    frame_count = 0
    
    while True:
        try:
            # Check if file exists
            if os.path.exists(args.file):
                # Get file modification time
                current_modified = os.path.getmtime(args.file)
                
                # Read image if file was modified or first read
                if current_modified != last_modified or last_modified == 0:
                    img = cv2.imread(args.file)
                    
                    if img is not None:
                        # Display image
                        cv2.imshow(args.window_name, img)
                        
                        if last_modified != 0:  # Not first load
                            frame_count += 1
                            print(f"[Viewer] Updated frame {frame_count} - File modified at {time.ctime(current_modified)}")
                        else:
                            print(f"[Viewer] Initial load - Image size: {img.shape}")
                        
                        last_modified = current_modified
                    else:
                        print(f"[Viewer] Warning: Failed to read image from {args.file}")
                else:
                    # No update, just refresh window
                    pass
            else:
                # File doesn't exist yet
                if last_modified == 0:
                    print(f"[Viewer] Waiting for file: {args.file}")
                    last_modified = -1  # Mark as waiting
                
                # Show waiting message
                waiting_img = cv2.imread(args.file) if os.path.exists(args.file) else None
                if waiting_img is None:
                    # Create a blank image with text
                    waiting_img = (50, 50, 50) * cv2.ones((400, 600, 3), dtype='uint8')
                    cv2.putText(
                        waiting_img,
                        "Waiting for image...",
                        (150, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (200, 200, 200),
                        2
                    )
                    cv2.imshow(args.window_name, waiting_img)
            
            # Check for quit key
            key = cv2.waitKey(int(args.interval * 1000)) & 0xFF
            if key == ord('q'):
                print("\n[Viewer] Quitting...")
                break
            
        except KeyboardInterrupt:
            print("\n[Viewer] Interrupted by user")
            break
        except Exception as e:
            print(f"[Viewer] Error: {e}")
            time.sleep(args.interval)
    
    cv2.destroyAllWindows()
    print(f"[Viewer] Total updates: {frame_count}")


if __name__ == "__main__":
    main()

