"""
Example server code to run inside Docker container.
This streams Aria camera observations to the AriaCamera client via UDP.

Usage:
    python aria_camera_server_example.py --port 5555 --control-port 5556
"""

import socket
import struct
import time
import numpy as np
import cv2
import argparse
import threading


class AriaCameraServer:
    """
    Streams Aria camera observations over UDP.
    
    Protocol:
    - Receives (control_port): traj_record_state (bool) + traj_ended (bool)
    - Sends (data_port): frame_id + timestamp + rgb_image (JPEG) + eye_image (JPEG) + gaze_point
    
    Note: Large frames are split into multiple UDP packets.
    """
    
    def __init__(self, data_port=5555, control_port=5556, max_packet_size=60000):
        self.data_port = data_port
        self.control_port = control_port
        self.max_packet_size = max_packet_size
        self.frame_id = 0
        
        # Control state
        self.traj_record_state = False
        self.traj_ended = False
        self.client_address = None
        
        # Create UDP sockets
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_socket.bind(('0.0.0.0', self.control_port))
        self.control_socket.settimeout(0.1)  # Non-blocking with timeout
        
        # TODO: Initialize your actual Aria camera here
        # self.aria_device = ...
        
        print(f"[AriaCameraServer] UDP data sender ready on port {data_port}")
        print(f"[AriaCameraServer] UDP control receiver listening on port {control_port}")
        
    def get_aria_observation(self):
        """
        Get observation from Aria camera.
        Replace this with your actual Aria camera interface.
        
        Returns:
            rgb_image: (H, W, 3) numpy array, RGB format
            eye_image: (H, W, 3) numpy array, RGB format
            gaze_point: (2,) numpy array with (x, y) coordinates
        """
        # TODO: Replace with actual Aria camera capture
        # For now, generate dummy data
        rgb_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        eye_image = np.random.randint(0, 255, (240, 640, 3), dtype=np.uint8)
        gaze_point = np.array([128.0, 128.0], dtype=np.float32)
        
        return rgb_image, eye_image, gaze_point
    
    def _receive_control_signals(self):
        """Background thread to receive control signals."""
        while True:
            try:
                data, addr = self.control_socket.recvfrom(1024)
                self.client_address = addr
                
                # Parse control data
                traj_record_state, traj_ended = struct.unpack("??", data)
                self.traj_record_state = traj_record_state
                self.traj_ended = traj_ended
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[AriaCameraServer] Control receive error: {e}")
    
    def _send_observation(self, frame_id, timestamp, rgb_image, eye_image, gaze_point):
        """Send observation via UDP, splitting into multiple packets if needed."""
        if self.client_address is None:
            return  # No client connected yet
        
        try:
            # Compress images to JPEG
            _, rgb_jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, eye_jpeg = cv2.imencode('.jpg', cv2.cvtColor(eye_image, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            rgb_jpeg_data = rgb_jpeg.tobytes()
            eye_jpeg_data = eye_jpeg.tobytes()
            
            # Prepare observation data
            obs_header = struct.pack(
                "dIIff",
                timestamp,                  # d: timestamp
                len(rgb_jpeg_data),         # I: rgb_image length
                len(eye_jpeg_data),         # I: eye_image length
                float(gaze_point[0]),       # f: gaze_x
                float(gaze_point[1])        # f: gaze_y
            )
            
            complete_data = obs_header + rgb_jpeg_data + eye_jpeg_data
            
            # Split into packets
            packets = self._split_into_packets(frame_id, complete_data)
            
            # Send all packets
            for packet in packets:
                self.data_socket.sendto(packet, self.client_address)
                
        except Exception as e:
            print(f"[AriaCameraServer] Send error: {e}")
    
    def _split_into_packets(self, frame_id, data):
        """Split data into multiple UDP packets."""
        # Calculate packet header size
        packet_header_fmt = "QHH"  # frame_id, total_packets, packet_num
        packet_header_size = struct.calcsize(packet_header_fmt)
        
        # Calculate payload size per packet
        payload_size = self.max_packet_size - packet_header_size
        
        # Calculate number of packets needed
        total_packets = (len(data) + payload_size - 1) // payload_size
        
        packets = []
        for packet_num in range(total_packets):
            start = packet_num * payload_size
            end = min(start + payload_size, len(data))
            payload = data[start:end]
            
            # Create packet header
            header = struct.pack(packet_header_fmt, frame_id, total_packets, packet_num)
            
            packets.append(header + payload)
        
        return packets
    
    def serve_forever(self):
        """Start the server and stream observations."""
        # Start control receiver thread
        control_thread = threading.Thread(target=self._receive_control_signals, daemon=True)
        control_thread.start()
        
        print("[AriaCameraServer] Server started. Waiting for clients...")
        
        # Main observation streaming loop
        while True:
            try:
                if self.client_address is not None:
                    # Get observation from Aria camera
                    rgb_image, eye_image, gaze_point = self.get_aria_observation()
                    
                    # Send observation
                    timestamp = time.time()
                    self._send_observation(self.frame_id, timestamp, rgb_image, eye_image, gaze_point)
                    
                    self.frame_id += 1
                    
                    # Control frame rate (e.g., 30 FPS)
                    time.sleep(1.0 / 30.0)
                else:
                    # Wait for first client
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"[AriaCameraServer] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Aria Camera UDP Server")
    parser.add_argument("--data-port", type=int, default=5555, 
                       help="Port for sending observation data (default: 5555)")
    parser.add_argument("--control-port", type=int, default=5556,
                       help="Port for receiving control signals (default: 5556)")
    parser.add_argument("--max-packet-size", type=int, default=60000,
                       help="Maximum UDP packet size in bytes (default: 60000)")
    args = parser.parse_args()
    
    server = AriaCameraServer(
        data_port=args.data_port,
        control_port=args.control_port,
        max_packet_size=args.max_packet_size
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

