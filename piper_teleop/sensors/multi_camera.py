
import socket
import struct
import threading
import time
import numpy as np


class RemoteCamera:
    """
    Receives only frame metadata (frame_id + timestamp) over TCP.
    Initialization blocks until the first metadata packet arrives.
    """

    def __init__(self, serial_number="REMOTE_Aria",
                 host=None, port=None,
                 timeout=60,visualize=False):

        print("A Remote RealSense CAMERA (metadata only) is ADDED")

        self.serial_number = serial_number
        self.host = host
        self.port = port

        self.traj_record_state = False
        self.traj_ended = False

        # Latest metadata storage
        self.latest_frame_id = None
        self.latest_timestamp = None
        self.running = True

        # Event to wait for first metadata
        self.first_metadata_event = threading.Event()

        # Start background receiver thread
        self._start_receiver_thread()

        # BLOCK until the first metadata arrives or timeout
        print("[RemoteRealSense] Waiting for first metadata...")
        if not self.first_metadata_event.wait(timeout=timeout):
            raise RuntimeError(
                f"[RemoteRealSense] ERROR: No metadata received within {timeout} seconds"
            )

        print("[RemoteRealSense] First metadata received. Initialization complete.")

    # ============================================================
    # SOCKET RECEIVER
    # ============================================================

    def _start_receiver_thread(self):
        t = threading.Thread(target=self._receive_stream, daemon=True)
        t.start()

    def _recv_all(self, sock, length):
        data = b""
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _receive_stream(self):
        metadata_size = struct.calcsize("Qd")  # frame_id (Q) + timestamp (d)

        while self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, self.port))
                print("[RemoteRealSense] Connected to metadata stream")

                while self.running:
                    flags = {
                        "traj_record_state": self.traj_record_state,
                        "traj_ended": self.traj_ended
                    }
                    data_to_send = struct.pack("Qd", self.get_traj_record_state(), self.get_traj_ended())

                    # Send the packed data over TCP
                    s.sendall(data_to_send)
                            
                    metadata = self._recv_all(s, metadata_size)
                    if not metadata:
                        break

                    frame_id, timestamp = struct.unpack("Qd", metadata)

                    # Store latest metadata
                    self.latest_frame_id = frame_id
                    self.latest_timestamp = timestamp

                    # 🔥 FIRST METADATA RECEIVED → UNBLOCK __init__
                    if not self.first_metadata_event.is_set():
                        self.first_metadata_event.set()

                s.close()

            except Exception as e:
                print("[RemoteRealSense] Stream error:", e)

            time.sleep(1)

    # ============================================================
    # get_metadata() — returns latest frame_id + timestamp
    # ============================================================

    def get_image(self):
        """
        Returns (frame_id, timestamp) of the latest received metadata.
        """
        return np.array([self.latest_frame_id])#, self.latest_timestamp])

    def get_traj_record_state(self):
        return self.traj_record_state

    def get_traj_ended(self):
        return self.traj_ended
    
    def set_traj_record_state(self, state):
        self.traj_record_state = state

    def set_traj_ended(self, state):
        self.traj_ended = state

    def set_host(self, host):
        self.host = host

    def set_port(self, port):
        self.port = port