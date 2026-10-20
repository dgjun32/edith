from collections import deque
import os
import queue
import socket
import struct
import subprocess
import tempfile
import threading
import time
import numpy as np
import cv2


INFERENCE_V2_VERSION = 2
INFERENCE_V2_AUDIO_PRESENT_FLAG = 0x1
INFERENCE_V2_HEADER_FMT = "<BBHQffQIII"
INFERENCE_V2_HEADER_SIZE = struct.calcsize(INFERENCE_V2_HEADER_FMT)
DEFAULT_AUDIO_SAMPLE_RATE_HZ = 48000
INIT_TIMEOUT_SEC = 30.0
SOCKET_TIMEOUT_SEC = 5.0
MAX_CONSECUTIVE_SOCKET_TIMEOUTS = 3
RECONNECT_DELAY_SEC = 1.0
CONNECT_ERROR_LOG_INTERVAL_SEC = 5.0
RAW_ARIA_VIDEO_SAVE_ENABLED = True
RAW_ARIA_VIDEO_DIR = "/home/alin/projects/piper_project/temp/aria_raw"
RAW_ARIA_VIDEO_FPS = 15
RAW_ARIA_VIDEO_SIZE = (1408, 1408)
RAW_ARIA_VIDEO_CODEC = "mp4v"
RAW_ARIA_CROPPED_VIDEO_SIZE = (512, 512)
# Max frames buffered for the async video writer (~8s of raw+cropped at 15fps).
# When full, new frames are DROPPED so the enqueueing thread never blocks.
VIDEO_WRITE_QUEUE_MAX = 256
_FFMPEG = "/home/alin/miniconda3/envs/piper/bin/ffmpeg"

def _reencode_h264(path: str):
    """Re-encode a cv2-written mp4v file to H264 for broad playback compatibility."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=os.path.dirname(path))
    tmp.close()
    try:
        subprocess.run(
            [_FFMPEG, "-y", "-i", path,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", tmp.name],
            check=True, capture_output=True,
        )
        os.replace(tmp.name, path)
    except Exception as exc:
        print(f"[AriaCamera] ffmpeg re-encode failed for {path}: {exc}")
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


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
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    # circle
    cv2.circle(image, (x, y), radius, color, thickness)

    # crosshair
    cv2.line(image, (x - cross_size, y), (x + cross_size, y), color, thickness)
    cv2.line(image, (x, y - cross_size), (x, y + cross_size), color, thickness)

    return image

def rotate_coordinate(coordinate, image_width, image_height, rotation='cw_90'):
    """
    Rotate coordinate by various angles.
    Coordinate system: (0,0) at upper-left
    """
    x, y = coordinate
    if rotation == 'cw_90':
        # 시계방향 90도
        x_new = image_height - 1 - y
        y_new = x
    
    elif rotation == 'cw_180':
        # 180도
        x_new = image_width - 1 - x
        y_new = image_height - 1 - y
    
    elif rotation == 'cw_270' or rotation == 'ccw_90':
        # 시계방향 270도 = 반시계방향 90도
        x_new = y
        y_new = image_width - 1 - x
    
    else:
        # 회전 없음
        x_new = x
        y_new = y
    
    return np.array([x_new, y_new])

def scale_coordinate(coordinate, scale_factor=256/1408):
    """
    Scale coordinate by a factor.
    """
    x, y = coordinate
    x_new = x * scale_factor
    y_new = y * scale_factor
    return np.array([x_new, y_new])


CROP_SIZE = 512

def gaze_crop(frame, gx, gy, crop_size=CROP_SIZE):
    """
    Crop a crop_size x crop_size region centered on gaze point (gx, gy).
    Clamps to image bounds.
    Returns: (cropped_frame, x0, y0) where x0,y0 is the top-left of the crop box.
    """
    h, w = frame.shape[:2]

    # Clamp center so the crop box stays within the image
    cx = int(np.clip(gx, crop_size // 2, w - crop_size // 2))
    cy = int(np.clip(gy, crop_size // 2, h - crop_size // 2))

    x0 = cx - crop_size // 2
    y0 = cy - crop_size // 2

    return frame[y0:y0 + crop_size, x0:x0 + crop_size].copy(), gx, gy


DOWNSAMPLED_CROP_SIZE = 256

def downsampled_crop(frame, gx, gy, crop_size=DOWNSAMPLED_CROP_SIZE):
 
    h, w = frame.shape[:2]

    #also return the gaze point in the downsampled image
    gx_new = gx * crop_size / w
    gy_new = gy * crop_size / h
    return cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR), gx_new, gy_new


class AriaCamera:
    """
    Receives Aria RGB image and gaze point over TCP.
    Initialization blocks until the first data arrives.
    """

    def __init__(self, serial_number="ARIA_CAMERA",
                 host=None, port=None,
                 timeout=60, visualize=False,
                 original_size=1408, target_size=256,
                 rotation='cw_90'):

        print(f"[AriaCamera] Initializing Aria Camera from {host}:{port}")

        self.serial_number = serial_number
        self.host = host
        self.port = port
        self.visualize = visualize
        
        # Image size parameters
        self.original_size = original_size
        self.target_size = target_size
        self.scale_factor = target_size / original_size
        self.rotation = rotation

        self.traj_record_state = False
        self.traj_ended = False

        # Gaze calibration offsets (full-resolution pixels), applied to the
        # gaze point before gaze-based cropping. Default 0 = no change.
        self.gaze_offset_x = 0.0
        self.gaze_offset_y = 0.0

        # Latest data storage
        self.latest_frame_id = None
        self.latest_timestamp = None
        self.latest_rgb_image = None
        self.latest_gaze_point = None
        self.latest_audio_timestamp = 0
        self.latest_audio_sample_rate_hz = DEFAULT_AUDIO_SAMPLE_RATE_HZ
        self.latest_frame_monotonic = 0.0
        self.audio_buffer_max_sec = 0.5
        self.audio_chunks = deque()
        self.running = True
        self.raw_video_writer = None
        self.raw_video_path = None
        self.raw_video_frame_count = 0
        self.raw_video_lock = threading.Lock()
        self.cropped_video_writer = None
        self.cropped_video_path = None
        self.cropped_video_frame_count = 0

        # Async video writer: disk writes must never block the socket
        # receiver thread (a stalled receiver causes TCP backpressure that
        # can wedge the remote publisher). Frames are dropped when the
        # writer falls behind — recording quality is sacrificed, never the
        # live stream.
        self._video_write_queue = queue.Queue(maxsize=VIDEO_WRITE_QUEUE_MAX)
        self._video_dropped_frames = 0
        self._video_writer_thread = None
        if RAW_ARIA_VIDEO_SAVE_ENABLED:
            self._video_writer_thread = threading.Thread(
                target=self._video_writer_loop, daemon=True,
                name="AriaVideoWriter")
            self._video_writer_thread.start()

        # Thread-safe lock
        self.lock = threading.Lock()

        # Event to wait for first data
        self.first_data_event = threading.Event()

        # Start background receiver thread
        self._start_receiver_thread()

        # BLOCK until the first data arrives or timeout
        print("[AriaCamera] Waiting for first data...")
        if not self.first_data_event.wait(timeout=timeout):
            raise RuntimeError(
                f"[AriaCamera] ERROR: No data received within {timeout} seconds"
            )

        print("[AriaCamera] First data received. Initialization complete.")

    # ============================================================
    # SOCKET RECEIVER
    # ============================================================

    def _start_receiver_thread(self):
        t = threading.Thread(target=self._receive_stream, daemon=True)
        t.start()

    def _recv_all(self, sock, size):
        """Receive exactly 'size' bytes from socket."""
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def _rotate_coordinate(self, coordinate, image_width, image_height):
        """Rotate coordinate based on self.rotation setting."""
        x, y = coordinate
        if self.rotation == 'cw_90':
            x_new = image_height - 1 - y
            y_new = x
        elif self.rotation == 'cw_180':
            x_new = image_width - 1 - x
            y_new = image_height - 1 - y
        elif self.rotation in ['cw_270', 'ccw_90']:
            x_new = y
            y_new = image_width - 1 - x
        else:
            x_new = x
            y_new = y
        return np.array([x_new, y_new])
    
    def _scale_coordinate(self, coordinate):
        """Scale coordinate by scale_factor."""
        return coordinate * self.scale_factor

    def _enqueue_video_frame(self, kind: str, frame_bgr):
        """Non-blocking enqueue for the async video writer.

        Called from the receiver thread ("raw") and get_observation
        ("cropped"). Never blocks: drops the frame when the writer is
        behind, so the socket receiver always keeps draining the stream.
        """
        if not RAW_ARIA_VIDEO_SAVE_ENABLED or frame_bgr is None:
            return
        try:
            self._video_write_queue.put_nowait((kind, frame_bgr))
        except queue.Full:
            self._video_dropped_frames += 1
            if self._video_dropped_frames % 100 == 1:
                print(f"[AriaCamera] Video writer behind — dropped "
                      f"{self._video_dropped_frames} frames so far")

    def _video_writer_loop(self):
        """Dedicated thread: the only place that touches the disk writers."""
        while True:
            try:
                item = self._video_write_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break
                continue
            if item is None:  # sentinel from stop()
                break
            kind, frame = item
            try:
                if kind == "raw":
                    self._write_raw_video_frame(frame)
                else:
                    self._write_cropped_video_frame(frame)
            except Exception as exc:
                print(f"[AriaCamera] Video write failed ({kind}): {exc}")

    def _ensure_raw_video_writer_locked(self):
        if (not RAW_ARIA_VIDEO_SAVE_ENABLED
                or self.raw_video_writer is not None):
            return

        os.makedirs(RAW_ARIA_VIDEO_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.raw_video_path = os.path.join(
            RAW_ARIA_VIDEO_DIR,
            f"aria_raw_{self.serial_number}_{stamp}.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*RAW_ARIA_VIDEO_CODEC)
        self.raw_video_writer = cv2.VideoWriter(
            self.raw_video_path,
            fourcc,
            RAW_ARIA_VIDEO_FPS,
            RAW_ARIA_VIDEO_SIZE,
        )
        if not self.raw_video_writer.isOpened():
            print(f"[AriaCamera] Failed to open raw video writer: "
                  f"{self.raw_video_path}")
            self.raw_video_writer.release()
            self.raw_video_writer = None
            self.raw_video_path = None
            return
        print(f"[AriaCamera] Saving raw Aria video -> {self.raw_video_path}")

    def _write_raw_video_frame(self, frame_bgr):
        if not RAW_ARIA_VIDEO_SAVE_ENABLED or frame_bgr is None:
            return

        with self.raw_video_lock:
            self._ensure_raw_video_writer_locked()
            if self.raw_video_writer is None:
                return

            frame = frame_bgr
            if frame.shape[1] != RAW_ARIA_VIDEO_SIZE[0] or frame.shape[0] != RAW_ARIA_VIDEO_SIZE[1]:
                frame = cv2.resize(
                    frame,
                    RAW_ARIA_VIDEO_SIZE,
                    interpolation=cv2.INTER_AREA,
                )
            self.raw_video_writer.write(frame)
            self.raw_video_frame_count += 1

    def _release_raw_video_writer(self):
        with self.raw_video_lock:
            if self.raw_video_writer is not None:
                self.raw_video_writer.release()
                self.raw_video_writer = None
                raw_path = self.raw_video_path
                raw_count = self.raw_video_frame_count
            else:
                raw_path = None
                raw_count = 0
            if self.cropped_video_writer is not None:
                self.cropped_video_writer.release()
                self.cropped_video_writer = None
                cropped_path = self.cropped_video_path
                cropped_count = self.cropped_video_frame_count
            else:
                cropped_path = None
                cropped_count = 0

        if raw_path:
            print(f"[AriaCamera] Re-encoding raw video ({raw_count} frames)...")
            _reencode_h264(raw_path)
            print(f"[AriaCamera] Raw video saved: {raw_path}")
        if cropped_path:
            print(f"[AriaCamera] Re-encoding cropped video ({cropped_count} frames)...")
            _reencode_h264(cropped_path)
            print(f"[AriaCamera] Cropped video saved: {cropped_path}")

    def _ensure_cropped_video_writer_locked(self):
        if (not RAW_ARIA_VIDEO_SAVE_ENABLED
                or self.cropped_video_writer is not None):
            return

        os.makedirs(RAW_ARIA_VIDEO_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.cropped_video_path = os.path.join(
            RAW_ARIA_VIDEO_DIR,
            f"aria_cropped_{self.serial_number}_{stamp}.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*RAW_ARIA_VIDEO_CODEC)
        self.cropped_video_writer = cv2.VideoWriter(
            self.cropped_video_path,
            fourcc,
            RAW_ARIA_VIDEO_FPS,
            RAW_ARIA_CROPPED_VIDEO_SIZE,
        )
        if not self.cropped_video_writer.isOpened():
            print(f"[AriaCamera] Failed to open cropped video writer: "
                  f"{self.cropped_video_path}")
            self.cropped_video_writer.release()
            self.cropped_video_writer = None
            self.cropped_video_path = None
            return
        print(f"[AriaCamera] Saving cropped Aria video -> {self.cropped_video_path}")

    def _write_cropped_video_frame(self, frame_bgr):
        if not RAW_ARIA_VIDEO_SAVE_ENABLED or frame_bgr is None:
            return

        with self.raw_video_lock:
            self._ensure_cropped_video_writer_locked()
            if self.cropped_video_writer is None:
                return

            frame = frame_bgr
            if (frame.shape[1] != RAW_ARIA_CROPPED_VIDEO_SIZE[0]
                    or frame.shape[0] != RAW_ARIA_CROPPED_VIDEO_SIZE[1]):
                frame = cv2.resize(
                    frame,
                    RAW_ARIA_CROPPED_VIDEO_SIZE,
                    interpolation=cv2.INTER_AREA,
                )
            self.cropped_video_writer.write(frame)
            self.cropped_video_frame_count += 1

    def _buffer_audio_chunk_locked(self, audio_ts_ns, sample_rate_hz, pcm16):
        if sample_rate_hz <= 0 or pcm16.size == 0:
            return

        if self.audio_chunks and self.audio_chunks[-1]["sample_rate_hz"] != sample_rate_hz:
            self.audio_chunks.clear()

        chunk = {
            "audio_ts_ns": int(audio_ts_ns),
            "sample_rate_hz": int(sample_rate_hz),
            "chunk_pcm16": np.asarray(pcm16, dtype=np.int16).copy(),
            "num_samples": int(pcm16.size),
        }
        self.audio_chunks.append(chunk)
        self.latest_audio_timestamp = int(audio_ts_ns)
        self.latest_audio_sample_rate_hz = int(sample_rate_hz)
        self._prune_audio_chunks_locked()

    def _prune_audio_chunks_locked(self):
        total_duration_sec = 0.0
        for chunk in self.audio_chunks:
            if chunk["sample_rate_hz"] > 0:
                total_duration_sec += chunk["num_samples"] / chunk["sample_rate_hz"]

        while len(self.audio_chunks) > 1 and total_duration_sec > self.audio_buffer_max_sec:
            dropped = self.audio_chunks.popleft()
            if dropped["sample_rate_hz"] > 0:
                total_duration_sec -= dropped["num_samples"] / dropped["sample_rate_hz"]

    def _parse_inference_v2_payload(self, payload):
        if len(payload) < INFERENCE_V2_HEADER_SIZE:
            raise ValueError("Inference v2 payload header is incomplete")

        (
            version,
            flags,
            _reserved,
            rgb_ts_ns,
            gaze_x,
            gaze_y,
            audio_ts_ns,
            audio_sample_rate_hz,
            audio_num_samples,
            rgb_jpg_len,
        ) = struct.unpack_from(INFERENCE_V2_HEADER_FMT, payload, 0)

        if version != INFERENCE_V2_VERSION:
            raise ValueError(
                f"Unsupported inference payload version: {version} "
                f"(expected {INFERENCE_V2_VERSION})"
            )

        rgb_start = INFERENCE_V2_HEADER_SIZE
        rgb_end = rgb_start + rgb_jpg_len
        audio_bytes_len = int(audio_num_samples) * np.dtype(np.int16).itemsize
        payload_end = rgb_end + audio_bytes_len
        if rgb_end > len(payload) or payload_end != len(payload):
            raise ValueError("Inference v2 payload size mismatch")
        if not (flags & INFERENCE_V2_AUDIO_PRESENT_FLAG) and audio_num_samples > 0:
            raise ValueError("Inference v2 audio flag mismatch")

        return {
            "rgb_ts_ns": int(rgb_ts_ns),
            "gaze_x": float(gaze_x),
            "gaze_y": float(gaze_y),
            "audio_ts_ns": int(audio_ts_ns),
            "audio_sample_rate_hz": int(audio_sample_rate_hz),
            "audio_pcm16": np.frombuffer(
                payload[rgb_end:payload_end], dtype="<i2"
            ).copy(),
            "rgb_bytes": payload[rgb_start:rgb_end],
        }


    def _receive_stream(self):
        """Background thread that receives RGB images and gaze points."""
        first_frame_received = False
        last_connect_error_log_time = 0.0

        while self.running:
            client = None
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(SOCKET_TIMEOUT_SEC)
                client.connect((self.host, self.port))
                print(f"[AriaCamera] Connected to {self.host}:{self.port}")
                connection_start_time = time.time()
                consecutive_timeouts = 0

                while self.running:
                    # Calculate remaining timeout for initialization
                    if not first_frame_received:
                        elapsed = time.time() - connection_start_time
                        remaining_timeout = INIT_TIMEOUT_SEC - elapsed
                        
                        if remaining_timeout <= 0:
                            print("[AriaCamera] Timeout: No data received within 30 seconds")
                            break
                        
                        client.settimeout(min(remaining_timeout, SOCKET_TIMEOUT_SEC))
                    else:
                        client.settimeout(SOCKET_TIMEOUT_SEC)
                    
                    try:
                        # Receive header (payload size)
                        header = self._recv_all(client, 4)
                        if header is None:
                            if not first_frame_received:
                                elapsed = time.time() - connection_start_time
                                print(f"[AriaCamera] Waiting for Aria initialization... ({elapsed:.1f}s / {INIT_TIMEOUT_SEC:.0f}s)")
                                time.sleep(0.5)
                                continue
                            else:
                                print("[AriaCamera] Connection closed")
                                break

                        payload_size = struct.unpack("<I", header)[0]

                        # Receive payload
                        payload = self._recv_all(client, payload_size)
                        if payload is None:
                            if not first_frame_received:
                                elapsed = time.time() - connection_start_time
                                print(f"[AriaCamera] Waiting for Aria initialization... ({elapsed:.1f}s / {INIT_TIMEOUT_SEC:.0f}s)")
                                time.sleep(0.5)
                                continue
                            else:
                                print("[AriaCamera] Connection closed")
                                break

                        parsed = self._parse_inference_v2_payload(payload)
                        
                        # Decode RGB image
                        rgb_image = cv2.imdecode(
                            np.frombuffer(parsed["rgb_bytes"], np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        
                        if rgb_image is None:
                            print("[AriaCamera] RGB decode failed")
                            continue

                        self._enqueue_video_frame("raw", rgb_image)
                        
                        # Rotate and scale gaze point
                        gaze_point = np.array([parsed["gaze_x"], parsed["gaze_y"]])
                        rotated_gaze = self._rotate_coordinate(gaze_point, self.original_size, self.original_size)
                        #scaled_gaze = self._scale_coordinate(rotated_gaze)
                        
                        # Update latest data (thread-safe)
                        with self.lock:
                            self.latest_timestamp = parsed["rgb_ts_ns"]
                            self.latest_rgb_image = rgb_image
                            self.latest_gaze_point = rotated_gaze
                            self.latest_frame_monotonic = time.monotonic()
                            self._buffer_audio_chunk_locked(
                                parsed["audio_ts_ns"],
                                parsed["audio_sample_rate_hz"],
                                parsed["audio_pcm16"],
                            )
                        
                        # Mark first frame received
                        consecutive_timeouts = 0
                        if not first_frame_received:
                            first_frame_received = True
                            print("[AriaCamera] First frame received! Aria initialized.")
                            if not self.first_data_event.is_set():
                                self.first_data_event.set()
                        
                    except socket.timeout:
                        if not first_frame_received:
                            elapsed = time.time() - connection_start_time
                            print(f"[AriaCamera] Still waiting... ({elapsed:.1f}s / {INIT_TIMEOUT_SEC:.0f}s)")
                            continue
                        else:
                            consecutive_timeouts += 1
                            stale_age = (
                                time.monotonic() - self.latest_frame_monotonic
                                if self.latest_frame_monotonic else 0.0
                            )
                            print(
                                f"[AriaCamera] Socket timeout "
                                f"({consecutive_timeouts}/{MAX_CONSECUTIVE_SOCKET_TIMEOUTS}); "
                                f"last frame {stale_age:.1f}s ago"
                            )
                            if consecutive_timeouts >= MAX_CONSECUTIVE_SOCKET_TIMEOUTS:
                                print("[AriaCamera] Reconnecting after repeated socket timeouts...")
                                break
                            continue

                if client is not None:
                    client.close()

            except (socket.timeout, TimeoutError) as e:
                now = time.monotonic()
                if now - last_connect_error_log_time >= CONNECT_ERROR_LOG_INTERVAL_SEC:
                    print(
                        f"[AriaCamera] Connect/read timed out while reconnecting to "
                        f"{self.host}:{self.port}: {e}"
                    )
                    last_connect_error_log_time = now
            except OSError as e:
                now = time.monotonic()
                if now - last_connect_error_log_time >= CONNECT_ERROR_LOG_INTERVAL_SEC:
                    print(
                        f"[AriaCamera] Socket error while reconnecting to "
                        f"{self.host}:{self.port}: {e}"
                    )
                    last_connect_error_log_time = now
            except Exception as e:
                print(f"[AriaCamera] Stream error: {e}")
            finally:
                if client is not None:
                    try:
                        client.close()
                    except OSError:
                        pass

            time.sleep(RECONNECT_DELAY_SEC)  # Wait before reconnecting

    # ============================================================
    # PUBLIC API
    # ============================================================

    def get_observation(self, mode='raw'):
        """
        Returns two images: (rgb_image, rgb_with_gaze_image)
        Both are numpy arrays of shape (H, W, 3) in RGB format.
        """
        with self.lock:
            if self.latest_rgb_image is None or self.latest_gaze_point is None:
                return None, None
            if mode == 'raw':
                return self.latest_rgb_image.copy(), self.latest_gaze_point.copy()
            elif mode == 'keyframe_cropped':
                #image, x0, y0 = gaze_crop(self.latest_rgb_image.copy(), self.latest_gaze_point[0], self.latest_gaze_point[1], crop_size=CROP_SIZE)
                image, x0, y0 = gaze_crop(self.latest_rgb_image.copy(), self.latest_gaze_point[0], self.latest_gaze_point[1]+0.0, crop_size=CROP_SIZE)
                #image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                return image, self.latest_gaze_point.copy()
            elif mode == 'downsampled':
                image, gx_new, gy_new = downsampled_crop(self.latest_rgb_image.copy(), self.latest_gaze_point[0], self.latest_gaze_point[1]+0.0, crop_size=DOWNSAMPLED_CROP_SIZE)
                return image, np.array([gx_new, gy_new])
            elif mode == 'human_view_streamed_and_keyframe_cropped':
                gx_adj = self.latest_gaze_point[0] + self.gaze_offset_x
                gy_adj = self.latest_gaze_point[1] + self.gaze_offset_y
                keyframe_image, gx, gy = gaze_crop(self.latest_rgb_image.copy(), gx_adj, gy_adj, crop_size=CROP_SIZE)
                streamed_image, gx_rescaled, gy_rescaled = downsampled_crop(self.latest_rgb_image.copy(), gx_adj, gy_adj, crop_size=512)
                self._enqueue_video_frame("cropped", keyframe_image)
                return keyframe_image, streamed_image, gx_adj, gy_adj, gx_rescaled, gy_rescaled
            else:
                raise ValueError(f"Invalid mode: {mode}")

    def get_audio_chunk(self, drain=True):
        """Return buffered inference audio chunks accumulated since the last read."""
        with self.lock:
            if not self.audio_chunks:
                return {
                    "chunk_pcm16": np.zeros((0,), dtype=np.int16),
                    "audio_ts_ns": int(self.latest_audio_timestamp),
                    "sample_rate_hz": int(self.latest_audio_sample_rate_hz),
                    "num_samples": 0,
                }

            chunks = list(self.audio_chunks)
            if drain:
                self.audio_chunks.clear()

        concatenated = np.concatenate(
            [chunk["chunk_pcm16"] for chunk in chunks],
            axis=0,
        )
        return {
            "chunk_pcm16": concatenated,
            "audio_ts_ns": int(chunks[0]["audio_ts_ns"]),
            "sample_rate_hz": int(chunks[-1]["sample_rate_hz"]),
            "num_samples": int(concatenated.size),
        }
    
    def set_gaze_offset(self, x_off: float, y_off: float):
        """Set gaze calibration offsets in FULL-RESOLUTION pixels.

        The offsets are added to the raw gaze point before gaze-based
        cropping in ``get_observation`` (human_view_streamed_and_keyframe_cropped
        mode). Call with (0, 0) to disable.
        """
        self.gaze_offset_x = float(x_off)
        self.gaze_offset_y = float(y_off)
        print(f"[AriaCamera] Gaze calibration offset set: "
              f"x={self.gaze_offset_x:+.1f}px y={self.gaze_offset_y:+.1f}px "
              f"(full-res)")

    def get_image(self, mode='raw'):
        """Returns the latest RGB image without gaze overlay."""
        with self.lock:
            if mode == 'raw':
                return self.latest_rgb_image.copy() if self.latest_rgb_image is not None else None
            elif mode == 'keyframe_cropped':
                image, x0, y0 = gaze_crop(self.latest_rgb_image.copy(), self.latest_gaze_point[0], self.latest_gaze_point[1], crop_size=CROP_SIZE)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                return image
            elif mode == 'downsampled':
                image, gx_new, gy_new = downsampled_crop(self.latest_rgb_image.copy(), self.latest_gaze_point[0], self.latest_gaze_point[1], crop_size=DOWNSAMPLED_CROP_SIZE)
                return image
            else:
                raise ValueError(f"Invalid mode: {mode}")
    
    
    def get_gaze_point(self,mode='raw'):
        """Returns the latest gaze point (x, y) in scaled coordinates."""
        with self.lock:
            if mode == 'raw':
                return self.latest_gaze_point.copy() if self.latest_gaze_point is not None else None
            elif mode == 'keyframe_cropped':
                return gaze_crop(self.latest_rgb_image.copy(), self.latest_gaze_point[0], self.latest_gaze_point[1], crop_size=CROP_SIZE)
            elif mode == 'downsampled':
                self.latest_gaze_point = self._scale_coordinate(self.latest_gaze_point)
                return self.latest_gaze_point.copy() if self.latest_gaze_point is not None else None
            else:
                raise ValueError(f"Invalid mode: {mode}")
    
    def get_timestamp(self):
        """Returns the latest timestamp."""
        with self.lock:
            return self.latest_timestamp

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
    
    def save_gaze(self, filename="home/alin/projects/piper_project/aria_gaze.png"):
        """
        Save current RGB image with gaze point overlay.
        
        Args:
            filename: Output filename (default: "aria_gaze.png")
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        with self.lock:
            if self.latest_rgb_image is None or self.latest_gaze_point is None:
                print("[AriaCamera] No image or gaze point available to save")
                return False
            
            # Create image with gaze overlay
            img_with_gaze = self.latest_rgb_image.copy()
            img_with_gaze = draw_gaze_point(
                img_with_gaze, 
                self.latest_gaze_point[0], 
                self.latest_gaze_point[1]
            )
            
            # Save image
            success = cv2.imwrite(filename, img_with_gaze)
            
            return success
    
    def stop(self):
        """Stop the receiver thread and close connection."""
        self.running = False
        if self._video_writer_thread is not None:
            try:
                self._video_write_queue.put_nowait(None)  # sentinel
            except queue.Full:
                pass
            self._video_writer_thread.join(timeout=5.0)
            self._video_writer_thread = None
        if self._video_dropped_frames:
            print(f"[AriaCamera] Video writer dropped "
                  f"{self._video_dropped_frames} frames total")
        self._release_raw_video_writer()
