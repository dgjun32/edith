"""eval_fam_hri.py — FAM-HRI (subtask, keyframe) pair executor.

Robot-control main script for the raw-streaming FAM-HRI pipeline
(HANDOFF_STREAMING.md). The SAM3-free server (:8001) turns one spoken session
into N (subtask, gaze-keyframe) pairs and POSTs them here; this script
executes them sequentially through the keyframe-conditioned pi0.5 policy.

  server (:8001) --POST /subtasks {subtasks, keyframes_b64}--> here (:8002)
                                                                 |
                                       pi0.5 policy server (ws, :8000)
                        observation: 3x RealSense + keyframe crop + prompt

Session keys (this terminal):
  s = start session   -> server clears buffers, Mac streams; SPEAK NOW
  e = end of speech   -> server runs ASR+GPT-4o+OPTICS pipeline; robot
                         executes the resulting pairs as they arrive
  r = reset episode   -> abort session, clear queue, reset robot, back to IDLE
  n = skip current subtask        q = quit

While the robot is executing you may press 's' again to speak the next
command; new pairs are appended to the FIFO.

The policy consumes the keyframe as
``observation.images.human_view_keyframe_cropped`` (RGB), same as
eval_openpi_ours_with_voice_260503.py — serve a keyframe-conditioned
checkpoint on :8000, not a language-only one.
"""

import argparse
import base64
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import cv2
import numpy as np

from edith.envs.piper_env import PiperRobotEnv
from edith.utils.time_utils import Timer

from openpi_client import websocket_client_policy
from ours_utils.keyboard import (
    poll_keypress as _poll_keypress,
    terminal_keypress_mode as _terminal_keypress_mode,
)

# ---------------------------------------------------------------------------
# Constants — edit these instead of CLI args
# ---------------------------------------------------------------------------

POLICY_HOST          = "localhost"
POLICY_PORT          = 8000
LISTEN_HOST          = "127.0.0.1"   # FAM-HRI server posts from this machine
LISTEN_PORT          = 8002
PIPELINE_URL         = "http://127.0.0.1:8001"  # FAM-HRI server (session ctrl)
ACTION_CHUNKSIZE     = 8
COMPLETION_THRESH    = 0.8           # completion_prob >= this → subtask done
MIN_SUBTASK_STEPS    = 40            # ignore early completion spikes
HOLD_POLL_SEC        = 0.2           # queue poll period while holding pose
KEYFRAME_SIZE        = 512           # fallback blank keyframe size

CAMERA_IDS = {
    'top_camera':         '242322070028',
    'left_wrist_camera':  '244222076964',
    'right_wrist_camera': '244222071848',
}


class EpisodeEnd(Exception):
    """'r' — abort session, reset robot, clear queue, back to IDLE."""


class ProgramQuit(Exception):
    """'q' — exit the program."""


# ---------------------------------------------------------------------------
# (subtask, keyframe) FIFO + HTTP listener
# ---------------------------------------------------------------------------

class SubtaskQueue:
    """Thread-safe FIFO of {'instruction', 'keyframe_bgr', 'referent'} items."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self.current: Optional[dict] = None
        self.done: List[str] = []
        self._accepting = False   # False while IDLE — POST /subtasks rejected

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    @accepting.setter
    def accepting(self, value: bool):
        with self._lock:
            self._accepting = bool(value)

    def extend_items(self, items: List[dict]) -> int:
        with self._lock:
            for it in items:
                self._pending.append(it)
            return len(items)

    def pop_next(self) -> Optional[dict]:
        with self._lock:
            if self.current is not None:
                self.done.append(self.current["instruction"])
                self.current = None
            if self._pending:
                self.current = self._pending.popleft()
            return self.current

    def clear(self):
        with self._lock:
            if self.current is not None:
                self.done.append(self.current["instruction"])
            self.current = None
            self._pending.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "session_active": self._accepting,
                "current": (self.current or {}).get("instruction"),
                "pending": [p["instruction"] for p in self._pending],
                "done": list(self.done),
            }


class KeyframeDisplay:
    """CV2 viewport showing the latest session's model-input keyframes.

    The HTTP listener thread only stores data here; render() must be called
    from the main thread (cv2 GUI is not thread-safe). Shows one thumbnail
    per (subtask, keyframe) pair: green border = currently executing,
    dimmed = done, plain = pending.
    """

    WINDOW = "FAM-HRI keyframes (model input)"
    THUMB = 320
    CAPTION_H = 46

    def __init__(self):
        self._lock = threading.Lock()
        self._batch_id = 0
        self._thumbs: List[np.ndarray] = []
        self._labels: List[str] = []
        self._status: List[str] = []       # pending | current | done
        self._enabled = True               # disabled on first imshow failure

    def set_batch(self, items: List[dict]) -> None:
        """Replace the display with a new session's pairs; tag the items."""
        with self._lock:
            self._batch_id += 1
            self._thumbs, self._labels, self._status = [], [], []
            for i, it in enumerate(items):
                it["disp_batch"] = self._batch_id
                it["disp_idx"] = i
                kf = it.get("keyframe_bgr")
                if kf is None:
                    kf = np.zeros((KEYFRAME_SIZE, KEYFRAME_SIZE, 3), np.uint8)
                self._thumbs.append(
                    cv2.resize(kf, (self.THUMB, self.THUMB)))
                self._labels.append(it["instruction"])
                self._status.append("pending")

    def mark_current(self, item: Optional[dict]) -> None:
        with self._lock:
            if item is None or item.get("disp_batch") != self._batch_id:
                return
            for i, s in enumerate(self._status):
                if s == "current":
                    self._status[i] = "done"
            self._status[item["disp_idx"]] = "current"

    def mark_all_current_done(self) -> None:
        with self._lock:
            self._status = ["done" if s == "current" else s
                            for s in self._status]

    def clear(self) -> None:
        with self._lock:
            self._thumbs, self._labels, self._status = [], [], []
        if self._enabled:
            try:
                cv2.destroyWindow(self.WINDOW)
            except Exception:  # noqa: BLE001
                pass

    def render(self) -> None:
        """Main thread only. Draw the strip and imshow it."""
        if not self._enabled:
            return
        with self._lock:
            if not self._thumbs:
                return
            thumbs = [t.copy() for t in self._thumbs]
            labels = list(self._labels)
            status = list(self._status)

        tiles = []
        th, cap = self.THUMB, self.CAPTION_H
        for i, (img, label, st) in enumerate(zip(thumbs, labels, status)):
            if st == "done":
                img = (img * 0.35).astype(np.uint8)
            tile = np.zeros((th + cap, th, 3), dtype=np.uint8)
            tile[:th] = img
            color = ((0, 255, 0) if st == "current"
                     else (160, 160, 160) if st == "done" else (255, 255, 255))
            text = f"[{i + 1}] {label}"
            if len(text) > 34:
                text = text[:33] + "…"
            cv2.putText(tile, text, (6, th + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
            cv2.putText(tile, st.upper(), (6, th + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            border = (0, 255, 0) if st == "current" else (60, 60, 60)
            cv2.rectangle(tile, (0, 0), (th - 1, th + cap - 1), border,
                          3 if st == "current" else 1)
            tiles.append(tile)

        strip = np.concatenate(tiles, axis=1)
        try:
            cv2.imshow(self.WINDOW, strip)
        except Exception as exc:  # noqa: BLE001
            print(f"[Display] disabled (no GUI available: {exc})")
            self._enabled = False


def start_subtask_listener(queue: SubtaskQueue, host: str, port: int,
                           display: Optional[KeyframeDisplay] = None
                           ) -> ThreadingHTTPServer:
    """HTTP endpoint the FAM-HRI server pushes pairs + progress events into."""

    class Handler(BaseHTTPRequestHandler):

        def _reply(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/status"):
                self._reply(200, {"ok": True, **queue.snapshot()})
            else:
                self._reply(404, {"ok": False, "error": "unknown path"})

        def do_POST(self):
            path = self.path.rstrip("/")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:  # noqa: BLE001
                self._reply(400, {"ok": False, "error": str(exc)})
                return

            # Pipeline progress events pushed by the FAM-HRI server.
            if path == "/event":
                msg = str(data.get("event", "")).strip()
                if msg:
                    print(f"\n[{time.strftime('%H:%M:%S')}] [Pipeline] {msg}")
                self._reply(200, {"ok": True})
                return

            if path != "/subtasks":
                self._reply(404, {"ok": False, "error": "unknown path"})
                return
            subtasks = data.get("subtasks", [])
            if not isinstance(subtasks, list):
                self._reply(400, {"ok": False,
                                  "error": "'subtasks' must be a list"})
                return
            if not queue.accepting:
                self._reply(409, {"ok": False,
                                  "error": "idle — press 's' in eval_fam_hri first"})
                return

            keyframes: List[Optional[np.ndarray]] = []
            for b64 in data.get("keyframes_b64", []):
                try:
                    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                    keyframes.append(cv2.imdecode(arr, cv2.IMREAD_COLOR))
                except Exception:  # noqa: BLE001
                    keyframes.append(None)

            referents = data.get("referents") or []
            items = []
            for i, s in enumerate(subtasks):
                s = str(s).strip()
                if not s:
                    continue
                items.append({
                    "instruction": s,
                    "keyframe_bgr": (keyframes[i]
                                     if i < len(keyframes) else None),
                    "referent": (referents[i] if i < len(referents) else ""),
                })
            if display is not None:
                display.set_batch(items)   # tag before queueing
            n = queue.extend_items(items)
            with_kf = sum(1 for k in keyframes if k is not None)
            print(f"\n[{time.strftime('%H:%M:%S')}] [Subtasks] +{n} pair(s) "
                  f"received ({with_kf} with keyframe): {subtasks}")
            self._reply(200, {"ok": True, "queued": n, **queue.snapshot()})

        def log_message(self, fmt, *args):  # silence per-request access logs
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="SubtaskListener").start()
    print(f"[Subtasks] Listening on http://{host}:{port}/subtasks")
    return server


# ---------------------------------------------------------------------------
# Session signalling + keys
# ---------------------------------------------------------------------------

def _notify_session(pipeline_url: str, action: str) -> bool:
    """POST /session/<action> on the FAM-HRI server (fail-soft)."""
    if not pipeline_url:
        return False
    import urllib.error
    import urllib.request
    url = f"{pipeline_url.rstrip('/')}/session/{action}"
    try:
        req = urllib.request.Request(url, data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2.0):
            pass
        print(f"[Session] {action} → {url}")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            # Not an error: 'e' with no active session (already finished).
            print(f"[Session] {action} ignored — no active session.")
            if action == "finish":
                print("[Eval]   → 's' = start a NEW session (speak again), "
                      "'r' = reset robot/episode.")
            return False
        print(f"[Session] WARNING: could not signal {action} ({url}): {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[Session] WARNING: could not signal {action} ({url}): {exc}")
        return False


def _handle_key(key: int, pipeline_url: str):
    """Global key handling inside a running episode."""
    if key in (ord('q'), ord('Q'), 27):
        raise ProgramQuit
    if key in (ord('r'), ord('R')):
        raise EpisodeEnd
    if key in (ord('s'), ord('S')):
        print("[Eval] ▶ session start — SPEAK NOW (press 'e' when done).")
        _notify_session(pipeline_url, "start")
    if key in (ord('e'), ord('E')):
        print("[Eval] ■ end of speech — running pipeline "
              "(watch [Pipeline] lines)...")
        _notify_session(pipeline_url, "finish")


def wait_for_start(pipeline_url: str):
    """IDLE state: block until 's' is pressed ('q' quits)."""
    print("[Eval] IDLE — press 's' to start a session (then speak, 'e' when "
          "done). 'q' quits.")
    while True:
        key = _poll_keypress(delay_ms=int(HOLD_POLL_SEC * 1000))
        if key in (ord('s'), ord('S')):
            print("[Eval] ▶ session start — SPEAK NOW (press 'e' when done).")
            _notify_session(pipeline_url, "start")
            return
        if key in (ord('q'), ord('Q'), 27):
            raise ProgramQuit
        time.sleep(HOLD_POLL_SEC)


# ---------------------------------------------------------------------------
# Robot control loop
# ---------------------------------------------------------------------------

def _new_policy_client():
    return websocket_client_policy.WebsocketClientPolicy(
        host=POLICY_HOST, port=POLICY_PORT)


def _policy_port_open(timeout: float = 0.3) -> bool:
    import socket
    try:
        with socket.create_connection((POLICY_HOST, POLICY_PORT),
                                      timeout=timeout):
            return True
    except OSError:
        return False


def _connect_policy_responsive(pipeline_url: str, render) -> object:
    """Connect to the policy server WITHOUT freezing the GUI/key loop.

    WebsocketClientPolicy blocks until the server is reachable, so we probe
    the TCP port in a loop first — rendering the keyframe viewport and
    handling s/e/r/q keys the whole time — and only construct the client
    once the port accepts connections.
    """
    announced = False
    while not _policy_port_open():
        if not announced:
            print(f"[Eval] policy ws://{POLICY_HOST}:{POLICY_PORT} is NOT up — "
                  "waiting for it (viewport + keys stay active)...")
            announced = True
        render()
        _handle_key(_poll_keypress(delay_ms=200), pipeline_url)
        time.sleep(0.1)
    if announced:
        print("[Eval] policy server is up — connecting.")
    return _new_policy_client()


def _blank_keyframe_rgb() -> np.ndarray:
    return np.zeros((KEYFRAME_SIZE, KEYFRAME_SIZE, 3), dtype=np.uint8)


def _build_observation(obs: dict, prompt: str,
                       keyframe_rgb: np.ndarray) -> dict:
    images = obs["images"]
    state = np.concatenate([v for v in obs["robot_state"].values()])
    return {
        "observation.images.top_view":                    images["top_camera_rgb"],
        "observation.images.left_wrist_view":             images["left_wrist_camera_rgb"],
        "observation.images.right_wrist_view":            images["right_wrist_camera_rgb"],
        "observation.images.human_view_keyframe_cropped": keyframe_rgb,
        "prompt":                                         prompt,
        "observation.state":                              state,
    }


def run_episode(env, queue: SubtaskQueue, args,
                display: Optional[KeyframeDisplay] = None):
    """Pop (subtask, keyframe) pairs and execute until EpisodeEnd/ProgramQuit."""
    # Lazy policy connection: WebsocketClientPolicy BLOCKS until the policy
    # server is reachable, so connecting eagerly would freeze the key loop
    # (s/e/r/q dead) whenever :8000 is down. Connect only when there is
    # actually a subtask to execute.
    client       = None
    timer        = Timer(env.control_hz)
    subtask_step = 0
    item         = queue.pop_next()
    keyframe_rgb = _blank_keyframe_rgb()
    waiting_announced = False

    def _render():
        if display is not None:
            display.render()

    def _activate(it: dict):
        nonlocal keyframe_rgb, subtask_step
        subtask_step = 0
        kf = it.get("keyframe_bgr")
        keyframe_rgb = (cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)
                        if kf is not None else _blank_keyframe_rgb())
        if display is not None:
            display.mark_current(it)
        snap = queue.snapshot()
        ref = f" (referent: {it['referent']})" if it.get("referent") else ""
        print(f"[Eval] ▶ Subtask: '{it['instruction']}'{ref} "
              f"(pending: {len(snap['pending'])})")

    if item is not None:
        _activate(item)

    while True:
        # ---- no subtask: hold pose, wait for pairs ----
        if item is None:
            if not waiting_announced:
                print("[Eval] Holding pose — waiting for (subtask, keyframe) "
                      "pairs (s/e/r/n/q keys active)...")
                waiting_announced = True
            _render()
            _handle_key(_poll_keypress(delay_ms=int(HOLD_POLL_SEC * 1000)),
                        args.pipeline_url)
            # cv2.waitKey returns immediately when no window exists; pace the
            # hold loop explicitly so it doesn't busy-spin headless.
            time.sleep(HOLD_POLL_SEC)
            item = queue.pop_next()
            if item is not None:
                waiting_announced = False
                _activate(item)
            continue

        # ---- observe → infer ----
        _render()
        if client is None:
            client = _connect_policy_responsive(args.pipeline_url, _render)
        obs = env.get_observation()["timestamp"]
        out = client.infer(_build_observation(
            obs, item["instruction"], keyframe_rgb))
        action_chunk    = out["actions"][:args.chunksize, :]
        completion_prob = (float(out["completion_prob"])
                           if "completion_prob" in out else None)
        cp_text = f"{completion_prob:.3f}" if completion_prob is not None else "N/A"
        print(f"[Eval] '{item['instruction']}' step={subtask_step} cp={cp_text}")

        # ---- execute chunk ----
        skip_requested = False
        for raw_action in action_chunk:
            env.step({
                'left_arm_positions':     raw_action[0:6],
                'left_gripper_position':  raw_action[6:7],
                'right_arm_positions':    raw_action[7:13],
                'right_gripper_position': raw_action[13:14],
            })
            subtask_step += 1
            key = _poll_keypress()
            if key in (ord('n'), ord('N')):
                skip_requested = True
                break
            _handle_key(key, args.pipeline_url)
            timer.sleep()

        # ---- advance decision ----
        advance = False
        if skip_requested:
            print(f"[Eval] Manual skip of '{item['instruction']}'")
            advance = True
        elif completion_prob is not None and completion_prob >= args.completion_thresh:
            if subtask_step < args.min_steps:
                print(f"  [Auto] cp={completion_prob:.3f} >= "
                      f"{args.completion_thresh} but step={subtask_step} < "
                      f"{args.min_steps} — skip")
            else:
                print(f"  [Auto] cp={completion_prob:.3f} — subtask done.")
                advance = True

        if advance:
            item = queue.pop_next()
            waiting_announced = False
            # Fresh websocket per subtask, mirroring the reference eval scripts
            # (clears any policy-server-side chunk state for the new prompt).
            # Lazy: reconnects on the next infer, keeping keys responsive.
            client = None
            if item is not None:
                _activate(item)
            elif display is not None:
                display.mark_all_current_done()
                _render()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="FAM-HRI pair executor.")
    p.add_argument("--listen-host", default=LISTEN_HOST)
    p.add_argument("--listen-port", type=int, default=LISTEN_PORT)
    p.add_argument("--policy-port", type=int, default=POLICY_PORT)
    p.add_argument("--chunksize", type=int, default=ACTION_CHUNKSIZE)
    p.add_argument("--completion-thresh", type=float, default=COMPLETION_THRESH)
    p.add_argument("--min-steps", type=int, default=MIN_SUBTASK_STEPS,
                   help="Min steps per subtask before completion_prob can advance it.")
    p.add_argument("--pipeline-url", default=PIPELINE_URL,
                   help="FAM-HRI server base URL for /session signals "
                        "('' to disable).")
    p.add_argument("--show-viewport", action="store_true",
                   help="Show the CV2 keyframe viewport. Off by default — the "
                        "server saves the combined strip to "
                        "server_debug/session_*/keyframes.jpg instead.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    POLICY_PORT = args.policy_port

    queue    = SubtaskQueue()
    display  = KeyframeDisplay() if args.show_viewport else None
    listener = start_subtask_listener(queue, args.listen_host, args.listen_port,
                                      display=display)
    env      = PiperRobotEnv(camera_ids=CAMERA_IDS)

    print("[Eval] FAM-HRI pair executor ready. "
          f"policy=ws://{POLICY_HOST}:{POLICY_PORT}  "
          f"completion_thresh={args.completion_thresh}  min_steps={args.min_steps}")
    print("[Eval] Keys: s=start session(speak)  e=end of speech(process)  "
          "n=skip  r=reset episode  q=quit")

    try:
        with _terminal_keypress_mode():
            while True:
                try:
                    wait_for_start(args.pipeline_url)     # IDLE until 's'
                    queue.accepting = True
                    run_episode(env, queue, args, display=display)
                except EpisodeEnd:
                    print("[Eval] Episode reset — aborting session, clearing "
                          "queue, back to IDLE.")
                    queue.accepting = False
                    _notify_session(args.pipeline_url, "stop")
                    queue.clear()
                    if display is not None:
                        display.clear()
                    env.reset()
                except ProgramQuit:
                    print("[Eval] Quit requested.")
                    if queue.accepting:
                        queue.accepting = False
                        _notify_session(args.pipeline_url, "stop")
                    break
    except KeyboardInterrupt:
        print("\n[Eval] Interrupted.")
        if queue.accepting:
            queue.accepting = False
            _notify_session(args.pipeline_url, "stop")
    finally:
        print("[Eval] Cleaning up...")
        listener.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        snap = queue.snapshot()
        if snap["done"]:
            print("[Eval] Executed subtasks:")
            for i, s in enumerate(snap["done"], 1):
                print(f"  [{i}] {s}")
        env.safe_end()
