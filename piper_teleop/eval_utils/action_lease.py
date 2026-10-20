"""Refresh the latest policy target so a stalled inference cannot starve control."""

import threading
import time

import numpy as np


class ActionLease:
    def __init__(self, robot, action, rate_hz):
        self.robot = robot
        self.period = 1.0 / rate_hz
        self.lock = threading.Lock()
        self.action = np.asarray(action, dtype=np.float64).copy()
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None

    def start(self):
        self.robot.command(self.action)
        self.thread = threading.Thread(
            target=self._run, name="policy-action-lease", daemon=True
        )
        self.thread.start()

    def update(self, action):
        with self.lock:
            self.action = np.asarray(action, dtype=np.float64).copy()

    def _run(self):
        next_tick = time.perf_counter() + self.period
        try:
            while not self.stop_event.wait(max(0, next_tick - time.perf_counter())):
                with self.lock:
                    action = self.action.copy()
                self.robot.command(action)
                next_tick += self.period
                if next_tick < time.perf_counter():
                    next_tick = time.perf_counter() + self.period
        except BaseException as error:
            self.error = error
            self.stop_event.set()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Robot action refresh failed") from self.error

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
