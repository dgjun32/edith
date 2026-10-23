"""Keyboard helpers for OpenCV + terminal driven eval loops."""

from __future__ import annotations

import contextlib
import select
import sys
import termios
import tty

import cv2


@contextlib.contextmanager
def terminal_keypress_mode():
    """Allow single-key terminal input while CV2 windows are displayed."""
    if not sys.stdin.isatty():
        yield False
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def poll_keypress(delay_ms: int = 1) -> int:
    """Read one key from the OpenCV window or the SSH terminal."""
    key = cv2.waitKey(max(1, int(delay_ms)))
    if key != -1:
        return key & 0xFF

    if not sys.stdin.isatty():
        return -1

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return -1

    ch = sys.stdin.read(1)
    return ord(ch) if ch else -1

