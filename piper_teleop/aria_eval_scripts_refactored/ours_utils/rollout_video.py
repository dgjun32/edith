"""Rollout debug video composition."""

from __future__ import annotations

import cv2
import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_rollout_video(
    top_frames,
    comp_prob_log,
    output_path,
    fps: float,
    vlm_event_log=None,
    keyframe_log=None,
    event_display_frames: int = 90,
):
    """Save rollout.mp4 with keyframe, top-view, and completion-prob rows."""
    if not top_frames:
        print("[rollout] No frames to save.")
        return

    frame_w, frame_h = top_frames[0].shape[1], top_frames[0].shape[0]
    dpi = 100

    comp_prob_log = sorted(comp_prob_log, key=lambda item: item[0])
    max_step = max((step for step, _ in comp_prob_log), default=len(top_frames))
    max_step = max(max_step, len(top_frames))

    fig, ax = plt.subplots(figsize=(frame_w / dpi, frame_h / dpi), dpi=dpi)

    def _render_plot(visible_log):
        ax.clear()
        xs, ys = zip(*visible_log) if visible_log else ([], [])
        ax.plot(xs, ys, color="steelblue", linewidth=1.5)
        ax.scatter(xs, ys, color="steelblue", s=12, zorder=3)
        ax.axhline(
            y=0.8,
            color="tomato",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="threshold = 0.8",
        )
        ax.set_xlim(0, max_step)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Global Step", fontsize=8)
        ax.set_ylabel("Completion Prob", fontsize=8)
        ax.set_title("Completion Probability over Time", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_RGBA2BGR),
            (frame_w, frame_h),
        )

    breakpoints = [-1] + [step for step, _ in comp_prob_log]
    rendered = {
        bp: _render_plot([(step, prob) for step, prob in comp_prob_log if step <= bp])
        for bp in breakpoints
    }
    plt.close(fig)

    def _get_plot(step):
        best = max((bp for bp in breakpoints if bp <= step), default=-1)
        return rendered[best]

    kf_entries = sorted(keyframe_log or [], key=lambda item: item[0])
    blank_keyframe = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    def _get_keyframe(step):
        keyframe_img = None
        for start_step, img in kf_entries:
            if start_step <= step:
                keyframe_img = img
            else:
                break
        if keyframe_img is None:
            return blank_keyframe
        return cv2.resize(
            keyframe_img, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

    vlm_events = sorted(vlm_event_log or [], key=lambda item: item[0])

    def _get_active_event(step):
        active = [
            (event_step, label)
            for event_step, label in vlm_events
            if event_step <= step < event_step + event_display_frames
        ]
        if not active:
            return None, None
        event_step, label = max(active, key=lambda item: item[0])
        return event_step, label

    print(f"[rollout] Compositing {len(top_frames)} frames "
          f"({len(vlm_events)} VLM events)...")
    combined_frames = []
    for idx, frame in enumerate(top_frames):
        row1 = _get_keyframe(idx)
        row2 = frame.copy()
        event_step, label = _get_active_event(idx)
        if label:
            cv2.putText(
                row2,
                f"step {event_step}: {label}"[:90],
                (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        row3 = _get_plot(idx)
        combined_frames.append(
            cv2.cvtColor(np.vstack([row1, row2, row3]), cv2.COLOR_BGR2RGB)
        )

    imageio.mimwrite(output_path, combined_frames, fps=fps)
    print(f"[rollout] Saved -> {output_path}")
