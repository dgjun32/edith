"""Verify a recorded observation file and export/replay its cameras with audio.

This program never imports a robot adapter or opens CAN. --play only opens a
video window and plays the exported WAV on the computer's default audio output.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import wave

import cv2
import h5py
import numpy as np


def audio_arrays(file, rows):
    if "aria_audio" in file:  # Early local capture format.
        g = file["aria_audio"]
        end = int(g["row_offsets"][rows])
        return (
            g["pcm"][:end],
            g["capture_timestamps_ns"][:end],
            g["source_timestamps_ns"][:end],
            int(g.attrs["sample_rate_hz"]),
            g["muted"][:end],
        )
    g = file["audio"]
    channels = int(g.attrs["num_channels"])
    pcm = np.concatenate(
        [g["pcm"][i].reshape(-1, channels) for i in range(rows)]
    ).astype(np.float32) / float(g.attrs.get("max_amplitude", 2147483647.0))
    host = np.concatenate([g["frame_capture_timestamps_ns"][i] for i in range(rows)])
    source = np.concatenate([g["frame_source_timestamps_ns"][i] for i in range(rows)])
    muted = np.concatenate([g["frame_muted"][i] for i in range(rows)])
    return pcm, host, source, int(g.attrs["sample_rate_hz"]), muted


def tile(file, name, index, t):
    image = cv2.imdecode(file[name]["rgb_jpeg"][index], cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable {name} image at row {index}")
    canvas = np.zeros((384, 512, 3), np.uint8)
    ratio = min(512 / image.shape[1], 350 / image.shape[0])
    small = cv2.resize(image, None, fx=ratio, fy=ratio)
    y, x = (350 - small.shape[0]) // 2, (512 - small.shape[1]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(
        canvas,
        f"{name} | {t:.2f}s | row {index}",
        (10, 375),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return canvas


def export(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    wav_path, video_path = directory / "audio.wav", directory / "replay.mp4"
    if wav_path.exists() or video_path.exists():
        raise FileExistsError(
            "Replay output already exists; choose another output directory"
        )
    with h5py.File(source) as file:
        rows = int(file.attrs["complete_rows"])
        metadata = json.loads(file.attrs["metadata"])
        names = list(dict.fromkeys(("aria", "top", "left", "right", *file.keys())))
        views = [name for name in names if name in file and "rgb_jpeg" in file[name]]
        if not views:
            raise ValueError("Recording contains no camera images")
        anchor = metadata.get("sync_anchor", "top" if "top" in views else views[0])
        columns, tile_rows = 2, (len(views) + 1) // 2
        video_width, video_height = 512 * columns, 384 * tile_rows
        times = file[anchor]["timestamp_ns"][:rows]
        if rows < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("Recording needs increasing camera timestamps")
        step = float(np.median(np.diff(times))) / 1e9
        fps = 1 / step
        duration = (times[-1] - times[0]) / 1e9 + step
        pcm, host, src, rate, muted = audio_arrays(file, rows)
        if rate <= 0 or not len(pcm) or np.any(np.diff(src) <= 0):
            raise ValueError("Audio is empty or has repeated/out-of-order samples")
        if len({len(pcm), len(host), len(src), len(muted)}) != 1:
            raise ValueError("Audio PCM/timestamp/mute lengths disagree")
        # Keep raw multichannel PCM in HDF5; export microphone 0 with a documented gain.
        mono = np.zeros(round(duration * rate), np.float32)
        offset = round((int(host[0]) - int(times[0])) / 1e9 * rate)
        indices = np.rint((src - src[0]) * rate / 1e9).astype(np.int64) + offset
        valid = (indices >= 0) & (indices < len(mono))
        mono[indices[valid]] = pcm[valid, 0]
        peak = float(np.max(np.abs(mono)))
        gain = min(16.0, 0.85 / peak) if peak > 0 else 1.0
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(
                np.rint(np.clip(mono * gain, -1, 1) * 32767).astype("<i2").tobytes()
            )
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{video_width}x{video_height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            str(wav_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(round(duration * fps)):
                t = frame / fps
                index = min(
                    rows - 1,
                    int(
                        np.searchsorted(times, times[0] + round(t * 1e9), side="right")
                        - 1
                    ),
                )
                tiles = [tile(file, name, index, t) for name in views]
                while len(tiles) < columns * tile_rows:
                    tiles.append(np.zeros((384, 512, 3), np.uint8))
                canvas = np.vstack(
                    [
                        np.hstack(tiles[row * columns : (row + 1) * columns])
                        for row in range(tile_rows)
                    ]
                )
                if frame == 0:
                    cv2.imwrite(str(directory / "preview.jpg"), canvas)
                proc.stdin.write(canvas.tobytes())
        finally:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("Replay video encoding failed")
        report = {
            "source": str(source.resolve()),
            "observations": rows,
            "duration_s": duration,
            "video_views": views,
            "audio_channels": pcm.shape[1],
            "audio_sample_rate_hz": rate,
            "audio_frames": len(pcm),
            "audio_muted_frames": int(np.sum(muted)),
            "audio_source_gaps": int(np.sum(np.diff(src) > 1.5e9 / rate)),
            "audio_channel_rms": np.sqrt(np.mean(pcm**2, axis=0)).tolist(),
            "replay_audio_channel": 0,
            "replay_gain": gain,
            "execute_actions": metadata.get("execute_actions", False),
            "robot_state_source": metadata.get("robot_state_source", "unspecified"),
            "sync_anchor": anchor,
        }
        if "audio" in file:
            report["audio_buffer_dropped_frames"] = int(
                file["audio"].attrs.get("dropped_samples", 0)
            )
            report["observations_with_audio_frames"] = sum(
                bool(len(file["audio/pcm"][i])) for i in range(rows)
            )
        if "policy_observation" in file:
            g = file["policy_observation"]
            report["all_policy_actions_zero"] = bool(np.all(g["action"][:rows] == 0))
            report["gaze_valid_fraction"] = float(
                np.isfinite(g["gaze_point"][:rows]).all(axis=1).mean()
            )
            report["policy_action_dimension"] = g["action"].shape[1]
            report["all_policy_observations_have_audio"] = all(
                json.loads(value).get("audio_present", False)
                for value in g["metadata_json"][:rows]
            )
        stamp = np.stack(
            [
                file[name]["timestamp_ns"][:rows]
                for name in (*views, "eye_gaze", "hand_pose")
                if name in file
            ]
        )
        report["max_bundle_spread_ms"] = float(np.max(np.ptp(stamp, axis=0)) / 1e6)
        (directory / "verification.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
    return video_path, wav_path


def play(video, wav):
    if not shutil.which("aplay"):
        raise RuntimeError("Install ALSA aplay or open replay.mp4 in a video player")
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    cv2.namedWindow("Edith observation replay (audio on)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edith observation replay (audio on)", 1024, 768)
    audio = subprocess.Popen(["aplay", "-q", str(wav)])
    start = time.monotonic()
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow("Edith observation replay (audio on)", frame)
            index += 1
            delay = max(1, round((start + index / fps - time.monotonic()) * 1000))
            if cv2.waitKey(delay) & 0xFF in (27, ord("q")):
                break
        if audio.poll() not in (None, 0):
            raise RuntimeError("Audio playback failed")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if audio.poll() is None:
            audio.terminate()
        audio.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args(argv)
    video, wav = export(args.recording, args.output_dir)
    if args.play:
        play(video, wav)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
