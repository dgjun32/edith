"""Audio-video muxing helpers for the ours eval scripts."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List

from .audio_processing import collect_audio_pcm16, write_pcm16_wav
from .video_processing import write_silent_video


def write_video_with_audio(frames_bgr, audio_records, output_path: str, fps: float):
    if not frames_bgr:
        return

    audio_pcm16, sample_rate_hz = collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate_hz <= 0:
        write_silent_video(frames_bgr, output_path, fps=fps)
        return

    silent_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    silent_tmp_path = silent_tmp.name
    wav_tmp_path = wav_tmp.name
    silent_tmp.close()
    wav_tmp.close()

    try:
        write_silent_video(frames_bgr, silent_tmp_path, fps=fps)
        write_pcm16_wav(audio_pcm16, sample_rate_hz, wav_tmp_path)

        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("[video] imageio_ffmpeg unavailable — saving silent video only.")
            os.replace(silent_tmp_path, output_path)
            silent_tmp_path = None
            return

        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    silent_tmp_path,
                    "-i",
                    wav_tmp_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_text = (exc.stderr or "").strip()
            print(
                "[video] audio mux failed — saving silent video only."
                + (f" {stderr_text}" if stderr_text else "")
            )
            os.replace(silent_tmp_path, output_path)
            silent_tmp_path = None
    finally:
        if silent_tmp_path and os.path.exists(silent_tmp_path):
            os.unlink(silent_tmp_path)
        if os.path.exists(wav_tmp_path):
            os.unlink(wav_tmp_path)


def merge_audio_into_video_bytes(video_bytes: bytes, wav_path: str) -> bytes:
    """Merge H264 video bytes with a WAV file -> MP4 with audio track."""
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.write(video_bytes)
    tmp_video.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-i", tmp_video.name,
         "-i", wav_path,
         "-c:v", "copy",
         "-ac", "1",
         "-c:a", "aac", "-b:a", "64k",
         "-shortest",
         tmp_out.name],
        check=True,
    )

    with open(tmp_out.name, "rb") as f:
        result = f.read()

    os.unlink(tmp_video.name)
    os.unlink(tmp_out.name)
    return result


def merge_audio_records_into_video_bytes(
    video_bytes: bytes,
    audio_records: List[dict],
) -> bytes:
    audio_pcm16, sample_rate = collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate <= 0:
        return video_bytes

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
    try:
        return merge_audio_into_video_bytes(video_bytes, wav_tmp.name)
    finally:
        os.unlink(wav_tmp.name)


def mux_audio_file_into_video_file(
    video_path: str,
    wav_path: str,
    output_path: str,
    audio_bitrate: str = "128k",
):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-i", video_path,
         "-i", wav_path,
         "-c:v", "copy",
         "-ac", "1",
         "-c:a", "aac", "-b:a", audio_bitrate,
         "-shortest",
         output_path],
        check=True,
    )


def mux_audio_records_into_video_file(
    video_path: str,
    audio_records: List[dict],
    audio_bitrate: str = "128k",
) -> bool:
    audio_pcm16, sample_rate = collect_audio_pcm16(audio_records)
    if audio_pcm16.size == 0 or sample_rate <= 0:
        return False

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    wav_tmp.close()
    tmp_out.close()
    write_pcm16_wav(audio_pcm16, sample_rate, wav_tmp.name)
    try:
        mux_audio_file_into_video_file(
            video_path, wav_tmp.name, tmp_out.name,
            audio_bitrate=audio_bitrate)
        os.replace(tmp_out.name, video_path)
        return True
    finally:
        if os.path.exists(wav_tmp.name):
            os.unlink(wav_tmp.name)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)
