"""Gemini planner using local synchronized frames, no camera/robot ownership."""

import json
import os
import re
import threading
import cv2
from .processing import local_file


class GeminiPlanner:
    def __init__(self, policy_dir, config):
        self.config = config
        self.model = config.get("gemini_model")
        if not self.model:
            raise ValueError("Set gemini_model explicitly when planner=gemini")
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("planner=gemini requires GEMINI_API_KEY")
        from google import genai
        from google.genai import types

        path = local_file(
            policy_dir, config.get("prompt_file", "prompts/passing_tools.txt")
        )
        text = path.read_text()

        def section(title, next_title=None):
            pattern = rf"^=+\s*\n{title}\s*\n=+\s*$"
            match = re.search(pattern, text, re.M)
            if match is None:
                raise ValueError(f"Missing prompt section {title}")
            end = len(text)
            if next_title:
                following = re.search(
                    rf"^=+\s*\n{next_title}\s*\n=+\s*$", text[match.end() :], re.M
                )
                if following is None:
                    raise ValueError(f"Missing prompt section {next_title}")
                end = match.end() + following.start()
            return text[match.end() : end].strip()

        self.initial = section("INITIAL SCRIPT", "ASYNCHRONOUS UPDATE SCRIPT")
        self.update = section("ASYNCHRONOUS UPDATE SCRIPT")
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=round(float(config.get("planner_timeout_s", 30)) * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self.lock = threading.Lock()
        self.closed = False

    def plan(self, frames, queue_state, transcript, initial):
        from google.genai import types

        duration = (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1e9
        current = next((i for i in queue_state if i["current"]), None)
        values = {
            "duration": duration,
            "segment_duration": duration,
            "fps": self.config.get("planner_fps", 4),
            "max_intervals": self.config.get("max_queue_items", 64),
            "transcript_section": f"Human instruction/transcript: {transcript}",
            "queue_state": "\n".join(
                f"[{i['index']}] {i['instruction']}" for i in queue_state
            ),
            "current_idx": current["index"] if current else len(queue_state),
            "current_instruction": current["instruction"] if current else "(waiting)",
            "visual_queue_section": "Queue keyframes are attached with their indices.",
            "t_start": 0,
            "t_end": duration,
        }
        template = self.initial if initial else self.update
        pattern = r"\{(" + "|".join(values) + r")(?::([^{}]+))?\}"
        prompt = re.sub(pattern, lambda m: format(values[m[1]], m[2] or ""), template)
        prompt = prompt.replace("{{", "{").replace("}}", "}")
        if self.config.get("planner_source") == "top":
            prompt = prompt.replace("first-person", "overhead robot-camera").replace(
                "egocentric", "overhead"
            )
            prompt += "\nThese are robot top-camera frames. Interpret left/right in image coordinates, not the wearer perspective."
        prompt += (
            "\nThe clip is supplied as sampled RGB frames, each labelled with its actual "
            "capture time in seconds relative to the first frame. Use those times for "
            "keyframe_interval; do not infer timing from frame count. "
            "Return the existing action_queue or command/new_elements JSON schema."
        )
        parts = [types.Part(text=prompt)]

        def image_part(image):
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("Planner JPEG encoding failed")
            return types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg")

        for frame in frames:
            parts.extend(
                [
                    types.Part(
                        text=f"Clip frame t={(frame.timestamp_ns - frames[0].timestamp_ns) / 1e9:.6f}s"
                    ),
                    image_part(frame.image),
                ]
            )
        if self.config.get("visual_queue", True):
            for item in queue_state[-8:]:
                if item["keyframe"] is not None:
                    parts.extend(
                        [
                            types.Part(
                                text=f"Queue [{item['index']}] {item['instruction']}"
                            ),
                            image_part(item["keyframe"]),
                        ]
                    )
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json"
            ),
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        return json.loads(text)

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.client.close()
