"""policy_edith: hierarchical keyframe policy for canonical Aria Gen 2/YAM observations.

High level: every ``planner_interval_s``, Gemini turns the egocentric clip, with
locally transcribed speech captioned onto its frames, into keyframe-grounded
subtasks on a queue. Low level: the VLA executes the current subtask and
reports its completion, which advances the queue.
"""

from edith.policies.common.keyframes import KeyframePolicy

from edith.policies.common.speech import AudioWindow, SpeechCaptionPlanner, WhisperTranscriber


class Policy(KeyframePolicy):
    use_keyframe = True
    planner_source = "aria"

    def __init__(
        self, policy_dir, config, *, client=None, planner=None, transcriber=None
    ):
        super().__init__(policy_dir, config, client=client, planner=planner)
        self.audio = None
        if config.get("speech_to_text") and config.get("planner") == "gemini":
            try:
                if transcriber is None:
                    transcriber = WhisperTranscriber(config)
                    transcriber.load()  # ready before the robot starts
                self.audio = AudioWindow(config.get("speech_channel", 0))
            except BaseException:
                self.close()
                raise
            self.planner = SpeechCaptionPlanner(
                self.planner,
                transcriber,
                self.audio,
                min_chars=int(config.get("speech_min_chars", 2)),
                font_path=config.get("caption_font"),
            )

    def reset(self, observation):
        super().reset(observation)
        if self.audio is not None:
            self.audio.clear()
            self.planner.restart()

    def act(self, observation):
        # Store this step's audio before a planner clip can be cut from it.
        if self.audio is not None and observation.get("audio") is not None:
            self.audio.add(observation["audio"])
        return super().act(observation)
