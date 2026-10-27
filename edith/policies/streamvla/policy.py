"""streamvla: the VLA sees the live egocentric view and follows spoken instructions.

Every step pi0.5 receives the robot cameras and state, the gaze-cropped
first-person view, and the current instruction. With ``speech_to_text``, each
utterance is transcribed locally (Whisper) when the speaker pauses and becomes
the instruction; until then the preset, ``--instruction`` or transcript text is used.
"""

from edith.policies.common.chunked import ChunkedPolicy
from edith.policies.common.speech import (
    AudioWindow,
    UtteranceListener,
    WhisperTranscriber,
    detect_speech,
    require_faster_whisper,
)


class Policy(ChunkedPolicy):
    view = "stream"

    def __init__(self, policy_dir, config, *, client=None, transcriber=None, detect=None):
        super().__init__(policy_dir, config, client=client)
        self.listener = None
        self.local_whisper = transcriber is None
        if config.get("speech_to_text"):
            # Nothing is loaded here; the model warms up in the background at reset.
            self.listener = UtteranceListener(
                transcriber or WhisperTranscriber(config),
                AudioWindow(config.get("speech_channel", 0)),
                detect or detect_speech,
                silence_s=config.get("utterance_silence_s", 0.8),
                max_utterance_s=config.get("max_utterance_s", 15),
                min_chars=int(config.get("speech_min_chars", 2)),
                mode=config.get("speech_mode", "latest"),
                timeout_s=config.get("speech_timeout_s", 60),
            )

    def reset(self, observation):
        super().reset(observation)
        if self.listener is not None:
            if self.local_whisper:
                require_faster_whisper()
            self.listener.reset()

    def _instruction(self, observation):
        spoken = self.listener.instruction if self.listener is not None else None
        return spoken or super()._instruction(observation)

    def act(self, observation):
        if self.listener is not None:
            self.listener.update(observation)
        return super().act(observation)
