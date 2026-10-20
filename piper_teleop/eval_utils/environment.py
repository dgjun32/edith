"""Robot-agnostic evaluation environment over synchronized local sensors."""

from collections.abc import Mapping

import numpy as np

from .observation import build_observation, legacy_observation


class RobotEvaluationEnv:
    """Join a sensor backend and a :class:`RobotAdapter` for evaluation.

    ``get_observation`` retains the shape used by the existing evaluation
    scripts. New directory policies should use ``get_policy_observation``.
    Neither method starts hardware; lifecycle and action authorization remain
    explicit in the runner.
    """

    def __init__(
        self,
        sensors,
        robot,
        fallback_state,
        *,
        instruction="",
        control_hz=15.0,
    ):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.sensors = sensors
        self.robot = robot
        self.fallback_state = fallback_state
        self.instruction = instruction
        self.control_hz = float(control_hz)
        self.last_bundle = None

    def observation_from_bundle(self, bundle) -> dict:
        projector = None
        if getattr(self.sensors, "aria", None) is not None:
            projector = self.sensors.aria.project_perception_to_rgb
        return build_observation(
            bundle,
            arm_order=self.robot.arm_order,
            fallback_state=self.fallback_state,
            instruction=self.instruction,
            projector=projector,
        )

    def get_policy_observation(self, timeout=5.0) -> dict:
        self.last_bundle = self.sensors.get_bundle(timeout)
        return self.observation_from_bundle(self.last_bundle)

    def get_observation(self, mode="raw", audio_drain=True, timeout=5.0) -> dict:
        """Return an existing-evaluator-compatible synchronized observation.

        ``mode`` and ``audio_drain`` are accepted for drop-in compatibility.
        The Gen 2 backend always returns its calibrated full RGB frame; the
        optional ``aria_cropped_rgb`` key aliases that frame unless a policy
        performs its own gaze crop.
        """

        del mode, audio_drain
        observation = self.get_policy_observation(timeout)
        return legacy_observation(observation, self.robot.arm_order)

    def _flat_action(self, action) -> np.ndarray:
        if not isinstance(action, Mapping):
            return self.robot.validate_action(action)
        parts = []
        for name in self.robot.arm_order:
            arm_key = f"{name}_arm_positions"
            gripper_key = f"{name}_gripper_position"
            if arm_key not in action or gripper_key not in action:
                raise ValueError(
                    f"Mapping action requires {arm_key!r} and {gripper_key!r}"
                )
            parts.extend(
                (
                    np.asarray(action[arm_key], dtype=np.float64).reshape(-1),
                    np.asarray(action[gripper_key], dtype=np.float64).reshape(-1),
                )
            )
        return self.robot.validate_action(np.concatenate(parts))

    def step(self, action) -> dict:
        command = self._flat_action(action)
        self.robot.command(command)
        return {"commanded_joint_position": command.copy()}

    def get_control_hz(self) -> float:
        return self.control_hz

    def safe_end(self) -> None:
        self.robot.close()
        self.sensors.safe_end()
