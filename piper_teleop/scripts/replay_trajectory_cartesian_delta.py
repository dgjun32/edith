"""Replay a recorded HDF trajectory using cartesian end-pose controllers.

The collected HDF only stores joint positions and gripper positions (see
``hdf5_to_lerobot/utils.py::REQUIRED_KEYS``). The cartesian state of the
robot at each timestep therefore has to be recovered from the recorded
**proprioceptive** joint trajectory
(``observation/timestamp/robot_state/{arm}_arm_positions``) via forward
kinematics. The ``action/{arm}_arm_positions`` stream stores the
**master arm / teleop target** joints, not the robot's own state, so it
is *not* what we want here.

Pipeline:

  1. Read the recorded *observation* joint trajectory (per arm) from the
     HDF, plus the *observation* gripper.
  2. Run the Piper forward-kinematics solver on each joint frame to
     obtain a sequence of end-effector poses ``s_t`` (xyz [mm], rpy
     [deg]) - this is the robot's cartesian state trajectory.
  3. Unwrap the FK Euler angles along the time axis so that branch
     crossings (e.g. yaw 179 -> -179 deg) do not become spurious
     ~360 deg per-step jumps.
  4. Compute per-step cartesian deltas ``Δ_t = s_{t+1} - s_t``.
  5. Ramp each arm to the recorded *observation* start joint
     configuration with ``safe_joint_control`` so the live end-effector
     pose matches the FK pose at step 0.
  6. Build a per-arm cartesian controller. Two are available:
       * :class:`CartesianDeltaController` (``--mode delta``): reads
         the live end-effector pose via ``GetArmEndPoseMsgs`` and
         commands ``current_pose + Δ`` via ``EndPoseCtrl``. Deltas are
         applied as-is, with no scaling or lag compensation.
       * :class:`CartesianAbsoluteController` (``--mode absolute``):
         forwards a caller-provided absolute pose directly to
         ``EndPoseCtrl`` without consulting the live pose.
  7. Before the loop starts, send a single ``EndPoseCtrl`` anchored at
     the live pose so MOVE-P has a coherent target during the
     mode-switch settle window.
  8. At every step feed ``Δ_t`` (delta mode) or ``s_{t+1}`` (absolute
     mode) to each arm's controller and replay the recorded gripper
     directly (gripper is not part of the cartesian pose).

Usage::

    # Default: cartesian-delta controller, sourced from observation
    python -m piper_teleop.scripts.replay_trajectory_cartesian_delta \
        --filepath /home/alin/projects/piper_project/data/hdf/.../demo0.hdf

    # Absolute pose replay (still sourced from observation)
    python -m piper_teleop.scripts.replay_trajectory_cartesian_delta \
        --filepath /home/alin/projects/piper_project/data/hdf/.../demo0.hdf \
        --mode absolute
"""

import argparse
import os
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as _R

from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
from piper_sdk import C_PiperInterface_V2


# Both arms recorded by ``PiperRobotEnv`` (see ``envs/piper_env.py``).
ARM_NAMES = ["left", "right"]


# --- Rotation helpers ------------------------------------------------------
# The Piper FK and SDK use ZYX-intrinsic Euler angles ``(RX, RY, RZ) =
# (roll, pitch, yaw)`` with the world-frame rotation
# ``R = Rz(yaw) * Ry(pitch) * Rx(roll)``. That matches scipy's
# ``from_euler('ZYX', [yaw, pitch, roll])``.
#
# We need rotation *composition* (not Euler subtraction) when computing
# cartesian deltas. The Piper home pose has pitch ~= 85 deg, very close
# to the ZYX gimbal-lock at 90 deg, so naively differencing Euler angles
# produces ~180 deg representational jumps even when the underlying
# rotation matrix changes by a fraction of a degree. Sending such a
# "delta" through ``EndPoseCtrl`` would command a real, violent
# rotation. Composing rotation matrices avoids this.


def _euler_zyx_deg_to_rot(rpy_deg: np.ndarray) -> _R:
    """Convert ``[RX, RY, RZ] = [roll, pitch, yaw]`` (deg) to a Rotation."""
    rpy = np.asarray(rpy_deg, dtype=np.float64)
    # scipy's 'ZYX' takes [yaw, pitch, roll].
    return _R.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]], degrees=True)


def _rot_to_euler_zyx_deg(rot: _R) -> np.ndarray:
    """Convert a Rotation back to ``[RX, RY, RZ] = [roll, pitch, yaw]`` (deg).

    Uses the rotation matrix directly to extract ZYX intrinsic Euler
    angles with an ``atan2``-based formula that handles the
    pitch = ±90° gimbal-lock singularity more gracefully than
    ``scipy``'s ``as_euler("ZYX")``, which sets roll to zero in that
    case and can produce discontinuous jumps.
    """
    m = rot.as_matrix()
    # ZYX intrinsic: R = Rz(yaw) Ry(pitch) Rx(roll)
    # m[2,0] = -sin(pitch)
    sp = -m[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)

    if np.abs(sp) < 1.0 - 1e-8:
        roll  = np.arctan2(m[2, 1], m[2, 2])
        yaw   = np.arctan2(m[1, 0], m[0, 0])
    else:
        # Gimbal lock: pitch ≈ ±90°. Attribute all remaining rotation
        # to yaw and set roll = 0 to stay deterministic.
        roll = 0.0
        yaw  = np.arctan2(-m[0, 1], m[1, 1])

    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def _clip_rotation(delta_rot: _R, max_angle_deg: float) -> _R:
    """Cap a rotation's axis-angle magnitude at ``max_angle_deg``.

    Returns ``delta_rot`` unchanged if its magnitude is already at or
    below the cap, or if ``max_angle_deg`` is non-positive.
    """
    if max_angle_deg is None or max_angle_deg <= 0:
        return delta_rot
    rotvec = delta_rot.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    angle_deg = np.rad2deg(angle)
    if angle_deg <= max_angle_deg or angle == 0.0:
        return delta_rot
    return _R.from_rotvec(rotvec * (max_angle_deg / angle_deg))


def _read_trajectory(filepath: str, source: str = "observation"):
    """Read joint and gripper trajectories for both arms from an HDF file.

    Args:
        filepath: Path to a ``demoX.hdf`` produced by ``collect_trajectory``.
        source: ``"observation"`` (default, recommended) reads the
            robot's proprioceptive state
            (``observation/timestamp/robot_state/{arm}_arm_positions``)
            - this is the **robot's own cartesian state** once passed
            through FK. ``"action"`` reads the master / teleop target
            (``action/{arm}_arm_positions``); use this only for
            debugging - the cartesian deltas it produces describe the
            master arm's motion, not the robot's.

    Returns:
        (data, horizon) where ``data[arm] = {"joints": (T, 6) [rad],
        "gripper": (T,) [normalised 0..1]}`` and ``horizon == T``.
    """
    if source not in {"action", "observation"}:
        raise ValueError(f"source must be 'action' or 'observation', got {source!r}")

    data = {}
    with h5py.File(filepath, "r") as hf:
        for arm in ARM_NAMES:
            if source == "action":
                joint_key = f"action/{arm}_arm_positions"
                gripper_key = f"action/{arm}_gripper_position"
            else:
                joint_key = f"observation/timestamp/robot_state/{arm}_arm_positions"
                gripper_key = f"observation/timestamp/robot_state/{arm}_gripper_position"

            if joint_key not in hf:
                raise KeyError(f"Missing dataset '{joint_key}' in {filepath}")
            if gripper_key not in hf:
                raise KeyError(f"Missing dataset '{gripper_key}' in {filepath}")

            joints = np.asarray(hf[joint_key], dtype=np.float64)
            gripper = np.asarray(hf[gripper_key], dtype=np.float64).reshape(-1)

            if joints.ndim != 2 or joints.shape[1] != 6:
                raise ValueError(
                    f"Unexpected joint shape for {arm}: {joints.shape}, expected (T, 6)"
                )
            if gripper.shape[0] != joints.shape[0]:
                raise ValueError(
                    f"Length mismatch for {arm}: joints={joints.shape[0]} "
                    f"gripper={gripper.shape[0]}"
                )

            data[arm] = {"joints": joints, "gripper": gripper}

    horizons = {arm: data[arm]["joints"].shape[0] for arm in ARM_NAMES}
    if len(set(horizons.values())) != 1:
        raise ValueError(f"Per-arm horizon mismatch: {horizons}")
    return data, next(iter(horizons.values()))


def _joints_to_endpose_units(joints_traj: np.ndarray,
                             fk: C_PiperForwardKinematics) -> np.ndarray:
    """Forward-kinematics map a (T, 6) [rad] joint trajectory to a (T, 6)
    cartesian pose trajectory in EndPoseCtrl units (xyz in 0.001 mm, rpy
    in 0.001 deg).

    ``C_PiperForwardKinematics.CalFK`` returns ``[x_mm, y_mm, z_mm,
    rx_deg, ry_deg, rz_deg]`` for the end-effector. Both
    ``EndPoseCtrl`` and ``GetArmEndPoseMsgs`` use units of 0.001 mm /
    0.001 deg, so we multiply by 1000 to match.

    The Euler angles produced by ``CalFK`` are ``atan2``-based and live
    in ``[-180, 180]`` deg. We unwrap them along the time axis so that
    branch crossings (e.g. 179 -> -179 deg) do not produce spurious
    ~360 deg per-step jumps when callers compute differences.
    """
    poses = np.zeros((joints_traj.shape[0], 6), dtype=np.float64)
    for i, q in enumerate(joints_traj):
        ee_pose = np.asarray(fk.CalFK(q.tolist())[-1], dtype=np.float64)
        poses[i] = ee_pose

    # Unwrap RPY (cols 3..5) in radians, then restore degrees.
    rpy_rad = np.deg2rad(poses[:, 3:6])
    rpy_rad = np.unwrap(rpy_rad, axis=0)
    poses[:, 3:6] = np.rad2deg(rpy_rad)

    return poses * 1000.0


class _CartesianControllerBase:
    """Shared infrastructure for cartesian end-pose controllers.

    Wraps a single ``C_PiperInterface_V2`` and provides:
      * :meth:`reset` to switch the arm into MOVE-P (end-pose) mode and
        optionally prime the gripper / settle.
      * :meth:`get_current_pose` to read the live end-effector pose in
        EndPoseCtrl units (0.001 mm / 0.001 deg).
      * :meth:`_send_target` to dispatch a target pose + optional
        gripper command.

    Subclasses implement :meth:`step` to convert their domain-specific
    input (a delta or an absolute pose) into a target pose and call
    :meth:`_send_target`.
    """

    def __init__(self,
                 piper,
                 gripper_effort: int = 700,
                 move_spd_rate: int = 30):
        self.piper = piper
        self.gripper_effort = int(gripper_effort)
        self.move_spd_rate = int(np.clip(move_spd_rate, 0, 100))
        self._last_cmd = None

    def _enter_move_p(self) -> None:
        """(Re-)issue the MOVE-P switch with our configured speed rate."""
        # ctrl_mode=0x01 (CAN), move_mode=0x00 (MOVE P, end-pose),
        # is_mit_mode=0x00 (position-velocity).
        self.piper.MotionCtrl_2(0x01, 0x00, self.move_spd_rate, 0x00)

    def reset(self,
              anchor_pose: np.ndarray = None,
              initial_gripper_normalized: float = None,
              settle_time: float = 0.0) -> None:
        """Switch the arm into MOVE-P end-pose mode and send an anchor pose.

        Args:
            anchor_pose: ``(6,)`` pose in EndPoseCtrl units to send as
                an initial "hold here" target right after switching to
                MOVE-P. **Must come from our Python FK** (not from
                ``GetArmEndPoseMsgs``), because the firmware feedback
                can report Euler angles in a different branch than
                ``EndPoseCtrl`` accepts near pitch ≈ 90°. If ``None``,
                no anchor is sent (only the mode switch is performed).
            initial_gripper_normalized: If provided, also command the
                gripper to this normalized (0..1) value.
            settle_time: Optional sleep (s) after the mode switch.
        """
        self._enter_move_p()

        if anchor_pose is not None:
            self._send_target(np.asarray(anchor_pose, dtype=np.float64))

        if initial_gripper_normalized is not None:
            self._send_gripper(float(initial_gripper_normalized))

        if settle_time > 0:
            t0 = time.time()
            while time.time() - t0 < settle_time:
                self.resend()
                time.sleep(0.01)

    def get_current_pose(self) -> np.ndarray:
        """Return the live end-effector pose in EndPoseCtrl units."""
        msg = self.piper.GetArmEndPoseMsgs().end_pose
        return np.array(
            [msg.X_axis, msg.Y_axis, msg.Z_axis,
             msg.RX_axis, msg.RY_axis, msg.RZ_axis],
            dtype=np.float64,
        )

    def step(self, *args, **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _send_target(self,
                     target_pose: np.ndarray,
                     gripper_normalized: float = None) -> np.ndarray:
        X, Y, Z, RX, RY, RZ = (int(round(float(v))) for v in target_pose)
        self._enter_move_p()
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        if gripper_normalized is not None:
            self._send_gripper(float(gripper_normalized))
        self._last_cmd = (X, Y, Z, RX, RY, RZ)
        return target_pose

    def resend(self) -> None:
        """Re-issue the last EndPoseCtrl command to keep CAN alive.

        The Piper firmware requires continuous command refreshing at
        ~100 Hz (see ``piper_ctrl_end_pose.py`` SDK demo). If the CAN
        bus doesn't receive a new EndPoseCtrl within ~10-20 ms the arm
        stalls. Call this in a tight loop between trajectory steps.
        """
        if self._last_cmd is None:
            return
        self._enter_move_p()
        self.piper.EndPoseCtrl(*self._last_cmd)

    def _send_gripper(self, gripper_normalized: float) -> None:
        grip_int = int(round(gripper_normalized * 100000))
        self.piper.GripperCtrl(abs(grip_int), self.gripper_effort, 0x01, 0)


class CartesianDeltaController(_CartesianControllerBase):
    """Open-loop cartesian-delta controller for one Piper arm.

    Maintains an internal running pose ``p`` (in EndPoseCtrl units).
    At each ``step(prev_pose, next_pose)``:

      1. Compute the recorded translation delta ``Δp = next[:3] -
         prev[:3]`` (vector subtraction).
      2. Compute the recorded rotation delta as a rotation matrix
         ``ΔR = R(next) * R(prev)^{-1}`` (composition, NOT Euler
         subtraction — avoids the ~180 deg representational jumps
         near the Piper's pitch ~ 90 deg singularity).
      3. Optionally cap ``|ΔR|`` at :attr:`max_step_rot_deg`.
      4. Apply the delta to the internal pose:
         ``p_pos += Δp``, ``p_R = ΔR * p_R``.
      5. Convert ``p`` back to Euler and send to ``EndPoseCtrl``.

    The internal pose is seeded via :meth:`set_initial_pose` with the
    FK-computed start pose (which is in the same Euler branch as
    ``EndPoseCtrl``). We do **not** read ``GetArmEndPoseMsgs`` during
    the loop because the firmware feedback can report Euler angles in
    a different branch than what ``EndPoseCtrl`` expects (near the
    pitch ~ 90 deg gimbal-lock singularity). Avoiding that read makes
    the delta loop reliable.

    Before stepping, callers must invoke :meth:`reset` once to switch
    the arm into MOVE-P end-pose mode, and :meth:`set_initial_pose`
    to seed the internal running pose.
    """

    def __init__(self,
                 piper,
                 gripper_effort: int = 700,
                 move_spd_rate: int = 50,
                 max_step_rot_deg: float = 30.0,
                 delta_scale: float = 1.0):
        super().__init__(piper, gripper_effort=gripper_effort,
                         move_spd_rate=move_spd_rate)
        self.max_step_rot_deg = max_step_rot_deg
        self.delta_scale = delta_scale
        # Internal running pose: position in 0.001 mm, rotation as a
        # scipy Rotation object. Set via ``set_initial_pose``.
        self._pos = None        # (3,) float64, 0.001 mm
        self._R = None          # scipy Rotation

    def set_initial_pose(self, pose_units: np.ndarray) -> None:
        """Seed the internal running pose from an FK-computed pose.

        Args:
            pose_units: ``(6,)`` pose in EndPoseCtrl units
                (0.001 mm / 0.001 deg). Must be in the same Euler
                branch as ``EndPoseCtrl`` (i.e. from our FK, not from
                ``GetArmEndPoseMsgs``).
        """
        p = np.asarray(pose_units, dtype=np.float64)
        self._pos = p[:3].copy()
        self._R = _euler_zyx_deg_to_rot(p[3:6] * 1e-3)

    def step(self,
             prev_pose: np.ndarray,
             next_pose: np.ndarray,
             gripper_normalized: float = None) -> np.ndarray:
        """Apply one cartesian delta to the arm.

        Args:
            prev_pose: ``(6,)`` recorded FK pose ``s_t`` in EndPoseCtrl
                units (0.001 mm / 0.001 deg).
            next_pose: ``(6,)`` recorded FK pose ``s_{t+1}``.
            gripper_normalized: Optional gripper command in [0, 1].

        Returns:
            The commanded target pose in EndPoseCtrl units.
        """
        if self._pos is None or self._R is None:
            raise RuntimeError(
                "set_initial_pose() must be called before step()"
            )

        prev = np.asarray(prev_pose, dtype=np.float64)
        nxt = np.asarray(next_pose, dtype=np.float64)

        # Translation delta (0.001 mm).
        delta_pos = (nxt[:3] - prev[:3]) * self.delta_scale

        # Rotation delta via composition (not Euler subtraction).
        R_prev = _euler_zyx_deg_to_rot(prev[3:6] * 1e-3)
        R_next = _euler_zyx_deg_to_rot(nxt[3:6] * 1e-3)
        dR = R_next * R_prev.inv()
        if self.delta_scale != 1.0:
            rv = dR.as_rotvec() * self.delta_scale
            dR = _R.from_rotvec(rv)
        dR = _clip_rotation(dR, self.max_step_rot_deg)

        # Accumulate on internal running pose.
        self._pos = self._pos + delta_pos
        self._R = dR * self._R

        target_rpy_deg = _rot_to_euler_zyx_deg(self._R)
        target_pose = np.concatenate([self._pos, target_rpy_deg * 1000.0])

        return self._send_target(target_pose, gripper_normalized)


class CartesianAbsoluteController(_CartesianControllerBase):
    """Open-loop absolute cartesian controller for one Piper arm.

    Exposes ``step(target_pose, gripper_normalized=None)`` and simply
    forwards the caller-provided absolute pose to ``EndPoseCtrl``
    without ever consulting the live end-effector pose. Useful when
    the caller has already computed the target trajectory in the
    robot's frame (e.g. directly from FK on a recorded joint
    trajectory).

    Before stepping, callers must invoke :meth:`reset` once to switch
    the arm into MOVE-P (end-pose) mode.
    """

    def step(self,
             target_pose: np.ndarray,
             gripper_normalized: float = None) -> np.ndarray:
        """Command an absolute end-effector pose.

        Args:
            target_pose: ``(6,)`` array ``[X, Y, Z, RX, RY, RZ]`` in
                EndPoseCtrl units (0.001 mm / 0.001 deg).
            gripper_normalized: Optional gripper command in [0, 1].

        Returns:
            The commanded target pose in EndPoseCtrl units.
        """
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (6,):
            raise ValueError(
                f"target_pose must have shape (6,), got {target.shape}"
            )
        return self._send_target(target, gripper_normalized)


def replay_trajectory_cartesian_delta(piper_interfaces: dict,
                                      filepath: str,
                                      mode: str = "delta",
                                      source: str = "observation",
                                      warmup_time: float = 1.5,
                                      gripper_effort: int = 700,
                                      pre_replay_pause: float = 0.5,
                                      move_spd_rate: int = 50,
                                      max_step_rot_deg: float = 30.0,
                                      delta_scale: float = 1.0) -> None:
    """Replay an HDF trajectory through per-arm cartesian controllers.

    Args:
        piper_interfaces: ``{arm_name: C_PiperInterface_V2}`` dict with
            one entry per arm (``"left"`` and ``"right"``).  These must
            be freshly enabled and **never** put into MIT mode.
        filepath: Path to the ``demoX.hdf`` to replay.
        mode: Controller to use.
            * ``"delta"`` (default): per-arm
              :class:`CartesianDeltaController`. The replay loop
              feeds it ``Δ_t = s_{t+1} - s_t`` from the FK pose
              sequence; the controller commands
              ``current_pose + Δ`` each step.
            * ``"absolute"``: per-arm
              :class:`CartesianAbsoluteController`. The replay loop
              feeds it ``s_{t+1}`` directly; the controller commands
              that absolute pose without consulting the live pose.
        source: ``"observation"`` (default, recommended) uses the
            recorded robot proprioceptive joints, so the deltas come
            from the **robot's own** cartesian state. ``"action"`` uses
            the master/teleop joints; the deltas then describe the
            master arm's motion, not the robot's, and warming up to
            ``action[0]`` can yank the slave arm into a non-trivial
            reaching pose at the start.
        warmup_time: Seconds to ramp each arm from its current pose to
            the recorded start joint configuration before starting the
            replay loop.
        gripper_effort: Effort value passed to ``GripperCtrl`` while
            replaying (matches collection-time setting in
            ``PiperRobot``).
        pre_replay_pause: Settle delay between switching each arm into
            end-pose mode (post-warmup) and the first replay step.
        move_spd_rate: Cartesian motion speed-rate (0..100 %, default
            30) forwarded to every ``MotionCtrl_2(MOVE P)`` call. Lower
            values keep the arm tame even if the commanded target
            momentarily drifts away from the live pose.
        max_step_rot_deg: Per-step rotation safety cap (axis-angle
            magnitude, deg). Only used in ``delta`` mode. Set to
            ``0`` or ``None`` to disable.
    """
    if mode not in {"delta", "absolute"}:
        raise ValueError(f"mode must be 'delta' or 'absolute', got {mode!r}")

    traj, horizon = _read_trajectory(filepath, source=source)
    print(
        f"[Replay] Loaded trajectory: file={filepath} horizon={horizon} "
        f"source={source} mode={mode}"
    )

    fk_solvers = {arm: C_PiperForwardKinematics() for arm in ARM_NAMES}
    # s_t for each arm: (T, 6) end-effector pose trajectory in
    # EndPoseCtrl units (0.001 mm / 0.001 deg).
    ee_traj_units = {
        arm: _joints_to_endpose_units(traj[arm]["joints"], fk_solvers[arm])
        for arm in ARM_NAMES
    }
    n_steps = ee_traj_units[ARM_NAMES[0]].shape[0] - 1

    # Quick stats on the recorded deltas - useful sanity check.
    # Translation: simple vector norm of position diff.
    # Rotation: axis-angle magnitude of the composed rotation delta
    # (the *real* rotation; not the misleading Euler-difference norm).
    for arm in ARM_NAMES:
        ee = ee_traj_units[arm]
        pos_diff = np.diff(ee[:, :3], axis=0) * 1e-3              # mm
        pos_norm = np.linalg.norm(pos_diff, axis=1)
        rot_norm_deg = np.zeros(ee.shape[0] - 1)
        for k in range(ee.shape[0] - 1):
            R_a = _euler_zyx_deg_to_rot(ee[k, 3:6] * 1e-3)
            R_b = _euler_zyx_deg_to_rot(ee[k + 1, 3:6] * 1e-3)
            rot_norm_deg[k] = np.rad2deg(
                np.linalg.norm((R_b * R_a.inv()).as_rotvec())
            )
        print(
            f"[Replay] {arm}: |Δpos| max={pos_norm.max():.2f} mm "
            f"mean={pos_norm.mean():.2f} mm; "
            f"|ΔR| (axis-angle) max={rot_norm_deg.max():.2f} deg "
            f"mean={rot_norm_deg.mean():.2f} deg"
        )

    pipers = {arm: piper_interfaces[arm] for arm in ARM_NAMES}

    # -- Warmup: move each arm to the recorded start joint configuration
    # using MOVE J (regular joint control, NOT MIT). This keeps the
    # firmware in position-velocity mode so EndPoseCtrl works afterwards.
    RAD_TO_MDEG = 1000.0 * 180.0 / np.pi
    print(f"[Replay] Warming up to recorded start pose ({warmup_time:.2f}s)...")
    for arm in ARM_NAMES:
        start_joints_rad = np.asarray(traj[arm]["joints"][0], dtype=np.float64)
        start_joints_mdeg = (start_joints_rad * RAD_TO_MDEG).astype(int)
        j = start_joints_mdeg
        print(f"[Replay] {arm}: MOVE J target (0.001 deg) = "
              f"[{j[0]}, {j[1]}, {j[2]}, {j[3]}, {j[4]}, {j[5]}]")

    t0 = time.time()
    while time.time() - t0 < warmup_time:
        for arm in ARM_NAMES:
            start_joints_rad = np.asarray(traj[arm]["joints"][0], dtype=np.float64)
            j = [int(round(v)) for v in start_joints_rad * RAD_TO_MDEG]
            pipers[arm].MotionCtrl_2(0x01, 0x01, 30, 0x00)   # CAN + MOVE J
            pipers[arm].JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
        time.sleep(0.01)

    # -- Sanity check: send a clearly different EndPoseCtrl target to
    # verify that MOVE P actually works with these fresh interfaces.
    HOME_POSE = (57000, 0, 215000, 0, 85000, 0)
    TEST_POSE = (57000, 0, 235000, 0, 85000, 0)   # 20 mm higher
    print(f"[Sanity] Testing EndPoseCtrl: sending Z=235mm (20mm above home) for 2s ...")
    for arm in ARM_NAMES:
        ep = pipers[arm].GetArmEndPoseMsgs().end_pose
        print(f"[Sanity] {arm} BEFORE: X={ep.X_axis} Y={ep.Y_axis} Z={ep.Z_axis}")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        for arm in ARM_NAMES:
            pipers[arm].MotionCtrl_2(0x01, 0x00, 50, 0x00)
            pipers[arm].EndPoseCtrl(*TEST_POSE)
        time.sleep(0.01)
    for arm in ARM_NAMES:
        ep = pipers[arm].GetArmEndPoseMsgs().end_pose
        print(f"[Sanity] {arm} AFTER:  X={ep.X_axis} Y={ep.Y_axis} Z={ep.Z_axis}")
    # Return to home briefly
    t0 = time.time()
    while time.time() - t0 < 2.0:
        for arm in ARM_NAMES:
            pipers[arm].MotionCtrl_2(0x01, 0x00, 50, 0x00)
            pipers[arm].EndPoseCtrl(*HOME_POSE)
        time.sleep(0.01)

    controller_cls = (CartesianDeltaController if mode == "delta"
                      else CartesianAbsoluteController)
    print(
        f"[Replay] Initialising {controller_cls.__name__} per arm "
        f"(move_spd_rate={move_spd_rate}%"
        f"{f', max_step_rot_deg={max_step_rot_deg}' if mode == 'delta' else ''})..."
    )
    if delta_scale != 1.0:
        print(f"[Replay] delta_scale={delta_scale:.1f}x (translation + rotation)")
    if mode == "delta":
        controllers = {
            arm: CartesianDeltaController(
                pipers[arm],
                gripper_effort=gripper_effort,
                move_spd_rate=move_spd_rate,
                max_step_rot_deg=max_step_rot_deg,
                delta_scale=delta_scale,
            )
            for arm in ARM_NAMES
        }
    else:
        controllers = {
            arm: CartesianAbsoluteController(
                pipers[arm],
                gripper_effort=gripper_effort,
                move_spd_rate=move_spd_rate,
            )
            for arm in ARM_NAMES
        }
    for arm in ARM_NAMES:
        anchor = ee_traj_units[arm][0]
        anchor_int = tuple(int(round(float(v))) for v in anchor)
        print(
            f"[Replay] {arm}: anchor EndPoseCtrl"
            f"({anchor_int[0]}, {anchor_int[1]}, {anchor_int[2]}, "
            f"{anchor_int[3]}, {anchor_int[4]}, {anchor_int[5]})"
        )
        controllers[arm].reset(
            anchor_pose=anchor,
            initial_gripper_normalized=float(traj[arm]["gripper"][0]),
            settle_time=0.5,
        )
        if mode == "delta":
            controllers[arm].set_initial_pose(anchor)

    for arm in ARM_NAMES:
        fk0 = ee_traj_units[arm][0]
        live = controllers[arm].get_current_pose()
        diff_pos_mm = np.linalg.norm(fk0[:3] - live[:3]) * 1e-3
        R_fk0 = _euler_zyx_deg_to_rot(fk0[3:6] * 1e-3)
        R_live = _euler_zyx_deg_to_rot(live[3:6] * 1e-3)
        diff_rot_deg = np.rad2deg(
            np.linalg.norm((R_live * R_fk0.inv()).as_rotvec())
        )
        print(
            f"[Replay] {arm}: FK(joints[0]) vs live end-pose -> "
            f"Δpos={diff_pos_mm:.2f} mm, |ΔR|={diff_rot_deg:.2f} deg"
        )

    if pre_replay_pause > 0:
        t0 = time.time()
        while time.time() - t0 < pre_replay_pause:
            for arm in ARM_NAMES:
                controllers[arm].resend()
            time.sleep(0.01)

    CAN_REFRESH_HZ = 100
    CAN_REFRESH_DT = 1.0 / CAN_REFRESH_HZ
    CONTROL_HZ = 15
    step_dt = 1.0 / CONTROL_HZ
    print(
        f"[Replay] Cartesian replay: {n_steps} steps @ {CONTROL_HZ} Hz "
        f"(mode={mode}), CAN refresh @ {CAN_REFRESH_HZ} Hz"
    )
    for i in range(n_steps):
        for arm in ARM_NAMES:
            gripper_cmd = float(traj[arm]["gripper"][i + 1])
            if mode == "delta":
                target = controllers[arm].step(
                    ee_traj_units[arm][i],
                    ee_traj_units[arm][i + 1],
                    gripper_normalized=gripper_cmd,
                )
            else:
                target = controllers[arm].step(
                    ee_traj_units[arm][i + 1],
                    gripper_normalized=gripper_cmd,
                )
            if i < 3:
                cmd = controllers[arm]._last_cmd
                print(
                    f"\n  [{arm}] step {i}: EndPoseCtrl"
                    f"({cmd[0]}, {cmd[1]}, {cmd[2]}, "
                    f"{cmd[3]}, {cmd[4]}, {cmd[5]})"
                )

        t0 = time.time()
        while time.time() - t0 < step_dt:
            for arm in ARM_NAMES:
                controllers[arm].resend()
            time.sleep(CAN_REFRESH_DT)

        print(f"\r[Replay] step {i + 1}/{n_steps}", end="", flush=True)

    print(f"\n[Replay] Done.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path to a recorded demoX.hdf to replay.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="delta",
        choices=["delta", "absolute"],
        help="Cartesian controller. 'delta' (default) accumulates "
        "per-step cartesian deltas on an internal running pose "
        "seeded from FK(obs[0]). 'absolute' sends each FK-computed "
        "pose directly to EndPoseCtrl. Both avoid reading "
        "GetArmEndPoseMsgs for rotation (Euler-branch mismatch).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="observation",
        choices=["observation", "action"],
        help="Joint trajectory used as FK input. 'observation' (default, "
        "recommended) = robot's proprioceptive joints, so the "
        "cartesian deltas come from the robot's own state. 'action' = "
        "master/teleop joints (debug only); the deltas describe the "
        "master arm's motion, not the robot's.",
    )
    parser.add_argument(
        "--warmup_time",
        type=float,
        default=1.5,
        help="Seconds to ramp each arm to the recorded start pose.",
    )
    parser.add_argument(
        "--gripper_effort",
        type=int,
        default=700,
        help="Effort sent to GripperCtrl during replay.",
    )
    parser.add_argument(
        "--pre_replay_pause",
        type=float,
        default=0.5,
        help="Settle delay (s) between warmup and cartesian-delta loop.",
    )
    parser.add_argument(
        "--move_spd_rate",
        type=int,
        default=50,
        help="Cartesian motion speed-rate (0..100 %%) forwarded to "
        "MotionCtrl_2(MOVE P). Default 50. SDK demos use 100.",
    )
    parser.add_argument(
        "--max_step_rot_deg",
        type=float,
        default=30.0,
        help="Per-step rotation safety cap in degrees (axis-angle "
        "magnitude), used only in --mode delta. Set to 0 to disable. "
        "Default 30 deg/step is generous for 15 Hz teleop.",
    )
    parser.add_argument(
        "--delta_scale",
        type=float,
        default=1.0,
        help="Multiply all position and rotation deltas by this factor. "
        "Use >1.0 to amplify movement for debugging (e.g. 5.0 = 5x larger). "
        "Only used in --mode delta.",
    )
    return parser.parse_args()


def _enable_piper(piper) -> None:
    """Block until all 6 motors report enabled (mirrors PiperRobot.check_enable)."""
    timeout = 5.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok = all(
            getattr(piper.GetArmLowSpdInfoMsgs(), f"motor_{i}").foc_status.driver_enable_status
            for i in range(1, 7)
        )
        if ok:
            print("  Enable status: True")
            return
        piper.EnableArm(7)
        time.sleep(0.5)
    raise RuntimeError("Arm enable timed out")


if __name__ == "__main__":
    args = _parse_args()

    if not os.path.isfile(args.filepath):
        raise FileNotFoundError(args.filepath)

    CAN_PORTS = {"left": "can_left", "right": "can_right"}
    pipers = {}
    for arm, port in CAN_PORTS.items():
        p = C_PiperInterface_V2(port)
        p.ConnectPort()
        p.EnableArm(7)
        print(f"[Init] {arm} ({port}): enabling...")
        _enable_piper(p)
        pipers[arm] = p

    # Move both arms to home (all-zero joints) via MOVE J (non-MIT).
    print("[Init] Moving to home via MOVE J...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        for arm in ARM_NAMES:
            pipers[arm].MotionCtrl_2(0x01, 0x01, 30, 0x00)
            pipers[arm].JointCtrl(0, 0, 0, 0, 0, 0)
            pipers[arm].GripperCtrl(0, 700, 0x01, 0)
        time.sleep(0.01)

    def _go_home(duration: float = 3.0, spd: int = 50) -> None:
        """Move both arms to all-zero joints via MOVE J, then smoothly
        disable (mirrors PiperRobot.safe_end behaviour)."""
        print(f"[Cleanup] Returning to home ({duration:.1f}s, {spd}% speed)...")
        t0 = time.time()
        while time.time() - t0 < duration:
            try:
                for arm in ARM_NAMES:
                    pipers[arm].MotionCtrl_2(0x01, 0x01, spd, 0x00)
                    pipers[arm].JointCtrl(0, 0, 0, 0, 0, 0)
                    pipers[arm].GripperCtrl(0, 700, 0x01, 0)
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("\n[Cleanup] Please wait, returning to home...")
                continue
        # Smoothly ramp down stiffness using MIT mode so the arm
        # settles gently before disabling (same as PiperRobot.safe_disable).
        print("[Cleanup] Ramping down stiffness...")
        kp_base = np.array([2, 3.5, 3.5, 8, 8, 8])
        kd_base = np.array([0.2, 0.35, 0.35, 0.8, 0.8, 0.8])
        direction = np.array([-1, -1, 1, -1, 1, -1])
        hold = np.zeros(6)  # arm is already at home (all zeros)
        ramp_hz, ramp_steps = 30.0, 30
        for arm in ARM_NAMES:
            for s in range(ramp_steps):
                try:
                    frac = 1.0 - s / ramp_steps
                    kp = kp_base * frac
                    kd = kd_base * frac
                    pipers[arm].MotionCtrl_2(0x01, 0x04, 0, 0xAD)
                    for j in range(6):
                        pipers[arm].JointMitCtrl(
                            j + 1, direction[j] * hold[j], 0,
                            kp[j], kd[j], 0,
                        )
                    time.sleep(1.0 / ramp_hz)
                except KeyboardInterrupt:
                    continue
        for arm in ARM_NAMES:
            pipers[arm].DisableArm(7)
        print("[Cleanup] Done, arms disabled.")

    try:
        replay_trajectory_cartesian_delta(
            pipers,
            filepath=args.filepath,
            mode=args.mode,
            source=args.source,
            warmup_time=args.warmup_time,
            gripper_effort=args.gripper_effort,
            pre_replay_pause=args.pre_replay_pause,
            move_spd_rate=args.move_spd_rate,
            max_step_rot_deg=args.max_step_rot_deg,
            delta_scale=args.delta_scale,
        )
    except KeyboardInterrupt:
        print("\n[Replay] Interrupted.")
    finally:
        _go_home()
