"""PlaCo transaction planner for the MuJoCo-specific TB6-R5 URDF."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import placo

import config

JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
EE_FRAME = "ee_Link"


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _six(values: Any, name: str = "joint vector") -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite radians")
    return result.copy()


def _set_q(robot: placo.RobotWrapper, q: Any) -> None:
    values = _six(q)
    for name, value in zip(JOINT_NAMES, values):
        robot.set_joint(name, float(value))
    robot.update_kinematics()


def _load_robot() -> placo.RobotWrapper:
    if not config.MUJOCO_URDF_PATH.is_file():
        raise RuntimeError(f"MuJoCo URDF is unavailable: {config.MUJOCO_URDF_PATH}")
    return placo.RobotWrapper(str(config.MUJOCO_URDF_PATH), int(placo.Flags.ignore_collisions))


class IncrementalMujocoPlacoPlanner:
    """One temporary TCP candidate per XR frame; rejected work is never retained."""

    def __init__(self, q_initial: Any) -> None:
        self.robot = _load_robot()
        self.robot.set_velocity_limits(config.JOINT_SPEED_LIMIT_RAD_S)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.dt = config.DT
        self.solver.enable_joint_limits(True)
        self.solver.enable_velocity_limits(True)
        self.columns: list[int] = []
        for name in JOINT_NAMES:
            joint_id = self.robot.model.getJointId(name)
            if joint_id == 0 or self.robot.model.joints[joint_id].nv != 1:
                raise RuntimeError(f"required joint absent: {name}")
            self.columns.append(int(self.robot.model.joints[joint_id].idx_v))
        self.limits = np.asarray([self.robot.get_joint_limits(name) for name in JOINT_NAMES], dtype=float)
        self.q_commit = _six(q_initial, "q_initial")
        _set_q(self.robot, self.q_commit)
        self.task = self.solver.add_frame_task(EE_FRAME, self.current_tcp())
        self.task.configure("ee_6d_incremental", "scaled", 1.0, 1.0)
        self._solve_once = self.solver.solve
        self.last_failure: Optional[str] = None

    def current_q_robot(self) -> np.ndarray:
        return np.asarray([self.robot.get_joint(name) for name in JOINT_NAMES], dtype=float)

    def current_tcp(self) -> np.ndarray:
        _set_q(self.robot, self.q_commit)
        return np.asarray(self.robot.get_T_world_frame(EE_FRAME), dtype=float).copy()

    def _metrics(self) -> tuple[float, float]:
        values = np.linalg.svd(
            np.asarray(self.robot.frame_jacobian(EE_FRAME, "local_world_aligned"), dtype=float)[:, self.columns],
            compute_uv=False,
        )
        return float(values[-1]), float(values[0] / values[-1]) if values[-1] > 1e-12 else float("inf")

    def _restore_commit(self) -> None:
        _set_q(self.robot, self.q_commit)
        self.task.T_world_frame = self.current_tcp()

    def reset_to(self, q: Any) -> None:
        self.q_commit = _six(q)
        self._restore_commit()

    def propose(self, delta_position_robot: Any, delta_rotation_robot: Any) -> Optional[dict[str, Any]]:
        q_before = self.q_commit.copy()
        try:
            _set_q(self.robot, q_before)
            tcp_base = np.asarray(self.robot.get_T_world_frame(EE_FRAME), dtype=float).copy()
            delta_p = np.asarray(delta_position_robot, dtype=float)
            delta_r = np.asarray(delta_rotation_robot, dtype=float)
            if delta_p.shape != (3,) or delta_r.shape != (3, 3) or not np.all(np.isfinite(delta_p)) or not np.all(np.isfinite(delta_r)):
                raise ValueError("frame delta is invalid")
            tcp_candidate = tcp_base.copy()
            tcp_candidate[:3, 3] += delta_p
            tcp_candidate[:3, :3] = orthonormalize_rotation(delta_r @ tcp_base[:3, :3])
            sigma_min, condition = self._metrics()
            self.task.T_world_frame = tcp_candidate
            self._solve_once(True)
            self.robot.update_kinematics()
            q_candidate = self.current_q_robot()
            velocity = (q_candidate - q_before) / config.DT
            peak = float(np.max(np.abs(velocity)))
            if np.any(q_candidate < self.limits[:, 0]) or np.any(q_candidate > self.limits[:, 1]):
                raise RuntimeError("PlaCo candidate violates a joint limit")
            if peak > config.JOINT_SPEED_LIMIT_RAD_S + 1e-9:
                raise RuntimeError("PlaCo candidate exceeds its configured velocity limit")
        except Exception as exc:
            self._restore_commit()
            self.last_failure = f"{type(exc).__name__}: {exc}"
            return None
        self.last_failure = None
        return {"q_before": q_before, "q_candidate": q_candidate, "q_command": q_candidate.copy(),
                "joint_velocity": velocity, "requested_peak_velocity": peak, "tcp_base": tcp_base,
                "tcp_candidate": tcp_candidate, "placo_scale": float(self.solver.scale),
                "sigma_min": sigma_min, "condition": condition}

    def commit(self, pending: Mapping[str, Any]) -> np.ndarray:
        if not np.array_equal(_six(pending["q_before"]), self.q_commit):
            raise RuntimeError("stale transaction")
        self.q_commit = _six(pending["q_command"], "q_command")
        self._restore_commit()
        return self.q_commit.copy()

    def rollback(self) -> None:
        self._restore_commit()
