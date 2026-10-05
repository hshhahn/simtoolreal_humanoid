from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from termcolor import colored

OBSERVATIONS_DIM = 133
ACTIONS_DIM = 29

# This is the joint order in isaacgym originally from parsing the original urdf
# IsaacGym sorts by alphabetical order when "depth" is equal
# While other libraries simply use the order specified in the urdf when "depth" is equal
# We don't want to use this old one because the sharpa joint order from the real robot does not match this
OLD_JOINT_ORDER = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
]

# This is the joint order that matches the real robot
ADJUSTED_JOINT_ORDER = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

JOINT_NAMES_ISAACGYM = ADJUSTED_JOINT_ORDER

OBS_NAME_TO_NAMES = {
    "q": [f"{name}_q" for name in JOINT_NAMES_ISAACGYM],
    "qd": [f"{name}_qd" for name in JOINT_NAMES_ISAACGYM],
    "palm_center_pos": [f"palm_center_pos_{x}" for x in "xyz"],
    "palm_rot": [f"palm_rot_{x}" for x in "xyzw"],
    "palm_linvel": [f"palm_linvel_{x}" for x in "xyz"],
    "palm_angvel": [f"palm_angvel_{x}" for x in "xyz"],
    "object_rot": [f"object_rot_{x}" for x in "xyzw"],
    "object_linvel": [f"object_linvel_{x}" for x in "xyz"],
    "object_angvel": [f"object_angvel_{x}" for x in "xyz"],
    "fingertip_rel_pos": [
        f"fingertip_rel_pos_{finger}_{x}"
        for finger in ["index", "middle", "ring", "thumb", "pinky"]
        for x in "xyz"
    ],
    "keypoints_rel_palm": [
        f"keypoints_rel_palm_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "keypoints_rel_goal": [
        f"keypoints_rel_goal_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "object_scales": [f"object_scales_{x}" for x in "xyz"],
    "closest_keypoint_max_dist": ["closest_keypoint_max_dist"],
    "closest_fingertip_dist": [
        f"closest_fingertip_dist_{finger}"
        for finger in ["index", "middle", "ring", "thumb", "pinky"]
    ],
    "lifted_object": ["lifted_object"],
    "progress_obs": ["progress_obs"],
    "successes": ["successes_obs"],
    "reward_obs": ["reward_obs"],
}
OBS_NAMES = sum(OBS_NAME_TO_NAMES.values(), [])
ACTION_NAMES = JOINT_NAMES_ISAACGYM


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
    import time

    """
    Print a warning message at most once every n_seconds per unique key.
    Stores state inside the function itself (no globals).
    """
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}  # create on first call

    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)

    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


@dataclass
class RecordedData:
    robot_root_states_array: np.ndarray
    object_root_states_array: np.ndarray
    robot_joint_positions_array: np.ndarray
    time_array: np.ndarray
    robot_joint_names: list[str]

    table_root_states_array: Optional[np.ndarray] = None
    goal_root_states_array: Optional[np.ndarray] = None
    robot_joint_velocities_array: Optional[np.ndarray] = None
    robot_joint_pos_targets_array: Optional[np.ndarray] = None

    observations_array: Optional[np.ndarray] = None
    actions_array: Optional[np.ndarray] = None

    object_name: Optional[str] = None

    def __post_init__(self):
        T = self.T
        J = self.J
        ROOT_STATE_DIM = 13  # xyz, xyzw, linvel, angvel
        assert self.robot_root_states_array.shape == (T, ROOT_STATE_DIM), (
            f"Expected robot root states array to be (T, {ROOT_STATE_DIM}), got {self.robot_root_states_array.shape}"
        )
        assert self.object_root_states_array.shape == (T, ROOT_STATE_DIM), (
            f"Expected object root states array to be (T, {ROOT_STATE_DIM}), got {self.object_root_states_array.shape}"
        )
        assert self.robot_joint_positions_array.shape == (T, J), (
            f"Expected robot joint positions array to be (T, J), got {self.robot_joint_positions_array.shape}"
        )
        assert self.time_array.shape == (T,), (
            f"Expected time array to be (T,), got {self.time_array.shape}"
        )
        assert len(self.robot_joint_names) == J, (
            f"Expected robot joint names to have length J, got {len(self.robot_joint_names)} and {J}"
        )

        if self.table_root_states_array is not None:
            assert self.table_root_states_array.shape == (T, ROOT_STATE_DIM), (
                f"Expected table root states array to be (T, {ROOT_STATE_DIM}), got {self.table_root_states_array.shape}"
            )
        if self.goal_root_states_array is not None:
            assert self.goal_root_states_array.shape == (T, ROOT_STATE_DIM), (
                f"Expected goal root states array to be (T, {ROOT_STATE_DIM}), got {self.goal_root_states_array.shape}"
            )
        if self.robot_joint_velocities_array is not None:
            assert self.robot_joint_velocities_array.shape == (T, J), (
                f"Expected robot joint velocities array to be (T, J), got {self.robot_joint_velocities_array.shape}"
            )
        if self.robot_joint_pos_targets_array is not None:
            assert self.robot_joint_pos_targets_array.shape == (T, J), (
                f"Expected robot joint pos targets array to be (T, J), got {self.robot_joint_pos_targets_array.shape}"
            )

        if self.observations_array is not None:
            assert self.observations_array.shape == (T, self.observations_dim), (
                f"Expected observations array to be (T, {self.observations_dim}), got {self.observations_array.shape}"
            )
        if self.actions_array is not None:
            assert self.actions_array.shape == (T, self.actions_dim), (
                f"Expected actions array to be (T, {self.actions_dim}), got {self.actions_array.shape}"
            )

    def to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            file_path,
            robot_root_states_array=self.robot_root_states_array,
            object_root_states_array=self.object_root_states_array,
            robot_joint_positions_array=self.robot_joint_positions_array,
            time_array=self.time_array,
            robot_joint_names=self.robot_joint_names,
            table_root_states_array=self.table_root_states_array,
            goal_root_states_array=self.goal_root_states_array,
            robot_joint_velocities_array=self.robot_joint_velocities_array,
            robot_joint_pos_targets_array=self.robot_joint_pos_targets_array,
            observations_array=self.observations_array,
            actions_array=self.actions_array,
            object_name=self.object_name,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> RecordedData:
        assert file_path.exists(), f"File {file_path} does not exist"
        recorded_data = np.load(file_path, allow_pickle=True)

        def maybe_none(x):
            if isinstance(x, np.ndarray) and x.shape == ():
                # If 0-D array, just get the item
                # Handles case of None and string and other types
                item = x.item()
                return item
            return x

        return cls(
            robot_root_states_array=recorded_data["robot_root_states_array"],
            object_root_states_array=recorded_data["object_root_states_array"],
            robot_joint_positions_array=recorded_data["robot_joint_positions_array"],
            time_array=recorded_data["time_array"],
            robot_joint_names=recorded_data["robot_joint_names"],
            table_root_states_array=maybe_none(
                recorded_data["table_root_states_array"]
                if "table_root_states_array" in recorded_data
                else None
            ),
            goal_root_states_array=maybe_none(
                recorded_data["goal_root_states_array"]
                if "goal_root_states_array" in recorded_data
                else None
            ),
            robot_joint_velocities_array=maybe_none(
                recorded_data["robot_joint_velocities_array"]
                if "robot_joint_velocities_array" in recorded_data
                else None
            ),
            robot_joint_pos_targets_array=maybe_none(
                recorded_data["robot_joint_pos_targets_array"]
                if "robot_joint_pos_targets_array" in recorded_data
                else None
            ),
            observations_array=maybe_none(
                recorded_data["observations_array"]
                if "observations_array" in recorded_data
                else None
            ),
            actions_array=maybe_none(
                recorded_data["actions_array"]
                if "actions_array" in recorded_data
                else None
            ),
            object_name=maybe_none(
                recorded_data["object_name"] if "object_name" in recorded_data else None
            ),
        )

    def slice(
        self, start: int | None = None, end: int | None = None, reset_time: bool = True
    ) -> RecordedData:
        if start is None and end is None:
            raise ValueError("start and end cannot both be None")

        if start is None:
            start = 0
        if end is None:
            end = self.T

        return RecordedData(
            robot_root_states_array=self.robot_root_states_array[start:end],
            object_root_states_array=self.object_root_states_array[start:end],
            robot_joint_positions_array=self.robot_joint_positions_array[start:end],
            time_array=(
                self.time_array[start:end] - self.time_array[start]
                if reset_time
                else self.time_array[start:end]
            ),
            robot_joint_names=self.robot_joint_names,
            table_root_states_array=self.table_root_states_array[start:end]
            if self.table_root_states_array is not None
            else None,
            goal_root_states_array=self.goal_root_states_array[start:end]
            if self.goal_root_states_array is not None
            else None,
            robot_joint_velocities_array=self.robot_joint_velocities_array[start:end]
            if self.robot_joint_velocities_array is not None
            else None,
            robot_joint_pos_targets_array=self.robot_joint_pos_targets_array[start:end]
            if self.robot_joint_pos_targets_array is not None
            else None,
            observations_array=self.observations_array[start:end]
            if self.observations_array is not None
            else None,
            actions_array=self.actions_array[start:end]
            if self.actions_array is not None
            else None,
            object_name=self.object_name,
        )

    def __len__(self) -> int:
        return self.T

    def robot_joint_positions_reordered(
        self,
        to_order: list[str],
    ) -> np.ndarray:
        return self.change_joint_order(
            q=self.robot_joint_positions_array,
            from_order=self.robot_joint_names,
            to_order=to_order,
        )

    def robot_joint_pos_targets_reordered(
        self,
        to_order: list[str],
    ) -> np.ndarray:
        return self.change_joint_order(
            q=self.robot_joint_pos_targets_array,
            from_order=self.robot_joint_names,
            to_order=to_order,
        )

    # ###############
    # Complex Properties
    # ###############
    @cached_property
    def robot_joint_velocities_array_fd2(self) -> np.ndarray:
        q = self.robot_joint_positions_array
        t = self.time_array

        qd = np.zeros_like(q)

        # Interior points: 2nd order finite difference
        # q_i = (q_{i+1} - q_{i-1}) / (t_{i+1} - t_{i-1})
        qd[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2])[..., None]

        # First point: 1st order finite difference
        # q_0 = (q_1 - q_0) / (t_1 - t_0)
        qd[0] = (q[1] - q[0]) / (t[1] - t[0])

        # Last point: 1st order finite difference
        # q_N = (q_N - q_{N-1}) / (t_N - t_{N-1})
        qd[-1] = (q[-1] - q[-2]) / (t[-1] - t[-2])
        return qd

    @cached_property
    def robot_joint_velocities_array_fd1(self) -> np.ndarray:
        q = self.robot_joint_positions_array
        t = self.time_array
        qd = np.zeros_like(q)

        # Forward difference for all but the last point
        # q_i = (q_{i+1} - q_i) / (t_{i+1} - t_i)
        qd[1:] = (q[1:] - q[:-1]) / (t[1:] - t[:-1])[..., None]

        # Backward difference for the last point
        # q_N = (q_N - q_{N-1}) / (t_N - t_{N-1})
        qd[-1] = (q[-1] - q[-2]) / (t[-1] - t[-2])
        return qd

    @cached_property
    def robot_ee_pose_array(self) -> np.ndarray:
        return self._compute_ee_pose_array(
            q=self.robot_joint_positions_array, joint_names=self.robot_joint_names
        )

    @cached_property
    def robot_target_ee_pose_array(self) -> np.ndarray:
        return self._compute_ee_pose_array(
            q=self.robot_joint_pos_targets_array, joint_names=self.robot_joint_names
        )

    # ###############
    # Implementation Details
    # ###############
    def _compute_ee_pose_array(
        self, q: np.ndarray, joint_names: list[str]
    ) -> np.ndarray:
        from isaacgymenvs.utils.observation_action_utils_sharpa import (
            compute_fk_dict,
            create_urdf_object,
        )

        urdf = create_urdf_object(robot_name="iiwa14_left_sharpa_adjusted_restricted")
        assert list(joint_names) == list(urdf.actuated_joint_names), (
            f"joint_names: {joint_names} != urdf.actuated_joint_names: {urdf.actuated_joint_names}"
        )
        fk_dict = compute_fk_dict(urdf=urdf, q=q, link_names=["iiwa14_link_7"])
        ee_pose_matrix = fk_dict["iiwa14_link_7"]
        return self.T_to_pose(ee_pose_matrix)

    # ###############
    # Hardcoded Properties
    # ###############
    @cached_property
    def observation_names(self) -> list[str]:
        names = OBS_NAMES
        assert len(names) == OBSERVATIONS_DIM, (
            f"Expected {len(names)} observation names, got {OBSERVATIONS_DIM}"
        )
        return names

    @cached_property
    def action_names(self) -> list[str]:
        names = ACTION_NAMES
        assert len(names) == ACTIONS_DIM, (
            f"Expected {len(names)} action names, got {ACTIONS_DIM}"
        )
        return names

    # ###############
    # Simple Properties
    # ###############
    @cached_property
    def T(self) -> int:
        return self.robot_root_states_array.shape[0]

    @cached_property
    def dt(self) -> float:
        dt = self.time_array[1] - self.time_array[0]

        dt_array = np.diff(self.time_array)
        all_close = np.allclose(dt_array, dt)
        if not all_close:
            fps_array = 1.0 / dt_array
            WARN_EVERY_N_SECONDS = 100.0
            warn_every("~" * 100, WARN_EVERY_N_SECONDS)
            warn_every("Time array is not evenly spaced", WARN_EVERY_N_SECONDS)
            warn_every(f"Differences: {dt_array}", WARN_EVERY_N_SECONDS)
            warn_every(f"dt: {dt}", WARN_EVERY_N_SECONDS)
            warn_every(f"Min difference: {np.min(dt_array)}", WARN_EVERY_N_SECONDS)
            warn_every(
                f"Median difference: {np.median(dt_array)}", WARN_EVERY_N_SECONDS
            )
            warn_every(f"Mean difference: {np.mean(dt_array)}", WARN_EVERY_N_SECONDS)
            warn_every(f"Max difference: {np.max(dt_array)}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps array: {fps_array}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps: {1.0 / dt}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps mean: {np.mean(fps_array)}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps median: {np.median(fps_array)}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps max: {np.max(fps_array)}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps min: {np.min(fps_array)}", WARN_EVERY_N_SECONDS)
            warn_every(f"fps std: {np.std(fps_array)}", WARN_EVERY_N_SECONDS)
            ERROR = False
            if ERROR:
                raise ValueError("Time array is not evenly spaced")
            USE_MEDIAN_DT = True
            if USE_MEDIAN_DT:
                warn_every(
                    f"Using median dt: {np.median(dt_array)}", WARN_EVERY_N_SECONDS
                )
                dt = np.median(dt_array)
            warn_every("~" * 100, WARN_EVERY_N_SECONDS)

        return dt

    @cached_property
    def J(self) -> int:
        return len(self.robot_joint_names)

    @cached_property
    def total_time(self) -> float:
        return self.time_array[-1] - self.time_array[0]

    @cached_property
    def observations_dim(self) -> int:
        assert self.observations_array.shape[-1] == OBSERVATIONS_DIM, (
            f"Expected observations array to have shape (..., {OBSERVATIONS_DIM}), got {self.observations_array.shape}"
        )
        return OBSERVATIONS_DIM

    @cached_property
    def actions_dim(self) -> int:
        assert self.actions_array.shape[-1] == ACTIONS_DIM, (
            f"Expected actions array to have shape (..., {ACTIONS_DIM}), got {self.actions_array.shape}"
        )
        return ACTIONS_DIM

    # ###############
    # Static methods
    # ###############
    @staticmethod
    def change_joint_order(
        q: np.ndarray,
        from_order: list[str],
        to_order: list[str],
        require_all_joints: bool = True,
    ) -> np.ndarray:
        J = len(from_order)
        assert q.ndim in [1, 2], (
            f"Expected q to be either (N,) or (N, J), got {q.shape}"
        )
        assert q.shape[-1] == J, (
            f"Expected q to have the same length as from_order, got {q.shape[-1]} and {J}"
        )

        if require_all_joints:
            assert len(to_order) == J, (
                f"Expected to_order to have the same length as from_order, got {len(to_order)} and {len(from_order)}. If you don't want to require all joints, set require_all_joints to False."
            )

        assert set(to_order).issubset(set(from_order)), (
            f"Expected to_order to be a subset of from_order, got to_order: {to_order} and from_order: {from_order}. Only in to_order: {set(to_order) - set(from_order)}"
        )

        # q is given in the from_order
        joint_name_to_value = {from_order[i]: q[..., i] for i in range(J)}
        new_q = np.stack([joint_name_to_value[name] for name in to_order], axis=-1)

        assert new_q.shape == (q.shape[:-1] + (len(to_order),)), (
            f"Expected new_q to be {q.shape[:-1] + (len(to_order),)}, got {new_q.shape}"
        )
        if require_all_joints:
            assert new_q.shape == q.shape, (
                f"Expected new_q to be {q.shape}, got {new_q.shape}"
            )
        return new_q

    @staticmethod
    def pose_to_T(pose: np.ndarray) -> np.ndarray:
        assert pose.ndim in [1, 2], (
            f"Expected pose to be either (7,) or (N, 7), got {pose.shape}"
        )
        assert pose.shape[-1] == 7, (
            f"Expected pose to be (7,) or (N, 7), got {pose.shape}"
        )
        xyz = pose[..., :3]
        xyzw = pose[..., 3:7]
        T = (
            np.eye(4)
            if pose.ndim == 1
            else np.eye(4)[None, ...].repeat(repeats=pose.shape[0], axis=0)
        )
        T[..., :3, :3] = R.from_quat(xyzw).as_matrix()
        T[..., :3, 3] = xyz
        return T

    @staticmethod
    def T_to_pose(T: np.ndarray) -> np.ndarray:
        assert T.ndim in [2, 3], (
            f"Expected T to be either (4, 4) or (N, 4, 4), got {T.shape}"
        )
        assert T.shape[-2:] == (4, 4), (
            f"Expected T to be (4, 4) or (N, 4, 4), got {T.shape}"
        )
        xyz = T[..., :3, 3]
        xyzw = R.from_matrix(T[..., :3, :3]).as_quat()
        pose = np.concatenate([xyz, xyzw], axis=-1)
        assert pose.shape == (T.shape[:-2] + (7,)), (
            f"Expected pose to be {T.shape[:-2] + (7,)}, got {pose.shape}"
        )
        return pose
