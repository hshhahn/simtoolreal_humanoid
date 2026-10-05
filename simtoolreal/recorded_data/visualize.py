from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from termcolor import colored
from viser.extras import ViserUrdf

from isaacgymenvs.utils.utils import get_repo_root_dir
from recorded_data import RecordedData


def warn(message: str):
    print(colored(message, "yellow"))


# ###########
# Constants
# ###########
GREEN_RGBA = (0, 255, 0, 0.5)
AXES_LENGTH = 0.2
AXES_RADIUS = 0.01

DISABLE_AXES = True
if DISABLE_AXES:
    AXES_LENGTH = 0.00001
    AXES_RADIUS = 0.00001


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    assert xyzw.shape[-1] == 4, f"Expected xyzw to be (..., 4), got {xyzw.shape}"
    return xyzw[..., [3, 0, 1, 2]]


def find_closest_object_name(text: str) -> Optional[str]:
    from dextoolbench.metadata import ALL_OBJECT_NAMES

    """Find the closest object name from ALL_OBJECT_NAMES that matches the given text."""
    matching_names = []
    for obj_name in ALL_OBJECT_NAMES:
        if obj_name in text:
            matching_names.append(obj_name)

    if len(matching_names) == 0:
        warn(f"No object name found in text: {text}")
        return None
    elif len(matching_names) == 1:
        return matching_names[0]
    else:
        # If multiple matches, return the longest one (most specific)
        # e.g., "claw_hammer" should match over "hammer"
        longest_match = max(matching_names, key=len)
        warn(
            f"Multiple object names found in text '{text}': {matching_names}. Using longest match: {longest_match}"
        )
        return longest_match


@dataclass
class VisualizeArgs:
    file_path: Path = Path("recorded_data/2025-10-20_14-32-39_None_310.npz")
    """Path to the recorded data file."""

    object_name: Optional[str] = None
    """The name of the object to visualize. If None, try to infer the object name from the recorded data. If can't be done, then use a default object name."""


def main():
    args: VisualizeArgs = tyro.cli(VisualizeArgs)
    file_path = args.file_path
    object_name = args.object_name

    # ###########
    # Load recorded data
    # ###########
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)

    FILL_IN_EXPECTED_TABLE_ROOT_STATES_IF_MISSING = True
    if (
        FILL_IN_EXPECTED_TABLE_ROOT_STATES_IF_MISSING
        and recorded_data.table_root_states_array is None
    ):
        warn(
            f"Filling in expected table root states because table root states array is missing for file: {file_path}"
        )
        expected_table_root_states = np.zeros((len(recorded_data), 13))
        expected_table_root_states[:, :3] = np.array([0.0, 0.0, 0.38])[None]
        expected_table_root_states[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])[None]
        recorded_data.table_root_states_array = expected_table_root_states

    # ###########
    # Create viser server and create viser objects
    # ###########
    # Create server
    SERVER = viser.ViserServer()
    SERVER.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Set initial camera pose
    @SERVER.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, -1.0, 1.03)
        # client.camera.wxyz = (0, 0, 0, 1)
        client.camera.look_at = (0, 0, 0.53)

        USE_REAL_CAMERA_T_R_C = True
        if USE_REAL_CAMERA_T_R_C:
            T_W_R = np.eye(4)
            T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])
            T_R_C = np.array(
                [
                    [
                        0.95527630647288930,
                        -0.17920451516639435,
                        0.23522950502752071,
                        -0.50020504226664309,
                    ],
                    [
                        -0.28890230754832508,
                        -0.39580744250644329,
                        0.87170632964878869,
                        -1.43857156913606077,
                    ],
                    [
                        -0.06310812138518884,
                        -0.90067874972183481,
                        -0.42987806970668574,
                        1.02018932829980047,
                    ],
                    [
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        1.00000000000000000,
                    ],
                ]
            )
            T_W_C = T_W_R @ T_R_C
            client.camera.position = T_W_C[:3, 3]
            client.camera.wxyz = xyzw_to_wxyz(R.from_matrix(T_W_C[:3, :3]).as_quat())

    # Load assets into viser
    KUKA_SHARPA_URDF_PATH = (
        get_repo_root_dir()
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    assert KUKA_SHARPA_URDF_PATH.exists(), (
        f"KUKA_SHARPA_URDF_PATH not found: {KUKA_SHARPA_URDF_PATH}"
    )
    from dextoolbench.objects import NAME_TO_OBJECT

    if object_name is None and recorded_data.object_name is not None:
        object_name = recorded_data.object_name
        print(f"Using object_name: {object_name} from recorded data")

    if object_name is None:
        closest_object_name = find_closest_object_name(file_path.stem)
        if closest_object_name is not None:
            object_name = closest_object_name
            warn(
                f"Using closest object name: {object_name} from recorded data file path: {file_path.stem}"
            )

    DEFAULT_OBJECT_NAME = "claw_hammer"
    if object_name is not None and object_name not in NAME_TO_OBJECT:
        warn(
            f"Object name {object_name} not found in NAME_TO_OBJECT, using default object name: {DEFAULT_OBJECT_NAME}"
        )
        object_name = DEFAULT_OBJECT_NAME

    if object_name is None:
        warn(f"Using default object name: {DEFAULT_OBJECT_NAME}")
        object_name = DEFAULT_OBJECT_NAME

    assert object_name in NAME_TO_OBJECT, (
        f"Object name {object_name} not found in NAME_TO_OBJECT"
    )

    OBJECT_URDF_PATH = NAME_TO_OBJECT[object_name].urdf_path
    assert OBJECT_URDF_PATH.exists(), f"OBJECT_URDF_PATH not found: {OBJECT_URDF_PATH}"
    SHARPA_URDF_PATH = (
        get_repo_root_dir()
        / "assets/urdf/left_sharpa_ha4/left_sharpa_ha4_v2_1_adjusted_restricted.urdf"
    )
    assert SHARPA_URDF_PATH.exists(), f"SHARPA_URDF_PATH not found: {SHARPA_URDF_PATH}"
    TABLE_URDF_PATH = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
    assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

    # Robot
    kuka_sharpa_frame = SERVER.scene.add_frame(
        "/robot/state",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
    )
    kuka_sharpa_viser = ViserUrdf(
        SERVER, KUKA_SHARPA_URDF_PATH, root_node_name="/robot/state"
    )

    # Target robot
    if recorded_data.robot_joint_pos_targets_array is not None:
        target_kuka_sharpa_frame = SERVER.scene.add_frame(
            "/target_robot/state",
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        BLUE_RGBA = (0, 0, 255, 0.5)
        target_kuka_sharpa_viser = ViserUrdf(
            SERVER,
            KUKA_SHARPA_URDF_PATH,
            root_node_name="/target_robot/state",
            mesh_color_override=BLUE_RGBA,
        )

    # Object
    object_frame = SERVER.scene.add_frame(
        "/object", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )
    _object_viser = ViserUrdf(SERVER, OBJECT_URDF_PATH, root_node_name="/object")

    # Table
    if recorded_data.table_root_states_array is not None:
        table_frame = SERVER.scene.add_frame(
            "/table", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
        )
        _table_viser = ViserUrdf(SERVER, TABLE_URDF_PATH, root_node_name="/table")

    # Goal
    if recorded_data.goal_root_states_array is not None:
        goal_frame = SERVER.scene.add_frame(
            "/goal", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
        )
        INCLUDE_GOAL_OBJECT = True
        if INCLUDE_GOAL_OBJECT:
            _goal_object_viser = ViserUrdf(
                SERVER,
                OBJECT_URDF_PATH,
                root_node_name="/goal",
                mesh_color_override=GREEN_RGBA,
            )

    # Palm
    palm_frame = SERVER.scene.add_frame(
        "/robot_palm", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )

    # Floating sharpa hand
    sharpa_frame = SERVER.scene.add_frame(
        "/floating_sharpa_hand",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
    )
    sharpa_viser = ViserUrdf(
        SERVER, SHARPA_URDF_PATH, root_node_name="/floating_sharpa_hand"
    )

    # Object relative to floating sharpa hand
    object_in_sharpa_frame = SERVER.scene.add_frame(
        "/floating_sharpa_hand/object",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
    )
    _object_in_sharpa_viser = ViserUrdf(
        SERVER, OBJECT_URDF_PATH, root_node_name="/floating_sharpa_hand/object"
    )

    # Get joint names since the ordering of the urdf may not match the ordering of the robot_joint_names
    kuka_sharpa_viser_joint_names = kuka_sharpa_viser._urdf.actuated_joint_names
    sharpa_viser_joint_names = sharpa_viser._urdf.actuated_joint_names
    assert set(sharpa_viser_joint_names).issubset(set(kuka_sharpa_viser_joint_names)), (
        f"sharpa_viser_joint_names: {sharpa_viser_joint_names} is not a subset of kuka_sharpa_viser_joint_names: {kuka_sharpa_viser_joint_names}\n"
        f"Only in sharpa_viser_joint_names: {set(sharpa_viser_joint_names) - set(kuka_sharpa_viser_joint_names)}\n"
        f"Only in kuka_sharpa_viser_joint_names: {set(kuka_sharpa_viser_joint_names) - set(sharpa_viser_joint_names)}"
    )

    # ###########
    # Add controls
    # ###########
    def get_frame_idx_slider_text(
        recorded_data: RecordedData,
        idx: int,
    ) -> str:
        fps = 1 / recorded_data.dt
        current_time = recorded_data.time_array[idx] - recorded_data.time_array[0]
        total_time = recorded_data.total_time
        return f"{current_time:.3f}s/{total_time:.3f}s ({idx:04d}/{len(recorded_data):04d}) ({fps:.0f}fps)"

    with SERVER.gui.add_folder("Frame Controls"):
        frame_idx_slider = SERVER.gui.add_slider(
            label=get_frame_idx_slider_text(recorded_data=recorded_data, idx=0),
            min=0,
            max=len(recorded_data) - 1,
            step=1,
            initial_value=0,
        )
        pause_toggle_button = SERVER.gui.add_button(
            label="Pause",
        )
        increment_button = SERVER.gui.add_button(
            label="Increment",
        )
        decrement_button = SERVER.gui.add_button(
            label="Decrement",
        )
        reset_button = SERVER.gui.add_button(
            label="Reset",
        )

    # Loop state
    FRAME_IDX = frame_idx_slider.value
    PAUSED = False

    @frame_idx_slider.on_update
    def _(_) -> None:
        nonlocal FRAME_IDX, frame_idx_slider
        FRAME_IDX = int(
            np.clip(frame_idx_slider.value, a_min=0, a_max=len(recorded_data) - 1)
        )
        frame_idx_slider.label = get_frame_idx_slider_text(
            recorded_data=recorded_data, idx=FRAME_IDX
        )

    @pause_toggle_button.on_click
    def _(_) -> None:
        nonlocal PAUSED
        PAUSED = not PAUSED
        if PAUSED:
            pause_toggle_button.label = "Play"
        else:
            pause_toggle_button.label = "Pause"

    @increment_button.on_click
    def _(_) -> None:
        nonlocal PAUSED, frame_idx_slider, pause_toggle_button
        if not PAUSED:
            pause_toggle_button.value = True

        frame_idx_slider.value = int(
            np.clip(frame_idx_slider.value + 1, a_min=0, a_max=len(recorded_data) - 1)
        )

    @decrement_button.on_click
    def _(_) -> None:
        nonlocal PAUSED, frame_idx_slider, pause_toggle_button
        if not PAUSED:
            pause_toggle_button.value = True

        frame_idx_slider.value = int(
            np.clip(frame_idx_slider.value - 1, a_min=0, a_max=len(recorded_data) - 1)
        )

    @reset_button.on_click
    def _(_) -> None:
        nonlocal frame_idx_slider
        frame_idx_slider.value = 0

    # ###########
    # Main loop
    # ###########
    while True:
        start_loop_time = time.time()

        # Get data
        robot_root_state = recorded_data.robot_root_states_array[FRAME_IDX]
        object_root_state = recorded_data.object_root_states_array[FRAME_IDX]
        robot_joint_position = recorded_data.robot_joint_positions_array[FRAME_IDX]

        # ###########
        # Update viser objects
        # ###########
        # Robot
        kuka_sharpa_frame.position = robot_root_state[:3]
        kuka_sharpa_frame.wxyz = xyzw_to_wxyz(robot_root_state[3:7])
        kuka_sharpa_joint_pos_viser_order = RecordedData.change_joint_order(
            robot_joint_position,
            from_order=recorded_data.robot_joint_names,
            to_order=kuka_sharpa_viser_joint_names,
        )
        kuka_sharpa_viser.update_cfg(kuka_sharpa_joint_pos_viser_order)

        # Target robot
        if recorded_data.robot_joint_pos_targets_array is not None:
            robot_joint_pos_target = recorded_data.robot_joint_pos_targets_array[
                FRAME_IDX
            ]
            target_kuka_sharpa_frame.position = robot_root_state[:3]
            target_kuka_sharpa_frame.wxyz = xyzw_to_wxyz(robot_root_state[3:7])
            target_kuka_sharpa_joint_pos_viser_order = robot_joint_pos_target
            target_kuka_sharpa_viser.update_cfg(
                target_kuka_sharpa_joint_pos_viser_order
            )

        # Object
        object_frame.position = object_root_state[:3]
        object_frame.wxyz = xyzw_to_wxyz(object_root_state[3:7])

        # Table
        if recorded_data.table_root_states_array is not None:
            table_frame.position = recorded_data.table_root_states_array[FRAME_IDX, :3]
            table_frame.wxyz = xyzw_to_wxyz(
                recorded_data.table_root_states_array[FRAME_IDX, 3:7]
            )

        # Goal
        if recorded_data.goal_root_states_array is not None:
            goal_frame.position = recorded_data.goal_root_states_array[FRAME_IDX, :3]
            goal_frame.wxyz = xyzw_to_wxyz(
                recorded_data.goal_root_states_array[FRAME_IDX, 3:7]
            )

        # Floating hand
        sharpa_joint_pos_viser_order = RecordedData.change_joint_order(
            robot_joint_position,
            from_order=recorded_data.robot_joint_names,
            to_order=sharpa_viser_joint_names,
            require_all_joints=False,
        )
        sharpa_viser.update_cfg(sharpa_joint_pos_viser_order)

        # Palm
        palm_pose_R = kuka_sharpa_viser._urdf.get_transform(
            frame_to="left_hand_C_MC"
        ).copy()
        assert palm_pose_R.shape == (
            4,
            4,
        ), f"palm_pose_R.shape: {palm_pose_R.shape}"
        T_R_P = palm_pose_R
        T_W_R = RecordedData.pose_to_T(robot_root_state[:7])
        T_W_P = T_W_R @ T_R_P
        palm_xyz_xyzw = RecordedData.T_to_pose(T_W_P)
        palm_frame.position = palm_xyz_xyzw[:3]
        palm_frame.wxyz = xyzw_to_wxyz(palm_xyz_xyzw[3:7])

        # By default MOVE_FLOATING_SHARPA_HAND = False so we can see how the object is moving wrt a fixed sharpa hand
        # Can set to True to debug and make sure that everything aligns
        MOVE_FLOATING_SHARPA_HAND = False
        if MOVE_FLOATING_SHARPA_HAND:
            sharpa_frame.position = palm_xyz_xyzw[:3]
            sharpa_frame.wxyz = xyzw_to_wxyz(palm_xyz_xyzw[3:7])
        else:
            # Keep floating sharpa hand in a fixed position
            sharpa_frame.position = recorded_data.robot_root_states_array[
                0, :3
            ] + np.array([-0.5, -0.8, 0.7])
            # sharpa_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            sharpa_frame.wxyz = xyzw_to_wxyz(R.from_euler("z", -np.pi / 2).as_quat())

        # Object relative to floating sharpa hand
        T_W_O = RecordedData.pose_to_T(object_root_state[:7])
        T_P_W = np.linalg.inv(T_W_P)
        T_P_O = T_P_W @ T_W_O
        object_xyz_xyzw_P = RecordedData.T_to_pose(T_P_O)
        object_in_sharpa_frame.position = object_xyz_xyzw_P[:3]
        object_in_sharpa_frame.wxyz = xyzw_to_wxyz(object_xyz_xyzw_P[3:7])

        # ###########
        # Sleep and update frame index
        # ###########
        end_loop_time = time.time()
        loop_time = end_loop_time - start_loop_time
        sleep_time = recorded_data.dt - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            warn(
                f"Loop time {loop_time:.6f}s is greater than recorded data dt {recorded_data.dt:.6f}s, not keeping up with recorded data (desired FPS: {1.0 / recorded_data.dt:.1f}, actual FPS: {1.0 / loop_time:.1f})"
            )
        if not PAUSED:
            frame_idx_slider.value = int(
                np.clip(
                    frame_idx_slider.value + 1, a_min=0, a_max=len(recorded_data) - 1
                )
            )


if __name__ == "__main__":
    main()
