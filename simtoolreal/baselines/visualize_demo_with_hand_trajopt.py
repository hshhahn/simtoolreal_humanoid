import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pytorch_kinematics as pk
import torch
import tyro
import viser
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_axis_angle
from scipy.spatial.transform import Rotation as R
from termcolor import colored
from tqdm import tqdm
from viser.extras import ViserUrdf


def get_repo_root_dir() -> Path:
    return Path(__file__).parent.parent


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

AXES_LENGTH = 0.0
AXES_RADIUS = 0.0


def info(message: str) -> None:
    print(colored(message, "green"))


def warn(message: str) -> None:
    print(colored(message, "yellow"))


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z])


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([x, y, z, w])


def pose_to_T(pose: np.ndarray) -> np.ndarray:
    assert pose.shape == (7,), f"Expected pose to be (7,), got {pose.shape}"
    xyz = pose[:3]
    xyzw = pose[3:7]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = xyz
    return T


def T_to_pose(T: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), f"T.shape: {T.shape}"
    pose = np.zeros(7)
    pose[:3] = T[:3, 3]
    pose[3:7] = R.from_matrix(T[:3, :3]).as_quat()
    return pose


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize 1D vector
    """
    assert v.ndim == 1, f"v.shape: {v.shape}"
    norm = np.linalg.norm(v)
    assert norm > 0, f"norm: {norm}"
    return v / norm


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform point by transform T
    """
    assert point.shape == (3,)
    assert T.shape == (4, 4)
    point = np.concatenate([point, [1]])
    transformed_point = T @ point
    return transformed_point[:3]


def create_urdf(
    obj_filepath: Path,
    mass: float = 0.066,
    ixx: float = 1e-3,
    iyy: float = 1e-3,
    izz: float = 1e-3,
    color: Optional[Literal["white"]] = None,
) -> Path:
    """
    Create URDF file for new object from path to object mesh
    """
    if color == "white":
        color_material = (
            """<material name="white"> <color rgba="1. 1. 1. 1."/> </material>"""
        )
    elif color is None:
        color_material = ""
    else:
        raise ValueError(f"Invalid color {color}")

    assert obj_filepath.suffix == ".obj"
    urdf_filepath = obj_filepath.with_suffix(".urdf")
    urdf_text = f"""<?xml version="1.0" ?>
        <robot name="model.urdf">
        <link name="baseLink">
            <contact>
                <lateral_friction value="0.8"/>
                <rolling_friction value="0.001"/>g
                <contact_cfm value="0.0"/>
                <contact_erp value="1.0"/>
            </contact>
            <inertial>
                <mass value="{mass}"/>
                <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
            </inertial>
            <visual>
            <geometry>
                <mesh filename="{obj_filepath.name}" scale="1 1 1"/>
            </geometry>
            {color_material}
            </visual>
            <collision>
            <geometry>
                <mesh filename="{obj_filepath.name}" scale="1 1 1"/>
            </geometry>
            </collision>
        </link>
        </robot>"""
    with urdf_filepath.open("w") as f:
        f.write(urdf_text)
    return urdf_filepath


def filter_poses(poses: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    """
    Smooths a trajectory of homogeneous transformation matrices using a Gaussian kernel.
    Handles the proper interpolation of rotations via quaternions.

    Args:
        poses: (N, 4, 4) numpy array of homogeneous matrices.
        sigma: Standard deviation for Gaussian kernel. Start with 2.0.

    Returns:
        (N, 4, 4) numpy array of smoothed matrices.
    """
    N = poses.shape[0]
    if N < 2:
        return poses.copy()

    # 1. Decompose Matrix -> Translation + Quaternion
    translations = poses[:, :3, 3]
    rot_matrices = poses[:, :3, :3]
    quats = R.from_matrix(rot_matrices).as_quat()  # Returns (N, 4)

    # 2. Fix Quaternion Discontinuities (Sign Flips)
    # q and -q represent the same rotation. To smooth properly, we must
    # ensure consecutive quaternions lie on the same "hemisphere".
    fixed_quats = quats.copy()
    for i in range(1, N):
        # If dot product is negative, the vectors point in opposite directions
        if np.sum(fixed_quats[i] * fixed_quats[i - 1]) < 0:
            fixed_quats[i] = -fixed_quats[i]

    # 3. Apply Gaussian Smoothing
    smoothed_trans = gaussian_filter1d(translations, sigma=sigma, axis=0)
    smoothed_quats_raw = gaussian_filter1d(fixed_quats, sigma=sigma, axis=0)

    # 4. Re-normalize Quaternions
    # Averaging shrinks magnitude; restore to unit length to be valid rotations
    norms = np.linalg.norm(smoothed_quats_raw, axis=1, keepdims=True)
    smoothed_quats = smoothed_quats_raw / (norms + 1e-8)

    # 5. Reconstruct (N, 4, 4) Matrices
    smoothed_rots = R.from_quat(smoothed_quats).as_matrix()

    smoothed_poses = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    smoothed_poses[:, :3, :3] = smoothed_rots
    smoothed_poses[:, :3, 3] = smoothed_trans

    return smoothed_poses


def filter_positions(positions: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    """
    Smooths a trajectory of 3D positions using a Gaussian kernel.

    Args:
        positions: (N, 3) numpy array of XYZ coordinates.
        sigma: Standard deviation for Gaussian kernel. Higher = smoother.

    Returns:
        (N, 3) numpy array of smoothed positions.
    """
    # Safety check for short trajectories
    if positions.shape[0] < 2:
        return positions.copy()

    return gaussian_filter1d(positions, sigma=sigma, axis=0)


def filter_keypoint_to_xyzs(
    keypoint_to_xyzs: List[Dict], sigma: float = 2.0
) -> List[Dict]:
    # 1. Identify all keypoint names (e.g., 'index_tip', 'thumb_tip', etc.)
    keypoints = [
        "wrist_back",
        "wrist_front",
        "index_0_back",
        "index_0_front",
        "middle_0_back",
        "middle_0_front",
        "ring_0_back",
        "ring_0_front",
        "index_3",
        "middle_3",
        "ring_3",
        "thumb_3",
        "pinky_3",
        "PALM_TARGET",
    ]
    filtered_trajectories = {}

    # 2. Extract and filter each keypoint's history independently
    for keypoint in keypoints:
        # Stack into (N, 3) array
        raw_traj = np.array([frame[keypoint] for frame in keypoint_to_xyzs])

        # Apply the filter
        filtered_trajectories[keypoint] = filter_positions(raw_traj, sigma=sigma)

    # 3. Repack back into List[Dict] structure for the loop
    N_TIMESTEPS = len(keypoint_to_xyzs)
    keypoint_to_xyzs = [
        {keypoint: filtered_trajectories[keypoint][i] for keypoint in keypoints}
        for i in range(N_TIMESTEPS)
    ]
    return keypoint_to_xyzs


def control_ik(
    j_eef: torch.Tensor,
    dpose: torch.Tensor,
    damping: float = 0.1,
    pos_only: bool = False,
) -> torch.Tensor:
    # -------------------------------------------------------------------------
    # Damped Least Squares IK (task-space)
    #
    # Unweighted formulation:
    #
    #   Solve:
    #       min_{dq}  || J dq - e ||^2 + λ ||dq||^2
    #
    #   Closed-form solution:
    #       dq = Jᵀ ( J Jᵀ + λ I )⁻¹ e
    # -------------------------------------------------------------------------

    if pos_only:
        j_eef = j_eef[:, 0:3]
        dpose = dpose[:, 0:3]

    # solve damped least squares

    # Set controller parameters
    NUM_ENVS, NUM_EE_DIMS, NUM_JOINTS = j_eef.shape
    assert dpose.shape == (NUM_ENVS, NUM_EE_DIMS), f"dpose.shape: {dpose.shape}"

    j_eef_T = torch.transpose(j_eef, 1, 2)
    assert j_eef_T.shape == (
        NUM_ENVS,
        NUM_JOINTS,
        NUM_EE_DIMS,
    ), f"j_eef_T.shape: {j_eef_T.shape}"

    lmbda = torch.eye(NUM_EE_DIMS, device=j_eef.device) * (damping**2)
    assert lmbda.shape == (NUM_EE_DIMS, NUM_EE_DIMS), f"lmbda.shape: {lmbda.shape}"

    # u = J.T @ (J @ J.T + lambda)^-1 @ dpose
    u = torch.bmm(
        j_eef_T,
        torch.bmm(
            torch.inverse(torch.bmm(j_eef, j_eef_T) + lmbda[None]),
            dpose.unsqueeze(dim=-1),
        ),
    ).squeeze(dim=-1)
    assert u.shape == (NUM_ENVS, NUM_JOINTS), f"u.shape: {u.shape}"

    return u


def control_ik_weighted(
    j_eef: torch.Tensor,
    dpose: torch.Tensor,
    damping: float = 0.1,
    pos_only: bool = False,
    w_pos: float = 1.0,
    w_rot: float = 0.1,  # <--- smaller => care less about rotation
) -> torch.Tensor:
    # Example: 0.02 position error is 2cm, 0.2 radians is 11 degrees
    # Should rescale so that rotation is cared about less

    # -------------------------------------------------------------------------
    # Damped Least Squares IK (task-space)
    #
    # Weighted formulation (to prioritize position over orientation):
    #
    #   Let W = diag([w₁, ..., w₆]) be task-space weights
    #
    #   Solve:
    #       min_{dq}  || W (J dq - e) ||^2 + λ ||dq||^2
    #
    #   Equivalent to defining:
    #       J_w = W J
    #       e_w = W e
    #
    #   Then:
    #       dq = J_wᵀ ( J_w J_wᵀ + λ I )⁻¹ e_w
    #
    #   (implemented below by scaling the rows of J and components of e)
    # -------------------------------------------------------------------------
    if pos_only:
        j_eef = j_eef[:, 0:3]
        dpose = dpose[:, 0:3]
        w = torch.tensor([w_pos, w_pos, w_pos], device=j_eef.device, dtype=j_eef.dtype)
    else:
        w = torch.tensor(
            [w_pos, w_pos, w_pos, w_rot, w_rot, w_rot],
            device=j_eef.device,
            dtype=j_eef.dtype,
        )

    # Apply task-space weights (row scaling)
    W = w[None, :, None]  # (B, ee, 1)
    j_w = j_eef * W  # scale rows of J
    e_w = dpose * w[None, :]  # scale components of error

    B, EE, NJ = j_w.shape
    j_w_T = j_w.transpose(1, 2)

    lmbda = torch.eye(EE, device=j_w.device, dtype=j_w.dtype) * (damping**2)

    u = torch.bmm(
        j_w_T,
        torch.bmm(
            torch.inverse(torch.bmm(j_w, j_w_T) + lmbda[None]),
            e_w.unsqueeze(-1),
        ),
    ).squeeze(-1)

    return u


def compute_current_T_R_P(
    arm_pk_chain: pk.SerialChain, q_arm: np.ndarray
) -> np.ndarray:
    NUM_ARM_JOINTS = 7
    assert q_arm.shape == (NUM_ARM_JOINTS,), f"q_arm.shape: {q_arm.shape}"

    device = arm_pk_chain.device
    q_arm_torch = torch.from_numpy(q_arm).float().to(device)
    wrist_pose = arm_pk_chain.forward_kinematics(q_arm_torch).get_matrix()[0]
    assert wrist_pose.shape == (4, 4), f"wrist_pose.shape: {wrist_pose.shape}"
    return wrist_pose.cpu().numpy()


def compute_T_R_P(hand_keypoint_to_xyz: dict) -> np.ndarray:
    T_W_P = np.eye(4)
    T_W_P[:3, 3] = hand_keypoint_to_xyz["PALM_TARGET"]
    r_R_P = compute_r_R_P(keypoint_to_xyz=hand_keypoint_to_xyz)
    r_W_R = T_W_R[:3, :3]
    r_W_P = r_W_R @ r_R_P
    T_W_P[:3, :3] = r_W_P
    T_R_W = np.linalg.inv(T_W_R)
    T_R_P = T_R_W @ T_W_P
    return T_R_P


def compute_new_q_arm(
    arm_pk_chain: pk.SerialChain, target_wrist_pose: np.ndarray, q_arm: np.ndarray
) -> np.ndarray:
    NUM_ARM_JOINTS = 7
    assert target_wrist_pose.shape == (4, 4), (
        f"target_wrist_pose.shape: {target_wrist_pose.shape}"
    )
    assert q_arm.shape == (NUM_ARM_JOINTS,), f"q_arm.shape: {q_arm.shape}"

    device = arm_pk_chain.device
    q_arm_torch = torch.from_numpy(q_arm).float().to(device)
    target_wrist_pose_torch = torch.from_numpy(target_wrist_pose).float().to(device)

    # Compute current wrist pose
    wrist_pose = arm_pk_chain.forward_kinematics(q_arm_torch).get_matrix()[0]
    assert wrist_pose.shape == (
        4,
        4,
    ), f"wrist_pose.shape: {wrist_pose.shape}"

    # Tunable Parameters
    MAX_POS_ERROR = 0.02  # Max 2cm step per iteration
    MAX_ROT_ERROR = np.deg2rad(5.7)  # Max ~5.7 degrees step per iteration

    # 1. Compute Raw Errors
    wrist_pos_error = (target_wrist_pose_torch[:3, 3] - wrist_pose[:3, 3])[None]

    # Orientation error (Axis-Angle)
    wrist_rot_matrix_error = (target_wrist_pose_torch[:3, :3] @ wrist_pose[:3, :3].T)[
        None
    ]
    wrist_rot_error = matrix_to_axis_angle(wrist_rot_matrix_error)

    # 2. Rescale Position Error (Preserve Direction)
    pos_norm = torch.linalg.norm(wrist_pos_error, dim=-1, keepdim=True)
    # Calculate scale factor: if norm < max, factor is 1.0. If norm > max, factor < 1.0
    pos_scale = torch.clamp(MAX_POS_ERROR / (pos_norm + 1e-6), max=1.0)
    wrist_pos_error_scaled = wrist_pos_error * pos_scale

    # 3. Rescale Rotation Error (Preserve Axis, Reduce Angle)
    rot_norm = torch.linalg.norm(wrist_rot_error, dim=-1, keepdim=True)
    rot_scale = torch.clamp(MAX_ROT_ERROR / (rot_norm + 1e-6), max=1.0)
    wrist_rot_error_scaled = wrist_rot_error * rot_scale

    # 4. Concatenate and Solve
    wrist_error = torch.cat([wrist_pos_error_scaled, wrist_rot_error_scaled], dim=-1)

    # Compute jacobian
    jacobian = arm_pk_chain.jacobian(q_arm_torch[None])
    NUM_XYZRPY = 6
    assert jacobian.shape == (
        1,
        NUM_XYZRPY,
        NUM_ARM_JOINTS,
    ), f"jacobian.shape: {jacobian.shape}"

    # Compute delta arm joint position
    # delta_q_arm = control_ik(
    #     j_eef=jacobian,
    #     dpose=wrist_error,
    #     pos_only=False,
    # )
    delta_q_arm = control_ik_weighted(
        j_eef=jacobian,
        dpose=wrist_error,
        pos_only=False,
    )

    # Clamp delta arm joint position to avoid sudden movements
    MAX_DELTA_Q_ARM = np.deg2rad(2)
    new_q_arm = q_arm_torch + delta_q_arm.clamp(
        min=-MAX_DELTA_Q_ARM, max=MAX_DELTA_Q_ARM
    )
    assert new_q_arm.shape == (
        1,
        NUM_ARM_JOINTS,
    ), f"new_q_arm.shape: {new_q_arm.shape}"

    # Clamp joint position to joint limits with buffer
    BUFFER = np.deg2rad(7.5)
    lower_limits = arm_pk_chain.low.cpu().numpy() + BUFFER
    upper_limits = arm_pk_chain.high.cpu().numpy() - BUFFER
    new_q_arm = np.clip(new_q_arm, lower_limits[None], upper_limits[None])

    return new_q_arm.cpu().numpy()[0]


def set_robot_state(robot, q: np.ndarray) -> None:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    actuatable_joint_idxs = get_actuatable_joint_idxs(robot)
    num_actuatable_joints = len(actuatable_joint_idxs)

    assert len(q.shape) == 1, f"q.shape: {q.shape}"
    assert q.shape[0] <= num_actuatable_joints, (
        f"q.shape: {q.shape}, num_actuatable_joints: {num_actuatable_joints}"
    )

    for i, joint_idx in enumerate(actuatable_joint_idxs):
        # q may not contain all the actuatable joints, so we assume that the joints not in q are all 0
        if i < len(q):
            pb.resetJointState(robot, joint_idx, q[i])
        else:
            pb.resetJointState(robot, joint_idx, 0)


def get_link_name_to_idx(robot: int) -> dict:
    """
    Get link name to index mapping for robot
    """
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    link_name_to_idx = {}
    for i in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, i)
        link_name_to_idx[joint_info[12].decode("utf-8")] = i
    return link_name_to_idx


def get_actuatable_joint_idxs(robot) -> List[int]:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    num_total_joints = pb.getNumJoints(robot)
    actuatable_joint_idxs = [
        i
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]
    return actuatable_joint_idxs


def get_joint_names(
    robot: int,
) -> List[str]:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    joint_names = []
    for i in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, i)
        if joint_info[2] == pb.JOINT_FIXED:
            continue
        joint_names.append(joint_info[1])
    return joint_names


def solve_fingertip_ik(
    hand_pb: int,
    hand_keypoint_to_xyz: dict,
    target_wrist_pose: np.ndarray,
) -> np.ndarray:
    # HACK: Hide pybullet import in functions that use it to avoid
    import pybullet as pb

    target_wrist_pos = target_wrist_pose[:3, 3]
    target_wrist_quat = R.from_matrix(target_wrist_pose[:3, :3]).as_quat()
    reset_base_pose_pb(
        hand_pb, target_pos=target_wrist_pos, target_orn=target_wrist_quat
    )

    hand_link_name_to_id = get_link_name_to_idx(hand_pb)
    hand_joint_names = get_joint_names(hand_pb)

    # HACK: For numerous reasons, it was hard to do the hand IK while constraining the arm joints
    # Thus, we simply load the hand robot and use it for the hand IK
    # This code should change per hand
    target_q = []

    thumb_result = np.array(
        pb.calculateInverseKinematics(
            hand_pb,
            hand_link_name_to_id["left_thumb_fingertip"],
            hand_keypoint_to_xyz["thumb_3"],
            maxNumIterations=2000,
            residualThreshold=0.001,
        )
    )
    index_result = np.array(
        pb.calculateInverseKinematics(
            hand_pb,
            hand_link_name_to_id["left_index_fingertip"],
            hand_keypoint_to_xyz["index_3"],
            maxNumIterations=2000,
            residualThreshold=0.001,
        )
    )
    middle_result = np.array(
        pb.calculateInverseKinematics(
            hand_pb,
            hand_link_name_to_id["left_middle_fingertip"],
            hand_keypoint_to_xyz["middle_3"],
            maxNumIterations=2000,
            residualThreshold=0.001,
        )
    )
    ring_result = np.array(
        pb.calculateInverseKinematics(
            hand_pb,
            hand_link_name_to_id["left_ring_fingertip"],
            hand_keypoint_to_xyz["ring_3"],
            maxNumIterations=2000,
            residualThreshold=0.001,
        )
    )
    pinky_result = np.array(
        pb.calculateInverseKinematics(
            hand_pb,
            hand_link_name_to_id["left_pinky_fingertip"],
            hand_keypoint_to_xyz["pinky_3"],
            maxNumIterations=2000,
            residualThreshold=0.001,
        )
    )
    target_q = np.concatenate(
        [
            thumb_result[:5],
            index_result[5:9],
            middle_result[9:13],
            ring_result[13:17],
            pinky_result[17:],
        ]
    )
    set_robot_state(hand_pb, target_q)

    assert target_q.shape == (22,), f"target_q.shape: {target_q.shape}"
    return target_q


def reset_base_pose_pb(body_id, target_pos, target_orn=(0, 0, 0, 1)):
    """
    Moves the body so that its base link origin is at target_pos/target_orn,
    compensating for the inertial offset.
    """
    import pybullet as pb

    # 1. Get the offset of the CoM relative to the URDF link origin
    # dynamics_info[3] is localInertialFramePosition
    # dynamics_info[4] is localInertialFrameOrientation
    dynamics_info = pb.getDynamicsInfo(body_id, -1)
    inertial_pos = dynamics_info[3]
    inertial_orn = dynamics_info[4]

    # 2. Calculate the specific World CoM pose that results in the desired visual pose
    # Formula: Target_CoM = Target_Visual * Local_Inertial_Offset
    final_pos, final_orn = pb.multiplyTransforms(
        target_pos,
        target_orn,  # Where we want the visual origin to be
        inertial_pos,
        inertial_orn,  # The offset from visual to CoM
    )

    # 3. Apply the reset
    pb.resetBasePositionAndOrientation(body_id, final_pos, final_orn)


@dataclass
class Args:
    object_path: Path
    object_poses_json_path: Path
    hand_poses_dir: Path
    visualize_hand_meshes: bool = False
    retarget_robot: bool = False
    retarget_robot_using_object_relative_pose: bool = False
    save_retargeted_robot_to_file: bool = False
    output_retargeted_robot_path: Path = Path("retargeted_robot") / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".npz"
    )
    dt: float = 1.0 / 30
    start_idx: int = 0

    def __post_init__(self) -> None:
        if self.retarget_robot_using_object_relative_pose:
            assert self.retarget_robot, (
                "retarget_robot_using_object_relative_pose requires retarget_robot to be True"
            )


def set_keypoint_sphere_positions(
    hand_keypoint_to_xyz: dict, server: viser.ViserServer
) -> None:
    from baselines.colors import (
        BLACK_RGBA,
        BLUE_RGBA,
        BLUE_TRANSLUCENT_RGBA,
        CYAN_RGBA,
        GREEN_RGBA,
        GREEN_TRANSLUCENT_RGBA,
        MAGENTA_RGBA,
        RED_RGBA,
        RED_TRANSLUCENT_RGBA,
        YELLOW_RGBA,
        YELLOW_TRANSLUCENT_RGBA,
    )

    keypoint_to_rgba = {
        "wrist_back": RED_TRANSLUCENT_RGBA,
        "wrist_front": RED_RGBA,
        "index_0_back": GREEN_TRANSLUCENT_RGBA,
        "index_0_front": GREEN_RGBA,
        "middle_0_back": BLUE_TRANSLUCENT_RGBA,
        "middle_0_front": BLUE_RGBA,
        "ring_0_back": YELLOW_TRANSLUCENT_RGBA,
        "ring_0_front": YELLOW_RGBA,
        "index_3": GREEN_RGBA,
        "middle_3": BLUE_RGBA,
        "ring_3": YELLOW_RGBA,
        "thumb_3": MAGENTA_RGBA,
        "pinky_3": BLACK_RGBA,
        "PALM_TARGET": CYAN_RGBA,
    }
    keypoints = keypoint_to_rgba.keys()

    if not hasattr(set_keypoint_sphere_positions, "spheres"):
        set_keypoint_sphere_positions.spheres = [
            server.scene.add_icosphere(
                f"/hand/keypoint_{keypoint}",
                radius=0.01,
                color=keypoint_to_rgba[keypoint][:3],
                position=hand_keypoint_to_xyz[keypoint],
                opacity=keypoint_to_rgba[keypoint][3],
            )
            for keypoint in keypoints
        ]
    else:
        for keypoint, sphere in zip(keypoints, set_keypoint_sphere_positions.spheres):
            sphere.position = hand_keypoint_to_xyz[keypoint]


def create_transformed_keypoint_to_xyz(hand_json: dict, T_W_C: np.ndarray) -> dict:
    keypoint_to_xyz = hand_json

    keypoints = [
        "wrist_back",
        "wrist_front",
        "index_0_back",
        "index_0_front",
        "middle_0_back",
        "middle_0_front",
        "ring_0_back",
        "ring_0_front",
        "index_3",
        "middle_3",
        "ring_3",
        "thumb_3",
        "pinky_3",
    ]
    for keypoint in keypoints:
        assert keypoint in keypoint_to_xyz, (
            f"{keypoint} not in {keypoint_to_xyz.keys()}"
        )
        keypoint_to_xyz[keypoint] = np.array(keypoint_to_xyz[keypoint])

    # Shorthand for next computations
    kpt_map = keypoint_to_xyz

    # Palm target
    mean_middle_0 = np.mean(
        [
            kpt_map["middle_0_back"],
            kpt_map["middle_0_front"],
        ],
        axis=0,
    )
    mean_wrist = np.mean(
        [
            kpt_map["wrist_back"],
            kpt_map["wrist_front"],
        ],
        axis=0,
    )
    palm_normal = normalize(
        np.cross(
            normalize(kpt_map["index_0_front"] - kpt_map["ring_0_front"]),
            normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]),
        )
    )
    kpt_map["PALM_TARGET"] = (
        # VERSION 0
        mean_wrist
        # # VERSION 1
        # mean_middle_0
        # - normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]) * 0.03
        # - palm_normal * 0.03
        #
        # VERSION 2
        # - palm_normal * 0.03 * np.sqrt(2)
        #
        # VERSION 3
        # - normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]) * 0.03 * np.sqrt(2)
    )

    transformed_keypoint_to_xyz = {
        keypoint: transform_point(T=T_W_C, point=kpt_map[keypoint])
        for keypoint in keypoints + ["PALM_TARGET"]
    }

    # HACK: add global_orient
    transformed_keypoint_to_xyz["global_orient"] = kpt_map["global_orient"]
    return transformed_keypoint_to_xyz


def compute_r_R_P(keypoint_to_xyz: dict) -> np.ndarray:
    # Z = palm to middle finger
    # Y = thumb to palm
    # X = palm normal
    kpt_map = keypoint_to_xyz
    palm_to_middle_finger = normalize(
        kpt_map["middle_0_front"] - kpt_map["wrist_front"]
    )
    thumb_to_palm = normalize(kpt_map["ring_0_front"] - kpt_map["index_0_front"])
    _palm_normal = normalize(np.cross(thumb_to_palm, palm_to_middle_finger))

    Z = palm_to_middle_finger
    Y_not_orthogonal = thumb_to_palm
    Y = normalize(Y_not_orthogonal - np.dot(Y_not_orthogonal, Z) * Z)
    X = normalize(
        np.cross(
            Y,
            Z,
        )
    )
    r_R_P = np.stack(
        [X, Y, Z],
        axis=1,
    )
    return r_R_P


def save_to_file(
    file_path: Path, q_array: np.ndarray, object_pose_array: np.ndarray, dt: float
):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    info(f"Saving to file: {file_path}")

    T = len(q_array)
    assert q_array.shape == (T, 29), (
        f"q_array.shape: {q_array.shape}, expected: (T, 29)"
    )

    robot_root_states_array = np.zeros((T, 13))
    robot_root_states_array[:, 1] = 0.8
    robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
    object_root_states_array = np.zeros((T, 13))
    object_root_states_array[:, :7] = object_pose_array
    table_root_states_array = np.zeros((T, 13))
    table_root_states_array[:, :3] = np.array([0.0, 0.0, 0.38])[None]
    goal_root_states_array = np.zeros((T, 13))
    goal_root_states_array[:, :7] = object_pose_array

    robot_joint_positions = q_array
    robot_joint_velocities = np.zeros((T, 29))
    robot_joint_pos_targets = q_array
    time_array = np.linspace(0, T * dt, T)

    assert robot_joint_positions.shape == (T, 29), (
        f"robot_joint_positions.shape: {robot_joint_positions.shape}, expected: (T, 29)"
    )
    assert robot_joint_velocities.shape == (T, 29), (
        f"robot_joint_velocities.shape: {robot_joint_velocities.shape}, expected: (T, 29)"
    )
    assert robot_joint_pos_targets.shape == (T, 29), (
        f"robot_joint_pos_targets.shape: {robot_joint_pos_targets.shape}, expected: (T, 29)"
    )
    assert object_root_states_array.shape == (T, 13), (
        f"object_root_states_array.shape: {object_root_states_array.shape}, expected: (T, 13)"
    )
    assert time_array.shape == (T,), (
        f"time_array.shape: {time_array.shape}, expected: (T,)"
    )

    JOINT_NAMES = [
        "iiwa14_joint_1",
        "iiwa14_joint_2",
        "iiwa14_joint_3",
        "iiwa14_joint_4",
        "iiwa14_joint_5",
        "iiwa14_joint_6",
        "iiwa14_joint_7",
        "left_1_thumb_CMC_FE",
        "left_thumb_CMC_AA",
        "left_thumb_MCP_FE",
        "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_2_index_MCP_FE",
        "left_index_MCP_AA",
        "left_index_PIP",
        "left_index_DIP",
        "left_3_middle_MCP_FE",
        "left_middle_MCP_AA",
        "left_middle_PIP",
        "left_middle_DIP",
        "left_4_ring_MCP_FE",
        "left_ring_MCP_AA",
        "left_ring_PIP",
        "left_ring_DIP",
        "left_5_pinky_CMC",
        "left_pinky_MCP_FE",
        "left_pinky_MCP_AA",
        "left_pinky_PIP",
        "left_pinky_DIP",
    ]

    from recorded_data import RecordedData

    recorded_data = RecordedData(
        robot_root_states_array=robot_root_states_array,
        object_root_states_array=object_root_states_array,
        robot_joint_positions_array=robot_joint_positions,
        time_array=time_array,
        robot_joint_names=JOINT_NAMES,
        robot_joint_velocities_array=robot_joint_velocities,
        robot_joint_pos_targets_array=robot_joint_pos_targets,
        goal_root_states_array=goal_root_states_array,
    )
    recorded_data.to_file(file_path)


def solve_trajopt(
    T_R_Ps: np.ndarray,
    q_start: np.ndarray,
    dt: float,
    waypoint_buffer: int = 0,
    use_collision_avoidance: bool = True,
) -> np.ndarray:
    import pyroki as pk

    from baselines import pyroki_snippets as pks
    from baselines.trajopt_improved_waypoints_sharpa import xyzw_to_wxyz

    N_TIMESTEPS = T_R_Ps.shape[0]
    assert T_R_Ps.shape == (N_TIMESTEPS, 4, 4), (
        f"T_R_Ps.shape: {T_R_Ps.shape}, expected: (N_TIMESTEPS, 4, 4)"
    )
    assert q_start.shape == (29,), f"q_start.shape: {q_start.shape}, expected: (29,)"

    import yourdfpy

    # Load robot
    urdf = yourdfpy.URDF.load(
        "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    target_link_name = "left_hand_C_MC"
    sphere_json_path = (
        get_repo_root_dir() / "baselines" / "assets" / "sharpa_spheres.json"
    )
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=q_start)

    # Load collision spheres
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    if not use_collision_avoidance:
        # Needs to have >0 spheres, so just include the base link
        sphere_decomposition = {"iiwa14_link_0": sphere_decomposition["iiwa14_link_0"]}
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Define Obstacles
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )

    table_size = np.array([0.475, 0.4, 0.3])
    table_center = np.array([0.0, -0.8, 0.38])

    table_coll = pk.collision.Box.from_extent(
        extent=table_size,
        position=table_center,
    )
    whiteboard_size = np.array([0.02, 0.4, 0.4])
    whiteboard_center = np.array([0.325, -0.8, 0.35])
    whiteboard_coll = pk.collision.Box.from_extent(
        extent=whiteboard_size,
        position=whiteboard_center,
    )
    world_coll = [ground_coll, table_coll, whiteboard_coll]

    # Create waypoints
    WAYPOINT_BUFFER = waypoint_buffer
    waypoints = {}
    for i in range(N_TIMESTEPS):
        pos = T_R_Ps[i, :3, 3]
        quat_xyzw = R.from_matrix(T_R_Ps[i, :3, :3]).as_quat()
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        waypoints[WAYPOINT_BUFFER + i] = (pos, quat_wxyz)

    # Solve trajectory
    start_time = time.time()

    N_TIMESTEPS_WITH_BUFFER = N_TIMESTEPS + WAYPOINT_BUFFER
    info(
        f"Solving trajectory with {N_TIMESTEPS_WITH_BUFFER} waypoints, each {dt} seconds apart, for a total time of {N_TIMESTEPS_WITH_BUFFER * dt} seconds"
    )
    traj = pks.solve_waypoint_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        q_start,
        waypoints,
        N_TIMESTEPS_WITH_BUFFER,
        dt,
    )

    traj = np.array(traj)
    assert len(traj) == N_TIMESTEPS_WITH_BUFFER, (
        f"len(traj): {len(traj)}, expected: {N_TIMESTEPS_WITH_BUFFER}"
    )
    info(f"Solved in {time.time() - start_time:.4f}s")

    # Clamp joint position to joint limits with buffer
    BUFFER = np.deg2rad(7.5)
    J_arm = 7
    arm_lower_limits = np.array(robot.joints.lower_limits[:J_arm]) + BUFFER
    arm_upper_limits = np.array(robot.joints.upper_limits[:J_arm]) - BUFFER
    traj[:, :J_arm] = np.clip(
        traj[:, :J_arm], arm_lower_limits[None], arm_upper_limits[None]
    )

    return traj


def interpolate_traj(traj: np.ndarray, n_steps: int) -> np.ndarray:
    from scipy.interpolate import interp1d

    """
    Upsamples a trajectory by a factor of n_steps using linear interpolation.

    Args:
        traj: Numpy array of shape (T, J) where T is time steps and J is joint dims.
        n_steps: The scaling factor for interpolation (e.g., 10 means 10x more points).

    Returns:
        new_traj: Interpolated trajectory of shape (T * n_steps, J).
    """
    # 1. Get dimensions
    T = traj.shape[0]
    J = traj.shape[1]

    # Input assertion
    assert traj.shape == (T, J), (
        f"Input shape mismatch. Expected ({T}, {J}), got {traj.shape}"
    )

    # 2. Create the time indices for the original trajectory
    # We assume the original trajectory is spaced evenly from 0 to T-1
    old_time = np.arange(T)

    # 3. Create the time indices for the new trajectory
    # We want T * n_steps points, spanning the exact same timeframe (0 to T-1)
    new_T = T * n_steps
    new_time = np.linspace(0, T - 1, new_T)

    # 4. Interpolate
    # interp1d creates a function we can call with our new timestamps
    # axis=0 ensures we interpolate along the time axis, independent of J
    interpolator = interp1d(old_time, traj, kind="linear", axis=0)
    new_traj = interpolator(new_time)

    # Output assertion (Corrected to check Time dimension, not Joint dimension)
    assert new_traj.shape == (new_T, J), (
        f"Output shape mismatch. Expected ({new_T}, {J}), got {new_traj.shape}"
    )

    return new_traj


def main():
    args: Args = tyro.cli(Args)
    print("=" * 80)
    print(args)
    print("=" * 80)

    # Start visualizer
    SERVER = viser.ViserServer()

    @SERVER.on_client_connect
    def _(client):
        client.camera.position = T_W_C[:3, 3]
        client.camera.wxyz = xyzw_to_wxyz(R.from_matrix(T_W_C[:3, :3]).as_quat())

    # Load table
    TABLE_URDF_PATH = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
    assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

    _table_frame = SERVER.scene.add_frame(
        "/table",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
        position=(0, 0, 0.38),
        wxyz=(1, 0, 0, 0),
    )
    _table_viser = ViserUrdf(SERVER, TABLE_URDF_PATH, root_node_name="/table")

    # Load robot
    KUKA_SHARPA_URDF_PATH = (
        get_repo_root_dir()
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    assert KUKA_SHARPA_URDF_PATH.exists(), (
        f"KUKA_SHARPA_URDF_PATH not found: {KUKA_SHARPA_URDF_PATH}"
    )
    _kuka_sharpa_frame = SERVER.scene.add_frame(
        "/robot/state",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
        position=(0, 0.8, 0),
        wxyz=(1, 0, 0, 0),
    )
    kuka_sharpa_viser = ViserUrdf(
        SERVER, KUKA_SHARPA_URDF_PATH, root_node_name="/robot/state"
    )
    HOME_JOINT_POS_IIWA = np.array(
        [
            -1.571,
            1.571 - np.deg2rad(10),
            -0.000,
            1.376 + np.deg2rad(10),
            -0.000,
            1.485,
            1.308,
        ]
    )
    HOME_JOINT_POS_SHARPA = np.zeros(22)
    HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])
    kuka_sharpa_viser.update_cfg(HOME_JOINT_POS)

    SHARPA_URDF_PATH = (
        get_repo_root_dir()
        / "assets/urdf/left_sharpa_ha4/left_sharpa_ha4_v2_1_adjusted_restricted.urdf"
    )
    assert SHARPA_URDF_PATH.exists(), f"SHARPA_URDF_PATH not found: {SHARPA_URDF_PATH}"
    sharpa_frame = SERVER.scene.add_frame(
        "/sharpa",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
        position=(100, 0, 0),
        wxyz=(1, 0, 0, 0),
    )
    sharpa_viser = ViserUrdf(SERVER, SHARPA_URDF_PATH, root_node_name="/sharpa")
    sharpa_viser.update_cfg(HOME_JOINT_POS_SHARPA)

    # Load object poses
    assert args.object_poses_json_path.exists(), (
        f"Object poses json path {args.object_poses_json_path} does not exist"
    )
    with open(args.object_poses_json_path, "r") as f:
        object_poses_data = json.load(f)

    # Handle different jsons
    if "start_pose" in object_poses_data and "goals" in object_poses_data:
        info("Using start_pose and goals")
        T_W_O_start = pose_to_T(np.array(object_poses_data["start_pose"]))
        T_W_Os = np.array(
            [pose_to_T(np.array(pose)) for pose in object_poses_data["goals"]]
        )
    else:
        warn("Did not find start_pose and goals in object_poses_data")
        warn("Assuming data is just goals and are in robot frame")
        T_R_Os = np.array([pose_to_T(np.array(pose)) for pose in object_poses_data])
        T_W_Os = np.array([T_W_R @ T_R_O for T_R_O in T_R_Os])
        T_W_O_start = T_W_Os[0]

    T_W_Os = filter_poses(T_W_Os)

    # Load object
    assert args.object_path.exists(), f"Object path {args.object_path} does not exist"
    if args.object_path.suffix == ".obj":
        object_urdf_path = create_urdf(args.obj_path)
    elif args.object_path.suffix == ".urdf":
        object_urdf_path = args.object_path
    else:
        raise ValueError(f"Invalid object path: {args.object_path}")
    info(f"Loading object from {object_urdf_path}")
    object_frame_viser = SERVER.scene.add_frame(
        "/object",
        position=T_W_O_start[:3, 3],
        wxyz=R.from_matrix(T_W_O_start[:3, :3]).as_quat(),
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
    )
    _object_viser = ViserUrdf(
        SERVER, object_urdf_path, root_node_name=object_frame_viser.name
    )

    # Load hand poses
    assert args.hand_poses_dir.exists(), (
        f"Hand poses dir {args.hand_poses_dir} does not exist"
    )

    hand_json_files = sorted(list(args.hand_poses_dir.glob("*.json")))
    assert len(hand_json_files) > 0, f"No hand poses found in {args.hand_poses_dir}"
    hand_jsons = []
    for filename in tqdm(hand_json_files, desc="Loading hand poses"):
        with open(filename, "r") as f:
            hand_jsons.append(json.load(f))

    hand_keypoint_to_xyzs = [
        create_transformed_keypoint_to_xyz(hand_json, T_W_C) for hand_json in hand_jsons
    ]

    # Filter
    hand_keypoint_to_xyzs = filter_keypoint_to_xyzs(hand_keypoint_to_xyzs)

    FAR_AWAY_POSITION = np.ones(3) * 100
    # Load hand meshes
    hand_frames = []
    hand_visers = []
    if args.visualize_hand_meshes:
        # Each timestep has a different hand mesh because they can change shape
        # So this is slow to load
        hand_urdf_files = [
            create_urdf(hand_json_file.with_suffix(".obj"))
            for hand_json_file in hand_json_files
        ]

        SPAWN_HANDS_AT_FAR_AWAY_POSITION = False
        if SPAWN_HANDS_AT_FAR_AWAY_POSITION:
            hand_xyz, hand_quat_xyzw = FAR_AWAY_POSITION, [0, 0, 0, 1]
        else:
            hand_xyz, hand_quat_xyzw = (
                T_W_C[:3, 3],
                R.from_matrix(T_W_C[:3, :3]).as_quat(),
            )

        for i, hand_urdf_file in tqdm(
            enumerate(hand_urdf_files), desc="Loading hands", total=len(hand_urdf_files)
        ):
            hand_frame = SERVER.scene.add_frame(
                f"/hand/{i}",
                position=hand_xyz,
                wxyz=hand_quat_xyzw,
                show_axes=True,
                axes_length=AXES_LENGTH,
                axes_radius=AXES_RADIUS,
            )
            hand_viser = ViserUrdf(
                SERVER, hand_urdf_file, root_node_name=hand_frame.name
            )
            if not SPAWN_HANDS_AT_FAR_AWAY_POSITION:
                # Move hand to far away position after spawning
                hand_frame.position = FAR_AWAY_POSITION

            hand_frames.append(hand_frame)
            hand_visers.append(hand_viser)

    N_TIMESTEPS = min(len(T_W_Os), len(hand_keypoint_to_xyzs))
    msg = f"len(T_W_Os): {len(T_W_Os)}, len(hand_keypoint_to_xyzs): {len(hand_keypoint_to_xyzs)}, N_TIMESTEPS: {N_TIMESTEPS}"
    if np.abs(len(T_W_Os) - len(hand_keypoint_to_xyzs)) > 10:
        warn(msg)
    else:
        info(msg)
    T_W_Os = T_W_Os[:N_TIMESTEPS]
    hand_keypoint_to_xyzs = hand_keypoint_to_xyzs[:N_TIMESTEPS]
    T_R_Ps = filter_poses(
        np.array(
            [
                compute_T_R_P(hand_keypoint_to_xyz=hand_keypoint_to_xyz)
                for hand_keypoint_to_xyz in hand_keypoint_to_xyzs
            ]
        )
    )
    hand_frames = hand_frames[:N_TIMESTEPS]
    hand_visers = hand_visers[:N_TIMESTEPS]

    if args.retarget_robot:
        with open(KUKA_SHARPA_URDF_PATH, "rb") as f:
            urdf_str = f.read()
        DEVICE = "cpu"
        arm_pk_chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="left_hand_C_MC",
        ).to(device=DEVICE)
        import pybullet as pb

        SHARPA_URDF_PATH = (
            get_repo_root_dir()
            / "assets/urdf/left_sharpa_ha4/left_sharpa_ha4_v2_1_adjusted_restricted.urdf"
        )
        assert SHARPA_URDF_PATH.exists(), (
            f"SHARPA_URDF_PATH not found: {SHARPA_URDF_PATH}"
        )
        pb.connect(pb.DIRECT)
        # pb.connect(pb.GUI)
        hand_pb = pb.loadURDF(str(SHARPA_URDF_PATH))

        # Compute hand IKs and save them
        q_hands = []
        for i, (T_R_P, hand_keypoint_to_xyz) in tqdm(
            enumerate(zip(T_R_Ps, hand_keypoint_to_xyzs)),
            total=N_TIMESTEPS,
            desc="Computing hand IKs",
        ):
            T_W_P = T_W_R @ T_R_P
            new_q_hand = solve_fingertip_ik(
                hand_pb=hand_pb,
                hand_keypoint_to_xyz=hand_keypoint_to_xyz,
                target_wrist_pose=T_W_P,
            )
            q_hands.append(new_q_hand)

        # Run trajopt for the whole trajectory
        DOWNSAMPLE_FACTOR = 10
        retargeted_qs = solve_trajopt(
            T_R_Ps=T_R_Ps[::DOWNSAMPLE_FACTOR],
            q_start=HOME_JOINT_POS,
            dt=args.dt,
        )
        retargeted_qs = interpolate_traj(retargeted_qs, n_steps=DOWNSAMPLE_FACTOR)
        if retargeted_qs.shape[0] < N_TIMESTEPS:
            extra = N_TIMESTEPS - retargeted_qs.shape[0]
            retargeted_qs = np.concatenate(
                [retargeted_qs, retargeted_qs[-1][None].repeat(extra, axis=0)], axis=0
            )
        elif retargeted_qs.shape[0] > N_TIMESTEPS:
            extra = retargeted_qs.shape[0] - N_TIMESTEPS
            retargeted_qs = retargeted_qs[:-extra]
        assert retargeted_qs.shape == (N_TIMESTEPS, 29), (
            f"retargeted_qs.shape: {retargeted_qs.shape}, expected: (N_TIMESTEPS, 29)"
        )

        # Update hand joints positions
        retargeted_qs[:, 7:] = q_hands

    if args.retarget_robot_using_object_relative_pose:
        idx_lifted = None
        LIFTED_Z = 0.6
        for i, T_W_O in enumerate(T_W_Os):
            z = T_W_O[:3, 3][2]
            if z >= LIFTED_Z:
                idx_lifted = i
                break
        assert idx_lifted is not None, f"No object pose with z >= {LIFTED_Z} found"
        info(f"First object pose with z >= {LIFTED_Z} is at index {idx_lifted}")

        # Compute T_R_Ps using lifted object pose
        T_R_P_lifted = T_R_Ps[idx_lifted]
        T_W_P_lifted = T_W_R @ T_R_P_lifted
        T_W_O_lifted = T_W_Os[idx_lifted]
        T_O_P_lifted = np.linalg.inv(T_W_O_lifted) @ T_W_P_lifted

        T_W_Ps_using_lifted_object_pose = np.array(
            [T_W_O @ T_O_P_lifted for T_W_O in T_W_Os]
        )
        T_R_W = np.linalg.inv(T_W_R)
        T_R_Ps_using_lifted_object_pose = np.array(
            [T_R_W @ T_W_P for T_W_P in T_W_Ps_using_lifted_object_pose]
        )
        retargeted_q_lifted = retargeted_qs[idx_lifted]
        q_hand_using_lifted_pose = q_hands[idx_lifted]

        # Run trajopt for the trajectory after the object was lifted
        DOWNSAMPLE_FACTOR = 10
        retargeted_qs_using_lifted_pose = solve_trajopt(
            T_R_Ps=T_R_Ps_using_lifted_object_pose[idx_lifted:][::DOWNSAMPLE_FACTOR],
            q_start=retargeted_q_lifted,
            dt=args.dt,
        )
        retargeted_qs_using_lifted_pose = interpolate_traj(
            retargeted_qs_using_lifted_pose, n_steps=DOWNSAMPLE_FACTOR
        )
        if retargeted_qs_using_lifted_pose.shape[0] < N_TIMESTEPS - idx_lifted:
            extra = N_TIMESTEPS - idx_lifted - retargeted_qs_using_lifted_pose.shape[0]
            retargeted_qs_using_lifted_pose = np.concatenate(
                [
                    retargeted_qs_using_lifted_pose,
                    retargeted_qs_using_lifted_pose[-1][None].repeat(extra, axis=0),
                ],
                axis=0,
            )
        elif retargeted_qs_using_lifted_pose.shape[0] > N_TIMESTEPS - idx_lifted:
            extra = retargeted_qs_using_lifted_pose.shape[0] - (
                N_TIMESTEPS - idx_lifted
            )
            retargeted_qs_using_lifted_pose = retargeted_qs_using_lifted_pose[:-extra]

        # Connect it with the original trajectory before lifted
        # And fix hand joints positions after lifted
        retargeted_qs_using_lifted_pose = np.concatenate(
            [
                retargeted_qs_using_lifted_pose[:, :7],
                q_hand_using_lifted_pose[None].repeat(
                    len(retargeted_qs_using_lifted_pose), axis=0
                ),
            ],
            axis=1,
        )
        retargeted_qs_using_lifted_pose = np.concatenate(
            [retargeted_qs[:idx_lifted], retargeted_qs_using_lifted_pose], axis=0
        )
        assert retargeted_qs_using_lifted_pose.shape == (N_TIMESTEPS, 29), (
            f"retargeted_qs_using_lifted_pose.shape: {retargeted_qs_using_lifted_pose.shape}, expected: (N_TIMESTEPS, 29)"
        )

    # Reset robot to home position (once at start)
    kuka_sharpa_viser.update_cfg(HOME_JOINT_POS)
    sharpa_viser.update_cfg(HOME_JOINT_POS_SHARPA)

    # Solve arm IK first time
    if args.retarget_robot:
        N_IK_STEPS = 10
        for i in range(N_IK_STEPS):
            q = np.array(kuka_sharpa_viser._urdf.cfg)
            q_arm = q[:7]

            # Arm IK
            T_R_P = T_R_Ps[0]
            new_q_arm = compute_new_q_arm(
                arm_pk_chain=arm_pk_chain,
                target_wrist_pose=T_R_P,
                q_arm=q_arm,
            )

            q_hand = q[7:]
            new_q_hand = q_hand

            new_q = np.concatenate([new_q_arm, new_q_hand])
            kuka_sharpa_viser.update_cfg(new_q)

    # ###########
    # Add controls
    # ###########
    fps = 1.0 / args.dt

    def get_frame_idx_slider_text(idx: int) -> str:
        current_time = idx * args.dt
        total_time = (N_TIMESTEPS - 1) * args.dt
        return f"{current_time:.3f}s/{total_time:.3f}s ({idx:04d}/{N_TIMESTEPS:04d}) ({fps:.0f}fps)"

    with SERVER.gui.add_folder("Frame Controls"):
        frame_idx_slider = SERVER.gui.add_slider(
            label=get_frame_idx_slider_text(args.start_idx),
            min=0,
            max=N_TIMESTEPS - 1,
            step=1,
            initial_value=args.start_idx,
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
        nonlocal FRAME_IDX
        FRAME_IDX = int(np.clip(frame_idx_slider.value, a_min=0, a_max=N_TIMESTEPS - 1))
        frame_idx_slider.label = get_frame_idx_slider_text(FRAME_IDX)

    @pause_toggle_button.on_click
    def _(_) -> None:
        nonlocal PAUSED
        PAUSED = not PAUSED
        pause_toggle_button.label = "Play" if PAUSED else "Pause"

    @increment_button.on_click
    def _(_) -> None:
        nonlocal PAUSED
        if not PAUSED:
            PAUSED = True
            pause_toggle_button.label = "Play"
        frame_idx_slider.value = int(
            np.clip(frame_idx_slider.value + 1, a_min=0, a_max=N_TIMESTEPS - 1)
        )

    @decrement_button.on_click
    def _(_) -> None:
        nonlocal PAUSED
        if not PAUSED:
            PAUSED = True
            pause_toggle_button.label = "Play"
        frame_idx_slider.value = int(
            np.clip(frame_idx_slider.value - 1, a_min=0, a_max=N_TIMESTEPS - 1)
        )

    @reset_button.on_click
    def _(_) -> None:
        frame_idx_slider.value = 0

    # ###########
    # Main loop
    # ###########
    while True:
        start_loop_time = time.time()

        i = FRAME_IDX
        T_W_O = T_W_Os[i]
        T_R_P = T_R_Ps[i]
        hand_keypoint_to_xyz = hand_keypoint_to_xyzs[i]

        # Object
        obj_xyz, obj_quat_xyzw = (
            T_W_O[:3, 3],
            R.from_matrix(T_W_O[:3, :3]).as_quat(),
        )
        object_frame_viser.position = obj_xyz
        object_frame_viser.wxyz = xyzw_to_wxyz(obj_quat_xyzw)

        # Hand keypoints
        set_keypoint_sphere_positions(hand_keypoint_to_xyz, SERVER)

        # Hand meshes
        if args.visualize_hand_meshes:
            # Move previous hand to far away position
            # Works when i = 0 because it just moves the last one
            prev_hand_frame = hand_frames[i - 1]
            prev_hand_frame.position = FAR_AWAY_POSITION

            hand_frame = hand_frames[i]
            hand_xyz, hand_quat_xyzw = (
                T_W_C[:3, 3],
                R.from_matrix(T_W_C[:3, :3]).as_quat(),
            )
            hand_frame.position = hand_xyz
            hand_frame.wxyz = xyzw_to_wxyz(hand_quat_xyzw)

        # Retarget robot
        if args.retarget_robot:
            if args.retarget_robot_using_object_relative_pose and i >= idx_lifted:
                # Move floating hand to wrist pose
                T_W_P_using_lifted_object_pose = T_W_Ps_using_lifted_object_pose[i]
                sharpa_frame.position = T_W_P_using_lifted_object_pose[:3, 3]
                sharpa_frame.wxyz = xyzw_to_wxyz(
                    R.from_matrix(T_W_P_using_lifted_object_pose[:3, :3]).as_quat()
                )

                new_q = retargeted_qs_using_lifted_pose[i].copy()
                new_q_hand = new_q[7:]
                kuka_sharpa_viser.update_cfg(new_q)
                sharpa_viser.update_cfg(new_q_hand)
            else:
                # Move floating hand to wrist pose
                T_W_P = T_W_R @ T_R_P
                sharpa_frame.position = T_W_P[:3, 3]
                sharpa_frame.wxyz = xyzw_to_wxyz(R.from_matrix(T_W_P[:3, :3]).as_quat())

                new_q = retargeted_qs[i].copy()
                new_q_hand = new_q[7:]
                kuka_sharpa_viser.update_cfg(new_q)
                sharpa_viser.update_cfg(new_q_hand)

            if args.save_retargeted_robot_to_file:
                # Create lists to store values
                if i == 0:
                    q_list = []
                    object_pose_list = []

                # Append values to lists
                q_list.append(new_q.copy())
                object_pose_list.append(T_to_pose(T_W_O))

                # Save to file
                if i == N_TIMESTEPS - 1:
                    save_to_file(
                        file_path=args.output_retargeted_robot_path,
                        q_array=np.array(q_list),
                        object_pose_array=np.array(object_pose_list),
                        dt=args.dt,
                    )

        # Sleep and update frame index
        end_loop_time = time.time()
        loop_time = end_loop_time - start_loop_time
        sleep_time = args.dt - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            warn(
                f"Visualization is running slow, late by {-sleep_time * 1000:.2f} ms",
            )

        if not PAUSED:
            frame_idx_slider.value = int(
                np.clip(frame_idx_slider.value + 1, a_min=0, a_max=N_TIMESTEPS - 1)
            )


if __name__ == "__main__":
    main()
