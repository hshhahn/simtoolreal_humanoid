import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from termcolor import colored
from tqdm import tqdm
from viser.extras import ViserUrdf

from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir

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

AXES_LENGTH = 0.1
AXES_RADIUS = 0.005


def info(message: str) -> None:
    print(colored(message, "green"))


def warn(message: str) -> None:
    print(colored(message, "yellow"))


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z])


def pose_to_T(pose: np.ndarray) -> np.ndarray:
    assert pose.shape == (7,), f"Expected pose to be (7,), got {pose.shape}"
    xyz = pose[:3]
    xyzw = pose[3:7]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = xyz
    return T


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


def depth_to_points(
    depth_m: np.ndarray,
    K: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    stride: int = 2,
    max_depth_m: float = 5.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = depth_m.shape
    v_coords, u_coords = np.indices((h, w))
    if stride > 1:
        v_coords = v_coords[::stride, ::stride]
        u_coords = u_coords[::stride, ::stride]
        depth = depth_m[::stride, ::stride]
        if rgb is not None:
            colors = rgb[::stride, ::stride, :]
        else:
            colors = None
    else:
        depth = depth_m
        colors = rgb

    z = depth.reshape(-1)
    valid = (z > 0.0) & (z < max_depth_m)
    z = z[valid]

    u = u_coords.reshape(-1)[valid]
    v = v_coords.reshape(-1)[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_c = np.stack([x, y, z], axis=1)

    cols = colors.reshape(-1, 3)[valid]
    return pts_c, cols


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R_rc = T[:3, :3]
    t_rc = T[:3, 3]
    return (pts @ R_rc.T) + t_rc[None, :]


@dataclass
class VisualizeDemoLowLevelArgs:
    object_path: Path
    object_poses_json_path: Path
    dt: float = 1.0 / 30
    start_idx: int = 0
    rgb_path: Optional[Path] = None
    depth_path: Optional[Path] = None
    cam_intrinsics_path: Optional[Path] = None


@dataclass
class VisualizeDemoArgs:
    object_category: str = "hammer"
    object_name: str = "claw_hammer"
    task_name: str = "swing_down"

    @property
    def demo_dir(self) -> Path:
        return (
            Path("dextoolbench/data")
            / self.object_category
            / self.object_name
            / self.task_name
        )


def main():
    args_high_level: VisualizeDemoArgs = tyro.cli(VisualizeDemoArgs)
    args = VisualizeDemoLowLevelArgs(
        object_path=NAME_TO_OBJECT[args_high_level.object_name].urdf_path,
        object_poses_json_path=args_high_level.demo_dir / "poses.json",
        rgb_path=args_high_level.demo_dir / "rgb",
        depth_path=args_high_level.demo_dir / "depth",
        cam_intrinsics_path=args_high_level.demo_dir / "cam_K.txt",
    )

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

    # Read in camera stuff
    if (
        args.rgb_path is not None
        and args.depth_path is not None
        and args.cam_intrinsics_path is not None
    ):
        rgb_path = args.rgb_path
        depth_path = args.depth_path
        cam_intrinsics_path = args.cam_intrinsics_path
        assert rgb_path.exists(), f"RGB path {rgb_path} does not exist"
        assert depth_path.exists(), f"Depth path {depth_path} does not exist"
        assert cam_intrinsics_path.exists(), (
            f"Cam intrinsics path {cam_intrinsics_path} does not exist"
        )
        rgb_paths = sorted(list(rgb_path.glob("*.png")))
        depth_paths = sorted(list(depth_path.glob("*.png")))
        K = np.loadtxt(cam_intrinsics_path)
        assert K.shape == (3, 3), f"K.shape: {K.shape}, expected: (3, 3)"
        assert len(rgb_paths) == len(depth_paths), (
            f"len(rgb_paths): {len(rgb_paths)}, len(depth_paths): {len(depth_paths)}"
        )
        from PIL import Image

        rgbs = [
            np.array(Image.open(rgb_path))
            for rgb_path in tqdm(
                rgb_paths, total=len(rgb_paths), desc="Loading RGB images"
            )
        ]
        depths = [
            np.array(Image.open(depth_path)) / 1000.0
            for depth_path in tqdm(
                depth_paths, total=len(depth_paths), desc="Loading depth images"
            )
        ]
        print(f"Min depth: {np.min(depths)}, Max depth: {np.max(depths)}")
        print(f"Mean depth: {np.mean(depths)}, Median depth: {np.median(depths)}")

        pts_c_list, cols_list = [], []
        for rgb, depth in tqdm(
            zip(rgbs, depths), total=len(rgbs), desc="Processing RGBD images"
        ):
            pts_c, cols = depth_to_points(
                depth,
                K,
                rgb=rgb,
                stride=1,
                max_depth_m=5,
            )
            pts_c_list.append(pts_c)
            cols_list.append(cols)
        min_num_pts = min(len(pts_c) for pts_c in pts_c_list)
        pts_c_list = [pts_c[:min_num_pts] for pts_c in pts_c_list]
        cols_list = [cols[:min_num_pts] for cols in cols_list]
        pts_c_array = np.stack(pts_c_list, axis=0)
        cols_array = np.stack(cols_list, axis=0)
        SUBSAMPLE_FACTOR = 10
        pts_c_array = pts_c_array[:, ::SUBSAMPLE_FACTOR]
        cols_array = cols_array[:, ::SUBSAMPLE_FACTOR]
        print(
            f"pts_c_array.shape: {pts_c_array.shape}, cols_array.shape: {cols_array.shape}"
        )
        pts_w_array = transform_points(T=T_W_C, pts=pts_c_array)
        print(f"pts_w_array.shape: {pts_w_array.shape}")

        pcd_handle = SERVER.scene.add_point_cloud(
            "/zed_points_robot_frame",
            points=pts_w_array[0].astype(np.float32),
            colors=cols_array[0].astype(np.uint8),
            point_size=0.002,
        )

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
        object_urdf_path = create_urdf(args.object_path)
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

    N_TIMESTEPS = len(T_W_Os)
    has_pcd = (
        args.rgb_path is not None
        and args.depth_path is not None
        and args.cam_intrinsics_path is not None
    )
    if has_pcd:
        N_TIMESTEPS = min(N_TIMESTEPS, len(pts_w_array))
        T_W_Os = T_W_Os[:N_TIMESTEPS]
        pts_w_array = pts_w_array[:N_TIMESTEPS]
        cols_array = cols_array[:N_TIMESTEPS]

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
            label=get_frame_idx_slider_text(0),
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

        # Object
        obj_xyz, obj_quat_xyzw = (
            T_W_O[:3, 3],
            R.from_matrix(T_W_O[:3, :3]).as_quat(),
        )
        object_frame_viser.position = obj_xyz
        object_frame_viser.wxyz = xyzw_to_wxyz(obj_quat_xyzw)

        # Update point cloud
        if has_pcd:
            pcd_handle.points = pts_w_array[i].astype(np.float32)
            pcd_handle.colors = cols_array[i].astype(np.uint8)

        # Sleep and update frame index
        end_loop_time = time.time()
        loop_time = end_loop_time - start_loop_time
        sleep_time = args.dt - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            warn(f"Visualization is running slow, late by {-sleep_time * 1000:.2f} ms")

        if not PAUSED:
            frame_idx_slider.value = int(
                np.clip(frame_idx_slider.value + 1, a_min=0, a_max=N_TIMESTEPS - 1)
            )


if __name__ == "__main__":
    main()
