"""Trajectory Optimization with Waypoints

Demonstrates avoiding a Box obstacle while hitting specific waypoints.
"""

import json
import time
from pathlib import Path
from typing import Literal

# Enable caching for faster re-runs
import jax
import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from baselines import pyroki_snippets as pks

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    return xyzw[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    return wxyz[..., [1, 2, 3, 0]]


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


def main(robot_name: Literal["sharpa"] = "sharpa"):
    # Load robot
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

    if robot_name == "sharpa":
        import yourdfpy

        # Sharpa
        urdf = yourdfpy.URDF.load(
            "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        target_link_name = "left_hand_C_MC"
        _down_wxyz = np.array([0.5, 0.5, 0.5, -0.5])
        sphere_json_path = Path(__file__).parent / "assets" / "sharpa_spheres.json"
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=HOME_JOINT_POS)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    # Load collision spheres
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    USE_SPHERE_DECOMPOSITION = True
    if USE_SPHERE_DECOMPOSITION:
        robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
            sphere_decomposition=sphere_decomposition,
            urdf=urdf,
        )
    else:
        robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # Problem Setup
    timesteps, dt = 50, 0.05

    # Implicit start config based on current robot state (usually 0 unless set)
    # We will use the robot's default config as the "seed" for the start pose IK
    # but to pass a specific start_cfg to the solver, we can run a quick IK first
    # or just use zeros if compatible. Let's use zeros for Panda.
    start_cfg = HOME_JOINT_POS
    assert len(start_cfg) == robot.joints.num_actuated_joints, (
        f"start_cfg.shape: {start_cfg.shape}, expected: ({robot.joints.num_actuated_joints},)"
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
    world_coll = [ground_coll, table_coll]

    # Define Waypoints
    # mid_pos = np.array([0.0, -0.6, 0.8])
    # right_pos = np.array([-0.5, -0.6, 0.8])
    # right_low_pos = np.array([-0.5, -0.6, 0.4])
    # left_pos = np.array([0.5, -0.6, 0.8])
    # left_low_pos = np.array([0.5, -0.6, 0.4])

    # waypoints = {}
    # waypoints.update({
    #     5 + i: (mid_pos, down_wxyz)
    #     for i in range(5)
    # })
    # waypoints.update({
    #     25 + i: (right_pos, down_wxyz)
    #     for i in range(5)
    # })
    # waypoints.update({
    #     49 - i: (left_low_pos, down_wxyz)
    #     for i in range(5)
    # })
    T_R_Ps_path = Path(__file__).parent / "T_R_Ps.json"
    q_hands_path = Path(__file__).parent / "q_hands.json"
    assert T_R_Ps_path.exists(), f"T_R_Ps_path not found: {T_R_Ps_path}"
    assert q_hands_path.exists(), f"q_hands_path not found: {q_hands_path}"
    with open(T_R_Ps_path, "r") as f:
        T_R_Ps = np.array(json.load(f))
    with open(q_hands_path, "r") as f:
        q_hands = np.array(json.load(f))
    N_TIMESTEPS = T_R_Ps.shape[0]
    assert T_R_Ps.shape == (N_TIMESTEPS, 4, 4), (
        f"T_R_Ps.shape: {T_R_Ps.shape}, expected: ({N_TIMESTEPS}, 4, 4)"
    )
    assert q_hands.shape == (N_TIMESTEPS, 22), (
        f"q_hands.shape: {q_hands.shape}, expected: ({N_TIMESTEPS}, 22)"
    )
    DOWNSAMPLE = True
    if DOWNSAMPLE:
        T_R_Ps = T_R_Ps[::10]
        q_hands = q_hands[::10]
        N_TIMESTEPS = T_R_Ps.shape[0]
        assert T_R_Ps.shape == (N_TIMESTEPS, 4, 4), (
            f"T_R_Ps.shape: {T_R_Ps.shape}, expected: ({N_TIMESTEPS}, 4, 4)"
        )
        assert q_hands.shape == (N_TIMESTEPS, 22), (
            f"q_hands.shape: {q_hands.shape}, expected: ({N_TIMESTEPS}, 22)"
        )

    positions = T_R_Ps[:, :3, 3]
    quat_xyzws = R.from_matrix(T_R_Ps[:, :3, :3]).as_quat()
    quat_wxyzs = xyzw_to_wxyz(quat_xyzws)
    BUFFER = 10
    timesteps = N_TIMESTEPS + BUFFER

    waypoints = {BUFFER + i: (positions[i], quat_wxyzs[i]) for i in range(N_TIMESTEPS)}

    # Visualize problem setup
    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # Draw robot
    urdf_vis = ViserUrdf(server, urdf)
    urdf_vis.update_cfg(start_cfg)

    # Draw robot collision model ghost
    server.scene.add_mesh_trimesh(
        "robot_coll_ghost",
        robot_coll.at_config(robot, start_cfg).to_trimesh(),
    )

    # Draw Table
    server.scene.add_mesh_trimesh(
        "table_box",
        trimesh.creation.box(
            extents=table_size,
            transform=trimesh.transformations.translation_matrix(table_center),
        ),
    )

    # Draw Waypoints
    for t, (pos, wxyz) in waypoints.items():
        server.scene.add_frame(
            f"/waypoints/t{t}",
            position=pos,
            wxyz=wxyz,
            axes_length=0.1,
            axes_radius=0.01,
        )

    # Solve trajectory
    print("Solving trajectory with waypoints...")
    start_time = time.time()

    print(
        f"Solving trajectory with {timesteps} waypoints, each {dt} seconds apart, for a total time of {timesteps * dt} seconds"
    )
    traj = pks.solve_waypoint_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_cfg,
        waypoints,
        timesteps,
        dt,
    )

    traj = np.array(traj)
    assert len(traj) == timesteps, f"len(traj): {len(traj)}, expected: {timesteps}"
    print(f"Solved in {time.time() - start_time:.4f}s")

    # Repeat q_hand for BUFFER timesteps to match traj length
    first_q_hand = q_hands[0].copy()
    q_hands = np.concatenate([first_q_hand[None].repeat(BUFFER, axis=0), q_hands])
    print(f"traj.shape: {traj.shape}, q_hands.shape: {q_hands.shape}")
    assert len(traj) == len(q_hands), (
        f"len(traj): {len(traj)}, len(q_hands): {len(q_hands)}"
    )
    traj = interpolate_traj(traj=traj, n_steps=10)
    q_hands = interpolate_traj(traj=q_hands, n_steps=10)

    # Visualize trajectory
    slider = server.gui.add_slider(
        "Timestep", min=0, max=len(traj) - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        start_time = time.time()
        if playing.value:
            slider.value = (slider.value + 1) % len(traj)

        cfg = traj[slider.value].copy()
        cfg[7:] = q_hands[slider.value].copy()
        urdf_vis.update_cfg(cfg)

        # Update collision ghost occasionally to see fit
        # (Updating every frame might be slow for complex meshes)
        if slider.value % 5 == 0:
            server.scene.add_mesh_trimesh(
                "robot_coll_ghost",
                robot_coll.at_config(robot, cfg).to_trimesh(),
            )

        end_time = time.time()
        loop_dt = end_time - start_time
        sleep_dt = dt - loop_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            print(f"Loop too slow! Desired FPS = 10, Actual FPS = {1 / loop_dt:.1f}")


if __name__ == "__main__":
    tyro.cli(main)
