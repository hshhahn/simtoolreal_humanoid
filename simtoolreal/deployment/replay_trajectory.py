import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rospy
import tyro
from sensor_msgs.msg import JointState

from recorded_data import RecordedData

# Global variables for current joint positions
CURRENT_JOINT_POS_IIWA = None
CURRENT_JOINT_POS_SHARPA = None


def current_joint_pos_iiwa_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_IIWA
    CURRENT_JOINT_POS_IIWA = np.array(msg.position)


def current_joint_pos_sharpa_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_SHARPA
    CURRENT_JOINT_POS_SHARPA = np.array(msg.position)


def interpolate_joint_pos(
    init_joint_pos: np.ndarray, final_joint_pos: np.ndarray, num_steps: int
) -> np.ndarray:
    assert init_joint_pos.shape == final_joint_pos.shape, (
        f"init_joint_pos.shape: {init_joint_pos.shape}, final_joint_pos.shape: {final_joint_pos.shape}"
    )
    joint_positions_list = []
    for i in range(num_steps):
        joint_positions_list.append(
            init_joint_pos + (final_joint_pos - init_joint_pos) * (i + 1) / num_steps
        )
    joint_positions_array = np.array(joint_positions_list)
    assert joint_positions_array.shape == (num_steps, init_joint_pos.shape[0]), (
        f"joint_positions_array.shape: {joint_positions_array.shape}, expected: ({num_steps}, {init_joint_pos.shape[0]})"
    )
    return np.array(joint_positions_list)


def publish_joint_pos_targets(
    joint_pos_targets: np.ndarray,
    pub_iiwa: rospy.Publisher,
    pub_sharpa: rospy.Publisher,
) -> None:
    assert joint_pos_targets.shape == (29,), (
        f"joint_pos_targets.shape: {joint_pos_targets.shape}, expected: ({29},)"
    )
    iiwa_joint_pos = joint_pos_targets[:7]
    sharpa_joint_pos = joint_pos_targets[7:]

    iiwa_msg = JointState()
    iiwa_msg.header.stamp = rospy.Time.now()
    iiwa_msg.header.frame_id = ""
    iiwa_msg.name = [
        "iiwa_joint_1",
        "iiwa_joint_2",
        "iiwa_joint_3",
        "iiwa_joint_4",
        "iiwa_joint_5",
        "iiwa_joint_6",
        "iiwa_joint_7",
    ]
    sharpa_msg = JointState()
    sharpa_msg.header.stamp = rospy.Time.now()
    sharpa_msg.header.frame_id = ""
    sharpa_msg.name = [
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
    ]

    iiwa_msg.position = copy.deepcopy(iiwa_joint_pos.tolist())
    sharpa_msg.position = copy.deepcopy(sharpa_joint_pos.tolist())
    pub_iiwa.publish(iiwa_msg)
    pub_sharpa.publish(sharpa_msg)


def move_to_pose(
    target_pos: np.ndarray,
    pub_iiwa: rospy.Publisher,
    pub_sharpa: rospy.Publisher,
    move_time: float = 10.0,
    control_hz: int = 60,
) -> None:
    assert target_pos.shape == (29,), (
        f"target_pos.shape: {target_pos.shape}, expected: ({29},)"
    )
    current_sharpa_pos = CURRENT_JOINT_POS_SHARPA.copy()
    current_iiwa_pos = CURRENT_JOINT_POS_IIWA.copy()
    current_pos = np.concatenate([current_iiwa_pos, current_sharpa_pos])

    SECONDS_TO_MOVE = move_time
    CONTROL_HZ = control_hz
    interpolated_targets = interpolate_joint_pos(
        init_joint_pos=current_pos,
        final_joint_pos=target_pos,
        num_steps=int(CONTROL_HZ * SECONDS_TO_MOVE),
    )
    for target_pos in interpolated_targets:
        if rospy.is_shutdown():
            print("ROS shutdown, exiting")
            sys.exit(0)

        start_time = rospy.Time.now()
        publish_joint_pos_targets(target_pos, pub_iiwa=pub_iiwa, pub_sharpa=pub_sharpa)
        end_time = rospy.Time.now()

        loop_without_sleep_dt = (end_time - start_time).to_sec()
        sleep_dt = 1 / CONTROL_HZ - loop_without_sleep_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            print(
                f"Loop too slow! Desired FPS: {CONTROL_HZ}, Actual FPS: {1.0 / loop_without_sleep_dt:.1f}"
            )


@dataclass
class ReplayTrajectoryArgs:
    file_path: Path = Path("recorded_data/2025-10-20_14-32-39_None_310.npz")
    """Path to the recorded data file."""

    dt: float = 1 / 30
    """Time step between trajectory points."""

    SLOW_DOWN_FACTOR: float = 3.0
    """Factor to slow down the trajectory by."""

    mode: Literal["joint_positions", "joint_pos_targets"] = "joint_positions"
    """Mode to replay the trajectory in."""


def main():
    args: ReplayTrajectoryArgs = tyro.cli(ReplayTrajectoryArgs)

    # Read in trajectory
    file_path = args.file_path
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)
    joint_positions_array = recorded_data.robot_joint_positions_array
    joint_pos_targets_array = recorded_data.robot_joint_pos_targets_array
    joint_names = recorded_data.robot_joint_names
    T = joint_positions_array.shape[0]
    J = len(joint_names)
    assert joint_positions_array.shape == (T, J), (
        f"joint_positions_array.shape: {joint_positions_array.shape}, expected: ({T}, {J})"
    )
    assert joint_pos_targets_array.shape == (T, J), (
        f"joint_pos_targets_array.shape: {joint_pos_targets_array.shape}, expected: ({T}, {J})"
    )

    # Initialize ROS node
    rospy.init_node("replay_trajectory", anonymous=True)

    # Create subscribers and publishers
    _sub_iiwa = rospy.Subscriber(
        "/iiwa/joint_states", JointState, current_joint_pos_iiwa_callback, queue_size=1
    )
    _sub_sharpa = rospy.Subscriber(
        "/sharpa/joint_states",
        JointState,
        current_joint_pos_sharpa_callback,
        queue_size=1,
    )
    pub_iiwa = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=1)
    pub_sharpa = rospy.Publisher("/sharpa/joint_cmd", JointState, queue_size=1)

    # Wait for current joint positions to be available
    while not rospy.is_shutdown():
        if CURRENT_JOINT_POS_IIWA is None or CURRENT_JOINT_POS_SHARPA is None:
            print(
                f"Waiting: CURRENT_JOINT_POS_IIWA = {CURRENT_JOINT_POS_IIWA}, CURRENT_JOINT_POS_SHARPA = {CURRENT_JOINT_POS_SHARPA}"
            )
            rospy.sleep(0.1)
        else:
            print("=" * 100)
            print("Got CURRENT_JOINT_POS_IIWA and CURRENT_JOINT_POS_SHARPA")
            print("=" * 100)
            break

    # Move to initial pose
    print("Moving to initial pose")
    move_to_pose(
        joint_positions_array[0],
        pub_iiwa=pub_iiwa,
        pub_sharpa=pub_sharpa,
        move_time=10.0,
    )
    print("Reached initial pose")

    # Replay trajectory
    if args.mode == "joint_positions":
        trajectory_array = joint_positions_array
    elif args.mode == "joint_pos_targets":
        trajectory_array = joint_pos_targets_array
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    print("Replaying trajectory")
    start_time = time.time()
    for timestep in range(len(trajectory_array)):
        print(f"Replaying timestep: {timestep}")
        move_to_pose(
            trajectory_array[timestep],
            pub_iiwa=pub_iiwa,
            pub_sharpa=pub_sharpa,
            move_time=args.dt * args.SLOW_DOWN_FACTOR,
        )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
