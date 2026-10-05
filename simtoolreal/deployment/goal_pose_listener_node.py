#!/usr/bin/env python
import json
import signal
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose, PoseStamped
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
)


def info(message: str):
    print(colored(message, "green"))


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
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


def keypoint_distance(
    pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, object_scales: np.ndarray
) -> float:
    """Compute the distance between two keypoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_pose1_xyzw = torch.from_numpy(pose1_xyzw).float().to(device)
    object_pose2_xyzw = torch.from_numpy(pose2_xyzw).float().to(device)
    object_scales = torch.from_numpy(object_scales).float().to(device)

    object_keypoint_positions = _compute_keypoint_positions(
        pose=object_pose1_xyzw[None], scales=object_scales[None]
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=object_pose2_xyzw[None], scales=object_scales[None]
    )
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    N_KEYPOINTS = 4
    N = 1
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"
    )
    keypoint_distances_l2 = torch.norm(keypoints_rel_goal, dim=-1).max(dim=-1).values
    return keypoint_distances_l2.item()


class GoalPoseListenerNode:
    def __init__(self, output_path: Path):
        # Signal handling to save on shutdown
        # When in progress saving to file, stop updating latest joint states and commands
        self._is_in_progress_saving_to_file = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # ROS setup
        rospy.init_node("goal_pose_listener_ros_node")

        self.output_path = output_path

        # State
        self.goal_pose_history: list[Pose] = []

        # ROS msgs
        self.latest_current_object_pose: Optional[Pose] = None

        # Subscribers
        self.current_object_pose_sub = rospy.Subscriber(
            "/robot_frame/current_object_pose",
            PoseStamped,
            self.current_object_pose_callback,
            queue_size=1,
        )

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def current_object_pose_callback(self, msg: PoseStamped):
        """Callback to update the current object pose."""
        self.latest_current_object_pose = msg.pose

    def store_current_object_pose(self):
        """Publish the goal object pose."""
        latest_current_object_pose = deepcopy(self.latest_current_object_pose)
        p = latest_current_object_pose
        current_object_pose_xyzw = np.array(
            [
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ]
        )
        self.goal_pose_history.append(current_object_pose_xyzw)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""

        # Wait for the current object pose to be received
        while not rospy.is_shutdown():
            if self.latest_current_object_pose is None:
                warn_every("Waiting for current object pose", n_seconds=1.0)
                time.sleep(0.1)
            else:
                info("Current object pose received, starting goal pose listener node")
                break  # All messages received, exit loop

        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Store the current object pose
            self.store_current_object_pose()
            info(f"Stored {len(self.goal_pose_history)} goal poses")

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.dt)
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []

    def _signal_handler(self, signum, frame):
        if self._is_in_progress_saving_to_file:
            warn("Already in progress of saving to file, skipping")
            return

        self._is_in_progress_saving_to_file = True
        if len(self.goal_pose_history) == 0:
            warn("No data recorded, skipping")
        else:
            info(f"Received signal {signum}, saving to file")
            self.save_to_file()
            info(f"Saved to file: {self.output_path}")

        rospy.signal_shutdown("Shutting down")

    def save_to_file(self):
        info(f"Saving to file: {self.output_path}")

        T = len(self.goal_pose_history)
        goal_pose_history = np.array(self.goal_pose_history)

        assert goal_pose_history.shape == (T, 7), (
            f"goal_pose_history.shape: {goal_pose_history.shape}, expected: (T, 7)"
        )

        json.dump(goal_pose_history.tolist(), self.output_path.open("w"), indent=4)


if __name__ == "__main__":
    try:
        object_name = "black_spatula"
        output_dir = Path("goal_pose_listener_ros_node_output") / object_name
        output_dir.mkdir(parents=True, exist_ok=True)

        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = output_dir / f"{datetime_str}.json"
        # Create and run the GoalPoseListenerNode
        node = GoalPoseListenerNode(output_path=output_path)
        node.run()
    except rospy.ROSInterruptException:
        pass
