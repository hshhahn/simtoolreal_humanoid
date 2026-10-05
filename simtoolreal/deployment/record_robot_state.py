import copy
import datetime
import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored

from recorded_data import RecordedData


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


def info(message: str):
    print(colored(message, "green"))


JOINT_NAMES = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
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
    "joint_16.0",
    "joint_17.0",
    "joint_18.0",
    "joint_19.0",
    "joint_20.0",
    "joint_21.0",
]


class RecordRobotState:
    def __init__(self):
        rospy.init_node("record_robot_state")

        # Signal handling to save on shutdown
        # When in progress saving to file, stop updating latest joint states and commands
        self._is_in_progress_saving_to_file = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Store latest joint states and commands
        self.latest_iiwa_joint_state: Optional[JointState] = None
        self.latest_sharpa_joint_state: Optional[JointState] = None
        self.latest_iiwa_joint_cmd: Optional[JointState] = None
        self.latest_sharpa_joint_cmd: Optional[JointState] = None

        # Store history of joint states and commands
        self.time_history: list[float] = []
        self.iiwa_joint_position_history: list[np.ndarray] = []
        self.sharpa_joint_position_history: list[np.ndarray] = []
        self.iiwa_joint_velocity_history: list[np.ndarray] = []
        self.sharpa_joint_velocity_history: list[np.ndarray] = []
        self.iiwa_joint_pos_target_history: list[np.ndarray] = []
        self.sharpa_joint_pos_target_history: list[np.ndarray] = []

        # Subscribers
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states",
            JointState,
            self._iiwa_joint_state_callback,
            queue_size=1,
        )
        self.sharpa_joint_state_sub = rospy.Subscriber(
            "/sharpa/joint_states",
            JointState,
            self._sharpa_joint_state_callback,
            queue_size=1,
        )
        self.iiwa_joint_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self._iiwa_joint_cmd_callback, queue_size=1
        )
        self.sharpa_joint_cmd_sub = rospy.Subscriber(
            "/sharpa/joint_cmd",
            JointState,
            self._sharpa_joint_cmd_callback,
            queue_size=1,
        )

        # ROS rate
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def _iiwa_joint_state_callback(self, msg: JointState):
        self.latest_iiwa_joint_state = msg

    def _sharpa_joint_state_callback(self, msg: JointState):
        self.latest_sharpa_joint_state = msg

    def _iiwa_joint_cmd_callback(self, msg: JointState):
        self.latest_iiwa_joint_cmd = msg

    def _sharpa_joint_cmd_callback(self, msg: JointState):
        self.latest_sharpa_joint_cmd = msg

    def run(self):
        while not rospy.is_shutdown():
            if (
                self.latest_iiwa_joint_state is None
                or self.latest_sharpa_joint_state is None
                or self.latest_iiwa_joint_cmd is None
                or self.latest_sharpa_joint_cmd is None
            ):
                warn_every(
                    f"Waiting: latest_iiwa_joint_state = {self.latest_iiwa_joint_state}, latest_sharpa_joint_state = {self.latest_sharpa_joint_state}, latest_iiwa_joint_cmd = {self.latest_iiwa_joint_cmd}, latest_sharpa_joint_cmd = {self.latest_sharpa_joint_cmd}",
                    n_seconds=1.0,
                )
                self.rate.sleep()
            else:
                break  # All messages received, exit loop

        info("All messages received, starting to record robot state")

        start_time = time.time()
        while not rospy.is_shutdown():
            if self._is_in_progress_saving_to_file:
                warn("In progress of saving to file, exiting")
                break

            # Record time
            current_time = time.time()
            dt = current_time - start_time

            # Create copy of latest joint states and commands
            iiwa_joint_state = copy.copy(self.latest_iiwa_joint_state)
            sharpa_joint_state = copy.copy(self.latest_sharpa_joint_state)
            iiwa_joint_cmd = copy.copy(self.latest_iiwa_joint_cmd)
            sharpa_joint_cmd = copy.copy(self.latest_sharpa_joint_cmd)

            # Convert to numpy arrays
            iiwa_joint_position = np.array(iiwa_joint_state.position)
            iiwa_joint_velocity = np.array(iiwa_joint_state.velocity)
            sharpa_joint_position = np.array(sharpa_joint_state.position)
            sharpa_joint_velocity = np.array(sharpa_joint_state.velocity)
            iiwa_joint_cmd_position = np.array(iiwa_joint_cmd.position)
            sharpa_joint_cmd_position = np.array(sharpa_joint_cmd.position)

            # Store
            self.iiwa_joint_position_history.append(iiwa_joint_position)
            self.sharpa_joint_position_history.append(sharpa_joint_position)
            self.iiwa_joint_velocity_history.append(iiwa_joint_velocity)
            self.sharpa_joint_velocity_history.append(sharpa_joint_velocity)
            self.iiwa_joint_pos_target_history.append(iiwa_joint_cmd_position)
            self.sharpa_joint_pos_target_history.append(sharpa_joint_cmd_position)
            self.time_history.append(dt)

            info(f"Recorded {len(self.time_history)} steps over {dt:.2f} seconds")
            self.rate.sleep()

    def _signal_handler(self, signum, frame):
        if self._is_in_progress_saving_to_file:
            warn("Already in progress of saving to file, skipping")
            return

        self._is_in_progress_saving_to_file = True
        if len(self.time_history) == 0:
            warn("No data recorded, skipping")
        else:
            info(f"Received signal {signum}, saving to file")
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = Path("recorded_robot_state") / f"{datetime_str}.npz"
            self.save_to_file(output_path)
            info(f"Saved to file: {output_path}")

        rospy.signal_shutdown("Shutting down")

    def save_to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        info(f"Saving to file: {file_path}")

        T = len(self.time_history)
        robot_root_states_array = np.zeros((T, 13))
        robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
        object_root_states_array = np.zeros((T, 13))
        object_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1

        iiwa_joint_positions = np.array(self.iiwa_joint_position_history)
        sharpa_joint_positions = np.array(self.sharpa_joint_position_history)
        robot_joint_positions = np.concatenate(
            [iiwa_joint_positions, sharpa_joint_positions], axis=1
        )

        iiwa_joint_velocities = np.array(self.iiwa_joint_velocity_history)
        sharpa_joint_velocities = np.array(self.sharpa_joint_velocity_history)
        robot_joint_velocities = np.concatenate(
            [iiwa_joint_velocities, sharpa_joint_velocities], axis=1
        )

        iiwa_joint_pos_targets = np.array(self.iiwa_joint_pos_target_history)
        sharpa_joint_pos_targets = np.array(self.sharpa_joint_pos_target_history)
        robot_joint_pos_targets = np.concatenate(
            [iiwa_joint_pos_targets, sharpa_joint_pos_targets], axis=1
        )

        time_array = np.array(self.time_history)

        assert robot_joint_positions.shape == (T, 29), (
            f"robot_joint_positions.shape: {robot_joint_positions.shape}, expected: (T, 29)"
        )
        assert robot_joint_velocities.shape == (T, 29), (
            f"robot_joint_velocities.shape: {robot_joint_velocities.shape}, expected: (T, 29)"
        )
        assert robot_joint_pos_targets.shape == (T, 29), (
            f"robot_joint_pos_targets.shape: {robot_joint_pos_targets.shape}, expected: (T, 29)"
        )
        assert time_array.shape == (T,), (
            f"time_array.shape: {time_array.shape}, expected: (T,)"
        )

        recorded_data = RecordedData(
            robot_root_states_array=robot_root_states_array,
            object_root_states_array=object_root_states_array,
            robot_joint_positions_array=robot_joint_positions,
            time_array=np.array(self.time_history),
            robot_joint_names=JOINT_NAMES,
            robot_joint_velocities_array=robot_joint_velocities,
            robot_joint_pos_targets_array=robot_joint_pos_targets,
        )
        recorded_data.to_file(file_path)


def main():
    record_robot_state = RecordRobotState()
    record_robot_state.run()


if __name__ == "__main__":
    main()
