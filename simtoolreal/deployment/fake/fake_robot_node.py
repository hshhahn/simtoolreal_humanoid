#!/usr/bin/env python
import time
from typing import Literal

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 22

DEFAULT_ARM_Q = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308])

DEFAULT_HAND_Q = np.zeros(22)


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


class FakeRobotNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("fake_robot_node")

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.sharpa_joint_cmd = None

        # Publisher and subscriber
        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=1)
        self.sharpa_pub = rospy.Publisher(
            "/sharpa/joint_states", JointState, queue_size=1
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback, queue_size=1
        )
        self.sharpa_cmd_sub = rospy.Subscriber(
            "/sharpa/joint_cmd",
            JointState,
            self.sharpa_joint_cmd_callback,
            queue_size=1,
        )

        # State
        self.iiwa_joint_q = DEFAULT_ARM_Q
        self.sharpa_joint_q = DEFAULT_HAND_Q
        self.iiwa_joint_qd = np.zeros(NUM_ARM_JOINTS)
        self.sharpa_joint_qd = np.zeros(NUM_HAND_JOINTS)

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # When only testing the arm, set this to False to ignore the sharpa hand
        self.WAIT_FOR_SHARPA_CMD = True
        if not self.WAIT_FOR_SHARPA_CMD:
            warn("NOT WAITING FOR SHARPA CMD")
            self.sharpa_joint_cmd = np.zeros(NUM_HAND_JOINTS)

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def sharpa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.sharpa_joint_cmd = np.array(msg.position)

    def update_joint_states(self):
        """Update the PyBullet simulation with the commanded joint positions."""
        if self.iiwa_joint_cmd is None or self.sharpa_joint_cmd is None:
            warn_every(
                f"Waiting: iiwa_joint_cmd: {self.iiwa_joint_cmd}, sharpa_joint_cmd: {self.sharpa_joint_cmd}",
                n_seconds=1.0,
            )
            return

        delta_iiwa = self.iiwa_joint_cmd - self.iiwa_joint_q
        delta_sharpa = self.sharpa_joint_cmd - self.sharpa_joint_q

        MODE: Literal["INTERPOLATE", "PD_CONTROL"] = "INTERPOLATE"
        if MODE == "INTERPOLATE":
            delta_iiwa_norm = np.linalg.norm(delta_iiwa)
            delta_sharpa_norm = np.linalg.norm(delta_sharpa)

            MAX_DELTA_IIWA = 0.1
            MAX_DELTA_SHARPA = 0.1
            if delta_iiwa_norm > MAX_DELTA_IIWA:
                delta_iiwa = MAX_DELTA_IIWA * delta_iiwa / delta_iiwa_norm
            if delta_sharpa_norm > MAX_DELTA_SHARPA:
                delta_sharpa = MAX_DELTA_SHARPA * delta_sharpa / delta_sharpa_norm

            self.iiwa_joint_q += delta_iiwa
            self.sharpa_joint_q += delta_sharpa
            self.iiwa_joint_qd = delta_iiwa / self.dt
            self.sharpa_joint_qd = np.zeros(NUM_HAND_JOINTS)
        elif MODE == "PD_CONTROL":
            P = 10
            D = 0
            iiwa_qd_cmd = 0
            sharpa_qd_cmd = 0
            delta_iiwa_qd = iiwa_qd_cmd - self.iiwa_joint_qd
            delta_sharpa_qd = sharpa_qd_cmd - self.sharpa_joint_qd

            iiwa_qdd = P * delta_iiwa + D * delta_iiwa_qd
            sharpa_qdd = P * delta_sharpa + D * delta_sharpa_qd
            self.iiwa_joint_qd += iiwa_qdd * self.dt
            self.sharpa_joint_qd += sharpa_qdd * self.dt
            self.iiwa_joint_q += self.iiwa_joint_qd * self.dt
            self.sharpa_joint_q += self.sharpa_joint_qd * self.dt
        else:
            raise ValueError(f"Invalid mode: {MODE}")

    def publish_joint_states(self):
        """Publish the current joint states from PyBullet."""
        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.name = ["iiwa_joint_" + str(i) for i in range(NUM_ARM_JOINTS)]
        iiwa_msg.position = self.iiwa_joint_q.tolist()
        iiwa_msg.velocity = self.iiwa_joint_qd.tolist()
        self.iiwa_pub.publish(iiwa_msg)

        sharpa_msg = JointState()
        sharpa_msg.header.stamp = rospy.Time.now()
        sharpa_msg.name = ["sharpa_joint_" + str(i) for i in range(NUM_HAND_JOINTS)]
        sharpa_msg.position = self.sharpa_joint_q.tolist()
        sharpa_msg.velocity = self.sharpa_joint_qd.tolist()
        self.sharpa_pub.publish(sharpa_msg)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the joint states
            self.update_joint_states()

            # Publish the current joint states to ROS
            self.publish_joint_states()

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


if __name__ == "__main__":
    try:
        # Create and run the FakeRobotNode
        node = FakeRobotNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
