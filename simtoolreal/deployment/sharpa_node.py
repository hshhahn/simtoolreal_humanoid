import time
from copy import deepcopy

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored

# Must add this to system path before importing sharpa
# isort: off
from pathlib import Path

SHARPA_SDK_PATH = Path("/juno/u/tylerlum/Sharpa/SDK/SharpaWaveSDK_4.3.4/python")
assert SHARPA_SDK_PATH.exists(), f"SHARPA_SDK_PATH: {SHARPA_SDK_PATH} does not exist"

import sys

# import sharpa.so from SharpaWaveSDK python folder
sys.path.insert(0, str(SHARPA_SDK_PATH))
from sharpa import (
    ControlMode,
    ControlSource,
    SharpaWaveManager,
)
# isort: on

# Joint names (indices 0..21)
JOINT_NAMES = [
    "Thumb CMC Flexion/Extension",
    "Thumb CMC Abduction/Adduction",
    "Thumb MCP Flexion/Extension",
    "Thumb MCP Abduction/Adduction",
    "Thumb DIP Flexion/Extension",
    "Index MCP Flexion/Extension",
    "Index MCP Abduction/Adduction",
    "Index PIP Flexion/Extension",
    "Index DIP Flexion/Extension",
    "Middle MCP Flexion/Extension",
    "Middle MCP Abduction/Adduction",
    "Middle PIP Flexion/Extension",
    "Middle DIP Flexion/Extension",
    "Ring MCP Flexion/Extension",
    "Ring MCP Abduction/Adduction",
    "Ring PIP Flexion/Extension",
    "Ring DIP Flexion/Extension",
    "Pinky CMC Flexion/Extension",
    "Pinky MCP Flexion/Extension",
    "Pinky MCP Abduction/Adduction",
    "Pinky PIP Flexion/Extension",
    "Pinky DIP Flexion/Extension",
]

# Angle ranges (in degrees)
ANGLE_RANGES = [
    (0, 50),  # Thumb CMC Flexion/Extension
    (0, 10),  # Thumb CMC Abduction/Adduction
    (0, 30),  # Thumb MCP Flexion/Extension
    (0, 10),  # Thumb MCP Abduction/Adduction
    (0, 40),  # Thumb DIP Flexion/Extension
    (0, 20),  # Index MCP Flexion/Extension
    (-20, 20),  # Index MCP Abduction/Adduction
    (0, 20),  # Index PIP Flexion/Extension
    (0, 20),  # Index DIP Flexion/Extension
    (0, 20),  # Middle MCP Flexion/Extension
    (-20, 20),  # Middle MCP Abduction/Adduction
    (0, 20),  # Middle PIP Flexion/Extension
    (0, 20),  # Middle DIP Flexion/Extension
    (0, 20),  # Ring MCP Flexion/Extension
    (-20, 20),  # Ring MCP Abduction/Adduction
    (0, 20),  # Ring PIP Flexion/Extension
    (0, 20),  # Ring DIP Flexion/Extension
    (0, 10),  # Pinky CMC Flexion/Extension
    (0, 20),  # Pinky MCP Flexion/Extension
    (-20, 20),  # Pinky MCP Abduction/Adduction
    (0, 20),  # Pinky PIP Flexion/Extension
    (0, 20),  # Pinky DIP Flexion/Extension
]


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


def auto_detect_hand() -> None:
    """Automatically detect device and return device and device serial number"""
    print("Searching for devices...")

    try:
        manager = SharpaWaveManager.get_instance()
        time.sleep(1)  # Wait for 1 seconds for device discovery to complete
        while True:
            devices = manager.get_all_device_sn()
            if not devices:
                warn("No available devices found")
                time.sleep(1)
                continue
            else:
                info(f"Device found: {devices[0]}")
                return manager.connect(devices[0])
    except Exception as e:
        warn(f"Failed to connect to device: {str(e)}")
        exit(1)


def initialize(hand) -> bool:
    error = hand.set_control_mode(ControlMode.POSITION)
    if error.code != 0:
        warn(f"Failed to set control mode: {error.message}")
        return False
    error = hand.set_speed_coeff(0.3)
    if error.code != 0:
        warn(f"Failed to set speed coeff: {error.message}")
        return False

    error = hand.set_current_coeff(0.6)
    if error.code != 0:
        warn(f"Failed to set current coeff: {error.message}")
        return False
    error = hand.set_control_source(ControlSource.SDK)
    if error.code != 0:
        warn(f"Failed to set control source: {error.message}")
        return False
    return True


class SharpaRosNode:
    def __init__(self):
        rospy.init_node("sharpa_ros_node")

        self.rate_hz = 100
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # Store latest cmd
        self.latest_cmd = None

        # Listen to joint cmd
        self.joint_cmd_sub = rospy.Subscriber(
            "/sharpa/joint_cmd", JointState, self.joint_cmd_callback, queue_size=1
        )

        # Publish joint states
        self.joint_states_pub = rospy.Publisher(
            "/sharpa/joint_states", JointState, queue_size=1
        )

        # Initialize SharpaWave
        self._setup_sharpa_wave()

    def _setup_sharpa_wave(self):
        info("Sharpa Wave Example - Starting")
        # Try to automatically detect device
        self.hand = auto_detect_hand()
        if self.hand is None:
            warn("Error: No available device found")
            exit(1)
        info("Sharpa Wave Example - Init Hand Running Mode")
        if not initialize(self.hand):
            warn("Error: Failed to initialize hand")
            exit(1)

        info("Sharpa Wave Example - Run Demo Gestures")
        self.hand.start()

        enable_interpolation_mode = True
        self.hand.set_joint_position([0.0] * 22, enable_interpolation_mode)

    def joint_cmd_callback(self, msg: JointState):
        self.latest_cmd = msg

    def publish_joint_states(self):
        error, angles_deg = self.hand.get_joint_position_degree()
        if error.code != 0:
            print(f"Failed to get joint positions: {error.message}")
            return

        joint_states_msg = JointState()
        joint_states_msg.header.stamp = rospy.Time.now()
        joint_states_msg.name = JOINT_NAMES
        joint_states_msg.position = np.deg2rad(angles_deg).tolist()
        self.joint_states_pub.publish(joint_states_msg)

        DEBUG = False
        if DEBUG:
            print(f"Published joint states: {joint_states_msg.position}")
            print(f"Angles (deg): {angles_deg}")

    def run(self):
        while not rospy.is_shutdown():
            latest_cmd = deepcopy(self.latest_cmd)
            if latest_cmd is not None:
                enable_interpolation_mode = True
                self.hand.set_joint_position(
                    latest_cmd.position, enable_interpolation_mode
                )

            self.publish_joint_states()
            self.rate.sleep()

    def stop(self):
        info("Sharpa Wave Example - Stop Hand Running Mode")
        self.hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()
        info("Sharpa Wave Example - Stopped")


def main():
    node = SharpaRosNode()
    node.run()
    node.stop()


if __name__ == "__main__":
    main()
