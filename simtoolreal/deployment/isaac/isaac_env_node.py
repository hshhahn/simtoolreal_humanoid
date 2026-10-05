# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
import torch
# isort: on

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import rospy
import tyro
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from termcolor import colored

from deployment.isaac.isaac_env import create_env
from isaacgymenvs.utils.utils import get_repo_root_dir

N_IIWA_JOINTS = 7
N_SHARPA_JOINTS = 22

HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.1
HAND_DOF_SPEED_SCALE = 2.5

PUBLISH_GOAL_OBJECT_POSE = False

T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])

T_R_W = np.linalg.inv(T_W_R)


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacEnvNode:
    def __init__(
        self,
        env: SimToolReal,
        control_dt: float,
        update_and_publish_dt: float,
        device: str,
    ):
        self.env = env
        self.control_dt = control_dt
        self._update_and_publish_dt = update_and_publish_dt
        self._last_update_and_publish_time = time.time()
        self.device = device
        self._init_ros()

    def _init_ros(self):
        rospy.init_node("isaac_env_ros_node")

        self.latest_iiwa_joint_cmd = None
        self.latest_sharpa_joint_cmd = None

        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self._iiwa_joint_cmd_callback, queue_size=1
        )
        self.sharpa_cmd_sub = rospy.Subscriber(
            "/sharpa/joint_cmd",
            JointState,
            self._sharpa_joint_cmd_callback,
            queue_size=1,
        )

        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=1)
        self.sharpa_pub = rospy.Publisher(
            "/sharpa/joint_states", JointState, queue_size=1
        )
        self.object_pose_pub = rospy.Publisher(
            "/robot_frame/current_object_pose", PoseStamped, queue_size=1
        )
        if PUBLISH_GOAL_OBJECT_POSE:
            self.goal_object_pose_pub = rospy.Publisher(
                "/robot_frame/goal_object_pose", Pose, queue_size=1
            )

    def _iiwa_joint_cmd_callback(self, msg: JointState):
        self.latest_iiwa_joint_cmd = np.array(msg.position)

    def _sharpa_joint_cmd_callback(self, msg: JointState):
        self.latest_sharpa_joint_cmd = np.array(msg.position)

    def _publish(self, sim_state: Dict[str, np.ndarray]):
        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
        T_W_O = np.eye(4)
        T_W_O[:3, 3] = object_pos
        T_W_O[:3, :3] = R.from_quat(object_quat_xyzw).as_matrix()
        T_R_O = T_R_W @ T_W_O
        object_pos_R = T_R_O[:3, 3]
        object_quat_xyzw_R = R.from_matrix(T_R_O[:3, :3]).as_quat()

        object_pose_msg = PoseStamped()
        object_pose_msg.header.stamp = rospy.Time.now()
        object_pose_msg.header.frame_id = "robot_frame"
        object_pose_msg.pose.position.x = object_pos_R[0]
        object_pose_msg.pose.position.y = object_pos_R[1]
        object_pose_msg.pose.position.z = object_pos_R[2]
        object_pose_msg.pose.orientation.x = object_quat_xyzw_R[0]
        object_pose_msg.pose.orientation.y = object_quat_xyzw_R[1]
        object_pose_msg.pose.orientation.z = object_quat_xyzw_R[2]
        object_pose_msg.pose.orientation.w = object_quat_xyzw_R[3]
        self.object_pose_pub.publish(object_pose_msg)

        if PUBLISH_GOAL_OBJECT_POSE:
            raise NotImplementedError("Goal object pose not implemented for Isaac")

        joint_positions = sim_state["joint_positions"]
        joint_velocities = sim_state["joint_velocities"]
        joint_names = self.env.joint_names
        iiwa_joint_msg = JointState(
            header=object_pose_msg.header,
            name=joint_names[:N_IIWA_JOINTS],
            position=joint_positions[:N_IIWA_JOINTS],
            velocity=joint_velocities[:N_IIWA_JOINTS],
        )
        self.iiwa_pub.publish(iiwa_joint_msg)

        sharpa_joint_msg = JointState(
            header=object_pose_msg.header,
            name=joint_names[N_IIWA_JOINTS:],
            position=joint_positions[N_IIWA_JOINTS:],
            velocity=joint_velocities[N_IIWA_JOINTS:],
        )
        self.sharpa_pub.publish(sharpa_joint_msg)

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(
            torch.zeros((1, self.env.num_actions), device=self.device)
        )
        return obs["obs"]

    def run(self):
        self.reset()

        first_commands_received = False

        loop_no_sleep_dts, loop_dts = [], []

        joint_cmd = self.env.arm_hand_dof_pos.clone()
        while not rospy.is_shutdown():
            start_loop_no_sleep_time = time.time()

            update_and_publish = False
            if (
                time.time() - self._last_update_and_publish_time
                > self._update_and_publish_dt
            ):
                update_and_publish = True
                self._last_update_and_publish_time = time.time()

            if (
                self.latest_iiwa_joint_cmd is None
                or self.latest_sharpa_joint_cmd is None
            ):
                # Still run loop while waiting to start publishing sim state
                # Print waiting message every 1000 loops
                if len(loop_no_sleep_dts) % 1000 == 0:
                    warn(
                        f"Waiting: latest_iiwa_joint_cmd = {self.latest_iiwa_joint_cmd}, latest_sharpa_joint_cmd = {self.latest_sharpa_joint_cmd}"
                    )
            elif update_and_publish:
                if not first_commands_received:
                    info("=" * 100)
                    info("First commands received, starting to publish sim state")
                    info("=" * 100)
                    first_commands_received = True

                # Get latest joint commands
                iiwa_joint_cmd = self.latest_iiwa_joint_cmd.copy()
                sharpa_joint_cmd = self.latest_sharpa_joint_cmd.copy()
                joint_cmd[:] = (
                    torch.from_numpy(np.concatenate([iiwa_joint_cmd, sharpa_joint_cmd]))
                    .float()
                    .to(self.device)[None]
                )

            # Step simulation
            _, _, _, _ = self.env.step(
                actions=torch.zeros((1, self.env.num_actions), device=self.device),
                joint_pos_targets=joint_cmd,
            )

            # Publish sim state
            if update_and_publish:
                self._publish(self._get_sim_state())

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.control_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                warn(
                    f"Simulation is running slower than real time, desired FPS = {1.0 / self.control_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}"
                )
            loop_dts.append(loop_dt)

            # Get FPS
            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.control_dt)
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

    def _get_sim_state(self) -> Dict[str, np.ndarray]:
        joint_positions = self.env.arm_hand_dof_pos.squeeze(dim=0).cpu().numpy()
        joint_velocities = self.env.arm_hand_dof_vel.squeeze(dim=0).cpu().numpy()
        object_pose = self.env.object_pose.squeeze(dim=0).cpu().numpy()
        object_pos = object_pose[:3]
        object_quat_xyzw = object_pose[3:]
        object_quat_wxyz = object_quat_xyzw[[3, 0, 1, 2]]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "object_pos": object_pos,
            "object_quat_wxyz": object_quat_wxyz,
        }


@dataclass
class IsaacEnvNodeArgs:
    config_path: Path = Path("pretrained_policy/config.yaml")
    """Path to the config YAML."""

    object_category: str = "hammer"
    """Object category (e.g. hammer, marker, spatula)."""

    object_name: str = "claw_hammer"
    """Object name within the category."""

    task_name: str = "swing_down"
    """Task / trajectory name."""

    headless: bool = False
    """Run IsaacGym without rendering."""


def main():
    args: IsaacEnvNodeArgs = tyro.cli(IsaacEnvNodeArgs)

    CONTROL_DT = 1.0 / 60.0
    SUBSTEPS = 2
    assert args.config_path.exists()

    # NOTE: cpu has different physics than training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working

    trajectory_path = (
        get_repo_root_dir()
        / "dextoolbench/trajectories"
        / args.object_category
        / args.object_name
        / f"{args.task_name}.json"
    )
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    Z_OFFSET = 0.03
    traj_data["start_pose"][2] += Z_OFFSET

    env = create_env(
        config_path=str(args.config_path),
        headless=args.headless,
        device=DEVICE,
        overrides={
            # Turn off randomization noise
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.randomizeObjectRotation": False,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.tableResetZRange": 0.0,
            # Object
            "task.env.objectName": args.object_name,
            # Set up environment parameters
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": False,
            # Goal settings
            "task.env.goalObjectPose": traj_data["goals"][0],
            # Sim dt
            "task.sim.dt": CONTROL_DT,
            "task.sim.substeps": SUBSTEPS,
            # Reset
            "task.env.forceNoReset": True,
            # Initialization
            "task.env.useFixedInitObjectPose": True,
            "task.env.objectStartPose": traj_data["start_pose"],
            "task.env.startArmHigher": True,
            # Forces/torques (all zero for eval)
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            "task.env.forceOnlyWhenLifted": True,
            "task.env.torqueOnlyWhenLifted": True,
            "task.env.linVelImpulseOnlyWhenLifted": True,
            "task.env.angVelImpulseOnlyWhenLifted": True,
            "task.env.forceProbRange": [0.0001, 0.0001],
            "task.env.torqueProbRange": [0.0001, 0.0001],
            "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
            "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        },
    )

    isaac_env_node = IsaacEnvNode(
        env=env,
        control_dt=CONTROL_DT,
        update_and_publish_dt=0,  # Make this 0 to update and publish as fast as possible
        device=DEVICE,
    )
    isaac_env_node.run()


if __name__ == "__main__":
    main()
