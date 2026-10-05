"""Original SimToolReal joint control and rewards with a two-jaw gripper."""

import xml.etree.ElementTree as ET
import math
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import UsdFileCfg

from ..simtoolreal.simtoolreal_env import SimToolRealEnv
from ..simtoolreal.utils.action_utils import apply_action_pipeline, apply_wrench_dr
from ..simtoolreal.utils.obs_utils import compute_obs_dim
from ..simtoolreal.utils.reset_utils import allocate_state_buffers, reset_env_state
from ..simtoolreal.utils.scene_utils import (
    apply_physx_material_properties, ARM_DEFAULT_JOINT_POS,
    ARM_JOINT_STIFFNESS, ARM_JOINT_DAMPING,
)
from .env_cfg import KukaParallelGripperLiftEnvCfg


ARM_NAMES = tuple(f"iiwa14_joint_{i}" for i in range(1, 8))
JAW_NAMES = ("parallel_left_joint", "parallel_right_joint")
JOINT_NAMES = ARM_NAMES + JAW_NAMES
PAD_NAMES = ("parallel_left_finger", "parallel_right_finger")
OBS_SIZES = {"fingertip_pos_rel_palm": 6, "closest_fingertip_dist": 2}


class KukaParallelGripperEnv(SimToolRealEnv):
    cfg: KukaParallelGripperLiftEnvCfg
    palm_center_offset = (0., -.095, .12)  # Pad centre in the original wrist frame.
    fingertip_offset = (0., 0., 0.)
    joint_names_canonical = JOINT_NAMES
    fingertip_link_names = PAD_NAMES

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.action_space != 8:
            raise ValueError("Parallel gripper policy requires seven arm + one opening action")
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list, len(JOINT_NAMES), OBS_SIZES)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list, len(JOINT_NAMES), OBS_SIZES)
        root = ET.parse(cfg.assets.robot_urdf).getroot()
        adjacency = {link.get("name"): [] for link in root.findall("link")}
        for joint in root.findall("joint"):
            parent, child = (joint.find(tag).get("link") for tag in ("parent", "child"))
            adjacency[parent].append(child)
            adjacency[child].append(parent)
        # Fixed palm/pad links merge into wrist/fingers, as in the original
        # importer. Filter the remaining adjacent wrist/finger body pairs.
        for finger in PAD_NAMES:
            adjacency["iiwa14_link_7"].append(finger)
            adjacency[finger].append("iiwa14_link_7")
        self.robot_collision_adjacency = adjacency
        self.pose_viewer_robot_urdf = cfg.assets.robot_urdf
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        apply_physx_material_properties(self)
        allocate_state_buffers(self, joint_names=JOINT_NAMES, arm_joint_names=ARM_NAMES,
                               hand_joint_names=JAW_NAMES, palm_body_name="iiwa14_link_7",
                               fingertip_body_names=PAD_NAMES, num_fingertips=2, action_buffer_size=9)
        if self.robot.num_joints != 9 or self._arm_joint_ids != list(range(7)):
            raise ValueError(f"Unexpected KUKA/gripper joint layout: {self.robot.data.joint_names}")
        print(f"[KUKA/gripper] policy=8, robot=9, actor={cfg.observation_space}, critic={cfg.state_space}; original joint control/rewards", flush=True)

    def build_robot_articulation_cfg(self, usd_path):
        arm_start = dict(ARM_DEFAULT_JOINT_POS)
        if self.cfg.reset.start_arm_higher:
            arm_start["iiwa14_joint_2"] -= math.radians(10.)
            arm_start["iiwa14_joint_4"] += math.radians(10.)
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot", spawn=UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0., .8, 0.), joint_pos={**arm_start, **dict.fromkeys(JAW_NAMES, .05)},
                joint_vel={".*": 0.},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(joint_names_expr=list(ARM_NAMES),
                                           stiffness=ARM_JOINT_STIFFNESS, damping=ARM_JOINT_DAMPING),
                "gripper": ImplicitActuatorCfg(joint_names_expr=list(JAW_NAMES),
                                               stiffness=1500., damping=10., armature=.01,
                                               effort_limit_sim=40., velocity_limit_sim=.2),
            },
        )

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        DirectRLEnv._reset_idx(self, ids)
        reset_env_state(self, ids)
        # Position and velocity resets must also share a single jaw opening.
        pos = self.robot.data.joint_pos[ids].clone()
        vel = self.robot.data.joint_vel[ids].clone()
        pos[:, self._hand_joint_ids] = pos[:, self._hand_joint_ids].mean(dim=1, keepdim=True)
        vel[:, self._hand_joint_ids] = vel[:, self._hand_joint_ids].mean(dim=1, keepdim=True).clamp(-.2, .2)
        self.robot.write_joint_state_to_sim(pos, vel, env_ids=ids)
        self._cur_targets[ids] = pos
        self._prev_targets[ids] = pos

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 8):
            raise ValueError(f"Expected {self.num_envs} x 8 actions, got {actions.shape}")
        # Replicate the last coordinate; all original delay/smoothing math is reused.
        joint_actions = torch.cat((actions[:, :7], actions[:, 7:8].expand(-1, 2)), dim=1)
        apply_action_pipeline(self, joint_actions)
        apply_wrench_dr(self)
