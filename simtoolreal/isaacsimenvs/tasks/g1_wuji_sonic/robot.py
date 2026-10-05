"""G1 motor settings matching SONIC 1.1 and original Wuji Hand settings."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

from simtoolreal_shared.wuji_synergy import wuji_joint_names


MUJOCO_BODY_NAMES = tuple(
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
# This order must remain independent of the combined URDF parser order.
MUJOCO_TO_SONIC = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
BODY_JOINT_NAMES = tuple(MUJOCO_BODY_NAMES[i] for i in MUJOCO_TO_SONIC)
RIGHT_ARM_NAMES = tuple(
    f"right_{joint}_joint"
    for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )
)
LEFT_HAND_NAMES = wuji_joint_names("left")
RIGHT_HAND_NAMES = wuji_joint_names("right")
ALL_JOINT_NAMES = BODY_JOINT_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES
FINGERTIP_NAMES = tuple(f"right_finger{i}_tip_link" for i in range(1, 6))


def body_defaults() -> dict[str, float]:
    pose = dict.fromkeys(BODY_JOINT_NAMES, 0.0)
    for side, roll in (("left", 0.2), ("right", -0.2)):
        pose.update(
            {
                f"{side}_hip_pitch_joint": -0.312,
                f"{side}_knee_joint": 0.669,
                f"{side}_ankle_pitch_joint": -0.363,
                f"{side}_elbow_joint": 0.6,
                f"{side}_shoulder_pitch_joint": 0.2,
                f"{side}_shoulder_roll_joint": roll,
            }
        )
    return pose


def body_motor_parameters() -> dict[str, dict[str, float]]:
    """SONIC Isaac Lab gains, reflected inertia, torque and speed limits."""
    parameters = {}
    omega = 10.0 * 2.0 * math.pi
    for name in BODY_JOINT_NAMES:
        if "hip_yaw" in name or name == "waist_yaw_joint":
            armature, effort, velocity = 0.010177520, 88.0, 32.0
        elif "hip_" in name or "knee" in name:
            armature, effort, velocity = 0.025101925, 139.0, 20.0
        elif "ankle" in name or name in ("waist_roll_joint", "waist_pitch_joint"):
            armature, effort, velocity = 2.0 * 0.003609725, 50.0, 37.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            armature, effort, velocity = 0.00425, 5.0, 22.0
        else:
            armature, effort, velocity = 0.003609725, 25.0, 37.0
        stiffness = armature * omega**2
        parameters[name] = {
            "stiffness": stiffness,
            "damping": 4.0 * armature * omega,
            "armature": armature,
            "effort_limit_sim": effort,
            "velocity_limit_sim": velocity,
            "action_scale": 0.25 * effort / stiffness,
        }
    return parameters


def collision_adjacency(urdf_path: str) -> dict[str, list[str]]:
    """Filter joints and pairs separated by one intermediate rigid body.

    Closely connected CAD parts overlap at their joint interfaces. Other
    fingers and distant body parts retain physical self-collisions.
    """
    root = ET.parse(urdf_path).getroot()
    neighbors: dict[str, set[str]] = {
        link.get("name"): set() for link in root.findall("link")
    }
    for joint in root.findall("joint"):
        a, b = joint.find("parent").get("link"), joint.find("child").get("link")
        neighbors[a].add(b)
        neighbors[b].add(a)
    return {
        a: sorted((near | set().union(*(neighbors[b] for b in near))) - {a})
        for a, near in neighbors.items()
    }


def build_articulation_cfg(usd_path: str, cfg) -> ArticulationCfg:
    motors = body_motor_parameters()
    hand_params = json.loads(
        (Path(cfg.assets.robot_urdf).parent / "wuji_actuators.json").read_text()
    )
    urdf = ET.parse(cfg.assets.robot_urdf).getroot()
    defaults = body_defaults()
    for joint in urdf.findall("joint"):
        if joint.get("name") in LEFT_HAND_NAMES + RIGHT_HAND_NAMES:
            lower, upper = (
                float(joint.find("limit").get(k)) for k in ("lower", "upper")
            )
            defaults[joint.get("name")] = min(max(0.0, lower), upper)
    fields = (
        "stiffness",
        "damping",
        "armature",
        "effort_limit_sim",
        "velocity_limit_sim",
    )
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=cfg.sonic.base_position,
            rot=cfg.sonic.base_rotation,
            joint_pos=defaults,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "sonic_body": ImplicitActuatorCfg(
                joint_names_expr=list(BODY_JOINT_NAMES),
                **{f: {n: motors[n][f] for n in BODY_JOINT_NAMES} for f in fields},
            ),
            "wuji_hands": ImplicitActuatorCfg(
                joint_names_expr=list(LEFT_HAND_NAMES + RIGHT_HAND_NAMES),
                **{
                    f: {
                        n: hand_params[n][f] for n in LEFT_HAND_NAMES + RIGHT_HAND_NAMES
                    }
                    for f in fields
                },
            ),
        },
    )
