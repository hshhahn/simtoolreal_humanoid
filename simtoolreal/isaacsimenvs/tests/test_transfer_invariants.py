"""Tier-1 contract test: gym->sim transfer invariants survived the port.

Pins the conventions that make the pretrained Isaac Gym policy transferable
to Isaac Sim, against silent drift (e.g. an Isaac Lab upgrade changing
UrdfConverter or actuator behavior):

  1. Canonical joint order == isaacgymenvs JOINT_NAMES_ISAACGYM (joint
     reordering is the #1 transfer killer).
  2. lab<->canon permutations are mutually inverse bijections.
  3. Per-joint PD stiffness/damping (and hand armature) on the live
     articulation match the canonical tables in scene_utils.
  4. Hand joint limits match the gym-side Q_LOWER/UPPER_LIMITS tables
     (they parameterize action denormalization, not just clamping).
  5. Arm joints after a noise-free reset equal the training default pose.
  6. Object quaternion in the policy obs is xyzw (Isaac Gym convention),
     not Isaac Lab's native wxyz.
  7. USD physics flags: contact_offset=0.002, robot gravity disabled,
     self-collisions disabled.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_transfer_invariants.py \\
      --num_envs 2 --num_assets_per_type 1
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab.utils.math import convert_quat
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils import scene_utils
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES

    # Gym-side ground truth (pure numpy module, importable without isaacgym).
    from isaacgymenvs.utils.observation_action_utils_sharpa import (
        JOINT_NAMES_ISAACGYM,
        Q_LOWER_LIMITS_np,
        Q_UPPER_LIMITS_np,
    )

    # --- 1. Canonical order matches the gym-side policy order (static) ---
    assert tuple(scene_utils.JOINT_NAMES_CANONICAL) == tuple(JOINT_NAMES_ISAACGYM), (
        "JOINT_NAMES_CANONICAL diverged from isaacgymenvs JOINT_NAMES_ISAACGYM:\n"
        f"sim: {scene_utils.JOINT_NAMES_CANONICAL}\n"
        f"gym: {JOINT_NAMES_ISAACGYM}"
    )
    print("[test] 1. canonical joint order == JOINT_NAMES_ISAACGYM (29 joints)")

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    # Noise-free reset so the default arm pose is observable exactly.
    rs = cfg.reset
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    # Clean obs so the quaternion convention check is exact.
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = 0.0

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped

    # --- 2. Permutations are mutually inverse bijections ---
    p_c2l = inner._perm_canon_to_lab.cpu().numpy()
    p_l2c = inner._perm_lab_to_canon.cpu().numpy()
    assert sorted(p_c2l.tolist()) == list(range(29)), "canon_to_lab not a bijection"
    assert sorted(p_l2c.tolist()) == list(range(29)), "lab_to_canon not a bijection"
    assert (p_c2l[p_l2c] == np.arange(29)).all(), "permutations are not inverses"
    lab_names = list(inner.robot.data.joint_names)
    for lab_idx, name in enumerate(lab_names):
        assert scene_utils.JOINT_NAMES_CANONICAL[p_l2c.tolist().index(lab_idx)] == name
    print("[test] 2. lab<->canon permutations consistent with live joint names")

    # --- 3. PD gains / armature on the live articulation ---
    expected_stiffness = {**scene_utils.ARM_JOINT_STIFFNESS, **scene_utils.HAND_JOINT_STIFFNESS}
    expected_damping = {**scene_utils.ARM_JOINT_DAMPING, **scene_utils.HAND_JOINT_DAMPING}
    checked = 0
    for act_name, actuator in inner.robot.actuators.items():
        joint_names = actuator.joint_names
        stiff = actuator.stiffness[0].detach().cpu().numpy()
        damp = actuator.damping[0].detach().cpu().numpy()
        for j, jname in enumerate(joint_names):
            assert abs(stiff[j] - expected_stiffness[jname]) < 1e-3, (
                f"{act_name}/{jname}: stiffness {stiff[j]} != {expected_stiffness[jname]}"
            )
            assert abs(damp[j] - expected_damping[jname]) < 1e-3, (
                f"{act_name}/{jname}: damping {damp[j]} != {expected_damping[jname]}"
            )
            checked += 1
        if act_name == "hand":
            armature = getattr(actuator, "armature", None)
            if armature is not None:
                arm_np = armature[0].detach().cpu().numpy()
                for j, jname in enumerate(joint_names):
                    expected = scene_utils.HAND_JOINT_ARMATURE[jname]
                    assert abs(arm_np[j] - expected) < 1e-5, (
                        f"hand/{jname}: armature {arm_np[j]} != {expected}"
                    )
    assert checked == 29, f"only {checked}/29 joints covered by actuators"
    print("[test] 3. PD gains/armature match canonical tables on all 29 joints")

    # --- 4. Hand joint limits match the gym-side tables ---
    lower_canon = inner._joint_lower_canon.detach().cpu().numpy()
    upper_canon = inner._joint_upper_canon.detach().cpu().numpy()
    np.testing.assert_allclose(
        lower_canon[7:], Q_LOWER_LIMITS_np[7:], atol=1e-4,
        err_msg="hand lower limits diverge from gym Q_LOWER_LIMITS",
    )
    np.testing.assert_allclose(
        upper_canon[7:], Q_UPPER_LIMITS_np[7:], atol=1e-4,
        err_msg="hand upper limits diverge from gym Q_UPPER_LIMITS",
    )
    print("[test] 4. hand joint limits match gym Q_LOWER/UPPER_LIMITS")

    # --- 5. Default arm pose after noise-free reset ---
    obs, _ = env.reset()
    arm_ids = inner._arm_joint_ids
    arm_pos = inner.robot.data.joint_pos[:, arm_ids].detach().cpu().numpy()
    arm_names = [lab_names[i] for i in (arm_ids.tolist() if hasattr(arm_ids, "tolist") else list(arm_ids))]
    expected_pose = np.array([scene_utils.ARM_DEFAULT_JOINT_POS[n] for n in arm_names])
    np.testing.assert_allclose(
        arm_pos, np.tile(expected_pose, (args.num_envs, 1)), atol=1e-3,
        err_msg="arm joints after reset != training default pose",
    )
    print(f"[test] 5. arm reset pose == training default ({expected_pose.round(3).tolist()})")

    # --- 6. Object quaternion in obs is xyzw ---
    obs_list = list(cfg.obs.obs_list)
    offset = sum(OBS_FIELD_SIZES[f] for f in obs_list[: obs_list.index("object_rot")])
    obs_quat = obs["policy"][:, offset : offset + 4]
    expected_xyzw = convert_quat(inner.object.data.root_quat_w, to="xyzw")
    assert torch.allclose(obs_quat, expected_xyzw, atol=1e-5), (
        f"object_rot obs slice is not xyzw:\nobs: {obs_quat[0]}\nexpected: {expected_xyzw[0]}"
    )
    norms = obs_quat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    print("[test] 6. object_rot obs is xyzw (matches convert_quat of wxyz state)")

    # --- 7. USD physics flags on the env_0 robot prim ---
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
    from pxr import Usd, UsdPhysics

    stage = get_current_stage()
    robot_paths = find_matching_prim_paths("/World/envs/env_0/Robot")
    assert robot_paths, "robot prim not found at /World/envs/env_0/Robot"
    robot_prim = stage.GetPrimAtPath(robot_paths[0])

    n_collision = n_rb = n_art = 0
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attr = prim.GetAttribute("physxCollision:contactOffset")
            assert attr and attr.IsValid(), f"no contactOffset on {prim.GetPath()}"
            val = attr.Get()
            assert abs(val - 0.002) < 1e-9, (
                f"{prim.GetPath()}: contactOffset {val} != 0.002"
            )
            n_collision += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            attr = prim.GetAttribute("physxRigidBody:disableGravity")
            assert attr and attr.Get() is True, (
                f"{prim.GetPath()}: robot gravity not disabled"
            )
            n_rb += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            assert attr and attr.Get() is False, (
                f"{prim.GetPath()}: self-collisions not disabled"
            )
            n_art += 1
    assert n_collision > 0 and n_rb > 0 and n_art > 0, (
        f"missing physics prims: collision={n_collision} rb={n_rb} art={n_art}"
    )
    print(
        f"[test] 7. contact_offset=0.002 on {n_collision} collision prims, "
        f"gravity off on {n_rb} bodies, self-collision off on {n_art} articulation root(s)"
    )

    print("[test] transfer invariants test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
