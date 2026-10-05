"""Lightweight env-creation smoke test for SimToolReal.

Skips Hydra + rl_games + tensorboard. Just spins up AppLauncher, instantiates
the env via `gym.make`, runs a few steps with random actions, and asserts
the per-env Object prim count matches num_envs (the original cloner-drop
guard rail). Useful when iterating on scene_utils.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \\
      --num_envs 8 --num_assets_per_type 2 --steps 10
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaaclab.sim.utils import find_matching_prim_paths
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped

    object_prims = find_matching_prim_paths("/World/envs/env_.*/Object")
    goal_prims = find_matching_prim_paths("/World/envs/env_.*/GoalViz")
    assert len(object_prims) == args.num_envs, (
        f"Object prims: {len(object_prims)}, expected {args.num_envs}"
    )
    assert len(goal_prims) == args.num_envs, (
        f"GoalViz prims: {len(goal_prims)}, expected {args.num_envs}"
    )
    print(
        f"[smoke] OK — {len(object_prims)} Object + {len(goal_prims)} GoalViz prims"
    )

    # Diagnostic: all PhysX joint params that could dampen response —
    # stiffness, damping, armature (virtual inertia), friction, effort limit.
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import JOINT_NAMES_CANONICAL
    stiffness = inner.robot.data.joint_stiffness[0].cpu().numpy()
    damping = inner.robot.data.joint_damping[0].cpu().numpy()
    armature = inner.robot.data.joint_armature[0].cpu().numpy()
    friction = inner.robot.data.joint_friction_coeff[0].cpu().numpy()
    effort_lim = inner.robot.data.joint_effort_limits[0].cpu().numpy()
    perm = inner._perm_lab_to_canon.cpu().numpy()
    # Dump per-link masses / inertias Isaac Lab sees from the URDF.
    # URDF declares thumb chain masses around 1e-3 kg per link — if
    # PhysX 5 shows e.g. 1 kg per link we've found the inertia inflation
    # that explains the 100x slow PD response.
    masses = inner.robot.data.default_mass[0].cpu().numpy()  # (num_bodies,)
    inertias = inner.robot.data.default_inertia[0].cpu().numpy()  # (num_bodies, 9) or similar
    body_names = inner.robot.data.body_names
    print(f"[smoke] Body masses / inertias (all {len(body_names)} bodies):")
    print(f"  {'idx':>3}  {'name':<25s}  {'mass (kg)':>10s}  {'Ixx':>10s}  {'Iyy':>10s}  {'Izz':>10s}")
    for i, name in enumerate(body_names):
        ixx = inertias[i, 0] if inertias.ndim == 2 else inertias[i]
        iyy = inertias[i, 4] if inertias.ndim == 2 and inertias.shape[1] >= 5 else 0
        izz = inertias[i, 8] if inertias.ndim == 2 and inertias.shape[1] >= 9 else 0
        print(f"  [{i:2d}] {name:<25s}  {masses[i]:>10.5f}  {ixx:>10.2e}  {iyy:>10.2e}  {izz:>10.2e}")
    print()
    print(f"[smoke] PhysX joint parameters (canonical order):")
    print(f"  {'idx':>3}  {'name':<22s}  {'K':>8s}  {'D':>8s}  {'armature':>8s}  {'friction':>8s}  {'effort':>8s}")
    for i in range(29):
        lab_j = perm[i]
        print(
            f"  [{i:2d}] {JOINT_NAMES_CANONICAL[i]:<22s}  "
            f"{stiffness[lab_j]:>8.4f}  {damping[lab_j]:>8.4f}  "
            f"{armature[lab_j]:>8.5f}  {friction[lab_j]:>8.5f}  {effort_lim[lab_j]:>8.4f}"
        )

    obs, _ = env.reset()
    print(
        f"[smoke] reset → policy obs {obs['policy'].shape}, "
        f"critic obs {obs['critic'].shape}"
    )

    action_dim = env.unwrapped.action_space.shape[-1] if hasattr(env.unwrapped.action_space, "shape") else cfg.action_space
    for step in range(args.steps):
        action = torch.zeros(
            (args.num_envs, action_dim), device=inner.device, dtype=torch.float32
        )
        obs, reward, terminated, truncated, info = env.step(action)
        any_nan = (
            torch.isnan(obs["policy"]).any().item()
            or torch.isnan(obs["critic"]).any().item()
            or torch.isnan(reward).any().item()
        )
        if any_nan:
            raise RuntimeError(f"step {step}: NaN detected in obs/reward")
    print(f"[smoke] OK — {args.steps} steps, no NaN")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
