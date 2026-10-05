"""Tier-2 integration test: the pretrained gym policy rolls out in Isaac Sim.

Loads pretrained_policy/{config.yaml,model.pth} via deployment.rl_player
.RlPlayer and runs a deterministic rollout with DR / reset noise disabled
(mirrors the distributional-eval setup in debug_differences). Asserts:

  - observations and actions stay finite for the whole rollout
  - the policy reaches at least one goal (nonzero success hits)
  - at least one env lifts the object

This is the cheap gate before the full gym-vs-sim distributional parity run.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_pretrained_rollout.py \\
      --num_envs 8 --num_assets_per_type 2 --num_steps 600
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=600)
    parser.add_argument("--config", default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument("--success_tolerance", type=float, default=0.01)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    # All DR / reset noise off (mirrors debug_differences/policy_eval_isaacsim.py).
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    rs.fixed_start_pose = (
        0.0, 0.0, rs.table_reset_z + rs.table_object_z_offset, 1.0, 0.0, 0.0, 0.0,
    )

    if args.success_tolerance >= 0:
        cfg.termination.eval_success_tolerance = float(args.success_tolerance)

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    n_act = cfg.action_space
    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=n_act,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.rl_device,
        num_envs=args.num_envs,
    )
    player.player.init_rnn()

    obs, _ = env.reset()
    # Match gym driver's reset timing: advance one physics tick before the
    # first policy action (gym's first env.step(zeros) is the reset trigger).
    obs, _, _, _, _ = env.step(torch.zeros((args.num_envs, n_act), device=inner.device))

    total_hits = 0
    ever_lifted = torch.zeros(args.num_envs, dtype=torch.bool, device=inner.device)
    reward_sum = 0.0

    for step in range(args.num_steps):
        policy_obs = obs["policy"].to(args.rl_device)
        assert torch.isfinite(policy_obs).all(), f"non-finite obs at step {step}"

        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        assert torch.isfinite(action).all(), f"non-finite action at step {step}"

        obs, rew, terminated, truncated, _ = env.step(action.to(inner.device))
        assert torch.isfinite(rew).all(), f"non-finite reward at step {step}"

        total_hits += int(inner._is_success.sum().item())
        ever_lifted |= inner._lifted_object.bool()
        reward_sum += float(rew.sum().item())

        if (step + 1) % 100 == 0:
            print(
                f"[test] step {step + 1}/{args.num_steps}: "
                f"hits={total_hits}, lifted={int(ever_lifted.sum())}/{args.num_envs}, "
                f"mean_reward={reward_sum / ((step + 1) * args.num_envs):.3f}"
            )

    mean_reward = reward_sum / (args.num_steps * args.num_envs)
    n_lifted = int(ever_lifted.sum())
    print(
        f"[test] rollout done: total goal hits={total_hits}, "
        f"envs ever lifted={n_lifted}/{args.num_envs}, mean reward/step={mean_reward:.3f}"
    )

    assert total_hits > 0, "pretrained policy never reached a goal in Isaac Sim"
    assert n_lifted > 0, "pretrained policy never lifted the object in Isaac Sim"

    print("[test] pretrained rollout test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
