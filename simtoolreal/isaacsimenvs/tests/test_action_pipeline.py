"""Tier-1 contract test: sim action pipeline matches gym compute_joint_pos_targets.

Feeds identical (normalized actions, prev targets) through the Isaac Sim
action pipeline (`apply_action_pipeline`, with action delay disabled) and the
gym-side numpy reference (`compute_joint_pos_targets`), then compares the
resulting joint position targets in canonical order.

Targets are initialized mid-range and actions kept small so no joint-limit
clamp binds — inside the limits the two pipelines must agree exactly; the
limit tables themselves are covered by test_transfer_invariants.py.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_action_pipeline.py \\
      --num_envs 4 --num_assets_per_type 1
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.action_utils import apply_action_pipeline

    from isaacgymenvs.utils.observation_action_utils_sharpa import (
        compute_joint_pos_targets,
    )

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.domain_randomization.use_action_delay = False

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None
    env.reset()

    p_c2l = inner._perm_canon_to_lab
    p_l2c = inner._perm_lab_to_canon

    lower_canon = inner._joint_lower_canon.detach().cpu().numpy()
    upper_canon = inner._joint_upper_canon.detach().cpu().numpy()
    mid_canon = 0.5 * (lower_canon + upper_canon)

    rng = np.random.default_rng(args.seed)
    N = args.num_envs

    prev_canon = np.tile(mid_canon, (N, 1)).astype(np.float32)
    dt = float(inner.step_dt)
    act_cfg = inner.cfg.action

    max_err = 0.0
    for step in range(args.steps):
        actions_canon = rng.uniform(-0.2, 0.2, size=(N, 29)).astype(np.float32)

        # --- sim side ---
        prev_lab = torch.tensor(prev_canon, device=inner.device)[:, p_c2l]
        inner._prev_targets = prev_lab.clone()
        inner._cur_targets = prev_lab.clone()
        apply_action_pipeline(inner, torch.tensor(actions_canon, device=inner.device))
        sim_targets_canon = inner._cur_targets[:, p_l2c].detach().cpu().numpy()

        # --- gym reference (numpy, canonical order) ---
        gym_targets = compute_joint_pos_targets(
            actions=actions_canon.astype(np.float64),
            prev_targets=prev_canon.astype(np.float64),
            hand_moving_average=act_cfg.hand_moving_average,
            arm_moving_average=act_cfg.arm_moving_average,
            hand_dof_speed_scale=act_cfg.dof_speed_scale,
            dt=dt,
        )

        err = np.abs(sim_targets_canon - gym_targets).max()
        max_err = max(max_err, float(err))
        assert err < 1e-4, (
            f"step {step}: max target divergence {err:.2e}\n"
            f"sim: {sim_targets_canon[0].round(4)}\n"
            f"gym: {gym_targets[0].round(4)}"
        )

        # Chain: next step continues from the sim targets (both agreed).
        prev_canon = sim_targets_canon.astype(np.float32)

    print(f"[test] {args.steps} chained steps, max |sim - gym| target error = {max_err:.2e}")
    print("[test] action pipeline test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
