"""Tier-1 contract test: obs/action dimensions match the pretrained policy.

The dims are read from the pretrained checkpoint itself (running_mean_std
shape for obs, mu-head shape for actions) rather than hardcoded, so this test
fails if either the env obs layout or the checkpoint changes incompatibly.
Also asserts observations are finite after reset and after random-action steps.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_obs_action_spec.py \\
      --num_envs 4 --num_assets_per_type 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dims_from_checkpoint(checkpoint_path: str) -> tuple[int, int]:
    """Pull (num_obs, num_actions) out of an rl_games checkpoint."""
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # SAPG checkpoints are rank-keyed: {0: {...}, 1: {...}}; take rank 0.
    if isinstance(state, dict) and 0 in state:
        state = state[0]
    model = state["model"] if "model" in state else state
    num_obs = None
    num_act = None
    for key, tensor in model.items():
        if key.endswith("running_mean_std.running_mean") and "value" not in key:
            assert tensor.dim() == 1, f"{key}: {tensor.shape}"
            num_obs = int(tensor.shape[0])
        if key.endswith("a2c_network.mu.bias"):
            num_act = int(tensor.shape[0])
    assert num_obs is not None, "obs running_mean_std not found in checkpoint"
    assert num_act is not None, "mu head not found in checkpoint"
    return num_obs, num_act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--checkpoint", default=str(REPO_ROOT / "pretrained_policy/model.pth")
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_obs_dim

    expected_obs, expected_act = _dims_from_checkpoint(args.checkpoint)
    print(f"[test] checkpoint expects obs={expected_obs}, actions={expected_act}")

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    assert compute_obs_dim(cfg.obs.obs_list) == expected_obs, (
        f"obs_list sums to {compute_obs_dim(cfg.obs.obs_list)}, "
        f"checkpoint expects {expected_obs}"
    )

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped

    assert inner.cfg.observation_space == expected_obs, (
        f"env observation_space={inner.cfg.observation_space}, expected {expected_obs}"
    )
    assert inner.cfg.action_space == expected_act, (
        f"env action_space={inner.cfg.action_space}, expected {expected_act}"
    )

    obs, _ = env.reset()
    policy_obs = obs["policy"]
    assert policy_obs.shape == (args.num_envs, expected_obs), policy_obs.shape
    assert torch.isfinite(policy_obs).all(), "non-finite obs after reset"
    print(f"[test] reset obs shape {tuple(policy_obs.shape)}, all finite")

    for step in range(args.steps):
        actions = torch.rand(
            (args.num_envs, expected_act), device=inner.device
        ) * 2.0 - 1.0
        obs, rew, terminated, truncated, _ = env.step(actions)
        policy_obs = obs["policy"]
        assert policy_obs.shape == (args.num_envs, expected_obs), policy_obs.shape
        assert torch.isfinite(policy_obs).all(), f"non-finite obs at step {step}"
        assert torch.isfinite(rew).all(), f"non-finite reward at step {step}"
    print(f"[test] {args.steps} random steps: obs/reward finite")

    print("[test] obs/action spec test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
