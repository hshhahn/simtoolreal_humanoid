"""Verify the parallel-gripper task, coupled command and success signal on GPU."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--report", type=Path, default=Path("../.logs/kuka-gripper-verification.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    import gymnasium as gym
    import torch
    import isaacsimenvs
    from isaacsimenvs.tasks.kuka_parallel_gripper.env_cfg import KukaParallelGripperLiftEnvCfg
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
    from isaacsimenvs.tasks.kuka_parallel_gripper.env import KukaParallelGripperEnv
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import ARM_DEFAULT_JOINT_POS
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values

    cfg = KukaParallelGripperLiftEnvCfg()
    original_cfg = SimToolRealEnvCfg()
    for section in ("reset", "action", "reward", "termination", "domain_randomization",
                    "sim", "obs", "student_obs", "viewer", "scene"):
        assert getattr(cfg, section).to_dict() == getattr(original_cfg, section).to_dict(), section
    asset_diff = {key for key, value in cfg.assets.to_dict().items()
                  if value != original_cfg.assets.to_dict().get(key)}
    assert asset_diff == {"robot_urdf", "handle_head_types", "num_assets_per_type", "shuffle_assets"}
    assert KukaParallelGripperEnv._get_rewards is SimToolRealEnv._get_rewards
    assert KukaParallelGripperEnv._get_dones is SimToolRealEnv._get_dones
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0", cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset()
    assert raw.robot.num_joints == 9 and cfg.action_space == 8
    assert obs["policy"].shape == (args.num_envs, 71)
    assert obs["critic"].shape == (args.num_envs, 90)
    assert raw._closest_fingertip_dist.shape == (args.num_envs, 2)
    assert len(raw._object_urdf_paths) == 2
    assert raw._action_queue.shape[-1] == 9
    assert abs(raw.step_dt - 1/60) < 1e-8
    expected_arm = torch.tensor(list(ARM_DEFAULT_JOINT_POS.values()), device=raw.device)
    torch.testing.assert_close(raw.robot.data.default_joint_pos[0, raw._arm_joint_ids], expected_arm)
    expected_root = torch.tensor([0., .8, 0.], device=raw.device).expand(args.num_envs, -1)
    torch.testing.assert_close(raw.robot.data.root_pos_w-raw.scene.env_origins, expected_root)
    assert torch.isfinite(raw.robot.data.body_state_w).all()
    pad_positions = raw.robot.data.body_pos_w[:, raw._fingertip_body_ids] - raw.scene.env_origins[:, None]
    # Preserve the original joint-reset noise even when it puts a fingertip
    # below the table. Test the inherited settings rather than changing them.
    # With delay active, target equality must still hold after many commands.
    for step in range(args.steps):
        actions = torch.zeros(args.num_envs, 8, device=raw.device)
        actions[:, 7] = -1 if (step // 60) % 2 else 1
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        targets = raw._cur_targets[:, raw._hand_joint_ids]
        torch.testing.assert_close(targets[:, 0], targets[:, 1], atol=1e-6, rtol=0)
        assert (targets >= 0).all() and (targets <= .050001).all()
    # Goal counter uses actual object/goal geometry. Place one diagnostic
    # object at the goal and check the inherited accumulated-step criterion.
    state = raw.object.data.root_state_w.clone()
    state[0, :7] = raw.goal_viz.data.root_state_w[0, :7]
    state[0, 7:] = 0
    raw.object.write_root_state_to_sim(state[0:1], env_ids=torch.tensor([0], device=raw.device))
    raw.object.update(0.)
    raw._near_goal_steps[0] = 0
    for _ in range(cfg.termination.success_steps):
        compute_intermediate_values(raw)
    assert raw._is_success[0], "The unchanged goal success signal did not fire"
    report = {
        "task": "Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0", "environments": args.num_envs,
        "steps": args.steps, "policy_actions": 8, "physical_joints": 9,
        "actor_observation": 71, "critic_observation": 90, "fingertips": 2,
        "shared_jaw_command": True, "opening_range_m": [0., .1],
        "original_reward_settings": cfg.reward.to_dict(), "success_signal_verified": True,
        "original_reward_and_termination_methods": True,
        "original_configuration_sections_verified": True,
        "initial_pad_positions": pad_positions.cpu().tolist(),
        "start_pose": cfg.reset.fixed_start_pose, "goal_pose": cfg.reset.fixed_goal_pose,
        "table_surface_z_range": [.52, .54], "robot_base": [0., .8, 0.],
        "table_urdf": cfg.assets.table_urdf, "controller_hz": 60,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print("[KUKA/gripper test] PASS", flush=True)
    env.close()
    del app
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
