"""DexToolBench policy evaluation — Isaac Sim (Isaac Lab) backend.

Clone of ``dextoolbench/eval_isaacgym.py`` (Isaac Gym) running on
``Isaacsimenvs-SimToolReal-Direct-v0``. Loads a specific DexToolBench object
URDF + task trajectory, rolls the policy for N episodes, and writes the same
``eval.json`` schema (avg_goal_pct / avg_time_sec / per-episode lists).

Run inside the Isaac Sim venv:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python dextoolbench/eval_isaacsim.py \\
        --object_category hammer \\
        --object_name claw_hammer \\
        --task_name swing_down \\
        --config_path pretrained_policy/config.yaml \\
        --checkpoint_path pretrained_policy/model.pth \\
        --num_episodes 10 \\
        --output_dir evals_isaacsim/hammer/claw_hammer/swing_down
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL_HZ = 60.0

TABLE_URDF = "assets/urdf/table_narrow.urdf"
TABLE_WHITEBOARD_URDF = "assets/urdf/table_narrow_whiteboard.urdf"
TABLE_NAIL_URDF = "assets/urdf/table_narrow_nail.urdf"
TABLE_BOWL_PLATE_URDF = "assets/urdf/table_narrow_bowl_plate.urdf"

OBJECT_CATEGORY_TO_TABLE_URDF = {
    "hammer": TABLE_NAIL_URDF,
    "spatula": TABLE_BOWL_PLATE_URDF,
    "eraser": TABLE_WHITEBOARD_URDF,
    "screwdriver": TABLE_URDF,
    "marker": TABLE_WHITEBOARD_URDF,
    "brush": TABLE_URDF,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_category", required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--config_path", default=str(REPO_ROOT / "pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint_path", default=str(REPO_ROOT / "pretrained_policy/model.pth"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--downsample_factor", type=int, default=1,
                        help="Downsample factor for trajectory goals.")
    parser.add_argument("--z_offset", type=float, default=0.03,
                        help="Z offset added to start pose to avoid the table.")
    parser.add_argument("--force_table_urdf", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Always use the default table URDF regardless of category.")
    parser.add_argument("--rl_device", default="cuda")
    return parser


def _launch_app():
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def _write_isaac_trajectory(goals_xyzw: list[list[float]], out_dir: Path) -> str:
    """Convert gym-format goals ([x,y,z,qx,qy,qz,qw] list) into the
    fixed_trajectory_file format (pos (1,K,3), quat wxyz (1,K,4))."""
    pos = [[g[:3] for g in goals_xyzw]]
    quat_wxyz = [[[g[6], g[3], g[4], g[5]] for g in goals_xyzw]]
    path = out_dir / "trajectory_isaac_format.json"
    with open(path, "w") as f:
        json.dump({"pos": pos, "quat_wxyz": quat_wxyz}, f)
    return str(path)


def main() -> None:
    args = _args

    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    from dextoolbench.objects import NAME_TO_OBJECT

    # --- Object + trajectory ---
    obj = NAME_TO_OBJECT[args.object_name]
    trajectory_path = (
        REPO_ROOT / "dextoolbench/trajectories" / args.object_category
        / args.object_name / f"{args.task_name}.json"
    )
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)
    traj_data["start_pose"][2] += args.z_offset
    traj_data["goals"] = traj_data["goals"][:: args.downsample_factor]
    n_goals = len(traj_data["goals"])

    tmp_dir = Path(tempfile.mkdtemp(prefix="dextoolbench_eval_"))
    traj_file = _write_isaac_trajectory(traj_data["goals"], tmp_dir)

    table_urdf = TABLE_URDF if args.force_table_urdf else (
        OBJECT_CATEGORY_TO_TABLE_URDF[args.object_category]
    )

    # --- Env config (mirrors dextoolbench/eval_isaacgym.py overrides) ---
    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1

    cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
    cfg.assets.object_scale = tuple(obj.scale)
    cfg.assets.table_urdf = table_urdf

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z = 0.38
    rs.table_reset_z_range = 0.0
    rs.start_arm_higher = True
    # gym start_pose is (x,y,z, qx,qy,qz,qw); cfg wants (x,y,z, qw,qx,qy,qz).
    sp = traj_data["start_pose"]
    rs.fixed_start_pose = (sp[0], sp[1], sp[2], sp[6], sp[3], sp[4], sp[5])
    rs.fixed_trajectory_file = traj_file

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

    term = cfg.termination
    term.eval_success_tolerance = 0.01
    term.success_steps = 1
    term.max_consecutive_successes = n_goals

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    n_act = cfg.action_space
    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=n_act,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.rl_device,
        num_envs=1,
    )

    episode_goal_pcts: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(args.num_episodes):
        player.player.init_rnn()
        obs, _ = env.reset()
        # Match gym driver's reset timing: one physics tick before the first
        # policy action (gym's first step doubles as the reset trigger).
        obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))

        step, done, goals_reached = 0, False, 0
        # Loop on the env's own done signal only (matches eval_isaacgym.py and the
        # interactive worker). episode_length_s is a PER-GOAL timeout that the env
        # resets on each goal reached, so a multi-goal trajectory legitimately runs
        # far longer than episode_length_s; a fixed max-step cap here would guillotine
        # slow multi-goal tasks (e.g. flip_over) after the first goal's budget.
        while not done:
            policy_obs = obs["policy"].to(args.rl_device)
            action = player.get_normalized_action(policy_obs, deterministic_actions=True)
            obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
            done = bool(terminated[0].item() or truncated[0].item())
            if done:
                # step() already reset the env: the episode's final count was
                # copied into _prev_episode_successes before _successes zeroed.
                goals_reached = max(goals_reached, int(inner._prev_episode_successes[0].item()))
            else:
                goals_reached = max(goals_reached, int(inner._successes[0].item()))
            step += 1

        goal_pct = 100.0 * goals_reached / n_goals
        episode_goal_pcts.append(goal_pct)
        episode_lengths.append(step)
        print(f"[eval] episode {ep + 1}/{args.num_episodes}: "
              f"{goals_reached}/{n_goals} goals ({goal_pct:.0f}%), "
              f"{step / CONTROL_HZ:.1f}s")

    avg_goal_pct = float(np.mean(episode_goal_pcts))
    avg_time_sec = float(np.mean(episode_lengths) / CONTROL_HZ)
    print(f"[eval] DONE: avg_goal_pct={avg_goal_pct:.1f}, avg_time_sec={avg_time_sec:.1f}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval.json", "w") as f:
            json.dump(
                {
                    "avg_goal_pct": avg_goal_pct,
                    "avg_time_sec": avg_time_sec,
                    "episode_goal_pcts": episode_goal_pcts,
                    "episode_lengths": episode_lengths,
                },
                f,
                indent=4,
            )
        print(f"[eval] wrote {output_dir / 'eval.json'}")

    import os
    import sys
    # Skip Kit teardown (it hangs); os._exit makes cleanup moot.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
