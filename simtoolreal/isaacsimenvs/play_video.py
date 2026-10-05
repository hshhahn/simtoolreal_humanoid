"""Run a trained policy and record an mp4 of the rollout.

Unlike ``train.py --test`` (which calls rl_games' ``player.run()`` and gives us
no hook to capture frames), this script builds a manual rollout loop so we can
``camera.update(dt)`` and append rgb frames each step.

    python isaacsimenvs/play_video.py \
        --checkpoint runs/<experiment>/nn/last_<experiment>_ep_<...>_.pth \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --agent rl_games_cfg_entry_point \
        --num_envs 4 --steps 300

Output: ``isaacsimenvs/videos/<task>_rollout.mp4``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


VIDEO_DIR = Path(__file__).resolve().parent / "videos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to rl_games .pth checkpoint")
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0", help="Gym task id")
    parser.add_argument(
        "--agent",
        default="rl_games_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML (PPO vs SAPG).",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_idx", type=int, default=0, help="Which env the camera follows (0..num_envs-1)")
    parser.add_argument("--steps", type=int, default=300, help="Physics steps to record")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output mp4 path (default: videos/<task>_rollout.mp4)")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (mean)")
    parser.add_argument(
        "--goal_mode",
        default=None,
        help="Override env_cfg.peg_in_hole.goal_mode before instantiation (e.g. 'dense').",
    )
    my_args = parser.parse_args()

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import gymnasium as gym
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env

    # --- Env ---
    # Direct instantiation (not gym.make) — play loop is manual, no gym
    # step/reset semantics needed, and DirectRLEnv exposes .scene / .sim directly.
    env_cfg = load_cfg_from_registry(my_args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = my_args.num_envs
    if my_args.goal_mode is not None and hasattr(env_cfg, "peg_in_hole"):
        env_cfg.peg_in_hole.goal_mode = my_args.goal_mode

    spec = gym.spec(my_args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    env = env_cls(cfg=env_cfg)

    # --- Camera sensor ---
    # Spawn with a placeholder pose; re-aim using env_cfg.record_camera_{eye,target}
    # after the env's initial sim.reset.
    camera_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),  # placeholder
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    env.sim.reset()

    # Aim the camera using the env cfg's recording pose (env-local frame).
    env_origin = env.scene.env_origins[my_args.env_idx]
    eye = env_origin + torch.tensor(env_cfg.record_camera_eye, device=env.device)
    target = env_origin + torch.tensor(env_cfg.record_camera_target, device=env.device)
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    # Step the sim once to flush the pose change through PhysX/Hydra so
    # camera.data.pos_w updates.
    env.sim.step()
    camera.update(0.0)

    print(f"[diag] env_origin = {env_origin.cpu().tolist()}")
    print(f"[diag] camera eye desired = {eye.cpu().tolist()}")
    print(f"[diag] camera target = {target.cpu().tolist()}")
    print(f"[diag] camera pos_w actual = {camera.data.pos_w[0].cpu().tolist()}")
    print(f"[diag] camera quat_w actual = {camera.data.quat_w_world[0].cpu().tolist()}")

    # --- Agent cfg + rl_games wrap ---
    agent_cfg = load_cfg_from_registry(my_args.task, my_args.agent)
    import math

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env,
        rl_device=my_args.rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )

    agent_cfg["params"]["config"]["device"] = my_args.rl_device
    agent_cfg["params"]["config"]["device_name"] = my_args.rl_device

    # --- Player ---
    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(my_args.checkpoint)
    # `has_batch_dimension` is only set inside `player.run()`, which we bypass.
    # Our obs are always batched (num_envs, obs_dim), so set it explicitly.
    player.has_batch_dimension = True
    # Allocate LSTM hidden state (no-op for non-RNN models). Without this,
    # `player.get_action` for an LSTM checkpoint crashes inside
    # `network_builder.py` with `len(states)` on None.
    player.reset()

    # --- Rollout + capture ---
    obs = player.env_reset(wrapped)
    # Use the policy dt (sim_dt × decimation), not the raw physics dt:
    # the rollout loop advances one policy step (= `decimation` physics steps)
    # per `env.step()`. Capturing every Nth physics step would oversample.
    physics_dt = env.sim.get_physics_dt()
    decimation = int(getattr(env.cfg, "decimation", 1) or 1)
    policy_dt = physics_dt * decimation
    capture_every = max(1, round((1.0 / my_args.video_fps) / policy_dt))

    frames = []
    print(f"[play_video] Rolling out {my_args.steps} steps on {my_args.num_envs} envs...", flush=True)
    for step_i in range(my_args.steps):
        action = player.get_action(obs, is_deterministic=my_args.deterministic)
        obs, rew, dones, infos = player.env_step(wrapped, action)

        if step_i % capture_every == 0:
            camera.update(capture_every * policy_dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frame = rgb[0].cpu().numpy()[:, :, :3]
                if not frames:
                    print(
                        f"[diag] first frame: shape={frame.shape} dtype={frame.dtype}"
                        f" min={frame.min()} max={frame.max()} mean={frame.mean():.2f}"
                    )
                    import imageio as _io

                    _debug_png = VIDEO_DIR / "first_frame.png"
                    _debug_png.parent.mkdir(parents=True, exist_ok=True)
                    _io.imwrite(str(_debug_png), frame)
                    print(f"[diag] wrote first-frame png to {_debug_png}")
                frames.append(frame)

    # --- Save ---
    import imageio

    # Strip any "Isaac-" prefix and version suffix for the default filename.
    slug = my_args.task.lower().replace("isaac-", "").rsplit("-v", 1)[0]
    out_path = Path(my_args.out) if my_args.out else VIDEO_DIR / f"{slug}_rollout.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=my_args.video_fps)
    print(f"[play_video] Wrote {len(frames)} frames to {out_path}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
