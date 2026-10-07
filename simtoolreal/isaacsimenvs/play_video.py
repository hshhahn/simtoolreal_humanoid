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
import math
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
    parser.add_argument("--steps", type=int, default=300, help="Policy steps to record")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output mp4 path (default: videos/<task>_rollout.mp4)")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--saved_config", type=Path, help="Training Hydra config with the matching agent architecture")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera_eye", type=float, nargs=3)
    parser.add_argument("--camera_target", type=float, nargs=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kit_args", default="")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (mean)")
    parser.add_argument("--sapg_coefficient", type=float, help="Evaluate one SAPG coefficient in all environments")
    parser.add_argument("--all_tools_grid", action="store_true", help="Record one representative environment for each procedural tool family")
    parser.add_argument(
        "--checkpoint_tolerance", type=float, default=None,
        help="Recorded training success tolerance; overrides both replay tolerances when supplied.",
    )

    parser.add_argument(
        "--goal_mode",
        default=None,
        help="Override env_cfg.peg_in_hole.goal_mode before instantiation (e.g. 'dense').",
    )
    my_args = parser.parse_args()
    if my_args.checkpoint_tolerance is not None and (
        not math.isfinite(my_args.checkpoint_tolerance) or my_args.checkpoint_tolerance <= 0
    ):
        parser.error("--checkpoint_tolerance must be finite and positive")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    launcher_args.kit_args = my_args.kit_args
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
    if my_args.saved_config:
        import yaml
        from simtoolreal_shared.action_smoothing import restore_g1_smoothing_config
        from simtoolreal_shared.height_sampling import restore_g1_height_config

        saved_environment = yaml.safe_load(my_args.saved_config.read_text()).get("env", {})
        restore_g1_smoothing_config(env_cfg, saved_environment)
        restore_g1_height_config(env_cfg, saved_environment)
    env_cfg.scene.num_envs = my_args.num_envs
    env_cfg.seed = my_args.seed
    if my_args.checkpoint_tolerance is not None:
        env_cfg.termination.success_tolerance = my_args.checkpoint_tolerance
        env_cfg.termination.eval_success_tolerance = my_args.checkpoint_tolerance
        print(f"[play_video] Recorded goal tolerance: {my_args.checkpoint_tolerance}", flush=True)
    if not 0 <= my_args.env_idx < my_args.num_envs:
        raise ValueError("env_idx must select an existing environment")
    if my_args.goal_mode is not None and hasattr(env_cfg, "peg_in_hole"):
        env_cfg.peg_in_hole.goal_mode = my_args.goal_mode

    spec = gym.spec(my_args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    env = env_cls(cfg=env_cfg)

    slug = my_args.task.lower().replace("isaac-", "").rsplit("-v", 1)[0]
    out_path = Path(my_args.out) if my_args.out else VIDEO_DIR / f"{slug}_rollout.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    view_env_ids = [my_args.env_idx]
    view_labels = ["rollout"]
    if my_args.all_tools_grid:
        families = tuple(env_cfg.assets.handle_head_types)
        selected = {}
        for env_id, asset_id in enumerate(env._object_asset_index_per_env.cpu().tolist()):
            asset = Path(env._object_urdf_paths[asset_id]).name
            for family in families:
                if f"_{family}_handle_" in asset:
                    selected.setdefault(family, (env_id, asset_id))
                    break
        missing = [family for family in families if family not in selected]
        if missing:
            raise ValueError(f"Increase --num_envs to cover missing tool families: {missing}")
        view_labels = list(families)
        view_env_ids = [selected[family][0] for family in families]
        view_metadata = [
            {"family": family, "env_id": selected[family][0],
             "asset_index": selected[family][1],
             "asset_urdf": env._object_urdf_paths[selected[family][1]]}
            for family in families
        ]
        import json
        (out_path.parent / "camera-views.json").write_text(json.dumps(view_metadata, indent=2) + "\n")
        print(f"[play_video] Selected tool views: {view_metadata}", flush=True)

    if my_args.all_tools_grid:
        if env_cfg.reset.fixed_start_pose is not None:
            raise ValueError("Tool-grid evaluation requires randomized initial object poses")
        print(
            "[play_video] Randomized object starts: centre "
            f"{env_cfg.reset.reset_position_center_xy}, XY half-widths "
            f"({env_cfg.reset.reset_position_noise_x}, {env_cfg.reset.reset_position_noise_y}) m",
            flush=True,
        )

    # --- Camera sensor ---
    # Spawn with a placeholder pose; re-aim using env_cfg.record_camera_{eye,target}
    # after the env's initial sim.reset.
    camera_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=my_args.height,
        width=my_args.width,
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
    cameras = [
        Camera(cfg=camera_cfg.replace(prim_path=f"/World/RecordCamera_{index}"))
        for index in range(len(view_env_ids))
    ]
    env.sim.reset()

    # Each recording camera uses the same view relative to its own environment.
    for camera, env_id in zip(cameras, view_env_ids):
        env_origin = env.scene.env_origins[env_id]
        eye = env_origin + torch.tensor(my_args.camera_eye or env_cfg.record_camera_eye, device=env.device)
        target = env_origin + torch.tensor(my_args.camera_target or env_cfg.record_camera_target, device=env.device)
        camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    env.sim.step()
    for camera in cameras:
        camera.update(0.0)
    camera = cameras[0]
    env_origin = env.scene.env_origins[view_env_ids[0]]
    eye = env_origin + torch.tensor(my_args.camera_eye or env_cfg.record_camera_eye, device=env.device)
    target = env_origin + torch.tensor(my_args.camera_target or env_cfg.record_camera_target, device=env.device)

    print(f"[diag] env_origin = {env_origin.cpu().tolist()}")
    print(f"[diag] camera eye desired = {eye.cpu().tolist()}")
    print(f"[diag] camera target = {target.cpu().tolist()}")
    print(f"[diag] camera pos_w actual = {camera.data.pos_w[0].cpu().tolist()}")
    print(f"[diag] camera quat_w actual = {camera.data.quat_w_world[0].cpu().tolist()}")

    # --- Agent cfg + rl_games wrap ---
    if my_args.saved_config:
        import yaml
        agent_cfg = yaml.safe_load(my_args.saved_config.read_text())["agent"]
    else:
        agent_cfg = load_cfg_from_registry(my_args.task, my_args.agent)
    agent_cfg["params"]["config"]["num_actors"] = my_args.num_envs
    agent_cfg["params"]["config"]["multi_gpu"] = False
    if hasattr(env, "configure_agent"):
        env.configure_agent(agent_cfg)

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
    if my_args.sapg_coefficient is not None:
        if player.intr_reward_coef_embd is None or "learn_param" not in player.expl_type:
            raise ValueError("--sapg_coefficient requires the learned SAPG coefficient input")
        player.intr_reward_coef_embd.fill_(my_args.sapg_coefficient)
        print(f"[play_video] SAPG coefficient {my_args.sapg_coefficient} for all {my_args.num_envs} environments", flush=True)

    # --- Rollout + capture ---
    obs = player.env_reset(wrapped)
    starting_poses = []

    def record_starting_poses(policy_step, env_ids):
        for label, env_id in zip(view_labels, view_env_ids):
            if env_id not in env_ids:
                continue
            local_position = env.object.data.root_pos_w[env_id] - env.scene.env_origins[env_id]
            starting_poses.append({
                "family": label, "env_id": env_id, "policy_step": policy_step,
                "position_m": local_position.detach().cpu().tolist(),
                "quaternion_wxyz": env.object.data.root_quat_w[env_id].detach().cpu().tolist(),
            })

    record_starting_poses(0, set(view_env_ids))
    # Use the policy dt (sim_dt × decimation), not the raw physics dt:
    # the rollout loop advances one policy step (= `decimation` physics steps)
    # per `env.step()`. Capturing every Nth physics step would oversample.
    physics_dt = env.sim.get_physics_dt()
    decimation = int(getattr(env.cfg, "decimation", 1) or 1)
    policy_dt = physics_dt * decimation
    capture_every = max(1, round((1.0 / my_args.video_fps) / policy_dt))

    frames = {label: [] for label in view_labels}
    print(f"[play_video] Rolling out {my_args.steps} steps on {my_args.num_envs} envs...", flush=True)
    for step_i in range(my_args.steps):
        action = player.get_action(obs, is_deterministic=my_args.deterministic)
        if step_i % 100 == 0:
            assert torch.isfinite(action).all(), f"Non-finite policy action at step {step_i}"
            print(f"[play_video] step {step_i}/{my_args.steps}", flush=True)
        obs, rew, dones, infos = player.env_step(wrapped, action)
        if bool(dones.any()):
            reset_ids = set(torch.nonzero(dones, as_tuple=False).flatten().cpu().tolist())
            record_starting_poses(step_i + 1, reset_ids)
        if player.is_rnn:
            for state in player.states:
                state[:, dones.to(device=state.device, dtype=torch.bool)] = 0

        if step_i % capture_every == 0:
            for label, camera in zip(view_labels, cameras):
                camera.update(capture_every * policy_dt)
                rgb = camera.data.output["rgb"]
                if rgb is not None and rgb.shape[0] > 0:
                    frame = rgb[0].cpu().numpy()[:, :, :3]
                    if not frames[label]:
                        import imageio as _io
                        debug_png = out_path.parent / f"{label}_first_frame.png"
                        _io.imwrite(str(debug_png), frame)
                        print(f"[diag] {label} first frame: shape={frame.shape} mean={frame.mean():.2f}", flush=True)
                    frames[label].append(frame)

    # --- Save ---
    import json
    (out_path.parent / "starting-poses.json").write_text(
        json.dumps(starting_poses, indent=2) + "\n"
    )
    import imageio
    for label in view_labels:
        clip = out_path if len(view_labels) == 1 else out_path.with_name(f"{out_path.stem}_{label}.mp4")
        imageio.mimwrite(
            str(clip), frames[label], fps=my_args.video_fps,
            codec="libx264", macro_block_size=2,
            output_params=["-threads", "1", "-preset", "fast", "-crf", "18"],
        )
        print(f"[play_video] Wrote {len(frames[label])} frames to {clip}", flush=True)

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
