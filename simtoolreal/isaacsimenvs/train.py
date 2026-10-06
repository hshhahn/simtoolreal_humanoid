"""Train an isaacsimenvs task with our vendored rl_games.

Pipeline:
    argparse (--task, --agent, AppLauncher, wandb/video flags)
        ↓
    AppLauncher (boots Kit; must precede any isaaclab.* import — see CLAUDE.md)
        ↓
    @hydra_task_config_with_yaml  (configclass defaults ← task YAML overlay ← Hydra CLI)
        ↓
    gym.make(task_id, cfg=env_cfg)  → DirectRLEnv wrapped by gym.Wrapper
        ↓
    isaaclab_rl.RlGamesVecEnvWrapper (via register_rlgames_env — clipping,
                                      device bridging, obs-group routing)
        ↓
    rl_games.torch_runner.Runner (PPO / SAPG — both live in ./rl_games/)

CLI shape:
    python isaacsimenvs/train.py \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --agent rl_games_sapg_cfg_entry_point \   # or rl_games_cfg_entry_point
        --headless --capture_viewer \
        --wandb_activate --wandb_project X --wandb_name Y \
        env.scene.num_envs=4096 \
        agent.params.config.max_epochs=200 \
        agent.params.config.minibatch_size=16384 \
        agent.params.seed=42
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train isaacsimenvs task via rl_games.")
    # --- Task/agent selection ---
    parser.add_argument("--task", required=True, help="Gym task id, e.g. Isaacsimenvs-SimToolReal-Direct-v0")
    parser.add_argument(
        "--agent",
        default="rl_games_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML. "
        "Use rl_games_sapg_cfg_entry_point for SAPG.",
    )
    # --- Runtime toggles ---
    parser.add_argument("--test", action="store_true", help="Run inference (player) instead of training")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth to restore")
    parser.add_argument(
        "--checkpoint_load_mode",
        choices=("resume", "weights"),
        default="resume",
        help="resume restores optimizer/rollout/env state; weights starts fresh from model weights.",
    )
    parser.add_argument("--distributed", action="store_true", help="Train one policy across torchrun GPU ranks")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    # --- Video ---
    parser.add_argument("--capture_video", action="store_true", help="Attach recording camera; implies --enable_cameras")
    parser.add_argument("--video_interval", type=int, default=10)
    parser.add_argument("--video_capture_frames", type=int, default=120)
    parser.add_argument("--video_fps", type=int, default=30)
    # --- Pose-only interactive HTML viewer (no cameras / no renderer) ---
    parser.add_argument(
        "--capture_viewer",
        action="store_true",
        help="Write periodic pose-only interactive HTML viewers; does not enable Isaac cameras.",
    )
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--capture_viewer_interval", type=int, default=6000)
    parser.add_argument("--capture_viewer_env_id", type=int, default=0)
    parser.add_argument("--capture_viewer_wandb_key", default="interactive_viewer")
    parser.add_argument(
        "--capture_viewer_github_raw_base",
        default="",
        help="GitHub raw base URL used by the browser to fetch robot URDF meshes.",
    )
    parser.add_argument(
        "--capture_viewer_url_check",
        choices=("skip", "warn", "error"),
        default="skip",
        help="Whether to HEAD-check the robot URDF URL before writing viewer HTML.",
    )
    # --- wandb ---
    parser.add_argument("--wandb_activate", action="store_true")
    parser.add_argument("--wandb_project", default="isaacsimenvs")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="", help="Defaults to agent_cfg.params.config.name")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", default="")
    parser.add_argument("--wandb_logcode_dir", default="")
    # --- AppLauncher flags (--headless, --enable_cameras, etc.) ---
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    global_rank = int(os.environ.get("RANK", "0")) if args_cli.distributed else 0
    if args_cli.distributed:
        if args_cli.test:
            parser.error("--distributed is for training")
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            parser.error("--distributed requires torchrun with at least two GPU ranks")
        local_rank = int(os.environ["LOCAL_RANK"])
        args_cli.sim_device = args_cli.rl_device = args_cli.device = f"cuda:{local_rank}"

    # Recording a video requires cameras even if user forgot --enable_cameras.
    if args_cli.capture_video:
        args_cli.enable_cameras = True

    # Hand the leftover key=value tokens to Hydra via sys.argv.
    sys.argv = [sys.argv[0]] + hydra_args

    app = AppLauncher(args_cli).app

    # 2. Safe to import isaaclab-backed modules now.
    import gymnasium as gym
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from rl_games.torch_runner import Runner

    import isaacsimenvs  # noqa: F401  triggers gym.register side effects
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import (
        EnvStatsAlgoObserver,
        MultiObserver,
        register_rlgames_env,
    )

    @hydra_task_config_with_yaml(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        hydra_run_dir = HydraConfig.get().runtime.output_dir

        # sim_device CLI flag still wins — it's a launcher-level concern, not
        # something we expect in the task YAML.
        env_cfg.sim.device = args_cli.sim_device
        # Keep simulator/object randomization and the policy runner on the same
        # seed. Previously only rl_games saw the agent seed override.
        env_cfg.seed = int(agent_cfg["params"]["seed"]) + global_rank
        if args_cli.distributed:
            agent_cfg["params"]["config"]["multi_gpu"] = True

        # render_mode="rgb_array" makes DirectRLEnv.render() lazily create a
        # single omni.replicator render_product at cfg.viewer.cam_prim_path —
        # one buffer, num_envs-independent. The custom attach_record_camera
        # path we used before created a Camera sensor and called sim.reset()
        # *after* env init, which momentarily doubled the PhysX scene state
        # (~400 GB at 24576 envs) and OOM'd the slurm cgroup. The Lab-
        # canonical pattern is render_mode + gym.wrappers.RecordVideo (see
        # IsaacLab tests/test_record_video.py).
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.capture_video else None,
        )
        if hasattr(env.unwrapped, "configure_agent"):
            env.unwrapped.configure_agent(agent_cfg)

        if args_cli.capture_video:
            from pathlib import Path

            video_folder = str(Path(hydra_run_dir) / "videos")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_capture_frames,
                disable_logger=True,
            )

        if args_cli.capture_viewer:
            from pathlib import Path

            from isaacsimenvs.tasks.simtoolreal.pose_viewer import SimToolRealPoseViewerWrapper

            env = SimToolRealPoseViewerWrapper(
                env,
                output_dir=Path(hydra_run_dir) / "interactive_viewer",
                capture_len=args_cli.capture_viewer_len,
                capture_interval=args_cli.capture_viewer_interval,
                env_id=args_cli.capture_viewer_env_id,
                wandb_key=args_cli.capture_viewer_wandb_key,
                github_raw_base=args_cli.capture_viewer_github_raw_base,
                url_check=args_cli.capture_viewer_url_check,
            )

        # Clip bounds live in the rl_games YAML (params.env.*). Default to
        # +inf if absent so a task without clip YAML just runs unbounded —
        # matches isaacgymenvs's `cfg["env"].get("clipObservations", np.Inf)`.
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        register_rlgames_env(
            env,
            rl_device=args_cli.rl_device,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
        )

        stats_observer = EnvStatsAlgoObserver()
        observers = [stats_observer]
        if args_cli.wandb_activate and global_rank == 0:
            from isaacsimenvs.utils.wandb_utils import WandbAlgoObserver

            # WandbAlgoObserver expects attribute access (cfg.wandb_project,
            # cfg.wandb_notes, …) and also passes cfg to omegaconf_to_dict
            # for wandb.config upload. OmegaConf satisfies both.
            wandb_cfg = OmegaConf.create(
                {
                    "wandb_activate": True,
                    "wandb_project": args_cli.wandb_project,
                    "wandb_group": args_cli.wandb_group,
                    "wandb_entity": args_cli.wandb_entity,
                    "wandb_name": args_cli.wandb_name or agent_cfg["params"]["config"]["name"],
                    "wandb_tags": list(args_cli.wandb_tags),
                    "wandb_notes": args_cli.wandb_notes,
                    "wandb_logcode_dir": args_cli.wandb_logcode_dir,
                }
            )
            observers.append(WandbAlgoObserver(wandb_cfg))

        runner = Runner(MultiObserver(observers))
        # Co-locate rl_games artifacts (checkpoints, summaries) with the Hydra
        # run dir so slurm logs + config + videos all live together.
        agent_cfg["params"]["config"]["train_dir"] = (
            os.environ.get("SIMTOOLREAL_RUN_DIR", hydra_run_dir)
            if args_cli.distributed else hydra_run_dir
        )
        agent_cfg["params"]["config"]["device"] = args_cli.rl_device
        agent_cfg["params"]["config"]["device_name"] = args_cli.rl_device

        runner.load(agent_cfg)
        runner.reset()
        try:
            runner.run(
                {
                    "train": not args_cli.test,
                    "play": args_cli.test,
                    "checkpoint": args_cli.checkpoint,
                    "checkpoint_load_mode": args_cli.checkpoint_load_mode,
                }
            )
            if args_cli.distributed:
                from isaacsimenvs.utils.distributed import verify_training_ranks
                verify_training_ranks(stats_observer.algo, env_cfg.seed, env.unwrapped.device)
        finally:
            # The forced Kit exit below bypasses TensorBoard's atexit cleanup.
            # Close the writer while queued events can still reach shared storage.
            if stats_observer.writer is not None:
                stats_observer.writer.close()

    run()

    if args_cli.distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    # Kit shutdown hangs (per CLAUDE.md + isaacsim_conversion/distill.py).
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
