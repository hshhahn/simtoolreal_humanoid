"""Wandb observer for rl_games Runner.

Self-contained port of `isaacgymenvs/utils/wandb_utils.py` — same behavior,
no cross-package imports.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from rl_games.common.algo_observer import AlgoObserver

from isaacsimenvs.utils.reformat import omegaconf_to_dict


def _retry(times: int, exceptions: tuple):
    """Retry decorator: calls `func` up to `times` times on the listed exceptions."""

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f"Exception thrown when running {func}, attempt {attempt}/{times}")
                    time.sleep(min(2 ** attempt, 30))
                    attempt += 1
            return func(*args, **kwargs)

        return newfn

    return decorator


class WandbAlgoObserver(AlgoObserver):
    """Initialize wandb before rl_games' summary writer so sync_tensorboard works."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        import wandb

        # rl_games' `experiment_name` is derived from `agent_cfg.params.config.name`
        # (yaml-side, pinned and prefixed with `0_` to satisfy its policy_idx parsing).
        # The CLI `--wandb_name` value lives on `self.cfg.wandb_name` — prefer that
        # if set so per-run sub-file `EXPERIMENT_NAME`s actually show up in wandb.
        chosen = getattr(self.cfg, "wandb_name", "") or experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_unique_id = f"{chosen}_{timestamp}"
        display_name = f"{chosen}_{timestamp}"
        print(f"[Wandb] unique id: {wandb_unique_id}")

        cfg = self.cfg
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # Default log_code scope is the isaacsimenvs package; the full repo is
        # 19 GB and `wandb.run.log_code` walks every file (slow init). Override
        # via `wandb_logcode_dir=<path>` to point elsewhere, or `=''` unchanged.
        default_logcode_dir = os.path.join(repo_root, "isaacsimenvs")
        logcode_dir = cfg.wandb_logcode_dir if cfg.wandb_logcode_dir else default_logcode_dir

        @_retry(3, (Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                notes=cfg.wandb_notes if hasattr(cfg, "wandb_notes") else "",
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=display_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )
            wandb.run.log_code(root=logcode_dir)
            print(f"[Wandb] run dir: {wandb.run.dir} (log_code root: {logcode_dir})")

        print("[Wandb] initializing...")
        try:
            init_wandb()
            wandb.define_metric("*", step_metric="global_step")
        except Exception as exc:
            print(f"[Wandb] init failed: {exc}")

        if wandb.run is None:
            print("[Wandb] run is None — skipping diff + config upload.")
            return

        # Capture a git diff so the run is reproducible from HEAD.
        diff_path = os.path.join(wandb.run.dir, "diff.patch")
        with open(diff_path, "w") as f:
            os.system(f"cd {repo_root} && git diff > {f.name}")
        diff_artifact = wandb.Artifact("diff", type="file", description="Git diff")
        diff_artifact.add_file(diff_path)
        wandb.run.log_artifact(diff_artifact)

        cfg_dict = self.cfg if isinstance(self.cfg, dict) else omegaconf_to_dict(self.cfg)
        wandb.config.update(cfg_dict, allow_val_change=True)
