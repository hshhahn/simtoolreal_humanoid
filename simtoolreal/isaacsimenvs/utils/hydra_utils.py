"""Hydra decorator variant: configclass defaults + YAML overlay + CLI overrides.

Isaac Lab's stock `hydra_task_config` uses `configclass defaults` as the sole
source; task-level YAML presets are not supported. We want the isaacgymenvs
composition pattern — a task YAML file sits between the configclass defaults
and the Hydra CLI — so this decorator adds that middle layer.

Precedence (lowest → highest):
    1. CartpoleEnvCfg() defaults (Python)
    2. cfg/task/Cartpole.yaml overlay (file — optional, via new gym kwarg
       `env_cfg_yaml_entry_point`)
    3. Hydra CLI overrides (e.g. `env.scene.num_envs=4096`)

Implementation mirrors `isaaclab_tasks.utils.hydra.register_task_to_hydra` /
`hydra_task_config`, with one inserted `env_cfg.from_dict(yaml_overlay)` step
before the ConfigStore write. The wrapped function receives `(env_cfg, agent_cfg)`
exactly as Isaac Lab's own decorator does.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import gymnasium as gym
import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from isaaclab.envs.utils.spaces import (
    replace_env_cfg_spaces_with_strings,
    replace_strings_with_env_cfg_spaces,
)
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def hydra_task_config_with_yaml(
    task_name: str,
    agent_cfg_entry_point: str,
    yaml_entry_point: str = "env_cfg_yaml_entry_point",
) -> Callable:
    """Like `isaaclab_tasks.utils.hydra.hydra_task_config`, but layers a task YAML
    overlay on top of the configclass defaults before Hydra CLI merging.

    Args:
        task_name: Gym task id (e.g. ``"Isaacsimenvs-Cartpole-Direct-v0"``). A colon-separated
            prefix is stripped to match Isaac Lab's convention.
        agent_cfg_entry_point: Key in ``gym.register(..., kwargs=...)`` whose value
            is the agent (rl_games) config entry point. Typically
            ``"rl_games_cfg_entry_point"`` or ``"rl_games_sapg_cfg_entry_point"``.
        yaml_entry_point: Key in the task's gym kwargs pointing to the task YAML
            overlay. If absent from the registered kwargs, no overlay is applied
            and behavior matches Isaac Lab's stock decorator.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            short_name = task_name.split(":")[-1]

            env_cfg = load_cfg_from_registry(short_name, "env_cfg_entry_point")
            agent_cfg = (
                load_cfg_from_registry(short_name, agent_cfg_entry_point)
                if agent_cfg_entry_point
                else None
            )
            # Overlay the task YAML onto the configclass defaults (if registered).
            # `gym.spec(...).kwargs` is where `gym.register(kwargs=...)` puts them.
            yaml_path = gym.spec(short_name).kwargs.get(yaml_entry_point)
            if yaml_path:
                with open(yaml_path) as f:
                    overlay = yaml.safe_load(f) or {}
                # `configclass.from_dict` walks the dict and applies it onto the
                # typed configclass, recursing into nested configclasses — exactly
                # what our hand-rolled `apply_task_overrides` walker does, but
                # upstream-supported.
                env_cfg.from_dict(overlay)

            env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
            env_cfg_dict = env_cfg.to_dict()
            if isinstance(agent_cfg, dict) or agent_cfg is None:
                agent_cfg_dict = agent_cfg
            else:
                agent_cfg_dict = agent_cfg.to_dict()
            cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
            cfg_dict = replace_slices_with_strings(cfg_dict)
            ConfigStore.instance().store(name=short_name, node=cfg_dict)

            @hydra.main(config_path=None, config_name=short_name, version_base="1.3")
            def hydra_main(
                hydra_cfg: DictConfig,
                env_cfg=env_cfg,
                agent_cfg=agent_cfg,
            ):
                resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
                resolved = replace_strings_with_slices(resolved)
                env_cfg.from_dict(resolved["env"])
                env_cfg_restored = replace_strings_with_env_cfg_spaces(env_cfg)
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    merged_agent = resolved["agent"]
                else:
                    agent_cfg.from_dict(resolved["agent"])
                    merged_agent = agent_cfg
                func(env_cfg_restored, merged_agent, *args, **kwargs)

            hydra_main()

        return wrapper

    return decorator
