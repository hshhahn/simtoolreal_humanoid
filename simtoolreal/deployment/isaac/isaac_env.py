# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
import torch
# isort: on

from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from deployment.rl_player_utils import (
    read_cfg_omegaconf,
)
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict


def create_env(
    config_path: str,
    device: str,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    merge_with_default_config: bool = True,
    episode_length: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SimToolReal:
    cfg = read_cfg_omegaconf(config_path=config_path, device=device)

    if merge_with_default_config:
        cfg = merge_cfg_with_default_config(cfg)
    return create_env_from_cfg(
        cfg=cfg,
        headless=headless,
        enable_viewer_sync_at_start=enable_viewer_sync_at_start,
        episode_length=episode_length,
        overrides=overrides,
    )


def merge_cfg_with_default_config(cfg: DictConfig) -> DictConfig:
    # Use this if the config from config path is missing fields
    # For example, say we recently added a new field "object_friction" to the config
    # If this wasn't in the config file, this would normally fail
    # Merging with the default config will add this field with the default value
    print("Merging with default config")

    # Should be path of the isaacgymenvs/cfg directory relative to this file's directory
    with initialize(version_base="1.1", config_path="../../isaacgymenvs/cfg"):
        init_cfg = compose(config_name="config", overrides=["task=SimToolRealLSTM"])

    # Disable struct mode to allow merging
    OmegaConf.set_struct(init_cfg, False)
    OmegaConf.set_struct(cfg, False)

    # Put cfg second to override init_cfg
    merged_cfg = OmegaConf.merge(init_cfg, cfg)
    assert isinstance(merged_cfg, DictConfig), (
        f"Expected DictConfig, got {type(merged_cfg)}"
    )

    # Print the differences
    diff = recursive_diff(
        OmegaConf.to_container(cfg, resolve=True),
        OmegaConf.to_container(merged_cfg, resolve=True),
    )
    print("Changes:")
    print("-" * 80)
    for key, change in diff.items():
        print(f"{key}: {change}")

    return merged_cfg


def create_env_from_cfg(
    cfg: DictConfig,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    episode_length: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SimToolReal:
    # Modify the config
    cfg.headless = headless
    cfg.task.sim.enable_viewer_sync_at_start = enable_viewer_sync_at_start
    cfg.task.env.numEnvs = 1
    if episode_length is not None:
        cfg.task.env.episodeLength = episode_length

    # HACK: Assume that graphics_device_id should be 0
    # This is a pretty reasonable assumption because we are typically doing this testing on a workstation with 1 GPU
    cfg.graphics_device_id = 0

    # Modify the config for the task
    if overrides is not None:
        print("-" * 80)
        print("Overriding config")
        print("-" * 80)
        print()

        # Example: overrides = {"task.env.asset.robot": "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"}
        # Note: We first check if the key exists and error out if it doesn't
        # Otherwise, we may silently update a key that doesn't exist and we won't know it
        for key, value in overrides.items():
            if isinstance(value, str):
                value = f'"{value}"'
            else:
                value = str(value)

            # Print the current value of the key
            # This makes sure we are modifying the correct key
            current_value_eval_str = f"cfg.{key}"
            print(
                f"Current value of {current_value_eval_str}: {eval(current_value_eval_str)}"
            )

            # Update the value of the key
            update_value_eval_str = f"cfg.{key} = {value}"
            print(f"Evaluating: {update_value_eval_str}")
            exec(update_value_eval_str)
            print()

    print_dict(omegaconf_to_dict(cfg))

    env = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=True,
    )
    return env


def recursive_diff(cfg1: dict, cfg2: dict, path: str = "") -> dict:
    """Recursively compare two DictConfigs and return differences."""
    differences = {}

    # Get the keys from both configs
    keys1 = set(cfg1.keys()) if isinstance(cfg1, dict) else set()
    keys2 = set(cfg2.keys()) if isinstance(cfg2, dict) else set()

    # Check for keys that are only in cfg1
    for key in keys1 - keys2:
        differences[f"{path}.{key}".lstrip(".")] = f"{cfg1[key]} -> None"

    # Check for keys that are only in cfg2
    for key in keys2 - keys1:
        differences[f"{path}.{key}".lstrip(".")] = f"None -> {cfg2[key]}"

    # Check for keys that are in both configs
    for key in keys1 & keys2:
        val1 = cfg1[key]
        val2 = cfg2[key]

        # Recursively compare dictionaries or lists
        if isinstance(val1, dict) and isinstance(val2, dict):
            diff = recursive_diff(val1, val2, path=f"{path}.{key}".lstrip("."))
            differences.update(diff)
        elif val1 != val2:
            # If values differ, record the difference
            differences[f"{path}.{key}".lstrip(".")] = f"{val1} -> {val2}"

    return differences


def main() -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working
    CONFIG_PATH = Path("pretrained_policy/config.yaml")
    assert Path(CONFIG_PATH).exists()

    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=DEVICE,
    )

    print(env)
    obs = env.reset()
    N_STEPS = 1000
    for _ in range(N_STEPS):
        action = torch.rand(
            (env.num_envs, env.num_acts), device=DEVICE, dtype=torch.float
        )
        obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
