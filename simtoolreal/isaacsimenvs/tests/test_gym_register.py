"""Smoke test Phase 1: verify gym.register + YAML overlay compose correctly.

Confirms:
  1. `import isaacsimenvs` triggers `gym.register("Isaacsimenvs-SimToolReal-Direct-v0", ...)`.
  2. `load_cfg_from_registry` resolves each entry_point to the expected type.
  3. `env_cfg.from_dict(task_yaml)` applies overrides onto the configclass cleanly
     (the merge the trainer relies on).

Does NOT build the sim — no `gym.make`. Boots a minimal headless AppLauncher
only because isaaclab modules require it at import time.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_gym_register.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # AppLauncher not needed for the test logic — this test is
    # config-registry-only — but isaaclab imports require a booted app.
    from isaaclab.app import AppLauncher
    import argparse

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    app = AppLauncher(args).app

    import gymnasium as gym
    import yaml

    import isaacsimenvs  # noqa: F401  triggers gym.register side effect
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    task_id = "Isaacsimenvs-SimToolReal-Direct-v0"

    # 1. Registration succeeded.
    spec = gym.spec(task_id)
    print(f"[test] gym.spec({task_id!r}) → {spec.entry_point}")
    for key in (
        "env_cfg_entry_point",
        "env_cfg_yaml_entry_point",
        "rl_games_cfg_entry_point",
        "rl_games_sapg_cfg_entry_point",
    ):
        assert key in spec.kwargs, f"missing {key} in gym.register kwargs"
        print(f"  {key}: {spec.kwargs[key]}")

    # 2. env_cfg entry point resolves to the configclass.
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert isinstance(env_cfg, SimToolRealEnvCfg), f"expected SimToolRealEnvCfg, got {type(env_cfg)}"
    print(f"[test] configclass default num_envs = {env_cfg.scene.num_envs}")
    print(f"[test] configclass default sim.dt   = {env_cfg.sim.dt}")

    # 3. rl_games entry point is a YAML → dict.
    agent_cfg = load_cfg_from_registry(task_id, "rl_games_cfg_entry_point")
    assert isinstance(agent_cfg, dict), f"expected dict from rl_games YAML, got {type(agent_cfg)}"
    assert "params" in agent_cfg and "config" in agent_cfg["params"]
    print(f"[test] agent_cfg['params']['config']['name'] = {agent_cfg['params']['config']['name']}")

    # 4. Task YAML overlay applies cleanly onto the configclass.
    yaml_path = Path(spec.kwargs["env_cfg_yaml_entry_point"])
    with open(yaml_path) as f:
        overlay = yaml.safe_load(f) or {}

    # `clip_observations` / `clip_actions` live in the task YAML (read directly
    # by train.py) but are not configclass fields; pop them so the strict
    # `from_dict` doesn't reject them.
    overlay.pop("clip_observations", None)
    overlay.pop("clip_actions", None)

    # Pick a known overlay key and confirm round-trip.
    pre = env_cfg.scene.num_envs
    overlay.setdefault("scene", {})["num_envs"] = 77  # sentinel value unlikely to match default
    env_cfg.from_dict(overlay)
    assert env_cfg.scene.num_envs == 77, f"overlay did not apply: got {env_cfg.scene.num_envs}"
    print(f"[test] overlay applied: scene.num_envs {pre} → {env_cfg.scene.num_envs}")

    print("[test] Phase 1 registration smoke test OK")
    import os
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
