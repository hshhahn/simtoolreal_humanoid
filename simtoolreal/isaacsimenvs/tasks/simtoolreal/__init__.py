"""SimToolReal task registration.

Registers ``Isaacsimenvs-SimToolReal-Direct-v0`` with the gymnasium registry
for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → SimToolRealEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/SimToolReal.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/SimToolRealPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/SimToolRealSAPG.yaml (default)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .simtoolreal_env import SimToolRealEnv
from .simtoolreal_env_cfg import SimToolRealEnvCfg

__all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-SimToolReal-Direct-v0",
    entry_point="isaacsimenvs.tasks.simtoolreal.simtoolreal_env:SimToolRealEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg:SimToolRealEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "SimToolReal.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)
