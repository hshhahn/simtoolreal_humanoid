"""G1/Wuji/SONIC task registration."""

from pathlib import Path

import gymnasium as gym

from .env import G1WujiSonicEnv
from .env_cfg import G1WujiSonicEnvCfg, G1WujiSonicLiftEnvCfg, G1WujiSonicBimanualEnvCfg

gym.register(
    id="Isaacsimenvs-G1-Wuji-Sonic-v0",
    entry_point="isaacsimenvs.tasks.g1_wuji_sonic.env:G1WujiSonicEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.g1_wuji_sonic.env_cfg:G1WujiSonicEnvCfg",
        "rl_games_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicPPO.yaml"
        ),
        "rl_games_lstm_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicLSTMPPO.yaml"
        ),
        "rl_games_sapg_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicSAPG.yaml"
        ),
    },
)

gym.register(
    id="Isaacsimenvs-G1-Wuji-Sonic-Lift-v0",
    entry_point="isaacsimenvs.tasks.g1_wuji_sonic.env:G1WujiSonicEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.g1_wuji_sonic.env_cfg:G1WujiSonicLiftEnvCfg",
        "rl_games_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicPPO.yaml"
        ),
        "rl_games_lstm_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicLSTMPPO.yaml"
        ),
        "rl_games_sapg_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicSAPG.yaml"
        ),
    },
)

gym.register(
    id="Isaacsimenvs-G1-Wuji-Sonic-Bimanual-v0",
    entry_point="isaacsimenvs.tasks.g1_wuji_sonic.env:G1WujiSonicEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.g1_wuji_sonic.env_cfg:G1WujiSonicBimanualEnvCfg",
        "rl_games_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicPPO.yaml"
        ),
        "rl_games_lstm_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicLSTMPPO.yaml"
        ),
        "rl_games_sapg_cfg_entry_point": str(
            Path(__file__).resolve().parents[2] / "cfg/train/G1WujiSonicSAPG.yaml"
        ),
    },
)

__all__ = [
    "G1WujiSonicEnv", "G1WujiSonicEnvCfg", "G1WujiSonicLiftEnvCfg",
    "G1WujiSonicBimanualEnvCfg",
]
