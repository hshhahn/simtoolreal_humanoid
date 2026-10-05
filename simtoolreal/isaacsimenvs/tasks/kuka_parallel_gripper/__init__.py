"""Fixed-arm parallel-gripper baseline for the simplified marker task."""

from pathlib import Path
import gymnasium as gym

gym.register(
    id="Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0",
    entry_point="isaacsimenvs.tasks.kuka_parallel_gripper.env:KukaParallelGripperEnv",
    order_enforce=False, disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.kuka_parallel_gripper.env_cfg:KukaParallelGripperLiftEnvCfg",
        "rl_games_cfg_entry_point": str(Path(__file__).resolve().parents[2] / "cfg/train/KukaParallelGripperLSTMPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(Path(__file__).resolve().parents[2] / "cfg/train/SimToolRealSAPG.yaml"),
    },
)
