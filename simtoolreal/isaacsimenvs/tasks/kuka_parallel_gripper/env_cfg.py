"""Original fixed-KUKA scene with a parallel gripper and two marker instances."""

from pathlib import Path
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import AssetsCfg, SimToolRealEnvCfg


REPO = Path(__file__).resolve().parents[3]


@configclass
class ParallelGripperAssetsCfg(AssetsCfg):
    robot_urdf: str = str(REPO / "assets/urdf/kuka_parallel_gripper/kuka_parallel_gripper.urdf")
    handle_head_types: tuple[str, ...] = ("marker",)
    num_assets_per_type: int = 2
    shuffle_assets: bool = False


@configclass
class KukaParallelGripperLiftEnvCfg(SimToolRealEnvCfg):
    action_space: int = 8  # Original 7 arm velocity commands + shared jaw opening.
    observation_space: int = 71
    state_space: int = 90
    assets: ParallelGripperAssetsCfg = ParallelGripperAssetsCfg()
    # The two-marker asset pool is the only task simplification. All table,
    # reset/goal, reward, physics, DR, observation-layout and termination
    # settings inherit the original fixed-arm SimToolReal defaults.
