import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tyro


@dataclass
class LaunchTrainingArgs:
    """Launch isaacgymenvs training with configurable parameters."""

    # === Experiment ===
    custom_experiment_name: str = "my_experiment"
    """Custom experiment name (datetime will be appended)."""

    seed: int = 0
    """Random seed."""

    checkpoint: Optional[Path] = None
    """Path to checkpoint .pth file for finetuning. If None, trains from scratch."""

    # === Forces/Torques ===
    force_scale: float = 20
    """Force scale."""

    torque_scale: float = 2.0
    """Torque scale."""

    # === Penalty ===
    object_ang_vel_penalty_scale: float = 0.0
    """Object angular velocity penalty scale."""

    # === SAPG ===
    num_envs: int = 24576
    """Number of environments. Reduce this to 12288 if you run out of GPU memory."""

    num_blocks: int = 6
    """Number of SAPG blocks."""

    # === Wandb ===
    wandb_entity: str = "tylerlum"
    """Wandb entity (user or team)."""

    wandb_project: str = "simtoolreal"
    """Wandb project name."""

    wandb_group: str = f"{datetime.now().strftime('%Y-%m-%d')}"
    """Wandb group name."""

    wandb_activate: bool = True
    """Whether to activate wandb logging."""

    wandb_tags: List[str] = field(default_factory=list)
    """Wandb tags."""

    wandb_notes: str = ""
    """Wandb notes."""

    @property
    def sapg_block_size(self) -> int:
        return self.num_envs // self.num_blocks

    def __post_init__(self) -> None:
        assert self.num_envs % self.num_blocks == 0, "num_envs must be divisible by num_blocks"


def launch_training(args: LaunchTrainingArgs) -> None:
    if args.checkpoint is not None:
        assert args.checkpoint.exists(), f"Checkpoint not found: {args.checkpoint}"

    now = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )  # Add this to avoid overwriting existing experiments
    experiment_name = f"{args.custom_experiment_name}_{now}"
    hydra_run_dir = (
        f"./train_dir/{args.wandb_project}/{args.wandb_group}/{experiment_name}"
    )

    wandb_tags_str = "[" + ",".join(args.wandb_tags) + "]"

    cmd_parts = [
        "python",
        "-m",
        "isaacgymenvs.train",
        "++task.env.useSparseReward=False",
        "headless=True",
        f"task.env.numEnvs={args.num_envs}",
        # === Training ===
        "train.params.config.minibatch_size=98304",
        "multi_gpu=False",
        "train.params.config.good_reset_boundary=0",
        "task.env.goodResetBoundary=0",
        "train.params.config.use_others_experience=lf",
        "train.params.config.off_policy_ratio=1.0",
        "train.params.config.expl_type=mixed_expl_learn_param",
        "train.params.config.expl_reward_type=entropy",
        f"train.params.config.expl_coef_block_size={args.sapg_block_size}",
        "train.params.config.expl_reward_coef_scale=0.002",
        "train.params.network.space.continuous.fixed_sigma=coef_cond",
        # === Wandb ===
        f"wandb_project={args.wandb_project}",
        f"wandb_entity={args.wandb_entity}",
        f"wandb_activate={args.wandb_activate}",
        f"wandb_group={args.wandb_group}",
        f"wandb_tags={wandb_tags_str}",
        f"++wandb_notes='{args.wandb_notes}'",
        # === Seed ===
        f"seed={args.seed}",
        # === Experiment ===
        f"experiment=00_{experiment_name}",
        f"hydra.run.dir={hydra_run_dir}",
        "task=SimToolRealLSTMAsymmetric",
        "task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]",
        "task.env.forceConsecutiveNearGoalSteps=True",
        f"task.env.forceScale={args.force_scale}",
        f"task.env.torqueScale={args.torque_scale}",
        f"task.env.objectAngVelPenaltyScale={args.object_ang_vel_penalty_scale}",
    ]

    if args.checkpoint is not None:
        cmd_parts.append(f"checkpoint={args.checkpoint}")

    cmd = " ".join(cmd_parts)
    print(f"Running command:\n{cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    args: LaunchTrainingArgs = tyro.cli(LaunchTrainingArgs)
    launch_training(args)


if __name__ == "__main__":
    main()
