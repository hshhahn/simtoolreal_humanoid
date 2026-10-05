"""Batch DexToolBench evaluation over all 24 combinations — Isaac Sim backend.

Clone of ``run_all_evals_isaacgym.py`` driving ``eval_isaacsim.py``. Each combination
runs in a fresh subprocess (Kit cannot cleanly tear down within a process,
same constraint as Isaac Gym). Run from the repo root:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python dextoolbench/run_all_evals_isaacsim.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from termcolor import colored
from tqdm import tqdm


def log_info(text):
    print(colored(text, "cyan"))


script_path = Path(__file__).parent / "eval_isaacsim.py"
assert script_path.exists(), f"Script not found: {script_path}"
DATE_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

object_category_to_object_names = {
    "hammer": ["claw_hammer", "mallet_hammer"],
    "spatula": ["flat_spatula", "spoon_spatula"],
    "eraser": ["flat_eraser", "handle_eraser"],
    "screwdriver": ["long_screwdriver", "short_screwdriver"],
    "marker": ["sharpie_marker", "staples_marker"],
    "brush": ["blue_brush", "red_brush"],
}

object_category_to_task_names = {
    "hammer": ["swing_down", "swing_side"],
    "spatula": ["serve_plate", "flip_over"],
    "eraser": ["wipe_smile", "wipe_c"],
    "screwdriver": ["spin_vertical", "spin_horizontal"],
    "marker": ["draw_smile", "write_c"],
    "brush": ["sweep_forward", "sweep_right"],
}

POLICY_NAME_TO_PATH = {
    "pretrained_policy": Path("pretrained_policy"),
}
DOWNSAMPLE_FACTOR = 1
NUM_EPISODES = 10

for policy_path in POLICY_NAME_TO_PATH.values():
    assert policy_path.exists(), f"Policy path not found: {policy_path}"

ALL_COMBINATIONS = []
for object_category, object_names in object_category_to_object_names.items():
    task_names = object_category_to_task_names[object_category]
    for object_name in object_names:
        for task_name in task_names:
            for policy_name in POLICY_NAME_TO_PATH:
                ALL_COMBINATIONS.append(
                    (object_category, object_name, task_name, policy_name)
                )

trajectories_dir = Path(__file__).parent / "trajectories"
for object_category, object_name, task_name, _ in ALL_COMBINATIONS:
    trajectory_path = (
        trajectories_dir / object_category / object_name / f"{task_name}.json"
    )
    assert trajectory_path.exists(), f"Trajectory path not found: {trajectory_path}"

print(
    f"Will evaluate {len(ALL_COMBINATIONS)} combinations for {NUM_EPISODES} episodes each"
)

total = len(ALL_COMBINATIONS)
for i, (object_category, object_name, task_name, policy_name) in tqdm(
    enumerate(ALL_COMBINATIONS), desc="Running evaluations", total=total
):
    start_time = time.time()
    log_info(
        f"{i}/{total} Running evaluation for {object_category} {object_name} {task_name} {policy_name}"
    )
    output_dir = Path(
        f"evals_isaacsim/{DATE_STR}/{object_category}/{object_name}/{task_name}/{policy_name}"
    )
    policy_path = POLICY_NAME_TO_PATH[policy_name]
    checkpoint_path = policy_path / "model.pth"
    config_path = policy_path / "config.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    # Fresh subprocess per combination: Kit does not tear down cleanly in-process.
    cmd = (
        f"{sys.executable} {script_path} "
        f"--object_category {object_category} "
        f"--object_name {object_name} "
        f"--task_name {task_name} "
        f"--checkpoint_path {checkpoint_path} "
        f"--config_path {config_path} "
        f"--output_dir {output_dir} "
        f"--num_episodes {NUM_EPISODES} "
        f"--downsample_factor {DOWNSAMPLE_FACTOR}"
    )
    log_info(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    log_info(f"{i}/{total} Done")
    log_info(f"Time taken for evaluation: {time.time() - start_time:.2f} seconds")
