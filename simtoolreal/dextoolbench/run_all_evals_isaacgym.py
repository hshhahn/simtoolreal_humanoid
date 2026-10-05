import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from termcolor import colored
from tqdm import tqdm


def log_info(text):
    print(colored(text, "cyan"))


script_path = Path(__file__).parent / "eval_isaacgym.py"
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

# Validate everything
for policy_path in POLICY_NAME_TO_PATH.values():
    assert policy_path.exists(), f"Policy path not found: {policy_path}"

# Make in one big list
ALL_OBJECT_CATEGORY_OBJECT_NAME_TASK_NAME_POLICY_NAME = []
for object_category in object_category_to_object_names.keys():
    object_names = object_category_to_object_names[object_category]
    task_names = object_category_to_task_names[object_category]
    for object_name in object_names:
        for task_name in task_names:
            for policy_name, policy_path in POLICY_NAME_TO_PATH.items():
                ALL_OBJECT_CATEGORY_OBJECT_NAME_TASK_NAME_POLICY_NAME.append(
                    (object_category, object_name, task_name, policy_name)
                )

# Make sure all trajectories exist
trajectories_dir = Path(__file__).parent / "trajectories"
for (
    object_category,
    object_name,
    task_name,
    _,
) in ALL_OBJECT_CATEGORY_OBJECT_NAME_TASK_NAME_POLICY_NAME:
    trajectory_path = (
        trajectories_dir / object_category / object_name / f"{task_name}.json"
    )
    assert trajectory_path.exists(), f"Trajectory path not found: {trajectory_path}"

print(
    f"Will evaluate {len(ALL_OBJECT_CATEGORY_OBJECT_NAME_TASK_NAME_POLICY_NAME)} combinations for {NUM_EPISODES} episodes each"
)

"""
Making output_directory structure like
evals/<datetime>
|--<object_category>
|   |--<object_name>
|   |   |--<task_name>
|   |   |   |--<policy_name>
|   |   |   |   |--<eval.json>
"""

total = len(ALL_OBJECT_CATEGORY_OBJECT_NAME_TASK_NAME_POLICY_NAME)
for i, (object_category, object_name, task_name, policy_name) in tqdm(
    enumerate(ALL_OBJECT_CATEGORY_OBJECT_NAME_TASK_NAME_POLICY_NAME),
    desc="Running evaluations",
    total=total,
):
    import time

    start_time = time.time()
    log_info(
        f"{i}/{total} Running evaluation for {object_category} {object_name} {task_name} {policy_name}"
    )
    output_dir = Path(
        f"evals/{DATE_STR}/{object_category}/{object_name}/{task_name}/{policy_name}"
    )
    policy_path = POLICY_NAME_TO_PATH[policy_name]
    policy_stem = policy_path.name
    checkpoint_path = policy_path / "model.pth"
    config_path = policy_path / "config.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    # Need to run eval script in separate subprocesses because isaac doesn't clean up the environment properly
    cmd = f"python \
        {script_path} \
        --object_category {object_category} \
        --object_name {object_name} \
        --task_name {task_name} \
        --checkpoint_path {checkpoint_path} \
        --config_path {config_path} \
        --output_dir {output_dir} \
        --num_episodes {NUM_EPISODES} \
        --downsample_factor {DOWNSAMPLE_FACTOR} \
        --policy_name {policy_name}"
    log_info(f"Running command: {cmd}")
    MODE: Literal["subprocess", "os"] = "subprocess"

    if MODE == "subprocess":
        # Do NOT use check=True: Isaac Gym reliably segfaults (exit 139) during
        # process teardown AFTER eval.json is written. Judge success by the
        # output file, not the exit code.
        result = subprocess.run(cmd, shell=True)
        eval_json = output_dir / "eval.json"
        if not eval_json.exists():
            raise RuntimeError(
                f"{object_category}/{object_name}/{task_name}: no eval.json "
                f"(exit {result.returncode})"
            )
        if result.returncode not in (0, 139, -11):
            log_info(f"  (eval.json written; ignoring teardown exit {result.returncode})")
    elif MODE == "os":
        os.system(cmd)
    else:
        raise ValueError(f"Invalid mode: {MODE}")
    log_info(f"{i}/{total} Done")
    end_time = time.time()
    log_info(f"Time taken for evaluation: {end_time - start_time:.2f} seconds")
