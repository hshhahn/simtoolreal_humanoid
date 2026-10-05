"""Process DexToolBench poses.json files into trajectory JSONs.

For each task, reads the raw poses.json (robot-frame), converts to world frame,
filters poses to start after z >= min_z, downsamples, and writes a single
output JSON to dextoolbench/trajectories/<object_category>/<object_name>/<task_name>.json
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import tyro

from dextoolbench.metadata import DEXTOOLBENCH_DATA_STRUCTURE


@dataclass
class ProcessPoseArgs:
    """Process DexToolBench poses into trajectory JSONs."""

    dextoolbench_data_dir: Path = Path(__file__).parent / "data"
    """Source directory containing the raw data (poses.json files)."""

    destination_dir: Path = Path(__file__).parent / "trajectories"
    """Output directory for processed trajectory JSONs."""

    min_z: float = 0.65
    """Discard poses before the first one with z >= this value."""

    downsample_factor: int = 10
    """Keep every Nth pose after the min_z filter."""


def process_poses(
    raw_poses: List[List[float]],
    min_z: float,
    downsample_factor: int,
) -> dict:
    """Convert robot-frame poses to world-frame, filter by min_z, and downsample.

    Args:
        raw_poses: List of [x, y, z, qx, qy, qz, qw] in robot frame.
        min_z: Minimum z value to start the trajectory.
        downsample_factor: Keep every Nth pose.

    Returns:
        Dict with "start_pose" and "goals".
    """
    # Convert robot frame -> world frame (y += 0.8)
    world_frame_poses = [
        [x, y + 0.8, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in raw_poses
    ]

    # Find first index where z >= min_z
    first_valid_idx = None
    for idx, pose in enumerate(world_frame_poses):
        if pose[2] >= min_z:
            first_valid_idx = idx
            break

    assert first_valid_idx is not None, (
        f"No pose with z >= {min_z} found. "
        f"Max z = {max(p[2] for p in world_frame_poses):.4f}"
    )

    # Filter and downsample
    filtered_poses = world_frame_poses[first_valid_idx:]
    downsampled_poses = filtered_poses[::downsample_factor]

    return {
        "start_pose": world_frame_poses[0],
        "goals": downsampled_poses,
    }


def main() -> None:
    args: ProcessPoseArgs = tyro.cli(ProcessPoseArgs)

    assert args.dextoolbench_data_dir.exists(), (
        f"Data directory not found: {args.dextoolbench_data_dir}"
    )

    num_processed = 0
    num_missing = 0

    for (
        object_category,
        object_name_to_task_names,
    ) in DEXTOOLBENCH_DATA_STRUCTURE.items():
        for object_name, task_names in object_name_to_task_names.items():
            for task_name in task_names:
                label = f"{object_category}/{object_name}/{task_name}"
                poses_json_path = (
                    args.dextoolbench_data_dir
                    / object_category
                    / object_name
                    / task_name
                    / "poses.json"
                )

                if not poses_json_path.exists():
                    print(f"[MISSING] {label}  —  {poses_json_path}")
                    num_missing += 1
                    continue

                # Read raw poses (robot frame)
                with open(poses_json_path, "r") as f:
                    raw_poses = json.load(f)

                # Process: robot→world, min_z filter, downsample
                result = process_poses(
                    raw_poses=raw_poses,
                    min_z=args.min_z,
                    downsample_factor=args.downsample_factor,
                )

                # Write output
                output_path = (
                    args.destination_dir
                    / object_category
                    / object_name
                    / f"{task_name}.json"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    json.dump(result, f, indent=4)

                n_raw = len(raw_poses)
                n_out = len(result["goals"])
                print(
                    f"[OK]      {label}  —  {n_raw} raw -> {n_out} goals -> {output_path}"
                )
                num_processed += 1

    # Summary
    total = num_processed + num_missing
    print()
    print("=" * 60)
    print(f"Total: {total}  |  Processed: {num_processed}  |  Missing: {num_missing}")
    print(f"min_z={args.min_z}  downsample_factor={args.downsample_factor}")
    print(f"Output: {args.destination_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
