from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from recorded_data import RecordedData


@dataclass
class SliceArgs:
    file_path: Path = Path("recorded_data/2025-11-06_17-09-47.npz")
    """Path to the recorded data file."""

    start: Optional[int] = None
    """Start index for slicing (inclusive). None means from the beginning."""

    end: Optional[int] = None
    """End index for slicing (exclusive). None means to the end."""


def main():
    args: SliceArgs = tyro.cli(SliceArgs)

    assert args.file_path.exists(), f"File {args.file_path} does not exist"
    recorded_data = RecordedData.from_file(args.file_path)

    recorded_data = recorded_data.slice(start=args.start, end=args.end)
    output_filepath = (
        args.file_path.parent / f"{args.file_path.stem}_{args.start}_{args.end}.npz"
    )
    print(f"Saving sliced recorded data to {output_filepath}")
    recorded_data.to_file(output_filepath)


if __name__ == "__main__":
    main()
