import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from dextoolbench.metadata import DEXTOOLBENCH_DATA_STRUCTURE

import requests
import tyro
from tqdm import tqdm

DEFAULT_DOWNLOAD_URL = "https://download.cs.stanford.edu/juno/simtoolreal/"
DEFAULT_DATA_FOLDER = Path(__file__).parent / "dextoolbench" / "data"

# ── Full DexToolBench data structure ──────────────────────────────────────────
# {object_category: {object_name: [task_name, ...]}}
DEXTOOLBENCH_DATA_STRUCTURE: Dict[str, Dict[str, List[str]]] = DEXTOOLBENCH_DATA_STRUCTURE

ALL_OBJECT_CATEGORIES = sorted(DEXTOOLBENCH_DATA_STRUCTURE.keys())
ALL_OBJECT_NAMES = sorted(
    object_name
    for object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.values()
    for object_name in object_name_to_task_names.keys()
)
ALL_TASK_NAMES = sorted(
    set(
        task_name
        for object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.values()
        for task_names in object_name_to_task_names.values()
        for task_name in task_names
    )
)

# Reverse lookup: object_name -> object_category
_OBJECT_NAME_TO_CATEGORY: Dict[str, str] = {
    object_name: object_category
    for object_category, object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.items()
    for object_name in object_name_to_task_names
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_all_entries() -> List[Tuple[str, str, str]]:
    """Return all (object_category, object_name, task_name) triples."""
    entries = []
    for object_category, object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.items():
        for object_name, task_names in object_name_to_task_names.items():
            for task_name in task_names:
                entries.append((object_category, object_name, task_name))
    return entries


def resolve_entries(
    object_category: Optional[str],
    object_name: Optional[str],
    task_name: Optional[str],
) -> List[Tuple[str, str, str]]:
    """Filter the data structure down to the requested entries with validation."""
    all_entries = get_all_entries()

    # ── Validate inputs ──
    if object_category is not None and object_category not in DEXTOOLBENCH_DATA_STRUCTURE:
        raise ValueError(
            f"Unknown object_category '{object_category}'. "
            f"Choose from: {ALL_OBJECT_CATEGORIES}"
        )

    if object_name is not None and object_name not in _OBJECT_NAME_TO_CATEGORY:
        raise ValueError(
            f"Unknown object_name '{object_name}'. "
            f"Choose from: {ALL_OBJECT_NAMES}"
        )

    if object_name is not None and object_category is not None:
        expected_category = _OBJECT_NAME_TO_CATEGORY[object_name]
        if expected_category != object_category:
            raise ValueError(
                f"object_name '{object_name}' belongs to category "
                f"'{expected_category}', not '{object_category}'"
            )

    if task_name is not None:
        # Collect valid task names for the selected scope
        valid_task_names: set[str] = set()
        for entry_object_category, object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.items():
            if object_category is not None and entry_object_category != object_category:
                continue
            for entry_object_name, task_names in object_name_to_task_names.items():
                if object_name is not None and entry_object_name != object_name:
                    continue
                valid_task_names.update(task_names)
        if task_name not in valid_task_names:
            raise ValueError(
                f"Unknown task_name '{task_name}' for the selected scope. "
                f"Valid tasks: {sorted(valid_task_names)}"
            )

    # ── Filter ──
    filtered = []
    for entry_object_category, entry_object_name, entry_task_name in all_entries:
        if object_category is not None and entry_object_category != object_category:
            continue
        if object_name is not None and entry_object_name != object_name:
            continue
        if task_name is not None and entry_task_name != task_name:
            continue
        filtered.append((entry_object_category, entry_object_name, entry_task_name))

    return filtered


def print_all_options() -> None:
    """Pretty-print every (object_category / object_name / task_name) triple."""
    print("=" * 70)
    print("DexToolBench Data Structure")
    print("=" * 70)
    total = 0
    for object_category in sorted(DEXTOOLBENCH_DATA_STRUCTURE):
        print(f"\n  {object_category}/")
        for object_name in sorted(DEXTOOLBENCH_DATA_STRUCTURE[object_category]):
            print(f"    {object_name}/")
            for task_name in DEXTOOLBENCH_DATA_STRUCTURE[object_category][object_name]:
                print(f"      - {task_name}")
                total += 1
    print(f"\nTotal: {total} task datasets")
    print("=" * 70)
    print()
    print("Filter examples:")
    print("  Download everything:          python download_dextoolbench_data.py")
    print("  One category:                 python download_dextoolbench_data.py --object-category hammer")
    print("  One object:                   python download_dextoolbench_data.py --object-name claw_hammer")
    print("  One task across a category:   python download_dextoolbench_data.py --object-category hammer --task-name swing_down")
    print("  One specific dataset:         python download_dextoolbench_data.py --object-name claw_hammer --task-name swing_down")
    print()


# ── Args ──────────────────────────────────────────────────────────────────────
@dataclass
class DownloadArgs:
    """Download DexToolBench task data.

    With no filters, downloads ALL datasets. Use --object-category,
    --object-name, and --task-name to narrow the selection.
    Pass --list to see every available option without downloading.
    """

    download_url: str = DEFAULT_DOWNLOAD_URL
    data_folder: Path = DEFAULT_DATA_FOLDER

    # Filters (None = all)
    object_category: Optional[str] = None
    object_name: Optional[str] = None
    task_name: Optional[str] = None

    # Utility
    list: bool = False
    """Print all available options and exit (no downloads)."""

    @property
    def download_url_no_trailing_slash(self) -> str:
        if self.download_url.endswith("/"):
            return self.download_url[:-1]
        return self.download_url


# ── Download logic ────────────────────────────────────────────────────────────
def download_and_extract_zip(url: str, extract_to: Path) -> None:
    assert url.endswith(".zip"), f"URL must end with .zip, got {url}"

    url_filename_without_ext = Path(urlparse(url).path).stem
    output_zip_path = extract_to / f"{url_filename_without_ext}.zip"
    output_folder = extract_to / url_filename_without_ext
    print("=" * 80)
    print("Planning to:")
    print(f"Download {url} => {output_zip_path}")
    print(f"Then extract {output_zip_path} => {extract_to}")
    print(f"Then expect to end with {output_folder}")
    print("=" * 80 + "\n")

    if output_folder.exists():
        print("!" * 80)
        print(f"Folder {output_folder} already exists, skipping download.")
        print("!" * 80 + "\n")
        return

    # Make the directory
    extract_to.mkdir(parents=True, exist_ok=True)

    # Stream the download and show progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with output_zip_path.open("wb") as file, tqdm(
        desc=f"Downloading {url}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    # Extract the zip file with progress bar
    with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(
            desc=f"Extracting {output_zip_path}",
            total=total_files,
            unit="file",
            dynamic_ncols=True,
        ) as bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, extract_to)
                bar.update(1)

    assert output_folder.exists(), f"Expected {output_folder} to exist"

    # Clean up the downloaded zip file
    print(f"Removing {output_zip_path}")
    output_zip_path.unlink()

    print("DONE!")
    print("~" * 80 + "\n")


def run_download(args: DownloadArgs) -> None:
    if args.list:
        print_all_options()
        return

    entries = resolve_entries(
        object_category=args.object_category,
        object_name=args.object_name,
        task_name=args.task_name,
    )

    if len(entries) == 0:
        print("No matching datasets found.")
        return

    print(f"Will download {len(entries)} dataset(s):\n")
    for object_category, object_name, task_name in entries:
        print(f"  {object_category}/{object_name}/{task_name}")
    print()

    for object_category, object_name, task_name in entries:
        # URL:  <base>/dextoolbench/data/<category>/<object>/<task>.zip
        url = (
            f"{args.download_url_no_trailing_slash}"
            f"/dextoolbench/data/{object_category}/{object_name}/{task_name}.zip"
        )
        # Extract into <data_folder>/<category>/<object>/  so the zip
        # creates  <data_folder>/<category>/<object>/<task>/
        extract_to = args.data_folder / object_category / object_name
        download_and_extract_zip(url=url, extract_to=extract_to)


def main() -> None:
    args: DownloadArgs = tyro.cli(DownloadArgs)
    run_download(args)


if __name__ == "__main__":
    main()
