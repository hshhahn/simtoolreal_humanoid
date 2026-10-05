import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
import tyro
from tqdm import tqdm

DEFAULT_DOWNLOAD_URL = "https://download.cs.stanford.edu/juno/simtoolreal/"
DEFAULT_DATA_FOLDER = Path(__file__).parent / "pretrained_policy"


@dataclass
class DownloadArgs:
    download_url: str = DEFAULT_DOWNLOAD_URL
    data_folder: Path = DEFAULT_DATA_FOLDER

    @property
    def download_url_no_trailing_slash(self) -> str:
        if self.download_url.endswith("/"):
            return self.download_url[:-1]
        return self.download_url


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
    url = f"{args.download_url_no_trailing_slash}/pretrained_policy.zip"
    extract_to = args.data_folder.parent
    download_and_extract_zip(url=url, extract_to=extract_to)


def main() -> None:
    args: DownloadArgs = tyro.cli(DownloadArgs)
    run_download(args)


if __name__ == "__main__":
    main()
