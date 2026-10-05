from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "index.template.html"
DEFAULT_SCENE_PATH = REPO_ROOT / "scene" / "scene_chain_local.json"
DEFAULT_TRAJECTORY_PATH = REPO_ROOT / "trajectory" / "trajectory_chain_demo.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated" / "viewer"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_template(template_path: Path, scene_payload: dict) -> str:
    return template_path.read_text(encoding="utf-8").replace(
        "__SCENE_JSON__",
        json.dumps(scene_payload, separators=(",", ":")),
    )


def build_robot_payload(robot_entry: dict) -> dict:
    robot_payload = {
        "name": robot_entry["name"],
        "position": robot_entry.get("position", [0.0, 0.0, 0.0]),
        "rpy": robot_entry.get("rpy", [0.0, 0.0, 0.0]),
        "animated": robot_entry.get("animated", False),
    }

    urdf_url = robot_entry.get("urdf_url")
    urdf_file = robot_entry.get("urdf_file")

    if urdf_url:
        robot_payload["urdf_path"] = urdf_url
    elif urdf_file:
        urdf_path = REPO_ROOT / "robot" / urdf_file
        if not urdf_path.exists():
            raise ValueError(f"Referenced URDF file does not exist: {urdf_file}")
        robot_payload["urdf_file"] = urdf_file
        robot_payload["urdf_path"] = f"robot/{urdf_file}"
    else:
        raise ValueError(f'Scene robot "{robot_entry["name"]}" must define either "urdf_file" or "urdf_url".')

    return robot_payload


def build_local_scene_payload(scene_config: dict, trajectory: dict) -> dict:
    return {
        "robots": [build_robot_payload(robot_entry) for robot_entry in scene_config["robots"]],
        "trajectory": trajectory,
    }


def build_embedded_scene_payload(scene_config: dict, trajectory: dict, urdf_text_by_file: dict[str, str]) -> dict:
    robots = []
    for robot_entry in scene_config["robots"]:
        urdf_file = robot_entry.get("urdf_file")
        if not urdf_file:
            raise ValueError(f'Scene robot "{robot_entry["name"]}" is missing "urdf_file".')
        robots.append(
            {
                "name": robot_entry["name"],
                "position": robot_entry.get("position", [0.0, 0.0, 0.0]),
                "rpy": robot_entry.get("rpy", [0.0, 0.0, 0.0]),
                "animated": robot_entry.get("animated", False),
                "urdf_file": urdf_file,
                "urdf_text": urdf_text_by_file[urdf_file],
            }
        )
    return {"robots": robots, "trajectory": trajectory}


def read_urdf_texts(scene_config: dict) -> dict[str, str]:
    urdf_text_by_file: dict[str, str] = {}
    for robot_entry in scene_config["robots"]:
        urdf_file = robot_entry.get("urdf_file")
        if not urdf_file or urdf_file in urdf_text_by_file:
            continue
        urdf_text_by_file[urdf_file] = (REPO_ROOT / "robot" / urdf_file).read_text(encoding="utf-8")
    return urdf_text_by_file


def rewrite_urdf_with_direct_urls(urdf_text: str, urdf_file: str, file_urls: dict[str, str]) -> str:
    root = ET.fromstring(urdf_text)
    urdf_parent = Path("robot") / Path(urdf_file).parent

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue

        artifact_path = str((urdf_parent / filename).as_posix())
        direct_url = file_urls.get(artifact_path)
        if direct_url is None:
            continue

        suffix = Path(filename).suffix.lower()
        mesh.attrib["filename"] = f"{direct_url}#ext={suffix}"

    return ET.tostring(root, encoding="unicode")


def copy_referenced_robot_assets(robot_entries: list[dict], output_robot_dir: Path) -> list[Path]:
    if output_robot_dir.exists():
        shutil.rmtree(output_robot_dir)
    output_robot_dir.mkdir(parents=True, exist_ok=True)

    copied_urdf_paths = []
    copied_directories: set[Path] = set()

    for robot_entry in robot_entries:
        urdf_file = robot_entry.get("urdf_file")
        if not urdf_file:
            continue

        relative_urdf_path = Path(urdf_file)
        source_urdf_path = REPO_ROOT / "robot" / relative_urdf_path
        destination_urdf_path = output_robot_dir / relative_urdf_path
        destination_urdf_path.parent.mkdir(parents=True, exist_ok=True)

        if len(relative_urdf_path.parts) > 1:
            top_level_dir = relative_urdf_path.parts[0]
            source_dir = REPO_ROOT / "robot" / top_level_dir
            destination_dir = output_robot_dir / top_level_dir
            if source_dir not in copied_directories:
                if destination_dir.exists():
                    shutil.rmtree(destination_dir)
                shutil.copytree(source_dir, destination_dir)
                copied_directories.add(source_dir)
        else:
            shutil.copy2(source_urdf_path, destination_urdf_path)

        copied_urdf_paths.append(destination_urdf_path)

    return copied_urdf_paths


def validate_scene_and_trajectory(scene_config: dict, trajectory: dict) -> None:
    robot_entries = scene_config.get("robots", [])
    timestamps = trajectory.get("timestamps", [])
    positions = trajectory.get("positions", [])
    joint_names = trajectory.get("joint_names", [])
    object_trajectories = trajectory.get("object_trajectories", {})

    if not robot_entries:
        raise ValueError("Scene config must contain at least one robot entry.")
    if not timestamps:
        raise ValueError("Trajectory must contain at least one timestamp.")
    if len(timestamps) != len(positions):
        raise ValueError("Trajectory timestamps and positions must have the same length.")
    if not joint_names:
        raise ValueError("Trajectory must contain at least one joint name.")
    if any(len(frame) != len(joint_names) for frame in positions):
        raise ValueError("Each trajectory frame must have one value per joint name.")

    for object_name, object_trajectory in object_trajectories.items():
        object_positions = object_trajectory.get("positions", [])
        object_rpys = object_trajectory.get("rpys", [])
        if len(object_positions) != len(timestamps):
            raise ValueError(f'Object trajectory "{object_name}" must match timestamp count.')
        if object_rpys and len(object_rpys) != len(timestamps):
            raise ValueError(f'Object trajectory "{object_name}" rpy list must match timestamp count.')
        if any(len(position) != 3 for position in object_positions):
            raise ValueError(f'Object trajectory "{object_name}" positions must be xyz triples.')
        if object_rpys and any(len(rpy) != 3 for rpy in object_rpys):
            raise ValueError(f'Object trajectory "{object_name}" rpys must be triples.')


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"
