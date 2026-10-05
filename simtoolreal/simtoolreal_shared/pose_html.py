"""Backend-neutral construction of SimToolReal Three.js pose viewers."""

from __future__ import annotations

import base64
import json
import mimetypes
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "isaacsimenvs/utils/interactive_viewer/index.template.html"


def portable_visual_urdf(
    urdf_text: str,
    source_urdf_path: Path,
    public_raw_base: str | None = None,
) -> str:
    """Strip physics XML and make visual meshes browser-portable.

    Files inside this repository use a public raw URL when one is supplied.
    Files outside the repository, such as temporary generated assets, are
    embedded as data URLs. Only visual geometry is retained because collision
    and inertial data are irrelevant to the pose viewer.
    """

    root = ET.fromstring(urdf_text)
    for link in root.findall(".//link"):
        for tag in ("collision", "inertial", "self_collision_checking"):
            for element in list(link.findall(tag)):
                link.remove(element)

    encoded_by_path: dict[Path, str] = {}
    for mesh in root.findall(".//visual//mesh"):
        filename = mesh.get("filename")
        if not filename or filename.startswith(("data:", "http://", "https://")):
            continue
        if filename.startswith("package://"):
            mesh_path = REPO_ROOT / filename[len("package://") :]
        else:
            candidate = Path(filename)
            mesh_path = candidate if candidate.is_absolute() else source_urdf_path.parent / candidate
        mesh_path = mesh_path.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Viewer mesh does not exist: {mesh_path}")

        try:
            relative_path = mesh_path.relative_to(REPO_ROOT)
        except ValueError:
            relative_path = None
        if public_raw_base and relative_path is not None:
            mesh.set(
                "filename",
                public_raw_base.rstrip("/") + "/" + relative_path.as_posix(),
            )
            continue

        if mesh_path not in encoded_by_path:
            mime = mimetypes.guess_type(mesh_path.name)[0] or "application/octet-stream"
            payload = base64.b64encode(mesh_path.read_bytes()).decode("ascii")
            encoded_by_path[mesh_path] = (
                f"data:{mime};base64,{payload}#ext={mesh_path.suffix.lower()}"
            )
        mesh.set("filename", encoded_by_path[mesh_path])

    return ET.tostring(root, encoding="unicode")


def _pose7_trajectory(poses) -> dict:
    return {
        "positions": [[round(float(value), 6) for value in pose[:3]] for pose in poses],
        "quats": [[round(float(value), 6) for value in pose[3:7]] for pose in poses],
    }


def render_pose_html(
    *,
    robots,
    joint_names,
    joint_positions,
    object_poses,
    robot_base_poses,
    dt,
) -> str:
    """Render robot and object pose trajectories into the shared HTML template."""

    frame_count = len(joint_positions)
    if frame_count == 0:
        raise ValueError("Cannot render a pose viewer with zero frames")
    if len(robot_base_poses) != frame_count:
        raise ValueError("robot_base_poses length must match joint_positions")
    for name, poses in object_poses.items():
        if len(poses) != frame_count:
            raise ValueError(f"object_poses[{name!r}] length must match joint_positions")

    payload = {
        "robots": robots,
        "trajectory": {
            "robot_name": "robot",
            "joint_names": list(joint_names),
            "dt": float(dt),
            "timestamps": [round(index * float(dt), 6) for index in range(frame_count)],
            "positions": [
                [round(float(value), 6) for value in positions]
                for positions in joint_positions
            ],
            "base_trajectory": _pose7_trajectory(robot_base_poses),
            "object_trajectories": {
                name: _pose7_trajectory(poses) for name, poses in object_poses.items()
            },
        },
    }
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "__SCENE_JSON__", json.dumps(payload, separators=(",", ":"))
    )
