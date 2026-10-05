from __future__ import annotations

from pathlib import Path
from typing import Tuple
import time
import urllib.request

import numpy as np

from .viewer_common import DEFAULT_TEMPLATE_PATH, render_template


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
ColorRGB = Tuple[float, float, float]
DEFAULT_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/tylerlum/simtoolreal/6809a978753e950913a7588bbeaef07d16f10b56/"


def _as_2d_float_array(values, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (T, N). Got shape {array.shape}.")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} must have shape (T, {width}). Got shape {array.shape}.")
    return array


def _as_1d_float_array(values, *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (T,). Got shape {array.shape}.")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}. Got shape {array.shape}.")
    return array


def _normalize_quaternions(quaternions: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-length quaternion.")
    return quaternions / norms


def _split_pose7_array(poses: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Split pose arrays of shape (T, 7) into positions (T, 3) and quaternions (T, 4)."""
    pose_array = _as_2d_float_array(poses, name=name, width=7)
    positions = pose_array[:, :3]
    quats = _normalize_quaternions(pose_array[:, 3:], name=f"{name} quaternion part")
    return positions, quats


def _serialize_array(array: np.ndarray, *, decimals: int = 6) -> list:
    return np.round(array.astype(float), decimals=decimals).tolist()


def _build_object_trajectories(object_poses: dict[str, np.ndarray] | None, *, num_frames: int) -> dict:
    if not object_poses:
        return {}

    object_trajectories = {}
    for object_name, pose_values in object_poses.items():
        positions, quats = _split_pose7_array(pose_values, name=f'object_poses["{object_name}"]')
        if positions.shape[0] != num_frames:
            raise ValueError(
                f'object_poses["{object_name}"] must have {num_frames} frames. Got {positions.shape[0]}.'
            )
        object_trajectories[object_name] = {
            "positions": _serialize_array(positions),
            "quats": _serialize_array(quats),
        }
    return object_trajectories


def build_trajectory_payload(
    *,
    joint_names: list[str],
    robot_joint_positions,
    timestamps=None,
    dt: float | None = None,
    object_poses: dict[str, np.ndarray] | None = None,
    robot_base_poses=None,
    robot_name: str = "robot",
) -> dict:
    """
    Build the viewer trajectory payload from array-shaped inputs.

    Expected shapes:
    - robot_joint_positions: (T, J)
    - timestamps: (T,)
    - object_poses[name]: (T, 7) with columns [x, y, z, qx, qy, qz, qw]
    - robot_base_poses: (T, 7) with columns [x, y, z, qx, qy, qz, qw]

    Derived shapes inside the payload:
    - joint positions: (T, J)
    - object positions: (T, 3)
    - object quaternions: (T, 4)
    - base positions: (T, 3)
    - base quaternions: (T, 4)
    """
    joint_positions = _as_2d_float_array(robot_joint_positions, name="robot_joint_positions")
    num_frames, num_joints = joint_positions.shape

    if len(joint_names) != num_joints:
        raise ValueError(
            f"joint_names must have length {num_joints} to match robot_joint_positions. Got {len(joint_names)}."
        )

    if timestamps is None:
        if dt is None:
            raise ValueError("Provide either timestamps or dt.")
        timestamps_array = np.arange(num_frames, dtype=float) * float(dt)
    else:
        timestamps_array = _as_1d_float_array(timestamps, name="timestamps", length=num_frames)
        if dt is None and num_frames > 1:
            dt = float(timestamps_array[1] - timestamps_array[0])

    trajectory = {
        "robot_name": robot_name,
        "joint_names": list(joint_names),
        "dt": float(dt) if dt is not None else None,
        "timestamps": _serialize_array(timestamps_array),
        "positions": _serialize_array(joint_positions),
        "object_trajectories": _build_object_trajectories(object_poses, num_frames=num_frames),
    }

    if robot_base_poses is not None:
        base_positions, base_quats = _split_pose7_array(robot_base_poses, name="robot_base_poses")
        if base_positions.shape[0] != num_frames:
            raise ValueError(
                f"robot_base_poses must have {num_frames} frames. Got {base_positions.shape[0]}."
            )
        trajectory["base_trajectory"] = {
            "positions": _serialize_array(base_positions),
            "quats": _serialize_array(base_quats),
        }

    return trajectory


def create_html(
    *,
    joint_names: list[str],
    robot_joint_positions,
    robots: list[dict],
    object_poses: dict[str, np.ndarray] | None = None,
    robot_base_poses=None,
    timestamps=None,
    dt: float | None = None,
    robot_name: str = "robot",
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> str:
    """
    Render a full viewer HTML string from array-shaped robot and object trajectories.

    Expected shapes:
    - robot_joint_positions: (T, J)
    - timestamps: (T,) if provided
    - object_poses[name]: (T, 7) as [x, y, z, qx, qy, qz, qw]
    - robot_base_poses: (T, 7) as [x, y, z, qx, qy, qz, qw]
    """
    trajectory = build_trajectory_payload(
        joint_names=joint_names,
        robot_joint_positions=robot_joint_positions,
        timestamps=timestamps,
        dt=dt,
        object_poses=object_poses,
        robot_base_poses=robot_base_poses,
        robot_name=robot_name,
    )
    return render_template(template_path, {"robots": robots, "trajectory": trajectory})


def make_embedded_robot(
    *,
    name: str,
    urdf_text: str,
    position: Vec3 = (0.0, 0.0, 0.0),
    rpy: Vec3 = (0.0, 0.0, 0.0),
    animated: bool = False,
    color_override: ColorRGB | None = None,
) -> dict:
    robot = {
        "name": name,
        "urdf_text": urdf_text,
        "position": list(position),
        "rpy": list(rpy),
        "animated": animated,
    }
    if color_override is not None:
        robot["color_override"] = list(color_override)
    return robot


def make_url_robot(
    *,
    name: str,
    urdf_url: str,
    position: Vec3 = (0.0, 0.0, 0.0),
    rpy: Vec3 = (0.0, 0.0, 0.0),
    animated: bool = False,
    color_override: ColorRGB | None = None,
) -> dict:
    robot = {
        "name": name,
        "urdf_path": urdf_url,
        "position": list(position),
        "rpy": list(rpy),
        "animated": animated,
    }
    if color_override is not None:
        robot["color_override"] = list(color_override)
    return robot


def _read_table_urdf() -> str:
    table_path = Path(__file__).resolve().parents[2] / "assets" / "urdf" / "table_narrow.urdf"
    return table_path.read_text(encoding="utf-8")


def _check_viewer_urls(urls: list[str], url_check: str) -> set[str]:
    if url_check == "skip":
        print("[viewer] URL check skipped; browser mesh loading may fail silently.")
        return set()
    failed: set[str] = set()
    for url in dict.fromkeys(urls):
        print(f"[viewer] URL check ({url_check}) -> {url}")
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=10)
            print(f"[viewer]   PASSED ({time.monotonic() - t0:.2f}s)")
        except Exception as exc:
            failed.add(url)
            msg = f"[viewer]   FAILED ({time.monotonic() - t0:.2f}s): {exc}"
            if url_check == "error":
                raise ValueError(msg) from exc
            print(msg)
    return failed


def write_pose_viewer_html(path: Path, payload: dict, *, title: str) -> str:
    """Write a mesh-based HTML viewer for Isaac Sim distillation rollout payloads."""
    del title  # The SimToolReal template reads the title from embedded trajectory data.
    github_raw_base = payload.get("github_raw_base", DEFAULT_GITHUB_RAW_BASE)
    if not github_raw_base.endswith("/"):
        github_raw_base += "/"
    url_check = payload.get("url_check", "warn")
    print(f"[viewer] GitHub raw base: {github_raw_base}")

    robot_urdf_url = (
        github_raw_base
        + "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    object_urdf_url = (
        github_raw_base
        + "assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf"
    )
    _check_viewer_urls([robot_urdf_url, object_urdf_url], url_check)

    robots = [
        make_url_robot(name="robot", urdf_url=robot_urdf_url, animated=True),
        make_embedded_robot(name="table", urdf_text=_read_table_urdf()),
        make_url_robot(name="object", urdf_url=object_urdf_url),
        make_url_robot(
            name="goal",
            urdf_url=object_urdf_url,
            color_override=(0.20, 0.72, 0.31),
        ),
    ]

    html_text = create_html(
        joint_names=payload["robot_joint_names"],
        robot_joint_positions=payload["robot_joint_positions"],
        robots=robots,
        object_poses={
            "table": np.asarray(payload["table_poses"], dtype=float),
            "object": np.asarray(payload["object_poses"], dtype=float),
            "goal": np.asarray(payload["goal_poses"], dtype=float),
        },
        robot_base_poses=np.asarray(payload["robot_base_poses"], dtype=float),
        timestamps=np.asarray(payload["timestamps"], dtype=float),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return html_text
