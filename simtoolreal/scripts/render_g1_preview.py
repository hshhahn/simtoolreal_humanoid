"""Render URDF visuals at a recorded simulation pose without the RTX renderer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import trimesh


def origin_matrix(node) -> np.ndarray:
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return result


def pose_matrix(pose) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return result


def visual_meshes(urdf_path: Path, pose, joints: dict, color=None) -> list:
    root = ET.parse(urdf_path).getroot()
    links = {node.get("name"): node for node in root.findall("link")}
    joint_nodes = root.findall("joint")
    children = {node.find("child").get("link") for node in joint_nodes}
    root_links = set(links) - children
    if len(root_links) != 1:
        raise ValueError(f"Expected one root link in {urdf_path}")
    transforms = {root_links.pop(): pose_matrix(pose)}
    pending = list(joint_nodes)
    while pending:
        progress = False
        for node in list(pending):
            parent = node.find("parent").get("link")
            if parent not in transforms:
                continue
            motion = np.eye(4)
            value = joints.get(node.get("name"), 0.0)
            axis = node.find("axis")
            if node.get("type") in ("revolute", "continuous"):
                vector = np.fromstring(
                    axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "
                )
                motion[:3, :3] = Rotation.from_rotvec(
                    vector * value / np.linalg.norm(vector)
                ).as_matrix()
            elif node.get("type") == "prismatic":
                motion[:3, 3] = np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") * value
            transforms[node.find("child").get("link")] = (
                transforms[parent] @ origin_matrix(node.find("origin")) @ motion
            )
            pending.remove(node)
            progress = True
        if not progress:
            raise ValueError(f"Disconnected URDF joints in {urdf_path}")

    materials = {node.get("name"): node for node in root.findall("material")}
    result = []
    cache = {}
    for name, link in links.items():
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh_node = geometry.find("mesh")
            if mesh_node is not None:
                mesh_path = Path(mesh_node.get("filename"))
                if not mesh_path.is_absolute():
                    mesh_path = urdf_path.parent / mesh_path
                if mesh_path not in cache:
                    cache[mesh_path] = trimesh.load_mesh(mesh_path, process=False)
                mesh = cache[mesh_path]
                triangles = mesh.triangles.copy()
                triangles *= np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" ")
            elif geometry.find("box") is not None:
                mesh = trimesh.creation.box(
                    extents=np.fromstring(geometry.find("box").get("size"), sep=" ")
                )
                vertices, faces = trimesh.remesh.subdivide_to_size(
                    mesh.vertices, mesh.faces, max_edge=0.04
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                triangles = mesh.triangles
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                mesh = trimesh.creation.cylinder(
                    radius=float(cylinder.get("radius")),
                    height=float(cylinder.get("length")),
                    sections=32,
                )
                triangles = mesh.triangles
            elif geometry.find("sphere") is not None:
                mesh = trimesh.creation.icosphere(
                    subdivisions=2, radius=float(geometry.find("sphere").get("radius"))
                )
                triangles = mesh.triangles
            else:
                raise ValueError(f"Unsupported visual geometry in {urdf_path}")
            transform = transforms[name] @ origin_matrix(visual.find("origin"))
            triangles = triangles @ transform[:3, :3].T + transform[:3, 3]
            rgba = (
                np.array((*color, 1.0))
                if color is not None
                else np.array((0.65, 0.68, 0.72, 1.0))
            )
            material = visual.find("material")
            if color is None and material is not None:
                color_node = material.find("color")
                if color_node is None and material.get("name") in materials:
                    color_node = materials[material.get("name")].find("color")
                if color_node is not None:
                    rgba = np.fromstring(color_node.get("rgba"), sep=" ")
            result.append((triangles, rgba))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview_directory", type=Path)
    args = parser.parse_args()
    out = args.preview_directory.resolve()
    frame = json.loads((out / "frame.json").read_text())
    manifest = json.loads((out / "environment_manifest.json").read_text())
    joint_values = dict(zip(frame["robot_joint_names"], frame["robot_joint_pos"]))
    meshes = visual_meshes(
        Path(manifest["robot_urdf"]), frame["robot_base_pose"], joint_values
    )
    meshes += visual_meshes(
        out / "table.urdf", frame["table_pose"], {}, color=(0.58, 0.38, 0.23)
    )
    meshes += visual_meshes(
        out / "object.urdf", frame["object_pose"], {}, color=(0.95, 0.42, 0.08)
    )
    meshes += visual_meshes(
        out / "object.urdf", frame["goal_pose"], {}, color=(0.20, 0.72, 0.31)
    )
    light = np.array((0.3, -0.5, 0.8))
    light /= np.linalg.norm(light)
    shaded = []
    for triangles, rgba in meshes:
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        brightness = 0.5 + 0.5 * np.abs(normals @ light)
        colors = np.tile(rgba, (len(triangles), 1))
        colors[:, :3] *= brightness[:, None]
        shaded.append((triangles, colors))
    all_triangles = np.concatenate([triangles for triangles, _ in shaded])
    all_colors = np.concatenate([colors for _, colors in shaded])

    def draw(name, limits, title):
        fig = plt.figure(figsize=(10, 8), facecolor="#f6f8fb")
        ax = fig.add_subplot(111, projection="3d", computed_zorder=True)
        ax.set_facecolor("#f6f8fb")
        # Sort all triangles together, including tabletop and tool surfaces.
        ax.add_collection3d(
            Poly3DCollection(
                all_triangles,
                facecolors=all_colors,
                edgecolors="none",
                linewidths=0,
                antialiased=False,
                axlim_clip=True,
            )
        )
        floor = np.array(
            [[[-0.7, -0.5, 0], [0.7, -0.5, 0], [0.7, 1.0, 0], [-0.7, 1.0, 0]]]
        )
        if limits[2][0] == 0.0:
            ax.add_collection3d(
                Poly3DCollection(
                    floor, facecolors="#dce2ea", edgecolors="#bdc7d3", alpha=0.4
                )
            )
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect([hi - lo for lo, hi in limits])
        ax.view_init(elev=32, azim=-65)
        for pose_name, label, color in (
            ("object_pose", "Tool", "#b9580b"),
            ("goal_pose", "Target", "#168241"),
        ):
            position = frame[pose_name][:3]
            ax.text(
                position[0],
                position[1],
                position[2] + 0.06,
                label,
                color=color,
                fontsize=11,
                fontweight="bold",
            )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("height (m)")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        fig.text(
            0.5,
            0.045,
            "Orange: physical tool   |   Green: desired tool pose",
            ha="center",
            fontsize=12,
            color="#344256",
        )
        fig.text(
            0.5,
            0.018,
            "URDF meshes at recorded simulation poses · SONIC standing reference",
            ha="center",
            fontsize=10,
            color="#627084",
        )
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.17, top=0.90)
        fig.savefig(out / name, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[preview mesh render] wrote {out / name}", flush=True)

    draw(
        "environment.png",
        ((-0.6, 0.6), (-0.4, 0.95), (0.0, 1.6)),
        "G1 + dual Wuji hands: tool pose task",
    )
    draw(
        "hands_and_tool.png",
        ((-0.45, 0.45), (-0.1, 0.75),
         (manifest["tabletop_height_m"] - 0.10, manifest["tabletop_height_m"] + 0.50)),
        "Hands, tabletop tool and target pose",
    )
    manifest["image_renderer"] = (
        "Matplotlib URDF mesh rendering at recorded simulation state"
    )
    manifest["visual_triangles"] = sum(len(triangles) for triangles, _ in shaded)
    (out / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
