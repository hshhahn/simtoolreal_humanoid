#!/usr/bin/env python3
"""Compose SONIC's G1 body and the official left/right Wuji URDFs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent


def revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def build(output: Path, wrist_offset: float = 0.0415) -> None:
    sonic = WORKSPACE / "third_party/GR00T-WholeBodyControl"
    wuji = WORKSPACE / "third_party/wuji-description"
    unitree = WORKSPACE / "third_party/unitree_ros"
    source = sonic / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
    root = ET.parse(source).getroot()
    root.set("name", "g1_dual_wuji_sonic")
    # Replace the fixed Dex3 hands used by SONIC's training model.
    removed = {
        x.get("name")
        for x in root.findall("link")
        if x.get("name", "").startswith(("left_hand_", "right_hand_"))
    }
    for item in list(root):
        if item.tag == "link" and item.get("name") in removed:
            root.remove(item)
        elif item.tag == "joint" and any(
            item.find(tag).get("link") in removed for tag in ("parent", "child")
        ):
            root.remove(item)
    output.mkdir(parents=True, exist_ok=True)

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename.startswith("package://robot_description/"):
            raise ValueError(f"Unexpected G1 mesh path: {filename}")
        relative = Path(filename.removeprefix("package://robot_description/"))
        origin = sonic / "gear_sonic/data/assets/robot_description" / relative
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if origin.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            # NVIDIA stores the Unitree meshes as Git LFS pointers. The official
            # Unitree checkout has the same blobs; verify before reusing them.
            expected = next(
                line.split(":", 1)[1]
                for line in origin.read_text().splitlines()
                if line.startswith("oid sha256:")
            )
            exact_mesh = unitree / "robots/g1_description/meshes" / origin.name
            if (
                not exact_mesh.is_file()
                or hashlib.sha256(exact_mesh.read_bytes()).hexdigest() != expected
            ):
                raise RuntimeError(
                    f"Missing exact SONIC mesh {origin.name}; fetch its Git LFS blob from NVIDIA"
                )
            origin = exact_mesh
        shutil.copy2(origin, destination)
        mesh.set("filename", str(relative))

    # The official G1 adapter STL uses millimeters. Center it and convert to m.
    adapter = trimesh.load(
        wuji / "hand/attachment/unitree-g1-attachment/unitree-g1-docking-adapter.stl",
        force="mesh",
    )
    adapter.apply_scale(0.001)
    adapter.apply_translation(-adapter.bounds.mean(axis=0))
    adapter.export(output / "meshes/unitree_g1_wuji_adapter.stl")
    adapter.density = 1240.0  # PLA approximation; replace with measured adapter properties for hardware transfer.
    inertia = adapter.moment_inertia
    half_length = float(adapter.extents[2]) / 2.0
    hand_offset = wrist_offset + 2.0 * half_length

    for side in ("left", "right"):
        hand_path = wuji / f"hand/body/urdf/{side}.urdf"
        hand = ET.parse(hand_path).getroot()
        for mesh in hand.findall(".//mesh"):
            origin = (hand_path.parent / mesh.get("filename")).resolve()
            relative = Path("meshes/wuji") / side / origin.name
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            mesh.set("filename", str(relative))
        for item in hand:
            if item.tag in ("link", "joint"):
                # Anonymous CAD materials confuse USD import; use unique names.
                for material in item.findall(".//material"):
                    if not material.get("name"):
                        material.set("name", f"{side}_wuji_material")
                root.append(copy.deepcopy(item))

        adapter_name = f"{side}_wuji_adapter_link"
        link = ET.SubElement(root, "link", name=adapter_name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(float(adapter.mass)))
        ET.SubElement(
            inertial,
            "inertia",
            **{
                key: str(float(inertia[i, j]))
                for key, i, j in [
                    ("ixx", 0, 0),
                    ("iyy", 1, 1),
                    ("izz", 2, 2),
                    ("ixy", 0, 1),
                    ("ixz", 0, 2),
                    ("iyz", 1, 2),
                ]
            },
        )
        for role in ("visual", "collision"):
            element = ET.SubElement(link, role)
            ET.SubElement(
                ET.SubElement(element, "geometry"),
                "mesh",
                filename="meshes/unitree_g1_wuji_adapter.stl",
            )
        joint = ET.SubElement(
            root, "joint", name=f"{side}_wuji_adapter_fixed", type="fixed"
        )
        ET.SubElement(joint, "parent", link=f"{side}_wrist_yaw_link")
        ET.SubElement(joint, "child", link=adapter_name)
        ET.SubElement(
            joint,
            "origin",
            xyz=f"{wrist_offset + half_length} 0 0",
            rpy=f"0 {math.pi / 2.0} 0",
        )
        joint = ET.SubElement(
            root, "joint", name=f"{side}_wuji_mount_fixed", type="fixed"
        )
        ET.SubElement(joint, "parent", link=adapter_name)
        ET.SubElement(joint, "child", link=f"{side}_palm_link")
        # Adapter's local +Z and palm +Z point along G1's wrist +X.
        ET.SubElement(joint, "origin", xyz=f"0 0 {half_length}", rpy="0 0 0")

    ET.indent(root, space="  ")
    target = output / "g1_wuji.urdf"
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    if len(joints) != 69:
        raise ValueError(f"Expected 29 G1 + 40 Wuji joints; got {len(joints)}")
    for mesh in root.findall(".//mesh"):
        if not (output / mesh.get("filename")).is_file():
            raise FileNotFoundError(mesh.get("filename"))
    licenses = output / "licenses"
    licenses.mkdir(exist_ok=True)
    for name, path in [("SONIC", sonic), ("Wuji", wuji), ("Unitree", unitree)]:
        shutil.copy2(path / "LICENSE", licenses / f"{name}.txt")
    provenance = {
        "sonic_source": {
            "url": "https://github.com/NVlabs/GR00T-WholeBodyControl",
            "commit": revision(sonic),
            "urdf": str(source.relative_to(sonic)),
        },
        "wuji_source": {
            "url": "https://github.com/wuji-technology/wuji-description",
            "commit": revision(wuji),
            "variant": "original Wuji Hand, hand/body",
        },
        "unitree_source": {
            "url": "https://github.com/unitreerobotics/unitree_ros",
            "commit": revision(unitree),
        },
        "joints": {"body": 29, "left_hand": 20, "right_hand": 20},
        "mount": {
            "wrist_flange_x_m": wrist_offset,
            "hand_origin_x_m": hand_offset,
            "palm_rpy": [0, math.pi / 2, 0],
            "adapter_density_kg_m3": 1240.0,
            "status": "simulation mounting estimate; hardware flange calibration required",
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    actuators = {}
    for side in ("left", "right"):
        mjcf = ET.parse(wuji / f"hand/body/mjcf/{side}.xml").getroot()
        mjcf_joints = {j.get("name"): j for j in mjcf.findall(".//body/joint")}
        hand_joints = {
            j.get("name"): j
            for j in root.findall("joint")
            if j.get("name", "").startswith(f"{side}_finger")
            and j.get("type") == "revolute"
        }
        for actuator in mjcf.findall("actuator/position"):
            name = actuator.get("joint")
            limit = hand_joints[name].find("limit")
            actuators[name] = {
                "stiffness": float(actuator.get("kp")),
                "damping": float(actuator.get("kv")),
                "armature": float(mjcf_joints[name].get("armature", "0.0002")),
                "effort_limit_sim": float(limit.get("effort")),
                "velocity_limit_sim": float(limit.get("velocity")),
            }
    (output / "wuji_actuators.json").write_text(json.dumps(actuators, indent=2) + "\n")
    print(f"Built {target}: 29 body joints + two 20-joint Wuji hands")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "assets/urdf/g1_wuji")
    parser.add_argument("--wrist-offset", type=float, default=0.0415)
    args = parser.parse_args()
    build(args.output, args.wrist_offset)
