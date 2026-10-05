#!/usr/bin/env python3
"""Replace the Sharpa hand with a parametric parallel-jaw simulation gripper."""

from pathlib import Path
import json
import math
import shutil
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
OUTPUT = REPO / "assets/urdf/kuka_parallel_gripper"


def box_link(root, name, size, mass, origin=(0.0, 0.0, 0.0), color="0.25 0.28 0.32 1"):
    link = ET.SubElement(root, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    xyz = " ".join(map(str, origin))
    ET.SubElement(inertial, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    x, y, z = size
    ET.SubElement(inertial, "inertia", ixx=str(mass*(y*y+z*z)/12),
                  iyy=str(mass*(x*x+z*z)/12), izz=str(mass*(x*x+y*y)/12),
                  ixy="0", ixz="0", iyz="0")
    for role in ("visual", "collision"):
        element = ET.SubElement(link, role)
        ET.SubElement(element, "origin", xyz=xyz, rpy="0 0 0")
        ET.SubElement(ET.SubElement(element, "geometry"), "box", size=" ".join(map(str, size)))
        if role == "visual":
            material = ET.SubElement(element, "material", name=f"{name}_material")
            ET.SubElement(material, "color", rgba=color)


def joint(root, name, parent, child, kind, xyz, axis=None):
    item = ET.SubElement(root, "joint", name=name, type=kind)
    ET.SubElement(item, "parent", link=parent)
    ET.SubElement(item, "child", link=child)
    ET.SubElement(item, "origin", xyz=" ".join(map(str, xyz)), rpy="0 0 0")
    if axis is not None:
        ET.SubElement(item, "axis", xyz=" ".join(map(str, axis)))
        ET.SubElement(item, "limit", lower="0", upper="0.05", effort="40", velocity="0.2")
        ET.SubElement(item, "dynamics", damping="0", friction="0")


def main():
    root = ET.parse(SOURCE).getroot()
    root.set("name", "kuka_iiwa14_parallel_gripper")
    retained = {f"iiwa14_link_{i}" for i in range(8)}
    for item in list(root):
        if item.tag == "link" and item.get("name") not in retained:
            root.remove(item)
        elif item.tag == "joint" and any(item.find(tag).get("link") not in retained for tag in ("parent", "child")):
            root.remove(item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for mesh in root.findall(".//mesh"):
        relative = Path(mesh.get("filename"))
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE.parent / relative, target)
    box_link(root, "parallel_palm", (.09, .115, .04), .6, (0, 0, .02))
    # A 90-degree adapter hangs the fingers down from the unchanged original
    # wrist pose. Its 0.12 m stand-off keeps the rear jaw clear of the wrist.
    joint(root, "parallel_mount", "iiwa14_link_7", "parallel_palm", "fixed", (0, 0, .12))
    root.find("joint[@name='parallel_mount']/origin").set("rpy", f"{math.pi/2} {-math.pi/2} 0")
    for side, sign in (("left", 1), ("right", -1)):
        finger = f"parallel_{side}_finger"
        pad = f"parallel_{side}_pad"
        # Place the finger link frame at the pad centre so the original
        # fingertip observations/rewards still use the contact location
        # after the original importer merges the fixed pad link.
        box_link(root, finger, (.012, .025, .08), .075, (sign * .008, 0, -.02))
        joint(root, f"parallel_{side}_joint", "parallel_palm", finger,
              "prismatic", (sign * .002, 0, .095), (sign, 0, 0))
        box_link(root, pad, (.004, .025, .04), .008, color="0.08 0.08 0.08 1")
        joint(root, f"parallel_{side}_pad_fixed", finger, pad,
              "fixed", (0, 0, 0))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUTPUT / "kuka_parallel_gripper.urdf", encoding="utf-8", xml_declaration=True)
    licenses = OUTPUT / "licenses"
    licenses.mkdir(exist_ok=True)
    shutil.copy2(REPO / "assets/licenses/kukaiiwa-LICENSE.txt", licenses / "KUKA.txt")
    (OUTPUT / "provenance.json").write_text(json.dumps({
        "arm_source": str(SOURCE.relative_to(REPO)),
        "arm_joints": 7, "gripper_joints": 2, "policy_actions": 8,
        "gripper": "Parametric simulation parallel-jaw gripper; no specific hardware model",
        "opening_range_m": [0, .1], "finger_travel_m": [0, .05],
        "finger_length_m": .08, "pad_size_m": [.004, .025, .04],
        "per_jaw_effort_limit_N": 40, "per_jaw_velocity_limit_m_s": .2,
        "control": "One shared opening target drives both prismatic joints",
        "mount": "Simulation flange adapter at +Z 0.12 m, RPY=(pi/2,-pi/2,0); hardware calibration not provided",
    }, indent=2) + "\n")
    print(f"Built {OUTPUT / 'kuka_parallel_gripper.urdf'}: 7 arm + 2 jaw joints")


if __name__ == "__main__":
    main()
