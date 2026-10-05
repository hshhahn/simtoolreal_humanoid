"""Build a Cartesian wrist fixture around the unchanged G1 right Wuji hand.

The six fixture joints position the wrist for a capability diagnostic. They are
not additional finger commands and are never used by a training environment.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


FIXTURE_JOINT_NAMES = tuple(f"fixture_{axis}" for axis in ("x", "y", "z", "yaw", "pitch", "roll"))


def build_fixture(source: Path, destination: Path) -> Path:
    source = source.resolve()
    original = ET.parse(source).getroot()
    root = ET.Element("robot", name="wuji_wrist_grasp_diagnostic")

    def link(name: str, mass: float) -> None:
        node = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(inertial, "inertia", ixx="0.1", iyy="0.1", izz="0.1",
                      ixy="0", ixz="0", iyz="0")

    link("fixture_base", 1.0)
    parent = "fixture_base"
    axes = ("1 0 0", "0 1 0", "0 0 1", "0 0 1", "0 1 0", "1 0 0")
    for i, (name, axis) in enumerate(zip(FIXTURE_JOINT_NAMES, axes)):
        child = f"{name}_link"
        link(child, 1.0)
        joint = ET.SubElement(root, "joint", name=name,
                              type="prismatic" if i < 3 else "revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(joint, "limit", lower="-3.15", upper="3.15",
                      effort="5000", velocity="2")
        parent = child
    mount = ET.SubElement(root, "joint", name="fixture_palm_fixed", type="fixed")
    ET.SubElement(mount, "parent", link=parent)
    ET.SubElement(mount, "child", link="right_palm_link")
    names = {"right_palm_link"} | {
        node.get("name") for node in original.findall("link")
        if node.get("name", "").startswith("right_finger")
    }
    for node in original:
        if node.tag == "link" and node.get("name") in names:
            copied = copy.deepcopy(node)
            for mesh in copied.findall(".//mesh"):
                path = Path(mesh.get("filename"))
                if not path.is_absolute():
                    mesh.set("filename", str(source.parent / path))
            root.append(copied)
        elif node.tag == "joint" and all(
            node.find(tag).get("link") in names for tag in ("parent", "child")
        ):
            root.append(copy.deepcopy(node))
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="unicode", xml_declaration=True)
    return destination


def independent_targets(commands, lower, upper):
    """Diagnostic alternative: thumb opposition/flexion + four finger flexions.

    Each finger uses precisely the same joint ratios as the production three
    command mapping. Only the coupling between the four fingers is removed.
    """
    import torch
    from simtoolreal_shared.wuji_synergy import synergy_targets

    if commands.shape[-1] != 6:
        raise ValueError("Independent finger diagnostic expects six hand commands")
    result = synergy_targets(commands[..., :3], lower, upper)
    for finger in range(1, 5):
        shared = torch.stack((commands[..., 0], commands[..., 1], commands[..., finger + 1]), dim=-1)
        targets = synergy_targets(shared, lower, upper)
        result[..., 4 * finger:4 * finger + 4] = targets[..., 4 * finger:4 * finger + 4]
    return result
