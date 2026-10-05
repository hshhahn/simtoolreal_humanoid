"""Pre-compute convex-decomposed collision geometry for DexToolBench objects.

Runs CoACD (offline, outside any simulator) on each object's visual mesh and
writes a SEPARATE ``<name>_decomposed.urdf`` next to the original:

  - visual   = the original ``<name>.obj`` (unchanged, pretty)
  - collision = the CoACD convex parts (``<name>_collision/decomp_*.obj``)
  - inertial  = explicit <mass> + <inertia>

The original ``<name>.urdf`` is never modified. Both Isaac Gym (with V-HACD
disabled) and Isaac Sim load the decomposed URDF and get identical, already
convex collision geometry — no backend-specific runtime decomposition.

The explicit inertial matters because the original URDFs specify mass via a
``<density>`` tag, which Isaac Gym honors but Isaac Sim's URDF importer
ignores (giving a default mass). We compute mass = density x decomposed-hull
volume (the hulls are convex => watertight => reliable, and this mirrors how
Isaac Gym derives mass from the V-HACD hull volume), so both backends agree.
Objects that already ship explicit <mass>+<inertia> keep theirs verbatim.

    python dextoolbench/generate_collision_meshes.py                  # all
    python dextoolbench/generate_collision_meshes.py --object_name claw_hammer

Run ONE instance at a time — concurrent runs race on the shared output dirs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro

from dextoolbench.objects import NAME_TO_OBJECT
from dextoolbench.run_coacd import compute_overshoot, run_coacd

MAX_CONVEX_HULL = 64  # matches Isaac Gym's V-HACD max_convex_hulls; bounds sim cost
THRESHOLD = 0.05      # CoACD concavity (tool-appropriate; fabrica's 0.03 is for thin pegs)


def _mesh_filename(elem: ET.Element | None) -> str | None:
    mesh = elem.find("geometry/mesh") if elem is not None else None
    return mesh.get("filename") if mesh is not None else None


def _inertial_xml(link: ET.Element, parts: list[trimesh.Trimesh], density: float) -> ET.Element:
    """Return an <inertial> element. Preserve the original if it already has an
    explicit <mass>; otherwise compute mass+inertia from density x hull volume."""
    orig = link.find("inertial")
    if orig is not None and orig.find("mass") is not None:
        return ET.fromstring(ET.tostring(orig))  # explicit mass already authored — keep it

    combined = trimesh.util.concatenate(parts)
    combined.density = density
    com = combined.center_mass
    it = combined.moment_inertia  # 3x3 about the center of mass, at this density

    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=f"{com[0]:.6g} {com[1]:.6g} {com[2]:.6g}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{combined.mass:.6g}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{it[0, 0]:.6g}", ixy=f"{it[0, 1]:.6g}", ixz=f"{it[0, 2]:.6g}",
        iyy=f"{it[1, 1]:.6g}", iyz=f"{it[1, 2]:.6g}", izz=f"{it[2, 2]:.6g}",
    )
    return inertial


def _density(link: ET.Element) -> float:
    d = link.find("inertial/density")
    return float(d.get("value")) if d is not None else 400.0


def decompose_object(object_name: str) -> int:
    urdf_path = NAME_TO_OBJECT[object_name].urdf_path
    assert urdf_path.exists(), urdf_path
    obj_dir = urdf_path.parent

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link = root.find("link")
    visual = link.find("visual")
    visual_mesh = _mesh_filename(visual)
    if visual_mesh is None:
        raise ValueError(f"{urdf_path}: no visual mesh")
    mesh_path = obj_dir / visual_mesh
    assert mesh_path.exists(), mesh_path

    collision_dir = obj_dir / f"{mesh_path.stem}_collision"
    parts = run_coacd(
        mesh_path=mesh_path, output_dir=collision_dir,
        max_convex_hull=MAX_CONVEX_HULL, threshold=THRESHOLD,
        mcts_iterations=100, resolution=2000,
    )

    # Build the decomposed URDF: same robot/link name, original visual,
    # N convex collisions, explicit inertial.
    robot = ET.Element("robot", name=root.get("name"))
    new_link = ET.SubElement(robot, "link", name=link.get("name"))
    new_link.append(ET.fromstring(ET.tostring(visual)))
    col_origin = link.find("collision/origin")
    rel = f"{mesh_path.stem}_collision"
    for i in range(len(parts)):
        col = ET.SubElement(new_link, "collision")
        if col_origin is not None:
            col.append(ET.fromstring(ET.tostring(col_origin)))
        geom = ET.SubElement(col, "geometry")
        ET.SubElement(geom, "mesh", filename=f"{rel}/decomp_{i}.obj", scale="1 1 1")
    new_link.append(_inertial_xml(link, parts, _density(link)))

    out_tree = ET.ElementTree(robot)
    ET.indent(out_tree, space="  ")
    out_path = obj_dir / f"{mesh_path.stem}_decomposed.urdf"
    out_tree.write(out_path, xml_declaration=True, encoding="utf-8")

    overshoot_mm = compute_overshoot(mesh_path, collision_dir) * 1000
    print(f"[{object_name}] {len(parts)} hulls, {overshoot_mm:.1f}mm overshoot -> {out_path.name}")
    return len(parts)


@dataclass
class Args:
    object_name: str | None = None
    """Single object to process; default = all DexToolBench objects."""


def main() -> None:
    args = tyro.cli(Args)
    names = [args.object_name] if args.object_name else list(NAME_TO_OBJECT)
    total = sum(decompose_object(n) for n in names)
    print(f"Done. {len(names)} objects, {total} convex parts total.")


if __name__ == "__main__":
    main()
