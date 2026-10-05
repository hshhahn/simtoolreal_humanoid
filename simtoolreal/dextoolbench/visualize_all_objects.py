"""Visualize all tools in the dextoolbench folder using viser.

This script finds all URDF and OBJ files in the dextoolbench assets folder
and displays them organized by tool type with section titles.

Keypoints are shown at corners of the handle bounding box based on scale from objects.py.
"""

import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import trimesh
import viser

from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir

# Base directory for all dextoolbench assets
ASSETS_DIR = get_repo_root_dir() / "assets/urdf/dextoolbench"

# Colors (RGB 0-255)
WOOD_COLOR = (139, 90, 43)  # Dark walnut brown
METAL_COLOR = (105, 105, 105)  # Dark steel gray
KEYPOINT_COLOR = (255, 50, 50)  # Bright red for keypoints

# Keypoint configuration
# KEYPOINT_RADIUS = 0.006  # Radius of keypoint spheres
KEYPOINT_RADIUS = 0.0  # Radius of keypoint spheres
KEYPOINT_SCALE = 1.0  # Scale factor for keypoint positions (1.0 = at handle corners)
OBJECT_BASE_SIZE = 0.04  # Base size used in training
KEYPOINT_OFFSETS = [
    [1, 1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
]  # Normalized keypoint positions (corners of bounding box)

# Tools organized by category (2 per category, displayed side by side)
# Each row is one category with 2 objects
TOOLS_BY_CATEGORY = [
    ("brush", ["blue_brush", "red_brush"]),
    ("hammer", ["claw_hammer", "mallet_hammer"]),
    ("eraser", ["handle_eraser", "flat_eraser"]),
    ("marker", ["staples_marker", "sharpie_marker"]),
    ("screwdriver", ["short_screwdriver", "long_screwdriver"]),
    ("spatula", ["flat_spatula", "spoon_spatula"]),
]


# If filename starts with "primitive_", sort it to the start
def sort_primitive_tools_to_start(tool: Tuple[Path, Optional[Path]]) -> Tuple[int, str]:
    urdf_path = tool[0]
    filename = urdf_path.stem
    if filename.startswith("primitive_"):
        return (0, filename)  # Sort to beginning
    else:
        return (1, filename)  # Sort to end


def get_object_name_from_urdf(urdf_path: Path) -> Optional[str]:
    """Try to find matching object name from NAME_TO_OBJECT based on URDF path."""
    urdf_stem = urdf_path.stem

    # Direct match by name
    if urdf_stem in NAME_TO_OBJECT:
        return urdf_stem

    # Try to match by filepath
    for name, obj in NAME_TO_OBJECT.items():
        if obj.urdf_path == urdf_path or obj.urdf_path.stem == urdf_stem:
            return name

    return None


def compute_keypoints_from_scale(
    scale: Tuple[float, float, float],
) -> List[Tuple[float, float, float]]:
    """Compute keypoint positions based on object scale from objects.py.

    The scale represents the normalized object size used in training.
    Keypoints are placed at corners of the handle bounding box.

    Returns:
        List of (x, y, z) keypoint positions relative to handle center.
    """
    # Convert scale to half-extents (matching training code logic)
    # keypoint[coord] = scale[coord] * object_base_size * keypoint_scale / 2
    hx = scale[0] * OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2
    hy = scale[1] * OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2
    hz = scale[2] * OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2

    # Generate keypoints at corners based on normalized offsets
    keypoints = []
    for offset in KEYPOINT_OFFSETS:
        kp = (
            offset[0] * hx,
            offset[1] * hy,
            offset[2] * hz,
        )
        keypoints.append(kp)

    return keypoints


def parse_urdf_for_mesh(urdf_path: Path) -> Optional[Path]:
    """Parse URDF file to find the mesh filename if it uses mesh geometry.

    Returns:
        Path to the mesh file if found, else None.
    """
    with open(urdf_path, "r") as f:
        content = f.read()

    # Look for mesh filename in visual geometry
    mesh_match = re.search(r'<visual>.*?<mesh filename="([^"]+)"', content, re.DOTALL)
    if mesh_match:
        mesh_filename = mesh_match.group(1)
        mesh_path = urdf_path.parent / mesh_filename
        if mesh_path.exists():
            return mesh_path

    return None


def parse_urdf_for_primitives(
    urdf_path: Path,
) -> Tuple[Optional[dict], Optional[dict], Optional[float]]:
    """Parse URDF file to check if it uses primitive shapes (box/cylinder).

    Returns:
        (handle_params, head_params, head_offset_x) if primitives found, else (None, None, None)
    """
    with open(urdf_path, "r") as f:
        content = f.read()

    # Check if this URDF uses mesh geometry (not primitives)
    if "<mesh filename=" in content:
        return None, None, None

    # Find all visual elements with primitive geometry
    visual_pattern = (
        r'<visual>\s*<origin xyz="([^"]+)"[^>]*/>.*?<geometry>\s*(.*?)\s*</geometry>'
    )
    visuals = re.findall(visual_pattern, content, re.DOTALL)

    if len(visuals) < 1:
        return None, None, None

    handle_params = None
    head_params = None
    head_offset = None

    for origin_xyz, geometry in visuals:
        xyz = [float(x) for x in origin_xyz.split()]
        x_offset = xyz[0]

        # Parse geometry
        box_match = re.search(r'<box size="([^"]+)"/>', geometry)
        cylinder_match = re.search(
            r'<cylinder length="([^"]+)" radius="([^"]+)"/>', geometry
        )

        params = None
        if box_match:
            sizes = [float(s) for s in box_match.group(1).split()]
            params = {
                "type": "box",
                "length": sizes[0],
                "width": sizes[1],
                "height": sizes[2],
            }
        elif cylinder_match:
            params = {
                "type": "cylinder",
                "length": float(cylinder_match.group(1)),
                "radius": float(cylinder_match.group(2)),
            }

        if params:
            if abs(x_offset) < 0.001:  # Handle is at origin
                handle_params = params
            else:  # Head is offset
                head_params = params
                head_offset = x_offset

    return handle_params, head_params, head_offset


def create_mesh_from_params(
    params: dict, is_head: bool = False
) -> Optional[trimesh.Trimesh]:
    """Create a trimesh from geometry parameters."""
    if params is None:
        return None

    if params["type"] == "box":
        mesh = trimesh.creation.box(
            extents=(params["length"], params["width"], params["height"])
        )
    elif params["type"] == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=params["radius"], height=params["length"]
        )
        if is_head:
            # Rotate 90 degrees around X axis to align with Y axis
            rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        else:
            # Rotate 90 degrees around Y axis to align with X axis
            rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
        mesh.apply_transform(rotation)
    else:
        return None

    return mesh


def find_all_tools_by_type(
    base_dir: Path,
) -> Dict[str, List[Tuple[Path, Optional[Path]]]]:
    """Find all tools grouped by tool type (parent folder).

    Returns:
        Dict mapping tool_type -> list of (urdf_path, obj_path) tuples.
    """
    tools_by_type = defaultdict(list)
    urdf_files = sorted(base_dir.rglob("*.urdf"))

    for urdf_path in urdf_files:
        # Get tool type from folder structure: base_dir / tool_type / tool_name / file.urdf
        relative_path = urdf_path.relative_to(base_dir)
        if len(relative_path.parts) >= 2:
            tool_type = relative_path.parts[0]
        else:
            tool_type = "other"

        # Look for matching OBJ file
        obj_path = urdf_path.with_suffix(".obj")
        if not obj_path.exists():
            obj_files = list(urdf_path.parent.glob("*.obj"))
            obj_path = obj_files[0] if obj_files else None

        tools_by_type[tool_type].append((urdf_path, obj_path))

    return dict(tools_by_type)


def main() -> None:
    """Visualize all 12 tools in a 2x6 grid layout, grouped by category."""

    # Get tools organized by category from NAME_TO_OBJECT
    tools_data = []  # List of (category, tool_name, urdf_path, obj_path)
    for category, tool_names in TOOLS_BY_CATEGORY:
        for tool_name in tool_names:
            if tool_name in NAME_TO_OBJECT:
                urdf_path = NAME_TO_OBJECT[tool_name].urdf_path
                # Look for matching OBJ file
                obj_path = urdf_path.with_suffix(".obj")
                if not obj_path.exists():
                    obj_files = list(urdf_path.parent.glob("*.obj"))
                    obj_path = obj_files[0] if obj_files else None
                tools_data.append((category, tool_name, urdf_path, obj_path))
                print(f"Found: {tool_name} ({category}) -> {urdf_path}")
            else:
                print(f"WARNING: Tool not found in NAME_TO_OBJECT: {tool_name}")

    total_tools = len(tools_data)
    print(f"\nUsing {total_tools} tools in {len(TOOLS_BY_CATEGORY)} categories")

    # Start viser server
    server = viser.ViserServer(port=8080)
    print("\nViser server running at http://localhost:8080")

    @server.on_client_connect
    def _(client):
        client.camera.position = (0.0, 0.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.0)

    # Grid layout parameters (2x6 grid: 2 columns, 6 rows)
    spacing_x = 0.35  # Spacing between objects in same category (side by side)
    spacing_y_between = 0.14  # Spacing between categories
    num_cols = 2
    num_categories = len(TOOLS_BY_CATEGORY)

    # Center the grid
    grid_width = (num_cols - 1) * spacing_x
    grid_height = (num_categories - 1) * spacing_y_between
    x_start = -grid_width / 2
    y_start = -grid_height / 2

    # Load tools in grid layout (each category is one row)
    idx = 0
    for cat_idx, (category, tool_names) in enumerate(TOOLS_BY_CATEGORY):
        y_pos = y_start + cat_idx * spacing_y_between

        for col, tool_name in enumerate(tool_names):
            x_pos = x_start + col * spacing_x

            # Find the tool data
            tool_entry = next((t for t in tools_data if t[1] == tool_name), None)
            if tool_entry is None:
                continue
            _, _, urdf_path, obj_path = tool_entry

            print(
                f"  [{idx + 1}/{total_tools}] Loading: {tool_name} ({category}) at row {cat_idx}, col {col}"
            )

            # Create frame for this tool
            server.scene.add_frame(
                f"/tool_{idx:04d}_{tool_name}",
                position=(x_pos, y_pos, 0.0),
                axes_length=0.0,
                axes_radius=0.0,
            )

            # Try to parse as primitive URDF first
            handle_params, head_params, head_offset = parse_urdf_for_primitives(
                urdf_path
            )

            if handle_params:
                # Primitive-based object: create colored meshes for handle and head
                handle_mesh = create_mesh_from_params(handle_params, is_head=False)
                if handle_mesh:
                    handle_mesh.visual.face_colors = [*WOOD_COLOR, 255]
                    server.scene.add_mesh_trimesh(
                        name=f"/tool_{idx:04d}_{tool_name}/handle",
                        mesh=handle_mesh,
                    )

                if head_params and head_offset:
                    head_mesh = create_mesh_from_params(head_params, is_head=True)
                    if head_mesh:
                        head_mesh.apply_translation([head_offset, 0, 0])
                        head_mesh.visual.face_colors = [*METAL_COLOR, 255]
                        server.scene.add_mesh_trimesh(
                            name=f"/tool_{idx:04d}_{tool_name}/head",
                            mesh=head_mesh,
                        )
            else:
                # Mesh-based object: try to find mesh from URDF or fallback to OBJ search
                mesh_path = parse_urdf_for_mesh(urdf_path)
                if mesh_path is None and obj_path and obj_path.exists():
                    mesh_path = obj_path

                if mesh_path and mesh_path.exists():
                    try:
                        mesh = trimesh.load(str(mesh_path), process=False)
                        server.scene.add_mesh_trimesh(
                            name=f"/tool_{idx:04d}_{tool_name}/mesh",
                            mesh=mesh,
                        )
                    except Exception as e:
                        print(f"    Warning: Failed to load mesh {mesh_path}: {e}")

            # Add keypoints based on object scale from objects.py
            object_name = get_object_name_from_urdf(urdf_path)
            if object_name and object_name in NAME_TO_OBJECT:
                scale = NAME_TO_OBJECT[object_name].scale
                keypoints = compute_keypoints_from_scale(scale)
                for kp_idx, kp_pos in enumerate(keypoints):
                    server.scene.add_icosphere(
                        name=f"/tool_{idx:04d}_{tool_name}/keypoint_{kp_idx}",
                        radius=KEYPOINT_RADIUS,
                        color=KEYPOINT_COLOR,
                        position=kp_pos,
                    )
            else:
                print(
                    f"    Warning: No scale found for {tool_name}, skipping keypoints"
                )

            idx += 1

    print(
        f"\nLoaded {total_tools} tools in {num_categories} categories (2x{num_categories} grid)"
    )
    print("Press Ctrl+C to exit.")

    # Keep the server running
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
