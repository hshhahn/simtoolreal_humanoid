"""Visualize all tools in the dextoolbench_training folder using viser.

This script finds all URDF files in the dextoolbench_training assets folder
and displays them in a RANDOMIZED grid layout to showcase diversity.

Handle is colored with wooden color, head with gray metallic color.
"""

import math
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import trimesh
import viser

from isaacgymenvs.utils.utils import get_repo_root_dir

# Base directory for training tools assets
ASSETS_DIR = get_repo_root_dir() / "assets/urdf/dextoolbench_training"

# Colors (RGB 0-255)
WOOD_COLOR = (139, 90, 43)  # Dark walnut brown
METAL_COLOR = (105, 105, 105)  # Dark steel gray

# Random seed for reproducibility (change this to get different arrangements)
RANDOM_SEED = 42


def parse_urdf_geometry(
    urdf_path: Path,
) -> Tuple[Optional[dict], Optional[dict], Optional[float]]:
    """Parse URDF file to extract handle and head geometry parameters.

    Returns:
        (handle_params, head_params, head_offset_x)
        handle_params: dict with 'type' ('box' or 'cylinder') and dimensions
        head_params: dict with 'type' ('box' or 'cylinder') and dimensions
        head_offset_x: x offset of head from origin
    """
    with open(urdf_path, "r") as f:
        content = f.read()

    # Find all visual elements
    visual_pattern = (
        r'<visual>\s*<origin xyz="([^"]+)"[^>]*/>.*?<geometry>\s*(.*?)\s*</geometry>'
    )
    visuals = re.findall(visual_pattern, content, re.DOTALL)

    if len(visuals) < 2:
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


def find_all_tools(base_dir: Path) -> List[Path]:
    """Find all URDF tools in the directory.

    Returns:
        List of urdf_paths.
    """
    urdf_files = sorted(base_dir.rglob("*.urdf"))
    return urdf_files


def main() -> None:
    """Visualize all tools in a randomized grid layout."""

    # Find all tools
    all_tools = find_all_tools(ASSETS_DIR)

    if not all_tools:
        print(f"No tools found in {ASSETS_DIR}")
        print(
            "You should run `python dextoolbench/generate_training_objects.py` to generate the tools."
        )
        return

    total_tools = len(all_tools)
    print(f"Found {total_tools} tools")

    # Shuffle tools for random arrangement
    random.seed(RANDOM_SEED)
    random.shuffle(all_tools)
    print(f"Shuffled with seed={RANDOM_SEED}")

    # Start viser server
    server = viser.ViserServer(port=8080)
    print("\nViser server running at http://localhost:8080")

    @server.on_client_connect
    def _(client):
        client.camera.position = (0.0, 0.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.0)

    # Grid layout parameters - TIGHT spacing for paper figure
    spacing_x = 0.35  # Horizontal spacing (wider for tool length)
    spacing_y = 0.12  # Tight vertical spacing

    # Compute grid dimensions (roughly square)
    num_cols = int(math.ceil(math.sqrt(total_tools)))
    num_rows = int(math.ceil(total_tools / num_cols))

    print(f"Grid layout: {num_rows} rows x {num_cols} cols")

    # Center the grid
    grid_width = (num_cols - 1) * spacing_x
    grid_height = (num_rows - 1) * spacing_y
    x_start = -grid_width / 2
    y_start = -grid_height / 2

    # Load tools in grid layout
    for idx, urdf_path in enumerate(all_tools):
        row = idx // num_cols
        col = idx % num_cols

        x_pos = x_start + col * spacing_x
        y_pos = y_start + row * spacing_y

        tool_name = urdf_path.stem
        print(f"  [{idx + 1}/{total_tools}] Loading: {tool_name} at ({row}, {col})")

        # Parse URDF to get handle and head geometry
        handle_params, head_params, head_offset = parse_urdf_geometry(urdf_path)

        # Create frame for this tool
        server.scene.add_frame(
            f"/tool_{idx:04d}_{tool_name}",
            position=(x_pos, y_pos, 0.0),
            axes_length=0.0,
            axes_radius=0.0,
        )

        # Create and add handle mesh (wooden color)
        if handle_params:
            handle_mesh = create_mesh_from_params(handle_params, is_head=False)
            if handle_mesh:
                handle_mesh.visual.face_colors = [*WOOD_COLOR, 255]
                server.scene.add_mesh_trimesh(
                    name=f"/tool_{idx:04d}_{tool_name}/handle",
                    mesh=handle_mesh,
                )

        # Create and add head mesh (metallic gray color)
        if head_params and head_offset:
            head_mesh = create_mesh_from_params(head_params, is_head=True)
            if head_mesh:
                head_mesh.apply_translation([head_offset, 0, 0])
                head_mesh.visual.face_colors = [*METAL_COLOR, 255]
                server.scene.add_mesh_trimesh(
                    name=f"/tool_{idx:04d}_{tool_name}/head",
                    mesh=head_mesh,
                )

    print(f"\nLoaded {total_tools} tools in {num_rows}x{num_cols} grid")
    print(f"Grid size: {grid_width:.2f} x {grid_height:.2f} meters")
    print("Press Ctrl+C to exit.")

    # Keep the server running
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
