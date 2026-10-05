"""View .obj and .urdf files using viser.

This script allows you to visualize .obj mesh files and .urdf robot description files
either separately or together in the same scene.

Usage:
    # View an OBJ file
    python view_object_urdf.py --obj-path /path/to/mesh.obj

    # View a URDF file
    python view_object_urdf.py --urdf-path /path/to/robot.urdf

    # View both in the same scene (offset from each other)
    python view_object_urdf.py --obj-path /path/to/mesh.obj --urdf-path /path/to/robot.urdf
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import trimesh
import tyro
import viser
from viser.extras import ViserUrdf


@dataclass
class VisualizeObjectArgs:
    """Arguments for viewing .obj and .urdf files."""

    obj_path: Optional[Path] = None
    """Path to .obj file to view."""

    urdf_path: Optional[Path] = None
    """Path to .urdf file to view."""

    port: int = 8080
    """Port for viser server."""

    obj_offset: float = -0.3
    """X offset for OBJ mesh when viewing both."""

    urdf_offset: float = 0.3
    """X offset for URDF when viewing both."""


def main() -> None:
    """View .obj and/or .urdf files in the same viser scene."""
    args: VisualizeObjectArgs = tyro.cli(VisualizeObjectArgs)

    if args.obj_path is None and args.urdf_path is None:
        raise ValueError("Must provide either --obj-path or --urdf-path (or both)")

    # Start viser server
    server = viser.ViserServer(port=args.port)
    print(f"Viser server running at http://localhost:{args.port}")

    @server.on_client_connect
    def _(client):
        client.camera.position = (0.0, 0.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.0)

    min_z = 0.0

    # Load OBJ if provided
    if args.obj_path is not None:
        assert args.obj_path.exists(), f"OBJ file not found: {args.obj_path}"
        print(f"Loading OBJ file: {args.obj_path}")

        mesh = trimesh.load(str(args.obj_path), force="mesh")
        print(
            f"Loaded OBJ mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
        )

        # Determine offset (only apply if both are provided)
        obj_x_offset = args.obj_offset if args.urdf_path is not None else 0.0

        # Add frame with axes for OBJ
        server.scene.add_frame(
            "/obj",
            position=(obj_x_offset, 0.0, 0.0),
            axes_length=0.1,
            axes_radius=0.005,
        )

        # Add mesh to scene as child of frame
        server.scene.add_mesh_simple(
            name="/obj/mesh",
            vertices=mesh.vertices,
            faces=mesh.faces,
        )

        if mesh.bounds is not None:
            min_z = min(min_z, mesh.bounds[0, 2])

    # Load URDF if provided
    if args.urdf_path is not None:
        assert args.urdf_path.exists(), f"URDF file not found: {args.urdf_path}"
        print(f"Loading URDF file: {args.urdf_path}")

        # Determine offset (only apply if both are provided)
        urdf_x_offset = args.urdf_offset if args.obj_path is not None else 0.0

        # Load URDF using ViserUrdf with a root node that has the offset
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=args.urdf_path,
            root_node_name="/urdf",
            load_meshes=True,
            load_collision_meshes=False,
        )

        # Set the URDF root position to apply offset with axes
        server.scene.add_frame(
            "/urdf",
            position=(urdf_x_offset, 0.0, 0.0),
            axes_length=0.1,
            axes_radius=0.005,
        )

        # Get trimesh scene for grid positioning
        trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
        if trimesh_scene is not None:
            min_z = min(min_z, trimesh_scene.bounds[0, 2])

    # Add grid for reference
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(0.0, 0.0, min_z),
    )

    print("\nLoaded:")
    if args.obj_path is not None:
        print(f"  OBJ:  {args.obj_path.name}")
    if args.urdf_path is not None:
        print(f"  URDF: {args.urdf_path.name}")
    print("\nPress Ctrl+C to exit.")

    # Keep the server running
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
