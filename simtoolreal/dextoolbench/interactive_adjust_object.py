#!/usr/bin/env python3
"""Interactive mesh converter with measurement tools.

Features:
- Load GLB/OBJ/PLY/STL mesh files
- Interactive measurement tool (click two points to measure distance)
- Scale mesh to match real-world dimensions
- Rotate and translate mesh origin
- Export to OBJ + STL + URDF

Usage:
    python interactive_adjust_object.py --mesh-path /path/to/mesh.glb
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro
import viser
from scipy.spatial.transform import Rotation as R


@dataclass
class InteractiveAdjustObjectArgs:
    mesh_path: Path
    """Path to the input mesh file (GLB/OBJ/PLY/STL)."""

    output_dir: Optional[Path] = None
    """Directory to save exported files. Defaults to the same directory as the input mesh."""

    name: Optional[str] = None
    """Name for the exported files. Defaults to the input mesh file stem."""

    center: bool = True
    """Center the mesh at the origin on load."""

    port: int = 8080
    """Viser server port."""


def load_mesh(mesh_path: Path, center: bool = True) -> trimesh.Trimesh:
    """Load mesh file and optionally center it."""
    data = trimesh.load(str(mesh_path))

    if isinstance(data, trimesh.Scene):
        mesh = data.dump(concatenate=True)
    elif isinstance(data, trimesh.Trimesh):
        mesh = data
    else:
        raise ValueError(f"Unsupported format: {type(data).__name__}")

    if center:
        mesh.vertices -= mesh.centroid

    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def create_urdf(path: Path, name: str, obj_file: str, density: float = 400.0) -> None:
    """Create URDF file referencing the OBJ mesh."""
    urdf = f'''<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_file}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_file}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>
</robot>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(urdf)


class MeshEditor:
    """Interactive mesh editor with measurement and transform tools."""

    def __init__(
        self, mesh: trimesh.Trimesh, output_dir: Path, name: str, port: int = 8080
    ):
        self.original_mesh = mesh
        self.mesh = mesh.copy()
        self.output_dir = output_dir
        self.name = name
        self.saved = False

        # Measurement state
        self.measure_points: list = []
        self.measure_mode = False

        # Guard flag to prevent recursive slider updates
        self._updating_sliders = False

        # Create server
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._setup()

        print(f"\n{'=' * 50}")
        print(f"Viser: http://localhost:{port}")
        print(f"Output: {output_dir}")
        print(f"{'=' * 50}\n")

    def _setup(self):
        """Initialize scene and GUI."""

        # Camera setup
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.5, -0.5, 0.3)
            client.camera.look_at = (0.0, 0.0, 0.0)

        # Scene elements
        self.server.scene.add_grid("/ground", width=1, height=1, cell_size=0.05)
        self.server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.003)

        # Add mesh
        self._update_mesh()
        self._update_bbox()

        # GUI
        self._setup_gui()

    def _setup_gui(self):
        """Setup all GUI controls."""

        # === MEASUREMENT TOOL ===
        with self.server.gui.add_folder("📏 Measurement"):
            self.server.gui.add_markdown(
                "Click **two points** on mesh to measure distance.\n"
                "Use to determine scale for real-world dimensions."
            )
            self.measure_btn = self.server.gui.add_button("Start Measuring")
            self.measure_btn.on_click(self._toggle_measure_mode)

            self.clear_measure_btn = self.server.gui.add_button("Clear Points")
            self.clear_measure_btn.on_click(self._clear_measurements)

            self.measure_info = self.server.gui.add_markdown("*Click Start to begin*")

            self.server.gui.add_markdown("---")
            self.server.gui.add_markdown("**Set Target Dimension:**")
            self.target_dim = self.server.gui.add_number(
                "Target size (m)", initial_value=0.1, min=0.001, max=10.0, step=0.001
            )
            self.apply_scale_btn = self.server.gui.add_button("Apply Scale to Match")
            self.apply_scale_btn.on_click(self._apply_measured_scale)

        # === SCALE ===
        with self.server.gui.add_folder("📐 Scale"):
            self.scale_x = self.server.gui.add_slider(
                "X", min=0.01, max=10.0, step=0.01, initial_value=1.0
            )
            self.scale_y = self.server.gui.add_slider(
                "Y", min=0.01, max=10.0, step=0.01, initial_value=1.0
            )
            self.scale_z = self.server.gui.add_slider(
                "Z", min=0.01, max=10.0, step=0.01, initial_value=1.0
            )

            self.uniform_scale = self.server.gui.add_checkbox(
                "Uniform scale", initial_value=True
            )

            self.scale_x.on_update(lambda _: self._on_scale_change("x"))
            self.scale_y.on_update(lambda _: self._on_scale_change("y"))
            self.scale_z.on_update(lambda _: self._on_scale_change("z"))

        # === ROTATION ===
        with self.server.gui.add_folder("🔄 Rotation"):
            self.rot_x = self.server.gui.add_slider(
                "X (deg)", min=-180.0, max=180.0, step=5.0, initial_value=0.0
            )
            self.rot_y = self.server.gui.add_slider(
                "Y (deg)", min=-180.0, max=180.0, step=5.0, initial_value=0.0
            )
            self.rot_z = self.server.gui.add_slider(
                "Z (deg)", min=-180.0, max=180.0, step=5.0, initial_value=0.0
            )

            self.rot_x.on_update(lambda _: self._apply_transforms())
            self.rot_y.on_update(lambda _: self._apply_transforms())
            self.rot_z.on_update(lambda _: self._apply_transforms())

        # === ORIGIN/TRANSLATION ===
        with self.server.gui.add_folder("📍 Origin Offset"):
            self.off_x = self.server.gui.add_slider(
                "X", min=-0.5, max=0.5, step=0.005, initial_value=0.0
            )
            self.off_y = self.server.gui.add_slider(
                "Y", min=-0.5, max=0.5, step=0.005, initial_value=0.0
            )
            self.off_z = self.server.gui.add_slider(
                "Z", min=-0.5, max=0.5, step=0.005, initial_value=0.0
            )

            self.off_x.on_update(lambda _: self._apply_transforms())
            self.off_y.on_update(lambda _: self._apply_transforms())
            self.off_z.on_update(lambda _: self._apply_transforms())

        # === INFO ===
        with self.server.gui.add_folder("📊 Mesh Info"):
            self.bounds_info = self.server.gui.add_markdown("--")
            self._update_info()

            print_btn = self.server.gui.add_button("Print Bounds to Console")
            print_btn.on_click(lambda _: self._print_bounds())

        # === EXPORT ===
        with self.server.gui.add_folder("💾 Export"):
            self.density = self.server.gui.add_slider(
                "Density (kg/m³)", min=100.0, max=2000.0, step=50.0, initial_value=400.0
            )

            self.save_btn = self.server.gui.add_button("Save OBJ + URDF")
            self.save_btn.on_click(lambda _: self._save())

            self.status = self.server.gui.add_markdown("*Ready*")

        # === RESET ===
        reset_btn = self.server.gui.add_button("Reset All")
        reset_btn.on_click(lambda _: self._reset())

    def _toggle_measure_mode(self, _):
        """Toggle measurement mode."""
        self.measure_mode = not self.measure_mode
        if self.measure_mode:
            self.measure_btn.name = "Stop Measuring"
            self.measure_info.content = "**Mode: ACTIVE**\nClick on mesh..."
            self._enable_mesh_click()
        else:
            self.measure_btn.name = "Start Measuring"
            self._disable_mesh_click()

    def _enable_mesh_click(self):
        """Enable click handler on mesh."""
        if hasattr(self, "mesh_handle") and self.mesh_handle is not None:

            @self.mesh_handle.on_click
            def on_click(event: viser.ScenePointerEvent):
                if not self.measure_mode:
                    return

                # Find click point on mesh via ray casting
                ray_origin = np.array(event.ray_origin)
                ray_dir = np.array(event.ray_direction)

                # Use trimesh ray casting
                locations, _, _ = self.mesh.ray.intersects_location(
                    [ray_origin], [ray_dir]
                )

                if len(locations) > 0:
                    point = locations[0]
                    self._add_measure_point(point)

    def _disable_mesh_click(self):
        """Disable mesh click (by not handling it)."""
        pass

    def _add_measure_point(self, point: np.ndarray):
        """Add a measurement point."""
        idx = len(self.measure_points)

        # Add sphere at point
        self.server.scene.add_icosphere(
            f"/measure/point_{idx}",
            radius=0.005,
            color=(255, 50, 50),
            position=tuple(point),
        )

        self.measure_points.append(point)

        if len(self.measure_points) == 1:
            self.measure_info.content = f"**Point 1:** ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})\nClick second point..."

        elif len(self.measure_points) >= 2:
            p1, p2 = self.measure_points[-2], self.measure_points[-1]
            dist = np.linalg.norm(p2 - p1)

            # Draw line between points
            self.server.scene.add_spline_catmull_rom(
                f"/measure/line_{idx}",
                positions=np.array([p1, p2], dtype=np.float32),
                color=(255, 100, 100),
                line_width=3.0,
            )

            self.measure_info.content = (
                f"**Point 1:** ({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f})\n"
                f"**Point 2:** ({p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f})\n"
                f"**Distance:** {dist:.4f} m ({dist * 100:.2f} cm)\n\n"
                f"Set target dimension above and click 'Apply Scale'"
            )

            print(f"Measured distance: {dist:.4f} m")

    def _clear_measurements(self, _):
        """Clear all measurement points."""
        for i in range(len(self.measure_points) + 1):
            try:
                self.server.scene.remove(f"/measure/point_{i}")
                self.server.scene.remove(f"/measure/line_{i}")
            except Exception as e:
                print(f"Error clearing measurements: {e}")
        self.measure_points = []
        self.measure_info.content = "*Cleared*"

    def _apply_measured_scale(self, _):
        """Apply scale based on measurement and target dimension."""
        if len(self.measure_points) < 2:
            self.measure_info.content = "**Error:** Need 2 points first!"
            return

        p1, p2 = self.measure_points[-2], self.measure_points[-1]
        current_dist = np.linalg.norm(p2 - p1)
        target_dist = self.target_dim.value

        if current_dist < 1e-6:
            self.measure_info.content = "**Error:** Points too close!"
            return

        scale_factor = target_dist / current_dist

        # Apply uniform scale
        self.scale_x.value = scale_factor
        self.scale_y.value = scale_factor
        self.scale_z.value = scale_factor

        self._apply_transforms()
        self._clear_measurements(None)

        self.measure_info.content = (
            f"**Applied scale:** {scale_factor:.4f}\nMesh now matches target dimension."
        )
        print(f"Applied scale factor: {scale_factor:.4f}")

    def _on_scale_change(self, axis: str):
        """Handle scale slider change."""
        if self._updating_sliders:
            return

        if self.uniform_scale.value:
            # Sync all scales with guard to prevent recursion
            self._updating_sliders = True
            val = getattr(self, f"scale_{axis}").value
            self.scale_x.value = val
            self.scale_y.value = val
            self.scale_z.value = val
            self._updating_sliders = False

        self._apply_transforms()

    def _apply_transforms(self):
        """Apply all transforms to mesh."""
        self.mesh = self.original_mesh.copy()

        # 1. Rotation (euler XYZ)
        rx = np.radians(self.rot_x.value)
        ry = np.radians(self.rot_y.value)
        rz = np.radians(self.rot_z.value)
        rot_matrix = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
        self.mesh.vertices = self.mesh.vertices @ rot_matrix.T

        # 2. Scale
        scale = np.array([self.scale_x.value, self.scale_y.value, self.scale_z.value])
        self.mesh.vertices *= scale

        # 3. Translation (origin offset)
        offset = np.array([self.off_x.value, self.off_y.value, self.off_z.value])
        self.mesh.vertices += offset

        self._update_mesh()
        self._update_bbox()
        self._update_info()

    def _update_mesh(self):
        """Update mesh visualization."""
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.astype(np.uint32)

        # Try to get vertex colors
        vertex_colors = None
        if (
            hasattr(self.mesh.visual, "vertex_colors")
            and self.mesh.visual.vertex_colors is not None
        ):
            vc = self.mesh.visual.vertex_colors
            if vc.shape[0] == len(vertices):
                vertex_colors = vc[:, :3].astype(np.uint8)

        if vertex_colors is not None:
            self.mesh_handle = self.server.scene.add_mesh_simple(
                "/mesh",
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                flat_shading=False,
            )
        else:
            self.mesh_handle = self.server.scene.add_mesh_simple(
                "/mesh",
                vertices=vertices,
                faces=faces,
                color=(180, 180, 180),
                flat_shading=True,
            )

        # Re-enable click handler if in measure mode
        if self.measure_mode:
            self._enable_mesh_click()

    def _update_bbox(self):
        """Update bounding box visualization."""
        bounds = self.mesh.bounds
        min_pt, max_pt = bounds[0], bounds[1]

        corners = np.array(
            [
                [min_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
            ]
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for i, (a, b) in enumerate(edges):
            self.server.scene.add_spline_catmull_rom(
                f"/bbox/edge_{i}",
                positions=np.array([corners[a], corners[b]], dtype=np.float32),
                color=(255, 200, 0),
                line_width=2.0,
            )

    def _update_info(self):
        """Update mesh info display."""
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]

        self.bounds_info.content = (
            f"**Bounds:**\n"
            f"- X: [{bounds[0][0]:.3f}, {bounds[1][0]:.3f}]\n"
            f"- Y: [{bounds[0][1]:.3f}, {bounds[1][1]:.3f}]\n"
            f"- Z: [{bounds[0][2]:.3f}, {bounds[1][2]:.3f}]\n\n"
            f"**Size:** {size[0]:.3f} × {size[1]:.3f} × {size[2]:.3f} m"
        )

    def _print_bounds(self):
        """Print bounds to console."""
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        print(f"\n{'=' * 40}")
        print(f"Mesh: {self.name}")
        print(f"X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}]  size: {size[0]:.4f} m")
        print(f"Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}]  size: {size[1]:.4f} m")
        print(f"Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}]  size: {size[2]:.4f} m")
        print(f"{'=' * 40}\n")

    def _reset(self):
        """Reset all transforms."""
        self._updating_sliders = True
        self.scale_x.value = 1.0
        self.scale_y.value = 1.0
        self.scale_z.value = 1.0
        self.rot_x.value = 0.0
        self.rot_y.value = 0.0
        self.rot_z.value = 0.0
        self.off_x.value = 0.0
        self.off_y.value = 0.0
        self.off_z.value = 0.0
        self._updating_sliders = False
        self._apply_transforms()
        self._clear_measurements(None)
        self.status.content = "*Reset complete*"

    def _save(self):
        """Save mesh and URDF."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        obj_path = self.output_dir / f"{self.name}.obj"
        stl_path = self.output_dir / f"{self.name}.stl"
        urdf_path = self.output_dir / f"{self.name}.urdf"

        # Export meshes
        self.mesh.export(str(obj_path), file_type="obj")
        self.mesh.export(str(stl_path), file_type="stl")

        # Create URDF
        create_urdf(
            urdf_path, self.name, f"{self.name}.obj", density=self.density.value
        )

        print("\nSaved:")
        print(f"  {obj_path}")
        print(f"  {stl_path}")
        print(f"  {urdf_path}")

        self.status.content = f"**✓ Saved to:**\n`{self.output_dir}`"
        self.saved = True

    def run(self):
        """Run editor until saved or interrupted."""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")


def main():
    args: InteractiveAdjustObjectArgs = tyro.cli(InteractiveAdjustObjectArgs)

    assert args.mesh_path.exists(), f"File not found: {args.mesh_path}"

    name = args.name if args.name is not None else args.mesh_path.stem
    output_dir = (
        args.output_dir if args.output_dir is not None else args.mesh_path.parent
    )

    print(f"Loading: {args.mesh_path}")
    mesh = load_mesh(args.mesh_path, center=args.center)

    editor = MeshEditor(mesh, output_dir, name, port=args.port)
    editor.run()


if __name__ == "__main__":
    main()
