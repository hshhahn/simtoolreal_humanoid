"""URDF Robot Sphere Editor

Visualize robot models and interactively create/edit collision spheres.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser._scene_handles import FrameHandle, SceneNodeHandle, TransformControlsHandle
from viser.extras import ViserUrdf

from isaacgymenvs.utils.utils import get_repo_root_dir


# --- Data Structures ---
class SphereDef:
    def __init__(self, center: List[float], radius: float):
        self.center = center
        self.radius = radius


class LinkSpheres:
    def __init__(self):
        self.spheres: List[SphereDef] = []


RobotSpheres = Dict[str, LinkSpheres]


# --- Helper Functions ---
def create_robot_control_sliders(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    link_name_to_frame: Dict[str, FrameHandle],
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []

    joints = viser_urdf.get_actuated_joint_limits()

    for joint_name in sorted(joints.keys()):
        lower, upper = joints[joint_name]
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0

        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )

        def slider_on_update(_):
            vals = np.array([s.value for s in slider_handles])
            viser_urdf.update_cfg(vals)
            update_frames(viser_urdf, link_name_to_frame)

        slider.on_update(slider_on_update)
        slider_handles.append(slider)
        initial_config.append(initial_pos)

    return slider_handles, initial_config


def update_frames(viser_urdf: ViserUrdf, link_name_to_frame: Dict[str, FrameHandle]):
    """Updates the attached coordinate frames to match the robot's current configuration."""
    for link_name, frame in link_name_to_frame.items():
        # Get transform from URDF base to link
        T_world_link = viser_urdf._urdf.get_transform(frame_to=link_name)

        xyz = T_world_link[:3, 3]
        # FIX: .copy() ensures the array is writable/contiguous for SciPy
        rot_matrix = T_world_link[:3, :3].copy()
        xyzw = R.from_matrix(rot_matrix).as_quat()
        wxyz = xyzw[[3, 0, 1, 2]]

        frame.position = xyz
        frame.wxyz = wxyz


# --- Main Editor Logic ---


class SphereEditor:
    def __init__(
        self,
        server: viser.ViserServer,
        viser_urdf: ViserUrdf,
        link_frames: Dict[str, FrameHandle],
        initial_data: Optional[Dict] = None,
    ):
        self.server = server
        self.viser_urdf = viser_urdf
        self.link_frames = link_frames
        self.data: RobotSpheres = {}

        # State for the currently editing sphere
        self.active_gizmo: Optional[TransformControlsHandle] = None
        self.active_sphere_mesh: Optional[SceneNodeHandle] = None
        self.gizmo_base_vertices: Optional[np.ndarray] = None  # For scaling
        self.active_link_name: Optional[str] = None

        # GUI Handles
        self.save_btn: Optional[viser.GuiInputHandle] = None
        self.cancel_btn: Optional[viser.GuiInputHandle] = None
        self.radius_slider: Optional[viser.GuiInputHandle] = None

        if initial_data:
            self._load_from_json(initial_data)

        self._create_gui()

    def _load_from_json(self, json_data: Dict):
        """Populate self.data and visualize from existing JSON."""
        print(f"Loading spheres for {len(json_data)} links...")
        for link_name, sphere_data in json_data.items():
            if link_name not in self.link_frames:
                continue

            if link_name not in self.data:
                self.data[link_name] = LinkSpheres()

            centers = sphere_data.get("centers", [])
            radii = sphere_data.get("radii", [])

            for c, r in zip(centers, radii):
                self.data[link_name].spheres.append(SphereDef(c, r))
                self._visualize_saved_sphere(link_name, c, r)

    def _visualize_saved_sphere(
        self, link_name: str, center: List[float], radius: float
    ):
        """Draws a static green sphere attached to the link frame."""
        # FIX: Use path naming for hierarchy (no parent= arg)
        frame_path = self.link_frames[link_name].name
        name = f"{frame_path}/saved_sphere_{time.time_ns()}"

        # Generate sphere with correct radius
        print(f"Visualizing sphere for {link_name} at {center} with radius {radius}")
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)

        self.server.scene.add_mesh_simple(
            name=name,
            vertices=sphere.vertices,
            faces=sphere.faces,
            position=np.array(center),
            color=(0.0, 1.0, 0.0),  # Green
            opacity=1.0,
        )

    def _create_gui(self):
        """Create the editor controls."""

        export_btn = self.server.gui.add_button("PRINT JSON TO CONSOLE", color="yellow")

        @export_btn.on_click
        def _(_):
            output = {}
            for link, s_list in self.data.items():
                if not s_list.spheres:
                    continue
                output[link] = {
                    "centers": [s.center for s in s_list.spheres],
                    "radii": [s.radius for s in s_list.spheres],
                }
            print("\n" + "=" * 20 + " SPHERES JSON " + "=" * 20)
            print(json.dumps(output, indent=2))
            print("=" * 54 + "\n")

        with self.server.gui.add_folder("Active Edit", visible=True):
            self.save_btn = self.server.gui.add_button(
                "STORE SPHERE", visible=False, color="green"
            )
            self.cancel_btn = self.server.gui.add_button(
                "CANCEL", visible=False, color="red"
            )
            self.radius_slider = self.server.gui.add_slider(
                "Radius",
                min=0.005,
                max=0.3,
                step=0.001,
                initial_value=0.05,
                visible=False,
            )

            @self.radius_slider.on_update
            def _(_):
                # FIX: Update vertices directly to handle scaling
                if self.active_sphere_mesh and self.gizmo_base_vertices is not None:
                    scale = self.radius_slider.value
                    # Use _queue_update as per reference script
                    self.active_sphere_mesh._queue_update(
                        "vertices", self.active_sphere_mesh.vertices * scale
                    )

            @self.save_btn.on_click
            def _(_):
                self._save_active_sphere()

            @self.cancel_btn.on_click
            def _(_):
                self._cleanup_active_gizmo()

        with self.server.gui.add_folder("Link Selector"):
            sorted_links = sorted(self.link_frames.keys())
            for link_name in sorted_links:
                btn = self.server.gui.add_button(f"Add Sphere: {link_name}")

                def make_handler(lname):
                    return lambda _: self._start_editing_sphere(lname)

                btn.on_click(make_handler(link_name))

    def _start_editing_sphere(self, link_name: str):
        """Spawn a gizmo sphere on the specified link."""
        self._cleanup_active_gizmo()

        self.active_link_name = link_name

        # FIX: Path hierarchy
        frame_path = self.link_frames[link_name].name
        gizmo_name = f"{frame_path}/gizmo_{link_name}"

        print(f"Editing new sphere for {link_name} at {gizmo_name}")

        # 1. Create Transform Control (Gizmo)
        self.active_gizmo = self.server.scene.add_transform_controls(
            name=gizmo_name,
            position=(0.0, 0.0, 0.0),
            scale=0.15,
        )

        # 2. Attach visual sphere to the gizmo (Radius = 1.0 for easy scaling)
        sphere = trimesh.creation.icosphere(radius=1.0, subdivisions=2)
        # Store base vertices for scaling calculations
        self.gizmo_base_vertices = sphere.vertices.copy()

        self.active_sphere_mesh = self.server.scene.add_mesh_simple(
            name=f"{gizmo_name}/visual",
            vertices=sphere.vertices,
            faces=sphere.faces,
            color=(0.0, 0.5, 1.0),  # Blue
            opacity=1.0,
        )

        # 3. Show Edit Controls
        self.save_btn.visible = True
        self.cancel_btn.visible = True
        self.radius_slider.visible = True
        self.radius_slider.value = 0.05  # Triggers update

    def _save_active_sphere(self):
        """Commit the active sphere to data and make it static."""
        if not self.active_link_name or not self.active_gizmo:
            return

        pos = self.active_gizmo.position
        radius = self.radius_slider.value
        center_list = [float(pos[0]), float(pos[1]), float(pos[2])]

        if self.active_link_name not in self.data:
            self.data[self.active_link_name] = LinkSpheres()

        self.data[self.active_link_name].spheres.append(
            SphereDef(center_list, float(radius))
        )
        self._visualize_saved_sphere(self.active_link_name, center_list, radius)

        print(f"Saved sphere for {self.active_link_name}: c={center_list}, r={radius}")
        self._cleanup_active_gizmo()

    def _cleanup_active_gizmo(self):
        if self.active_gizmo:
            self.active_gizmo.remove()
            self.active_gizmo = None

        if self.active_sphere_mesh:
            self.active_sphere_mesh.remove()
            self.active_sphere_mesh = None
            self.gizmo_base_vertices = None

        self.active_link_name = None
        self.save_btn.visible = False
        self.cancel_btn.visible = False
        self.radius_slider.visible = False


def main(
    urdf_path: Path = (
        get_repo_root_dir()
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    ),
    spheres_json_path: Optional[Path] = Path(__file__).parent
    / "assets"
    / "sharpa_spheres.json",
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:

    server = viser.ViserServer()

    @server.on_client_connect
    def _(client):
        DIST = 0.5
        client.camera.position = (DIST, DIST, DIST)
        client.camera.look_at = (0.0, 0.0, 0.0)

    assert urdf_path.exists(), f"URDF path {urdf_path} does not exist"
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        root_node_name="/robot",
        mesh_color_override=(0.0, 0.0, 1.0, 0.2),  # Make robot blue and translucent
    )

    # Attach frames to every link
    link_name_to_frame = {}
    AXES_LENGTH = 0.05
    AXES_RADIUS = 0.002
    HIDE_AXES = True
    if HIDE_AXES:
        AXES_LENGTH = 0.00001
        AXES_RADIUS = 0.00001

    for link_name in viser_urdf._urdf.link_map.keys():
        link_name_to_frame[link_name] = server.scene.add_frame(
            f"/robot/{link_name}",
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )

    with server.gui.add_folder("Joint Control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf, link_name_to_frame
        )

    # Override initial config with home joint positions
    HOME_JOINT_POS_IIWA = np.array(
        [
            -1.571,
            1.571 - np.deg2rad(10),
            -0.000,
            1.376 + np.deg2rad(10),
            -0.000,
            1.485,
            1.308,
        ]
    )
    HOME_JOINT_POS_SHARPA = np.zeros(22)
    HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])
    initial_config = HOME_JOINT_POS

    viser_urdf.update_cfg(np.array(initial_config))
    update_frames(viser_urdf, link_name_to_frame)

    initial_data = {}
    if spheres_json_path and spheres_json_path.exists():
        with open(spheres_json_path, "r") as f:
            initial_data = json.load(f)
        print(f"Loaded existing spheres from {spheres_json_path}")

    _editor = SphereEditor(
        server=server,
        viser_urdf=viser_urdf,
        link_frames=link_name_to_frame,
        initial_data=initial_data,
    )

    server.scene.add_grid("/grid", width=2, height=2)

    reset_button = server.gui.add_button("Reset Joints")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    print("Ready! Open the browser URL to start editing.")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
